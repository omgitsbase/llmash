#include "rco.h"

#include "gsq.h"
#include "platform.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <thread>

namespace fs = std::filesystem;

namespace llmash {
namespace rco {

namespace {

typedef void (*to_float_fn)(const void *, float *, int64_t);

// ggml's type traits. The loader checks it against known answers before any
// of it is trusted, since this is read out of a DLL the runtime ships.
struct Traits {
    const char * type_name;
    int64_t      blck_size;
    int64_t      blck_size_interleave;
    size_t       type_size;
    bool         is_quantized;
    to_float_fn  to_float;
    void *       from_float_ref;
};

using traits_fn = const Traits * (*) (int);
using quant_fn  = size_t (*) (int, const float *, void *, int64_t, int64_t, int64_t, const float *);
using row_fn    = size_t (*) (int, int64_t);
using blck_fn   = int64_t (*) (int);
using tsize_fn  = size_t (*) (int);

void * open_lib(const std::string & path) {
#ifdef _WIN32
    // ALTERED_SEARCH_PATH so the runtime's other DLLs resolve from beside it
    // rather than from wherever llmash happens to be.
    return LoadLibraryExA(path.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
#else
    return dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
}

void close_lib(void * h) {
#ifdef _WIN32
    FreeLibrary(static_cast<HMODULE>(h));
#else
    dlclose(h);
#endif
}

void * symbol(void * h, const char * name) {
#ifdef _WIN32
    return reinterpret_cast<void *>(GetProcAddress(static_cast<HMODULE>(h), name));
#else
    return dlsym(h, name);
#endif
}

const char * ggml_base_name() {
#ifdef _WIN32
    return "ggml-base.dll";
#else
    return "libggml-base.so";
#endif
}

} // namespace

Ggml::~Ggml() {
    if (lib_ != nullptr) {
        close_lib(lib_);
    }
}

bool Ggml::load(const Config & cfg, std::string & err) {
    if (cfg.llama_bin.empty()) {
        err = "no llama.cpp runtime is configured";
        return false;
    }
    const fs::path  lib = fs::path(cfg.llama_bin).parent_path() / ggml_base_name();
    std::error_code ec;
    if (!fs::is_regular_file(lib, ec)) {
        err = std::string(ggml_base_name()) + " is not beside llama-server (" + lib.string() + ")";
        return false;
    }
    lib_ = open_lib(fs::absolute(lib, ec).string());
    if (lib_ == nullptr) {
        err = "could not load " + lib.string();
        return false;
    }
    traits_ = symbol(lib_, "ggml_get_type_traits");
    quant_  = symbol(lib_, "ggml_quantize_chunk");
    row_    = symbol(lib_, "ggml_row_size");
    blck_   = symbol(lib_, "ggml_blck_size");
    tsize_  = symbol(lib_, "ggml_type_size");
    if (traits_ == nullptr || quant_ == nullptr || row_ == nullptr || blck_ == nullptr || tsize_ == nullptr) {
        err = "this runtime's ggml does not export the quantizer";
        close_lib(lib_);
        lib_ = nullptr;
        return false;
    }

    // What every ggml answers for these, so a changed layout is caught here
    // rather than as nonsense further on.
    struct Known {
        int          type;
        const char * name;
        int64_t      block;
    };
    for (const Known & k : {Known{GT_F32, "f32", 1}, Known{GT_Q8_0, "q8_0", 32}, Known{GT_Q4_K, "q4_K", 256},
                            Known{GT_IQ4_XS, "iq4_xs", 256}}) {
        const Traits * t = reinterpret_cast<traits_fn>(traits_)(k.type);
        if (t == nullptr || t->type_name == nullptr || std::strcmp(t->type_name, k.name) != 0 ||
            t->blck_size != k.block || block(k.type) != k.block || type_size(k.type) != t->type_size) {
            err = "this runtime's ggml is laid out differently than expected";
            close_lib(lib_);
            lib_ = nullptr;
            return false;
        }
    }
    return true;
}

int64_t Ggml::block(int type) const { return reinterpret_cast<blck_fn>(blck_)(type); }
size_t  Ggml::type_size(int type) const { return reinterpret_cast<tsize_fn>(tsize_)(type); }
size_t  Ggml::row_size(int type, int64_t n) const { return reinterpret_cast<row_fn>(row_)(type, n); }

const char * Ggml::name(int type) const {
    const Traits * t = reinterpret_cast<traits_fn>(traits_)(type);
    return t != nullptr && t->type_name != nullptr ? t->type_name : "?";
}

bool Ggml::quantized(int type) const {
    const Traits * t = reinterpret_cast<traits_fn>(traits_)(type);
    return t != nullptr && t->is_quantized;
}

size_t Ggml::quantize(int type, const float * src, void * dst, int64_t nrows, int64_t n_per_row,
                      const float * imatrix) const {
    return reinterpret_cast<quant_fn>(quant_)(type, src, dst, 0, nrows, n_per_row, imatrix);
}

void Ggml::dequantize(int type, const void * src, float * dst, int64_t n) const {
    const Traits * t = reinterpret_cast<traits_fn>(traits_)(type);
    if (t == nullptr || t->to_float == nullptr) {
        std::memset(dst, 0, static_cast<size_t>(n) * sizeof(float));
        return;
    }
    t->to_float(src, dst, n);
}

const std::vector<int> & candidates() {
    // Every type IST-DASLab's own allocation uses, plus q8_0 and q5_K above
    // them: the point of a budget is to spend it unevenly, so the tensors
    // that need the source's own precision have to be able to keep it.
    static const std::vector<int> all = {GT_Q8_0,  GT_Q6_K,   GT_Q5_K,    GT_Q4_K,  GT_IQ4_XS,  GT_IQ3_S,
                                         GT_Q2_K,  GT_IQ2_S,  GT_IQ3_XXS, GT_IQ2_XS, GT_IQ2_XXS, GT_IQ1_M};
    return all;
}

double bits_of_type(const Ggml & g, int type) {
    const int64_t b = g.block(type);
    return b > 0 ? static_cast<double>(g.type_size(type)) * 8.0 / static_cast<double>(b) : 32.0;
}

std::vector<Cost> measure(const Ggml & g, const float * data, int64_t nrows, int64_t n_per_row,
                          const std::vector<int> & types, const float * imatrix, int64_t sample_rows,
                          int nthread) {
    std::vector<Cost> out;
    if (nrows <= 0 || n_per_row <= 0) {
        return out;
    }
    const int64_t take = (std::min)(nrows, (std::max<int64_t>)(sample_rows, 1));
    const int64_t step = nrows / take;

    // The sampled rows, gathered so the quantizer sees them as one block.
    std::vector<float> sample(static_cast<size_t>(take * n_per_row));
    for (int64_t i = 0; i < take; i++) {
        std::memcpy(sample.data() + i * n_per_row, data + (i * step) * n_per_row,
                    static_cast<size_t>(n_per_row) * sizeof(float));
    }

    // Weighted by the imatrix: an error on a channel the model barely
    // activates is not the same as one on a channel it leans on, and
    // without that every tensor scores alike and the allocation comes out
    // uniform.
    double norm = 0;
    for (int64_t i = 0; i < take * n_per_row; i++) {
        const double w = imatrix != nullptr ? imatrix[i % n_per_row] : 1.0;
        norm += w * static_cast<double>(sample[i]) * sample[i];
    }
    if (norm <= 0) {
        norm = 1;
    }

    out.resize(types.size());
    const int workers = (std::max)(1, (std::min)(nthread, static_cast<int>(types.size())));
    std::vector<std::thread> pool;
    std::atomic<size_t>      next{0};
    for (int w = 0; w < workers; w++) {
        pool.emplace_back([&] {
            std::vector<unsigned char> q;
            std::vector<float>         back(static_cast<size_t>(take * n_per_row));
            for (;;) {
                const size_t k = next.fetch_add(1);
                if (k >= types.size()) {
                    return;
                }
                const int type = types[k];
                q.assign(g.row_size(type, n_per_row) * static_cast<size_t>(take) + 64, 0);
                g.quantize(type, sample.data(), q.data(), take, n_per_row, imatrix);
                g.dequantize(type, q.data(), back.data(), take * n_per_row);
                double e = 0;
                for (int64_t i = 0; i < take * n_per_row; i++) {
                    const double d = static_cast<double>(sample[i]) - back[i];
                    e += (imatrix != nullptr ? imatrix[i % n_per_row] : 1.0) * d * d;
                }
                out[k].type  = type;
                out[k].error = e / norm;
                out[k].bytes = static_cast<int64_t>(g.row_size(type, n_per_row)) * nrows;
            }
        });
    }
    for (auto & t : pool) {
        t.join();
    }
    // Widest first, by what the types actually weigh rather than by the
    // order they were listed in.
    std::sort(out.begin(), out.end(), [](const Cost & a, const Cost & b) { return a.bytes > b.bytes; });
    return out;
}

int pick(const Measured & m, double lambda) {
    int    best = 0;
    double low  = 0;
    for (size_t i = 0; i < m.costs.size(); i++) {
        // The error is relative, so a tensor's say in the total is its share
        // of the weights; without that every norm would argue as loudly as a
        // projection a thousand times its size.
        const double c = m.costs[i].error * static_cast<double>(m.elements) +
                         lambda * static_cast<double>(m.costs[i].bytes);
        if (i == 0 || c < low) {
            low  = c;
            best = m.costs[i].type;
        }
    }
    return best;
}

std::map<std::string, int> allocate(const std::vector<Measured> & m, int64_t budget_bytes) {
    const double               lambda = solve_lambda(m, budget_bytes);
    std::map<std::string, int> out;
    int64_t                    spent = 0;
    for (const Measured & one : m) {
        const int t = pick(one, lambda);
        out[one.name] = t;
        for (const Cost & c : one.costs) {
            if (c.type == t) {
                spent += c.bytes;
            }
        }
    }

    // Whatever the ladder's steps left over goes to the tensors that gain
    // the most error back per byte.
    for (;;) {
        const Measured * best   = nullptr;
        int              to     = 0;
        int64_t          cost   = 0;
        double           gain   = 0;
        for (const Measured & one : m) {
            const int now = out[one.name];
            double    e_now = 0;
            int64_t   b_now = 0;
            for (const Cost & c : one.costs) {
                if (c.type == now) {
                    e_now = c.error;
                    b_now = c.bytes;
                }
            }
            for (const Cost & c : one.costs) {
                if (c.bytes <= b_now || spent + (c.bytes - b_now) > budget_bytes) {
                    continue;
                }
                const double g = (e_now - c.error) * static_cast<double>(one.elements) /
                                 static_cast<double>(c.bytes - b_now);
                if (g > gain) {
                    gain = g;
                    best = &one;
                    to   = c.type;
                    cost = c.bytes - b_now;
                }
            }
        }
        if (best == nullptr) {
            return out;
        }
        out[best->name] = to;
        spent += cost;
    }
}

double solve_lambda(const std::vector<Measured> & m, int64_t budget_bytes) {
    const auto total_at = [&](double lambda) {
        int64_t sum = 0;
        for (const Measured & one : m) {
            const int t = pick(one, lambda);
            for (const Cost & c : one.costs) {
                if (c.type == t) {
                    sum += c.bytes;
                    break;
                }
            }
        }
        return sum;
    };

    // Larger lambda buys fewer bits, so the total falls as it rises.
    double lo = 1e-12, hi = 1e-12;
    while (total_at(hi) > budget_bytes && hi < 1e12) {
        lo = hi;
        hi *= 4;
    }
    if (total_at(hi) > budget_bytes) {
        return hi;
    }
    for (int i = 0; i < 60; i++) {
        const double mid = std::sqrt(lo * hi);
        if (total_at(mid) > budget_bytes) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    return hi;
}

namespace {

// llama.cpp's older imatrix: a count, then per entry a name, the number of
// calls it was summed over, and the sums themselves.
bool load_legacy(const std::string & path, std::map<std::string, std::vector<float>> & out) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return false;
    }
    int32_t entries = 0;
    in.read(reinterpret_cast<char *>(&entries), 4);
    if (!in || entries <= 0 || entries > 100000) {
        return false;
    }
    for (int32_t i = 0; i < entries; i++) {
        int32_t len = 0;
        in.read(reinterpret_cast<char *>(&len), 4);
        if (!in || len <= 0 || len > 4096) {
            return false;
        }
        std::string name(static_cast<size_t>(len), '\0');
        in.read(&name[0], len);
        int32_t ncall = 0, nval = 0;
        in.read(reinterpret_cast<char *>(&ncall), 4);
        in.read(reinterpret_cast<char *>(&nval), 4);
        if (!in || nval <= 0 || nval > (1 << 26)) {
            return false;
        }
        std::vector<float> v(static_cast<size_t>(nval));
        in.read(reinterpret_cast<char *>(v.data()), static_cast<std::streamsize>(nval) * 4);
        if (!in) {
            return false;
        }
        if (ncall > 0) {
            for (float & x : v) {
                x /= static_cast<float>(ncall);
            }
        }
        out[name] = std::move(v);
    }
    return !out.empty();
}

} // namespace

bool Imatrix::load(const std::string & path, std::string & err) {
    {
        std::ifstream probe(path, std::ios::binary);
        char          magic[4] = {};
        probe.read(magic, 4);
        if (std::memcmp(magic, "GGUF", 4) != 0) {
            if (load_legacy(path, by_tensor_)) {
                return true;
            }
            err = "could not read " + path + " as an imatrix";
            return false;
        }
    }
    const gsq::Layout l = gsq::read_layout(path);
    if (!l.error.empty()) {
        err = l.error;
        return false;
    }
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        err = "could not open " + path;
        return false;
    }

    const auto read_floats = [&](const gsq::TensorEntry & t) {
        std::vector<float> v(static_cast<size_t>(t.bytes / 4));
        in.seekg(l.data_start.front() + t.offset);
        in.read(reinterpret_cast<char *>(v.data()), t.bytes);
        return v;
    };

    std::map<std::string, std::vector<float>> counts;
    for (const gsq::TensorEntry & t : l.tensors) {
        if (t.type == GT_F32 && t.name.size() > 7 && t.name.compare(t.name.size() - 7, 7, ".counts") == 0) {
            counts[t.name.substr(0, t.name.size() - 7)] = read_floats(t);
        }
    }
    for (const gsq::TensorEntry & t : l.tensors) {
        const std::string suffix = ".in_sum2";
        if (t.type != GT_F32 || t.name.size() <= suffix.size() ||
            t.name.compare(t.name.size() - suffix.size(), suffix.size(), suffix) != 0) {
            continue;
        }
        const std::string base = t.name.substr(0, t.name.size() - suffix.size());
        std::vector<float> v   = read_floats(t);
        const auto         c   = counts.find(base);
        if (c != counts.end() && !c->second.empty()) {
            const int64_t per = static_cast<int64_t>(v.size()) / static_cast<int64_t>(c->second.size());
            for (size_t i = 0; i < v.size(); i++) {
                const float n = c->second[per > 0 ? static_cast<size_t>(i / per) : 0];
                if (n > 0) {
                    v[i] /= n;
                }
            }
        }
        by_tensor_[base] = std::move(v);
    }
    if (by_tensor_.empty()) {
        err = "no imatrix entries in " + path;
        return false;
    }
    return true;
}

const float * Imatrix::of(const std::string & tensor, int64_t n_per_row, int64_t expert) const {
    const auto it = by_tensor_.find(tensor);
    if (it == by_tensor_.end()) {
        return nullptr;
    }
    const int64_t want = n_per_row * (expert + 1);
    if (static_cast<int64_t>(it->second.size()) < want) {
        return static_cast<int64_t>(it->second.size()) >= n_per_row ? it->second.data() : nullptr;
    }
    return it->second.data() + n_per_row * expert;
}

} // namespace rco
} // namespace llmash
