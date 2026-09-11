#include "gsq.h"

#include "cli_process.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>

namespace fs = std::filesystem;

namespace llmash {
namespace gsq {

namespace {

std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

std::string upper(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
    return s;
}

std::string trim(const std::string & s) {
    const size_t b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) {
        return "";
    }
    return s.substr(b, s.find_last_not_of(" \t\r\n") - b + 1);
}

// The size in bytes of a tensor's data, from its ggml type and dims.
struct TypeInfo {
    int64_t block = 1;  // elements per block
    int64_t size  = 4;  // bytes per block
};

TypeInfo type_info(uint32_t t) {
    switch (t) {
        case 0:  return {1, 4};    // F32
        case 1:  return {1, 2};    // F16
        case 2:  return {32, 18};  // Q4_0
        case 3:  return {32, 20};  // Q4_1
        case 6:  return {32, 22};  // Q5_0
        case 7:  return {32, 24};  // Q5_1
        case 8:  return {32, 34};  // Q8_0
        case 9:  return {32, 36};  // Q8_1
        case 10: return {256, 84};   // Q2_K
        case 11: return {256, 110};  // Q3_K
        case 12: return {256, 144};  // Q4_K
        case 13: return {256, 176};  // Q5_K
        case 14: return {256, 210};  // Q6_K
        case 15: return {256, 292};  // Q8_K
        case 16: return {256, 66};   // IQ2_XXS
        case 17: return {256, 74};   // IQ2_XS
        case 18: return {256, 66};   // IQ3_XXS
        case 19: return {256, 50};   // IQ1_S
        case 20: return {32, 18};    // IQ4_NL
        case 21: return {256, 82};   // IQ3_S
        case 22: return {256, 82};   // IQ2_S
        case 23: return {256, 110};  // IQ4_XS
        case 24: return {1, 1};      // I8
        case 25: return {1, 2};      // I16
        case 26: return {1, 4};      // I32
        case 27: return {1, 8};      // I64
        case 28: return {1, 8};      // F64
        case 29: return {256, 56};   // IQ1_M
        case 30: return {1, 2};      // BF16
        case 39: return {32, 17};    // MXFP4
        default: return {1, 4};
    }
}

int64_t tensor_bytes(const TensorEntry & t) {
    int64_t n = 1;
    for (const int64_t d : t.dims) {
        n *= d;
    }
    const TypeInfo ti = type_info(t.type);
    return ti.block > 0 ? n / ti.block * ti.size : n * ti.size;
}

int64_t align_up(int64_t v, int64_t a) { return (v + a - 1) / a * a; }

// A cursor over a header held in memory, so the same code reads a local file
// and the first megabytes of a remote one.
class Cursor {
public:
    Cursor(const char * p, size_t n) : p_(p), n_(n) {}

    bool     bad() const { return bad_; }
    size_t   at() const { return i_; }
    uint32_t u32() { return read<uint32_t>(); }
    uint64_t u64() { return read<uint64_t>(); }

    std::string str() {
        const uint64_t n = u64();
        if (bad_ || n > (64ull << 20) || i_ + n > n_) {
            bad_ = true;
            return "";
        }
        std::string s(p_ + i_, static_cast<size_t>(n));
        i_ += static_cast<size_t>(n);
        return s;
    }

    void skip(size_t n) {
        if (i_ + n > n_) {
            bad_ = true;
            return;
        }
        i_ += n;
    }

    void skip_value(uint32_t t) {
        static const int fixed[] = {1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8};
        if (t <= 7 || (t >= 10 && t <= 12)) {
            skip(static_cast<size_t>(fixed[t]));
            return;
        }
        if (t == 8) {
            str();
            return;
        }
        if (t == 9) {
            const uint32_t et = u32();
            const uint64_t n  = u64();
            for (uint64_t k = 0; k < n && !bad_; k++) {
                skip_value(et);
            }
            return;
        }
        bad_ = true;
    }

private:
    template <typename T> T read() {
        if (i_ + sizeof(T) > n_) {
            bad_ = true;
            return 0;
        }
        T v;
        std::memcpy(&v, p_ + i_, sizeof(T));
        i_ += sizeof(T);
        return v;
    }

    const char * p_;
    size_t       n_;
    size_t       i_   = 0;
    bool         bad_ = false;
};

} // namespace

// ============================================================ allocations

const std::vector<Allocation> & known() {
    static const std::vector<Allocation> all = {
        {"GSQ-RCO 3.5 bpw (task-lossless)", "ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF",
         "tensor-allocation/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.rco-allocation.txt", "imatrix-qwen3.8-27b.gguf", 3.5,
         "qwen35", 5120, "IQ3_S"},
        {"GSQ-RCO 3.0 bpw", "ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF",
         "tensor-allocation/Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.rco-allocation.txt", "imatrix-qwen3.8-27b.gguf", 3.0,
         "qwen35", 5120, "IQ3_XXS"},
    };
    return all;
}

std::string quant_name(double bpw) {
    char buf[32];
    if (bpw == static_cast<int>(bpw)) {
        std::snprintf(buf, sizeof(buf), "GSQ-%d", static_cast<int>(bpw));
    } else {
        std::snprintf(buf, sizeof(buf), "GSQ-%.1f", bpw);
    }
    return buf;
}

double bpw_of_quant(const std::string & quant) {
    static const std::regex re(R"(gsq[-_]?([0-9]+(?:\.[0-9]+)?)?)");
    std::smatch             m;
    const std::string       low = lower(trim(quant));
    if (!std::regex_match(low, m, re)) {
        return 0.0;
    }
    if (!m[1].matched) {
        return 3.5;
    }
    try {
        return std::stod(m[1].str());
    } catch (const std::exception &) {
        return 0.0;
    }
}

std::vector<Allocation> all_for_arch(const std::string & arch, int64_t embd) {
    std::vector<Allocation> out;
    for (const Allocation & a : known()) {
        if (a.arch == arch && a.embd == embd) {
            out.push_back(a);
        }
    }
    std::stable_sort(out.begin(), out.end(), [](const Allocation & a, const Allocation & b) { return a.bpw > b.bpw; });
    return out;
}

const Allocation * for_model(const std::string & arch, int64_t embd, double bpw) {
    for (const Allocation & a : known()) {
        if (a.arch == arch && a.embd == embd && std::abs(a.bpw - bpw) < 0.01) {
            return &a;
        }
    }
    return nullptr;
}

const std::vector<std::string> & source_preference() {
    static const std::vector<std::string> pref = {"q8_0", "bf16", "f16", "q6_k", "q5_k_m"};
    return pref;
}

std::map<std::string, std::string> parse_allocation(const std::string & text) {
    std::map<std::string, std::string> out;
    std::istringstream                 in(text);
    std::string                        line;
    static const std::regex            re(R"(^(\S+):\s+(\S+)$)");
    while (std::getline(in, line)) {
        const std::string s = trim(line);
        std::smatch       m;
        if (!s.empty() && s[0] != '#' && std::regex_match(s, m, re)) {
            out[m[1].str()] = lower(m[2].str());
        }
    }
    return out;
}

// ================================================================== GGUF

namespace {

// Reads one GGUF header out of `head`. `keep_kv` is false for a shard after
// the first, whose metadata is only split bookkeeping.
bool read_header(const std::string & head, Layout & l, bool keep_kv, std::string & err) {
    if (head.size() < 24 || std::memcmp(head.data(), "GGUF", 4) != 0) {
        err = "not a GGUF file";
        return false;
    }
    Cursor c(head.data(), head.size());
    c.skip(4);
    const uint32_t version = c.u32();
    const uint64_t n_tens  = c.u64();
    const uint64_t n_kv    = c.u64();
    if (keep_kv) {
        l.version = version;
    }
    for (uint64_t i = 0; i < n_kv && !c.bad(); i++) {
        const size_t      from = c.at();
        const std::string key  = c.str();
        c.skip_value(c.u32());
        if (c.bad()) {
            break;
        }
        // A chunk is its own little model, so the split bookkeeping goes: it
        // would send llama-quantize looking for shards that are not there.
        if (keep_kv && key.rfind("split.", 0) != 0) {
            l.kv.push_back(KvEntry{key, head.substr(from, c.at() - from)});
        }
    }
    if (c.bad()) {
        err = "its header is longer than what was read";
        return false;
    }

    const size_t part = l.data_start.size();
    for (uint64_t i = 0; i < n_tens && !c.bad(); i++) {
        TensorEntry t;
        t.name            = c.str();
        const uint32_t nd = c.u32();
        if (c.bad() || nd > 4) {
            err = "a tensor with " + std::to_string(nd) + " dimensions";
            return false;
        }
        for (uint32_t d = 0; d < nd; d++) {
            t.dims.push_back(static_cast<int64_t>(c.u64()));
        }
        t.type   = c.u32();
        t.offset = static_cast<int64_t>(c.u64());
        t.bytes  = tensor_bytes(t);
        t.part   = part;
        l.tensors.push_back(std::move(t));
    }
    if (c.bad()) {
        err = "its header is longer than what was read";
        return false;
    }
    l.data_start.push_back(align_up(static_cast<int64_t>(c.at()), l.align));
    return true;
}

} // namespace

Layout layout_from(const std::string & head) {
    Layout l;
    read_header(head, l, true, l.error);
    return l;
}

bool append_part(Layout & l, const std::string & head) {
    std::string err;
    if (!read_header(head, l, false, err)) {
        l.error = err;
        return false;
    }
    return true;
}

Layout read_layout(const std::string & path) {
    Layout        l;
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        l.error = "could not open " + path;
        return l;
    }
    // Big enough for any header that precedes the tensor data.
    std::string head(64ull << 20, '\0');
    in.read(&head[0], static_cast<std::streamsize>(head.size()));
    head.resize(static_cast<size_t>(in.gcount()));
    return layout_from(head);
}

std::vector<Chunk> plan_chunks(const Layout & l, int64_t target_bytes) {
    std::vector<Chunk> out;
    if (l.tensors.empty()) {
        return out;
    }
    Chunk cur{0, 0, 0};
    for (size_t i = 0; i < l.tensors.size(); i++) {
        cur.last = i;
        cur.bytes += align_up(l.tensors[i].bytes, l.align);
        if (cur.bytes >= target_bytes && i + 1 < l.tensors.size()) {
            out.push_back(cur);
            cur = Chunk{i + 1, i + 1, 0};
        }
    }
    if (cur.bytes > 0 || out.empty()) {
        out.push_back(cur);
    }
    return out;
}

namespace {

void put_u32(std::string & s, uint32_t v) { s.append(reinterpret_cast<const char *>(&v), 4); }
void put_u64(std::string & s, uint64_t v) { s.append(reinterpret_cast<const char *>(&v), 8); }

void put_str(std::string & s, const std::string & v) {
    put_u64(s, v.size());
    s += v;
}

std::string chunk_header(const Layout & l, const Chunk & c) {
    std::string h = "GGUF";
    put_u32(h, l.version);
    put_u64(h, static_cast<uint64_t>(c.last - c.first + 1));
    put_u64(h, static_cast<uint64_t>(l.kv.size()));
    for (const KvEntry & e : l.kv) {
        h += e.raw;
    }
    int64_t cursor = 0;
    for (size_t i = c.first; i <= c.last; i++) {
        const TensorEntry & t = l.tensors[i];
        put_str(h, t.name);
        put_u32(h, static_cast<uint32_t>(t.dims.size()));
        for (const int64_t d : t.dims) {
            put_u64(h, static_cast<uint64_t>(d));
        }
        put_u32(h, t.type);
        put_u64(h, static_cast<uint64_t>(cursor));
        cursor += align_up(t.bytes, l.align);
    }
    h.append(static_cast<size_t>(align_up(static_cast<int64_t>(h.size()), l.align)) - h.size(), '\0');
    return h;
}

} // namespace

int64_t write_chunk_header(const std::string & path, const Layout & l, const Chunk & c, std::string & err) {
    const std::string h = chunk_header(l, c);
    std::ofstream     out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        err = "could not create " + path;
        return -1;
    }
    out.write(h.data(), static_cast<std::streamsize>(h.size()));
    if (!out) {
        err = "could not write " + path;
        return -1;
    }
    return static_cast<int64_t>(h.size());
}

std::vector<Piece> chunk_pieces(const Layout & l, const Chunk & c, int64_t header_bytes) {
    std::vector<Piece> out;
    int64_t            into = header_bytes;
    for (size_t i = c.first; i <= c.last; i++) {
        const TensorEntry & t = l.tensors[i];
        out.push_back(Piece{t.part, l.data_start[t.part] + t.offset, t.bytes, into});
        into += align_up(t.bytes, l.align);
    }
    return out;
}

bool assemble(const std::vector<std::string> & chunks, const std::string & dest, const ProgressFn & progress,
              std::string & err) {
    if (chunks.empty()) {
        err = "nothing to assemble";
        return false;
    }
    // Every chunk carries the same metadata; the whole model is the first
    // one's header with all of their tensors in it.
    std::vector<Layout> parts;
    for (const std::string & p : chunks) {
        Layout l = read_layout(p);
        if (!l.error.empty()) {
            err = fs::path(p).filename().string() + ": " + l.error;
            return false;
        }
        parts.push_back(std::move(l));
    }

    Layout whole = parts.front();
    whole.tensors.clear();
    for (const Layout & p : parts) {
        for (const TensorEntry & t : p.tensors) {
            whole.tensors.push_back(t);
        }
    }
    const Chunk       all{0, whole.tensors.size() - 1, 0};
    const std::string head = chunk_header(whole, all);

    std::ofstream out(dest, std::ios::binary | std::ios::trunc);
    if (!out) {
        err = "could not create " + dest;
        return false;
    }
    out.write(head.data(), static_cast<std::streamsize>(head.size()));

    std::vector<char> buf(4 << 20);
    int64_t           done = 0;
    std::error_code   ec;
    for (size_t i = 0; i < chunks.size(); i++) {
        std::ifstream in(chunks[i], std::ios::binary);
        if (!in) {
            err = "could not read " + chunks[i];
            return false;
        }
        in.seekg(parts[i].data_start.front());
        for (;;) {
            in.read(buf.data(), static_cast<std::streamsize>(buf.size()));
            const std::streamsize n = in.gcount();
            if (n <= 0) {
                break;
            }
            out.write(buf.data(), n);
            if (!out) {
                err = "could not write " + dest;
                return false;
            }
            done += n;
            if (progress) {
                progress(done);
            }
        }
        in.close();
        // The point of chunking: the pieces never all exist beside the whole.
        fs::remove(chunks[i], ec);
    }
    out.close();
    return true;
}

// ============================================================ quantizing

Plan plan_types(const Layout & l, const std::map<std::string, std::string> & alloc) {
    static const char * keep_full[] = {"F32", "BF16", "F16"};
    Plan                p;
    for (const TensorEntry & t : l.tensors) {
        if (t.dims.size() < 2 || t.name.size() < 7 || t.name.compare(t.name.size() - 7, 7, ".weight") != 0) {
            continue;
        }
        p.total++;
        const auto it = alloc.find(t.name);
        if (it == alloc.end()) {
            continue;
        }
        p.covered++;
        const std::string want = upper(it->second);
        bool              full = false;
        for (const char * k : keep_full) {
            if (want == k) {
                full = true;
            }
        }
        if (full) {
            continue; // llama-quantize rejects these in a type file
        }
        p.lines.push_back(t.name + "=" + it->second);
        p.by_type[want]++;
    }
    return p;
}

std::string quantize_exe(const Config & cfg) {
    if (cfg.llama_bin.empty()) {
        return "";
    }
#ifdef _WIN32
    const char * exe = "llama-quantize.exe";
#else
    const char * exe = "llama-quantize";
#endif
    const fs::path  p = fs::path(cfg.llama_bin).parent_path() / exe;
    std::error_code ec;
    return fs::is_regular_file(p, ec) ? p.string() : std::string();
}

bool quantize(const std::string & exe, const std::string & src, const std::string & dst,
              const std::string & type_file, const std::string & imatrix, const std::string & fallback,
              const std::function<void(int, int)> & progress, std::string & err) {
    std::vector<std::string> argv{exe, "--allow-requantize"};
    if (!imatrix.empty()) {
        argv.push_back("--imatrix");
        argv.push_back(imatrix);
    }
    argv.push_back("--tensor-type-file");
    argv.push_back(type_file);
    argv.push_back(src);
    argv.push_back(dst);
    argv.push_back(fallback);

    static const std::regex step(R"(^\[\s*(\d+)/\s*(\d+)\])");
    std::vector<std::string> tail;
    const int code = run_streaming(argv, [&](const std::string & line) {
        tail.push_back(line);
        if (tail.size() > 40) {
            tail.erase(tail.begin());
        }
        std::smatch m;
        const std::string s = trim(line);
        if (progress && std::regex_search(s, m, step)) {
            progress(std::stoi(m[1].str()), std::stoi(m[2].str()));
        }
    });
    if (code != 0) {
        err = "llama-quantize failed";
        for (size_t i = tail.size() > 6 ? tail.size() - 6 : 0; i < tail.size(); i++) {
            err += "\n  " + tail[i];
        }
        return false;
    }
    return true;
}

} // namespace gsq
} // namespace llmash
