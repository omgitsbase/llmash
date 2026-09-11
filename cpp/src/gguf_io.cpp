#include "gguf_io.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <ostream>

namespace llmash {
namespace ggufio {

namespace {

struct TypeInfo {
    int64_t block = 1; // elements per block
    int64_t size  = 4; // bytes per block
};

TypeInfo type_info(uint32_t t) {
    switch (t) {
        case 0:  return {1, 4};      // F32
        case 1:  return {1, 2};      // F16
        case 2:  return {32, 18};    // Q4_0
        case 3:  return {32, 20};    // Q4_1
        case 6:  return {32, 22};    // Q5_0
        case 7:  return {32, 24};    // Q5_1
        case 8:  return {32, 34};    // Q8_0
        case 9:  return {32, 36};    // Q8_1
        case 10: return {256, 84};   // Q2_K
        case 11: return {256, 110};  // Q3_K
        case 12: return {256, 144};  // Q4_K
        case 13: return {256, 176};  // Q5_K
        case 14: return {256, 210};  // Q6_K
        case 15: return {256, 292};  // Q8_K
        case 16: return {256, 66};   // IQ2_XXS
        case 17: return {256, 74};   // IQ2_XS
        case 18: return {256, 98};   // IQ3_XXS
        case 19: return {256, 50};   // IQ1_S
        case 20: return {32, 18};    // IQ4_NL
        case 21: return {256, 110};  // IQ3_S
        case 22: return {256, 82};   // IQ2_S
        case 23: return {256, 136};  // IQ4_XS
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

// `keep_kv` is false for a shard after the first, whose metadata is only
// split bookkeeping.
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
        // The split bookkeeping goes: what is written back is one whole
        // model, and those keys would send the loader after shards that do
        // not exist.
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

void put_u32(std::string & s, uint32_t v) { s.append(reinterpret_cast<const char *>(&v), 4); }
void put_u64(std::string & s, uint64_t v) { s.append(reinterpret_cast<const char *>(&v), 8); }

void put_str(std::string & s, const std::string & v) {
    put_u64(s, v.size());
    s += v;
}

std::string whole_header(const Layout & l) {
    std::string h = "GGUF";
    put_u32(h, l.version);
    put_u64(h, static_cast<uint64_t>(l.tensors.size()));
    put_u64(h, static_cast<uint64_t>(l.kv.size()));
    for (const KvEntry & e : l.kv) {
        h += e.raw;
    }
    int64_t cursor = 0;
    for (const TensorEntry & t : l.tensors) {
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

int64_t TensorEntry::rows() const {
    int64_t n = 1;
    for (size_t i = 1; i < dims.size(); i++) {
        n *= dims[i];
    }
    return n;
}

int64_t TensorEntry::experts() const { return dims.size() > 2 ? dims[2] : 1; }

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

int64_t write_header(const std::string & path, const Layout & l, std::string & err) {
    const std::string h = whole_header(l);
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

bool patch_entry(std::ostream & out, const Layout & l, size_t tensor, uint32_t type, int64_t offset) {
    // Everything in the header before this entry's type field is fixed-width
    // apart from the names, so its position is what came before it.
    int64_t at = 4 + 4 + 8 + 8;
    for (const KvEntry & e : l.kv) {
        at += static_cast<int64_t>(e.raw.size());
    }
    for (size_t i = 0; i < tensor && i < l.tensors.size(); i++) {
        at += 8 + static_cast<int64_t>(l.tensors[i].name.size()) + 4 +
              8 * static_cast<int64_t>(l.tensors[i].dims.size()) + 4 + 8;
    }
    at += 8 + static_cast<int64_t>(l.tensors[tensor].name.size()) + 4 +
          8 * static_cast<int64_t>(l.tensors[tensor].dims.size());
    out.seekp(at, std::ios::beg);
    out.write(reinterpret_cast<const char *>(&type), 4);
    const uint64_t off = static_cast<uint64_t>(offset);
    out.write(reinterpret_cast<const char *>(&off), 8);
    return out.good();
}

} // namespace ggufio
} // namespace llmash
