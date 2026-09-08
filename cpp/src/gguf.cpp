#include "gguf.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <vector>

namespace llmash {

namespace {

enum : uint32_t {
    T_UINT8 = 0, T_INT8, T_UINT16, T_INT16, T_UINT32, T_INT32,
    T_FLOAT32, T_BOOL, T_STRING, T_ARRAY, T_UINT64, T_INT64, T_FLOAT64,
};

struct Reader {
    std::ifstream in;
    bool bad = false;

    template <typename T> T num() {
        T v{};
        if (!in.read(reinterpret_cast<char *>(&v), sizeof(T))) {
            bad = true;
            return T{};
        }
        return v;
    }

    std::string str() {
        const uint64_t n = num<uint64_t>();
        if (bad || n > (1u << 20)) {
            bad = true;
            return "";
        }
        std::string s(static_cast<size_t>(n), '\0');
        if (n && !in.read(s.data(), static_cast<std::streamsize>(n))) {
            bad = true;
            return "";
        }
        return s;
    }

    // Advances past a value without keeping it.
    void skip_value(uint32_t type) {
        switch (type) {
            case T_UINT8: case T_INT8: case T_BOOL:   in.seekg(1, std::ios::cur); break;
            case T_UINT16: case T_INT16:              in.seekg(2, std::ios::cur); break;
            case T_UINT32: case T_INT32: case T_FLOAT32: in.seekg(4, std::ios::cur); break;
            case T_UINT64: case T_INT64: case T_FLOAT64: in.seekg(8, std::ios::cur); break;
            case T_STRING: str(); break;
            case T_ARRAY: {
                const uint32_t et = num<uint32_t>();
                const uint64_t n  = num<uint64_t>();
                if (bad || n > (1ull << 32)) { bad = true; return; }
                for (uint64_t i = 0; i < n && !bad; i++) {
                    skip_value(et);
                }
                break;
            }
            default: bad = true; break;
        }
    }
};

bool contains(const std::string & hay, const char * needle) {
    return hay.find(needle) != std::string::npos;
}

} // namespace

std::string file_type_name(uint32_t ft) {
    switch (ft) {
        case 0:  return "F32";
        case 1:  return "F16";
        case 2:  return "Q4_0";
        case 3:  return "Q4_1";
        case 7:  return "Q8_0";
        case 8:  return "Q5_0";
        case 9:  return "Q5_1";
        case 10: return "Q2_K";
        case 11: return "Q3_K_S";
        case 12: return "Q3_K_M";
        case 13: return "Q3_K_L";
        case 14: return "Q4_K_S";
        case 15: return "Q4_K_M";
        case 16: return "Q5_K_S";
        case 17: return "Q5_K_M";
        case 18: return "Q6_K";
        case 23: return "IQ4_NL";
        case 30: return "IQ4_XS";
        default: return "";
    }
}

GGUFInfo read_gguf(const std::string & path) {
    GGUFInfo info;

    Reader r;
    r.in.open(path, std::ios::binary);
    if (!r.in) {
        return info;
    }

    char magic[4] = {};
    if (!r.in.read(magic, 4) || std::memcmp(magic, "GGUF", 4) != 0) {
        return info;
    }

    const uint32_t version = r.num<uint32_t>();
    if (r.bad || version < 2 || version > 3) {
        return info;
    }
    const uint64_t n_tensors = r.num<uint64_t>();
    const uint64_t n_kv      = r.num<uint64_t>();
    if (r.bad || n_kv > (1u << 20) || n_tensors > (1u << 24)) {
        return info;
    }
    info.n_tensors = static_cast<uint32_t>(n_tensors);

    for (uint64_t i = 0; i < n_kv && !r.bad; i++) {
        const std::string key  = r.str();
        const uint32_t    type = r.num<uint32_t>();
        if (r.bad) {
            break;
        }
        if (key == "general.architecture" && type == T_STRING) {
            info.arch = r.str();
        } else if (key == "general.name" && type == T_STRING) {
            info.name = r.str();
        } else if (key == "general.repo_url" && type == T_STRING) {
            info.repo = r.str();
        } else if (key == "general.file_type" && (type == T_UINT32 || type == T_INT32)) {
            info.quant = file_type_name(r.num<uint32_t>());
        } else if (key == "general.parameter_count" && (type == T_UINT64 || type == T_INT64)) {
            info.n_params = r.num<uint64_t>();
        } else {
            r.skip_value(type);
        }
        if (contains(key, "vision") || contains(key, "clip")) {
            info.has_vision = true;
        }
    }
    if (r.bad || info.arch.empty()) {
        return info;
    }

    for (uint64_t i = 0; i < n_tensors && !r.bad; i++) {
        const std::string name = r.str();
        const uint32_t    dims = r.num<uint32_t>();
        if (r.bad || dims > 4) {
            break;
        }
        for (uint32_t d = 0; d < dims && !r.bad; d++) {
            r.num<uint64_t>();
        }
        r.num<uint32_t>(); // ggml type
        r.num<uint64_t>(); // offset
        if (contains(name, "nextn") || contains(name, "mtp")) {
            info.has_mtp = true;
        }
    }

    info.ok = !r.bad;
    return info;
}

} // namespace llmash
