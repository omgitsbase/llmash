#include "gguf.h"

#include <algorithm>
#include <cctype>
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

    // A scalar as one element, an array as all of them, capped; anything else
    // consumed as zeros.
    std::vector<uint64_t> ints(uint32_t type) {
        std::vector<uint64_t> out;
        if (type != T_ARRAY) {
            out.push_back(num_any(type));
            return out;
        }
        const uint32_t et = num<uint32_t>();
        const uint64_t n  = num<uint64_t>();
        if (bad || n > (1ull << 32)) {
            bad = true;
            return out;
        }
        for (uint64_t i = 0; i < n && !bad; i++) {
            const uint64_t v = num_any(et);
            if (out.size() < 4096) {
                out.push_back(v);
            }
        }
        return out;
    }
    uint64_t num_any(uint32_t type) {
        switch (type) {
            case T_UINT8:  case T_INT8:  case T_BOOL: { uint8_t v = num<uint8_t>();  return v; }
            case T_UINT16: case T_INT16:              { uint16_t v = num<uint16_t>(); return v; }
            case T_UINT32: case T_INT32:              { uint32_t v = num<uint32_t>(); return v; }
            case T_UINT64: case T_INT64:              { return num<uint64_t>(); }
            default: skip_value(type); return 0;
        }
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

bool ends_with(const std::string & s, const char * suffix) {
    const size_t n = std::strlen(suffix);
    return s.size() >= n && s.compare(s.size() - n, n, suffix) == 0;
}

bool contains(const std::string & hay, const char * needle) {
    return hay.find(needle) != std::string::npos;
}

} // namespace

double kv_type_scale(const std::string & kv_type) {
    std::string t;
    for (const char c : kv_type) {
        t += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    if (t == "f32") return 2.0;
    if (t == "q8_0") return 34.0 / 64.0;
    if (t == "q5_1") return 24.0 / 64.0;
    if (t == "q5_0") return 22.0 / 64.0;
    if (t == "q4_1") return 20.0 / 64.0;
    if (t == "q4_0" || t == "iq4_nl") return 18.0 / 64.0;
    return 1.0;  // f16, bf16
}

double GGUFInfo::kv_bytes_per_token() const {
    if (n_layer <= 0 || head_kv <= 0) {
        return 0;
    }
    const int kl = key_len > 0 ? key_len : (head_count > 0 && embd > 0 ? embd / head_count : 128);
    const int vl = value_len > 0 ? value_len : kl;
    int attend = attn_every > 0 ? n_layer / attn_every : n_layer;
    attend -= swa_layers;
    if (attend < 0) {
        attend = 0;
    }
    return static_cast<double>(attend) * head_kv * (kl + vl) * 2.0;
}

double GGUFInfo::state_bytes() const {
    double b = 0;
    if (attn_every > 0 && ssm_state > 0 && ssm_inner > 0) {
        b += static_cast<double>(n_layer - n_layer / attn_every) * ssm_inner * ssm_state * 4.0;
    }
    if (swa_window > 0 && swa_layers > 0) {
        const int kl = key_len_swa > 0 ? key_len_swa : (key_len > 0 ? key_len : 128);
        const int vl = value_len_swa > 0 ? value_len_swa : kl;
        const int hk = head_kv_swa > 0 ? head_kv_swa : head_kv;
        b += static_cast<double>(swa_layers) * hk * (kl + vl) * 2.0 * swa_window;
    }
    return b;
}

std::string file_type_name(uint32_t ft) {
    // llama_ftype, in the order llama.h declares it. A gap here is not
    // cosmetic: the field feeds clients that list models, and some of them
    // drop a model whose quantization reads as empty.
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
        case 19: return "IQ2_XXS";
        case 20: return "IQ2_XS";
        case 21: return "Q2_K_S";
        case 22: return "IQ3_XS";
        case 23: return "IQ3_XXS";
        case 24: return "IQ1_S";
        case 25: return "IQ4_NL";
        case 26: return "IQ3_S";
        case 27: return "IQ3_M";
        case 28: return "IQ2_S";
        case 29: return "IQ2_M";
        case 30: return "IQ4_XS";
        case 31: return "IQ1_M";
        case 32: return "BF16";
        case 36: return "TQ1_0";
        case 37: return "TQ2_0";
        case 38: return "MXFP4_MOE";
        case 39: return "NVFP4";
        case 40: return "Q1_0";
        case 41: return "Q2_0";
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
    std::vector<uint64_t> head_kv_per_layer;
    std::vector<uint64_t> swa_pattern;  // 1 where a layer sees only its window

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
        } else if (key == "general.basename" && type == T_STRING) {
            info.basename = r.str();
        } else if (key == "general.size_label" && type == T_STRING) {
            info.size_label = r.str();
        } else if (key == "tokenizer.chat_template" && type == T_STRING) {
            info.chat_template = r.str();
        } else if (ends_with(key, ".context_length")) {
            info.ctx_train = r.num_any(type);
        } else if (ends_with(key, ".expert_count")) {
            info.experts = static_cast<int>(r.num_any(type));
        } else if (ends_with(key, ".expert_used_count")) {
            info.experts_used = static_cast<int>(r.num_any(type));
        } else if (ends_with(key, ".pooling_type")) {
            info.has_pooling = r.num_any(type) != 0;
        } else if (ends_with(key, ".block_count")) {
            info.n_layer = static_cast<int>(r.num_any(type));
        } else if (ends_with(key, ".embedding_length")) {
            info.embd = static_cast<int>(r.num_any(type));
        } else if (ends_with(key, ".attention.head_count")) {
            const auto v = r.ints(type);
            info.head_count = v.empty() ? 0 : static_cast<int>(v[0]);
        } else if (ends_with(key, ".attention.head_count_kv")) {
            head_kv_per_layer = r.ints(type);
        } else if (ends_with(key, ".attention.key_length")) {
            info.key_len = static_cast<int>(r.num_any(type));
        } else if (ends_with(key, ".attention.value_length")) {
            info.value_len = static_cast<int>(r.num_any(type));
        } else if (ends_with(key, ".attention.key_length_swa")) {
            info.key_len_swa = static_cast<int>(r.num_any(type));
        } else if (ends_with(key, ".attention.value_length_swa")) {
            info.value_len_swa = static_cast<int>(r.num_any(type));
        } else if (ends_with(key, ".full_attention_interval")) {
            info.attn_every = static_cast<int>(r.num_any(type));
        } else if (ends_with(key, ".attention.sliding_window")) {
            info.swa_window = static_cast<int>(r.num_any(type));
        } else if (ends_with(key, ".attention.sliding_window_pattern")) {
            swa_pattern = r.ints(type);
        } else if (ends_with(key, ".ssm.state_size")) {
            info.ssm_state = static_cast<int>(r.num_any(type));
        } else if (ends_with(key, ".ssm.inner_size")) {
            info.ssm_inner = static_cast<int>(r.num_any(type));
        } else if (key == "general.base_model.0.repo_url" && type == T_STRING) {
            info.base_repo_url = r.str();
        } else if (key == "general.base_model.0.organization" && type == T_STRING) {
            info.base_org = r.str();
        } else if (key == "general.base_model.0.name" && type == T_STRING) {
            info.base_name = r.str();
        } else if (key == "clip.has_vision_encoder" && type == T_BOOL) {
            info.has_vision_encoder = r.num_any(type) != 0;
        } else if (key == "clip.has_audio_encoder" && type == T_BOOL) {
            info.has_audio_encoder = r.num_any(type) != 0;
        } else {
            r.skip_value(type);
        }
        if (contains(key, "vision") || contains(key, "clip")) {
            info.has_vision = true;
        }
    }
    if (r.bad) {
        return info;
    }
    // A later shard of a split carries no architecture: not a model on its
    // own, but its tensors still count.
    const bool shard_only = info.arch.empty();
    // the kv head count on a full layer, and on a windowed one, when they differ
    for (size_t i = 0; i < head_kv_per_layer.size(); i++) {
        const bool windowed = i < swa_pattern.size() && swa_pattern[i] != 0;
        int &      slot     = windowed ? info.head_kv_swa : info.head_kv;
        if (slot == 0) {
            slot = static_cast<int>(head_kv_per_layer[i]);
        }
    }
    for (const uint64_t w : swa_pattern) {
        info.swa_layers += w != 0;
    }
    if (info.head_kv == 0) {
        info.head_kv = info.head_kv_swa;
    }

    std::vector<std::pair<uint64_t, bool>> spans;  // data offset, and whether it is an input-layer tensor
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
        const uint64_t offset = r.num<uint64_t>();
        if (contains(name, "nextn") || contains(name, "mtp")) {
            info.has_mtp = true;
        }
        spans.emplace_back(offset, name == "token_embd.weight" || name == "per_layer_token_embd.weight");
    }
    // A tensor's bytes are the gap to the next offset. The data begins where
    // the header ends, at the next 32-byte boundary.
    if (!r.bad && !spans.empty()) {
        const uint64_t header_end = static_cast<uint64_t>(r.in.tellg());
        const uint64_t data_start = (header_end + 31) / 32 * 32;
        r.in.seekg(0, std::ios::end);
        const uint64_t file_size = static_cast<uint64_t>(r.in.tellg());
        const uint64_t data_size = file_size > data_start ? file_size - data_start : 0;
        std::sort(spans.begin(), spans.end());
        for (size_t i = 0; i < spans.size(); i++) {
            if (!spans[i].second) {
                continue;
            }
            const uint64_t end = i + 1 < spans.size() ? spans[i + 1].first : data_size;
            info.input_bytes += end > spans[i].first ? end - spans[i].first : 0;
        }
    }

    info.ok = !r.bad && !shard_only;
    return info;
}

} // namespace llmash
