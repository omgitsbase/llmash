#pragma once

#include <cstdint>
#include <string>

namespace llmash {

// Read from the header, not guessed from the file name.
struct GGUFInfo {
    bool        ok = false;
    std::string arch;
    std::string name;
    std::string quant;
    uint64_t    n_params  = 0;
    uint32_t    n_tensors = 0;
    uint64_t    input_bytes = 0;  // token_embd and per_layer_token_embd: llama.cpp keeps these in RAM as the input layer
    bool        has_mtp = false;
    bool        has_vision = false;
    std::string repo;

    std::string basename;      // general.basename
    std::string size_label;    // general.size_label
    std::string chat_template; // tokenizer.chat_template
    uint64_t    ctx_train    = 0;  // <arch>.context_length
    int         experts      = 0;  // <arch>.expert_count
    int         experts_used = 0;  // <arch>.expert_used_count
    bool        has_pooling  = false; // an embedding model

    bool        has_vision_encoder = false; // clip.has_vision_encoder, mmproj files
    bool        has_audio_encoder  = false; // clip.has_audio_encoder
    std::string base_repo_url;              // general.base_model.0.repo_url
    std::string base_org;                   // general.base_model.0.organization
    std::string base_name;                  // general.base_model.0.name

    // The attention geometry, for sizing the cache a context costs.
    int n_layer     = 0;  // <arch>.block_count
    int head_count  = 0;  // attention.head_count
    int head_kv     = 0;  // attention.head_count_kv on a layer that attends over the whole context
    int head_kv_swa = 0;  // the same on a sliding-window layer
    int key_len     = 0;  // attention.key_length, value_length
    int value_len   = 0;
    int key_len_swa = 0;
    int value_len_swa = 0;
    int embd        = 0;  // embedding_length
    int attn_every  = 0;  // full_attention_interval: the other layers are recurrent (0: all attend)
    int swa_window  = 0;  // attention.sliding_window
    int swa_layers  = 0;  // layers on the window, from sliding_window_pattern
    int ssm_state   = 0;  // ssm.state_size
    int ssm_inner   = 0;  // ssm.inner_size

    // Bytes of K/V one token costs across the layers that see the whole
    // context, at an f16 cache; and the per-sequence state that does not grow
    // with it: recurrent layers, and windowed layers up to their window.
    double kv_bytes_per_token() const;
    double state_bytes() const;
};

// How many bytes a cache type spends per f16 byte: 1 for f16, about half for q8_0.
double kv_type_scale(const std::string & kv_type);

// ok=false for anything that is not a GGUF.
GGUFInfo read_gguf(const std::string & path);

std::string file_type_name(uint32_t file_type);

} // namespace llmash
