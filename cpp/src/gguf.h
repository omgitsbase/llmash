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
};

// ok=false for anything that is not a GGUF.
GGUFInfo read_gguf(const std::string & path);

std::string file_type_name(uint32_t file_type);

} // namespace llmash
