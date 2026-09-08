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
};

// ok=false for anything that is not a GGUF.
GGUFInfo read_gguf(const std::string & path);

std::string file_type_name(uint32_t file_type);

} // namespace llmash
