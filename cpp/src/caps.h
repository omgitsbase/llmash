#pragma once

#include "gguf.h"

#include <string>
#include <vector>

namespace llmash {

std::vector<std::string> projector_caps(const std::string & mmproj_path);

// `known` is false when the hub does not say, which is not the same as "nothing".
std::vector<std::string> hf_caps(const std::string & repo, const std::string & cache_dir, bool & known);

std::string hf_repo_of(const GGUFInfo & g);

std::vector<std::string> caps_for(const GGUFInfo & g, const std::string & projector,
                                  const std::string & cache_dir);

} // namespace llmash
