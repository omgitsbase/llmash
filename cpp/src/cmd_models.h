#pragma once

// `llmash models` and `llmash models set DIR`.

#include <string>
#include <utility>
#include <vector>

namespace llmash {

// `llmash models [set] [DIR]`.
void cmd_models(const std::vector<std::string> & args);

std::pair<int, int> count_library(const std::string & dir);

} // namespace llmash
