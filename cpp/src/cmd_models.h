#pragma once

// `llmash models` and `llmash models set DIR`.

#include <string>
#include <utility>
#include <vector>

namespace llmash {

// `llmash models [set] [DIR]`.
void cmd_models(const std::vector<std::string> & args);

// models.go's countLibrary, also called from doctor.go: how many GGUFs in
// dir would be read as models (sidecars, and shards after the first, are
// never counted at all), and how many are GGUFs whose header would not read.
std::pair<int, int> count_library(const std::string & dir);

} // namespace llmash
