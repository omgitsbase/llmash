#pragma once

// Port of cmd/llmash/models.go: `llmash models` and `llmash models set DIR`.

#include <string>
#include <utility>
#include <vector>

namespace llmash {

// `llmash models [set] [DIR]`. Loads its own Config, the way the Go command
// does (loadConfig() at the top of cmdModels), so it needs nothing from a
// caller beyond the words after "models" on the command line.
void cmd_models(const std::vector<std::string> & args);

// models.go's countLibrary, also called from doctor.go: how many GGUFs in
// dir would be read as models (sidecars, and shards after the first, are
// never counted at all), and how many are GGUFs whose header would not
// read. {0, 0} for a folder with nothing usable in it.
std::pair<int, int> count_library(const std::string & dir);

} // namespace llmash
