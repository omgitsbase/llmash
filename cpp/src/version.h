#pragma once

#include "config.h"

#include <string>

namespace llmash {

// The version compiled in, from the VERSION file at the repository root.
const char * built_version();

// What `-v` prints: root\VERSION when present, else the running server's,
// else built_version().
std::string version_string(const Config & cfg);

} // namespace llmash
