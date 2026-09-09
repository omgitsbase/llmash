#pragma once

#include "config.h"

#include <string>
#include <vector>

namespace llmash {

// `llmash update [--force]`: fetch the latest release's installer and run it
// over this install.
int cmd_update(const std::vector<std::string> & args, const Config & cfg);

} // namespace llmash
