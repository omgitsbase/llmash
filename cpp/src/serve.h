#pragma once

#include <string>
#include <vector>

namespace llmash {

// `llmash serve [--port N] [--host H]`: the server itself. Returns when it
// has shut down.
int cmd_serve(const std::vector<std::string> & args);

} // namespace llmash
