#pragma once

// The help text for every command, byte-for-byte what the real ollama.exe
// prints, with llmash's own commands slotted in.

#include <string>

namespace llmash {

// The top-level `ollama [flags]` / `ollama [command]` usage block.
const std::string & help_text();

const std::string * command_help(const std::string & topic);

std::string prog_text(const std::string & text, const std::string & prog);

} // namespace llmash
