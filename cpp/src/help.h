#pragma once

// The help text for every command, byte-for-byte what the real ollama.exe
// prints, with llmash's own commands slotted in.

#include <string>

namespace llmash {

// The top-level `ollama [flags]` / `ollama [command]` usage block.
const std::string & help_text();

// The `ollama <command> --help` text for one command, or nullptr for a
// command with none (an unknown command is a caller's problem, not this
// lookup's).
const std::string * command_help(const std::string & topic);

// Re-addresses the real binary's help to whatever name was actually typed,
// the way main.go's progText does: "ollama" only changes where a user would
// read it as this program's own name, not inside a URL or a model tag.
std::string prog_text(const std::string & text, const std::string & prog);

} // namespace llmash
