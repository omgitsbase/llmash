#pragma once

#include <filesystem>
#include <string>

namespace llmash {

// Files holding a secret (the link key) are readable by their owner alone: mode 0600 on Linux, and on Windows
// a DACL of the current user, SYSTEM and Administrators with nothing inherited from the folder.
bool write_private_file(const std::filesystem::path & p, const std::string & content);

// Narrows an existing file the same way, for one an older version left open to everyone.
void make_private(const std::filesystem::path & p);

}  // namespace llmash
