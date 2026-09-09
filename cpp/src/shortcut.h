#pragma once

#include <string>

namespace llmash {

std::string read_shortcut_target(const std::string & lnk_path);

bool write_shortcut(const std::string & lnk_path, const std::string & target, const std::string & args,
                    const std::string & working_dir, const std::string & icon_path,
                    const std::string & description);

} // namespace llmash
