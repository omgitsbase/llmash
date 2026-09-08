#pragma once

// A tiny RAII wrapper around vendor/subprocess.h, shared by the two places
// cmds.go shells out to something: win.go's hiddenPowerShell (uninstall's
// registry/shortcut work, link/unlink's tailscale calls) and console.go's
// clip() (link's "copy" keys).

#include <string>
#include <vector>

namespace llmash {

struct ProcessResult {
    bool        started   = false;
    int         exit_code = -1;
    std::string out; // captured stdout, when requested
};

// argv[0] is the program (found via PATH, as Windows always does); the rest
// are its arguments, each passed as one Win32 argument (no shell involved,
// so no quoting to get wrong).
ProcessResult run_hidden(const std::vector<std::string> & argv, const std::string * stdin_data, bool capture_stdout,
                          bool wait);

} // namespace llmash
