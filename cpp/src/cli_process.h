#pragma once

// A tiny RAII wrapper around vendor/subprocess.h, shared by the two places
// cmds.go shells out to something: win.go's hiddenPowerShell (uninstall's
// registry/shortcut work, link/unlink's tailscale calls) and console.go's
// clip() (link's "copy" keys).

#include <functional>
#include <string>
#include <vector>

namespace llmash {

struct ProcessResult {
    bool        started   = false;
    int         exit_code = -1;
    std::string out; // captured stdout, when requested
};

ProcessResult run_hidden(const std::vector<std::string> & argv, const std::string * stdin_data, bool capture_stdout,
                          bool wait);

// The same, with each line of output handed over as it arrives; returns the
// exit code, or -1 if the program could not be started.
int run_streaming(const std::vector<std::string> & argv, const std::function<void(const std::string &)> & on_line);

} // namespace llmash
