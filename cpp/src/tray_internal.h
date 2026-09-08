#pragma once

// Split out of tray.cpp so this translation unit never sees <windows.h>:
// subprocess.h hand-declares its own Win32 prototypes, which collide with
// the real ones once both are visible in the same file.

#include <string>

namespace llmash {

struct PowerShellRun {
    bool        started = false; // false only if powershell.exe itself could not be spawned
    int         exit_code = 0;
    std::string stdout_text;
};

// Runs `script` hidden via `powershell -NoProfile -NonInteractive -WindowStyle
// Hidden -Command`, waits for it to finish, and returns what it printed.
PowerShellRun run_hidden_powershell(const std::string & script);

} // namespace llmash
