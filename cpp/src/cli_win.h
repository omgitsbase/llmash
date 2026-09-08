#pragma once

// Port of the Windows-integration half of cmd/llmash/win.go: the bits
// `uninstall` and `link`/`unlink` need (registry, PATH, a hidden PowerShell,
// a .lnk's target) and nothing else -- win.go's fileExists/dirExists/which
// are plain std::filesystem/PATH lookups here instead, folded into
// cli_commands.cpp where they're used.

#include <string>

namespace llmash {

bool reg_delete_hkcu_key(const std::string & subkey);

// Removes every entry equal to bin_dir (case-insensitively) from HKCU's
// Environment\Path; returns whether anything changed.
bool remove_from_user_path(const std::string & bin_dir);

// A hidden PowerShell (-NoProfile -NonInteractive -WindowStyle Hidden).
std::string hidden_powershell(const std::string & script, bool wait, int * exit_code = nullptr);

// Escapes a single-quoted PowerShell string literal.
std::string ps_quote(const std::string & s);

// The full path of an executable found on PATH, or "" (a thin std::string
// wrapper so callers read the same as win.go's which()).
std::string which_exe(const std::string & exe);

// A .lnk's target path, or "" if it doesn't exist or can't be read (shells
// out to the same WScript.Shell COM one-liner win.go's shortcutTarget does).
std::string shortcut_target(const std::string & lnk_path);

} // namespace llmash
