#pragma once

#include "config.h"

#include <nlohmann/json.hpp>

#include <string>

namespace llmash {

// `llmash tray`: hands off to "llmashw.exe tray", hidden and detached, the
// same program built without a console. Returns false with a user-facing
// message in `err` when it could not be started, including the antivirus
// false positive real users have hit on the unsigned exe.
bool cmd_tray(const Config & cfg, std::string & err);

// `llmashw tray`: the notification-area icon. Starts the server if it is
// not already running, then blocks pumping its own message loop until
// "Close llmash" is picked from the menu. Must run from the console-less
// build; it owns no window a user ever sees other than the tray icon.
int tray_main(const Config & cfg);

// ------------------------------------------------- testable in isolation
//
// Everything below has no window and touches no live process; it is the
// string- and JSON-shaped logic that showMenu(), the server watchdog and
// the login-shortcut toggle build on.

std::string ps_quote(const std::string & s);

// The PowerShell that stops this install's server and, first, the
// llama-server.exe engines it owns (parented by its PID) so nothing is
// orphaned holding VRAM.
std::string stop_script(const std::string & root);

// The PowerShell that counts this install's own "serve" processes.
std::string process_exists_script(const std::string & root);

std::string startup_shortcut_path();

// The PowerShell that creates the Startup-folder shortcut ("Start at
// login" is a shortcut, not a registry Run key, so an antivirus scan of
// the Run key never finds llmash). `icon_path` may be empty.
std::string enable_startup_script(const std::string & root, const std::string & shortcut_path,
                                  const std::string & icon_path);

// "5s left" / "12m left" / "1.3h left" / "pinned" / "unloading" / "" for
// a /api/ps model entry, read from its expires_at.
std::string expires_text(const nlohmann::json & model);

// "3.2 GB" from size_vram, falling back to size; "" when neither is set.
std::string size_text(const nlohmann::json & model);

// Exercises the exact Shell_NotifyIconW add-then-remove sequence tray_main
// uses, with its own throwaway window class, so it can be checked outside a
// message loop and without a live server. A false return means Windows
// itself rejected the call; it cannot confirm a human actually saw the icon.
bool self_test_tray_icon();

} // namespace llmash
