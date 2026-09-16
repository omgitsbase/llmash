#pragma once

#include "config.h"

#include <nlohmann/json.hpp>

#include <string>

namespace llmash {

// `llmash tray`: hands off to "llmashw.exe tray", hidden and detached, the
// same program built without a console.
bool cmd_tray(const Config & cfg, std::string & err);
// `llmash start`: the tray, which brings the server up, then a word on where it answers.
int  cmd_start(const Config & cfg);

int tray_main(const Config & cfg);

// Below here: no window, no live process, so it can be tested directly.

std::string ps_quote(const std::string & s);


std::string startup_shortcut_path();

// "5s left" / "12m left" / "1.3h left" / "pinned" / "unloading" / "" for
// a /api/ps model entry, read from its expires_at.
std::string expires_text(const nlohmann::json & model);

// "3.2 GB" from size_vram, falling back to size; "" when neither is set.
std::string size_text(const nlohmann::json & model);

bool self_test_tray_icon();

} // namespace llmash
