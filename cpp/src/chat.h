#pragma once

#include "config.h"
#include "manager.h"
#include "registry.h"

#include <httplib.h>

namespace llmash {

// Implemented in chat.cpp, a separate module ported concurrently from
// chat.go.
void handle_chat(const httplib::Request & req, httplib::Response & res, Config & cfg, Manager & mgr, Registry & reg);
void handle_generate(const httplib::Request & req, httplib::Response & res, Config & cfg, Manager & mgr, Registry & reg);
void handle_embed(const httplib::Request & req, httplib::Response & res, Config & cfg, Manager & mgr, Registry & reg);
void handle_v1_chat_completions(const httplib::Request & req, httplib::Response & res, Config & cfg, Manager & mgr, Registry & reg);
void handle_v1_completions(const httplib::Request & req, httplib::Response & res, Config & cfg, Manager & mgr, Registry & reg);

} // namespace llmash
