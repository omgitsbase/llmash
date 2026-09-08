#pragma once

#include "config.h"
#include "manager.h"
#include "registry.h"

#include <httplib.h>

namespace llmash {

// Mounts every /api, /v1 and /cli route onto srv. Owns nothing; cfg, mgr and
// reg must outlive the server.
void register_routes(httplib::Server & srv, Config & cfg, Manager & mgr, Registry & reg);

} // namespace llmash
