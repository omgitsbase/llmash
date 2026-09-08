#pragma once

#include <nlohmann/json.hpp>

#include <string>

namespace llmash {

// Implemented elsewhere, ported from table.go/render.go (~385 lines of
// terminal table layout: column widths, unicode display width, wrapping,
// human byte/duration formatting). /cli/list and /cli/ps just need this
// text; the layout itself is not HTTP route wiring. Not implemented here.
std::string render_list(const nlohmann::json & rows);
std::string render_ps(const nlohmann::json & rows);

} // namespace llmash
