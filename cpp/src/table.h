#pragma once

#include <nlohmann/json.hpp>

#include <string>

namespace llmash {

std::string render_list(const nlohmann::json & rows);
std::string render_ps(const nlohmann::json & rows);

} // namespace llmash
