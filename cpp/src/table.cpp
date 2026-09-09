#include "table.h"

#include "cli_commands.h"

#include <vector>

namespace llmash {
namespace {

std::vector<nlohmann::json> rows_of(const nlohmann::json & rows) {
    std::vector<nlohmann::json> out;
    if (rows.is_array()) {
        for (const auto & r : rows) {
            out.push_back(r);
        }
    }
    return out;
}

} // namespace

std::string render_list(const nlohmann::json & rows) { return render_list(rows_of(rows)); }
std::string render_ps(const nlohmann::json & rows) { return render_ps(rows_of(rows)); }

} // namespace llmash
