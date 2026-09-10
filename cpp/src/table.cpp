#include "table.h"

#include "cli_commands.h"
#include "cli_format.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

namespace llmash {
namespace {

using nlohmann::json;

std::string field(const json & m, const std::string & k) {
    if (!m.is_object()) {
        return "";
    }
    const auto it = m.find(k);
    if (it == m.end() || it->is_null()) {
        return "";
    }
    return it->is_string() ? it->get<std::string>() : it->dump();
}

double number(const json & m, const std::string & k) {
    if (!m.is_object()) {
        return 0;
    }
    const auto it = m.find(k);
    return it != m.end() && it->is_number() ? it->get<double>() : 0;
}

double seconds_now() {
    return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
}

std::string first12(const std::string & digest) {
    const size_t colon = digest.find(':');
    const std::string d = colon == std::string::npos ? digest : digest.substr(colon + 1);
    return d.size() > 12 ? d.substr(0, 12) : d;
}

std::vector<json> rows_of(const json & rows) {
    std::vector<json> out;
    if (rows.is_array()) {
        for (const auto & r : rows) {
            out.push_back(r);
        }
    }
    return out;
}

} // namespace

std::string render_list(const std::vector<json> & rows) {
    const double now = seconds_now();
    Table        t({"NAME", "ID", "SIZE", "MODIFIED"});
    for (const auto & m : rows) {
        t.add({field(m, "name"), first12(field(m, "digest")),
               human_bytes(static_cast<int64_t>(number(m, "size"))),
               human_time_iso(field(m, "modified_at"), "Never", now)});
    }
    return t.str();
}

std::string render_ps(const std::vector<json> & rows) {
    const double now = seconds_now();
    Table        t({"NAME", "ID", "SIZE", "PROCESSOR", "CONTEXT", "UNTIL"});
    for (const auto & m : rows) {
        const int64_t size = static_cast<int64_t>(number(m, "size"));
        const int64_t vram = static_cast<int64_t>(number(m, "size_vram"));
        std::string   proc;
        if (vram == 0) {
            proc = "100% CPU";
        } else if (vram == size) {
            proc = "100% GPU";
        } else if (vram > size || size == 0) {
            proc = "Unknown";
        } else {
            const int cpu = static_cast<int>(
                std::llround(static_cast<double>(size - vram) / static_cast<double>(size) * 100.0));
            char buf[64];
            std::snprintf(buf, sizeof(buf), "%d%%/%d%% CPU/GPU", cpu, 100 - cpu);
            proc = buf;
        }
        std::string  until = "Never";
        const double exp   = parse_rfc3339(field(m, "expires_at"));
        if (exp >= 0) {
            until = now - exp > 0 ? "Stopping..."
                                  : human_time(static_cast<std::time_t>(exp), false, "Never", now);
        }
        t.add({field(m, "name"), first12(field(m, "digest")), human_bytes(size), proc,
               std::to_string(static_cast<long long>(number(m, "context_length"))), until});
    }
    return t.str();
}

std::string render_list(const json & rows) { return render_list(rows_of(rows)); }
std::string render_ps(const json & rows) { return render_ps(rows_of(rows)); }

} // namespace llmash
