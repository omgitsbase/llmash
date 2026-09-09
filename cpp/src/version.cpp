#include "version.h"

#include "cli_util.h"

#include <nlohmann/json.hpp>

#include <cctype>
#include <filesystem>
#include <fstream>
#include <iterator>

#ifndef LLMASH_VERSION
#define LLMASH_VERSION "0.0.0"
#endif

namespace fs = std::filesystem;

namespace llmash {

const char * built_version() { return LLMASH_VERSION; }

std::string version_string(const Config & cfg) {
    std::ifstream in(fs::path(cfg.root) / "VERSION");
    if (in) {
        std::string v((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        while (!v.empty() && std::isspace(static_cast<unsigned char>(v.back()))) {
            v.pop_back();
        }
        size_t start = 0;
        while (start < v.size() && std::isspace(static_cast<unsigned char>(v[start]))) {
            start++;
        }
        v = v.substr(start);
        if (!v.empty()) {
            return v;
        }
    }
    nlohmann::json out;
    std::string    err;
    if (clidoc::call_json("GET", "/api/version", nullptr, 2, out, err) && out.contains("version") &&
        out["version"].is_string() && !out["version"].get<std::string>().empty()) {
        return out["version"].get<std::string>();
    }
    return built_version();
}

} // namespace llmash
