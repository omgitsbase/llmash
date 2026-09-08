#include "config.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>

#ifdef _WIN32
#include <windows.h>
#endif

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace llmash {

std::string env_str(const char * name, const std::string & fallback) {
    const char * v = std::getenv(name);
    if (v == nullptr || *v == '\0') {
        return fallback;
    }
    return v;
}

int env_int(const char * name, int fallback) {
    const std::string v = env_str(name);
    if (v.empty()) {
        return fallback;
    }
    try {
        return std::stoi(v);
    } catch (const std::exception &) {
        return fallback;
    }
}

std::string exe_dir() {
#ifdef _WIN32
    wchar_t buf[MAX_PATH];
    const DWORD n = GetModuleFileNameW(nullptr, buf, MAX_PATH);
    if (n == 0 || n == MAX_PATH) {
        return fs::current_path().string();
    }
    return fs::path(std::wstring(buf, n)).parent_path().string();
#else
    return fs::current_path().string();
#endif
}

bool same_dir(const std::string & a, const std::string & b) {
    if (a.empty() || b.empty()) {
        return false;
    }
    std::error_code ec;
    std::string x = fs::weakly_canonical(fs::path(a), ec).string();
    if (ec) {
        x = a;
    }
    ec.clear();
    std::string y = fs::weakly_canonical(fs::path(b), ec).string();
    if (ec) {
        y = b;
    }
    const auto fold = [](std::string s) {
        std::transform(s.begin(), s.end(), s.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        while (!s.empty() && (s.back() == '\\' || s.back() == '/')) {
            s.pop_back();
        }
        return s;
    };
    return fold(x) == fold(y);
}

static json read_local(const std::string & root) {
    const fs::path p = fs::path(root) / "local.json";
    std::ifstream in(p, std::ios::binary);
    if (!in) {
        return json::object();
    }
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    // the installer used to write a byte-order mark, which the parser rejects
    if (text.size() >= 3 && static_cast<unsigned char>(text[0]) == 0xEF &&
        static_cast<unsigned char>(text[1]) == 0xBB && static_cast<unsigned char>(text[2]) == 0xBF) {
        text.erase(0, 3);
    }
    json j = json::parse(text, nullptr, /*allow_exceptions=*/false);
    if (j.is_discarded() || !j.is_object()) {
        return json::object();
    }
    return j;
}

static std::string j_str(const json & j, const char * key) {
    const auto it = j.find(key);
    if (it == j.end() || !it->is_string()) {
        return "";
    }
    return it->get<std::string>();
}

Config load_config() {
    Config c;
    c.root = exe_dir();

    const json local = read_local(c.root);

    c.models_root = env_str("OLLAMA_MODELS", j_str(local, "models_root"));
    c.gguf_dir    = env_str("LLMASH_GGUF", j_str(local, "gguf_dir"));
    c.llama_bin   = env_str("LLAMA_BIN", j_str(local, "llama_bin"));

    const auto it = local.find("extra_roots");
    if (it != local.end() && it->is_array()) {
        for (const auto & e : *it) {
            if (e.is_string()) {
                c.extra_roots.push_back(e.get<std::string>());
            }
        }
    }

    c.port        = env_int("LLMASH_PORT", 11434);
    c.public_port = env_int("LLMASH_PUBLIC_PORT", 11435);
    c.ctx         = env_int("LLMASH_CTX", 8192);
    c.parallel    = env_int("LLMASH_PARALLEL", 1);
    c.kv_type     = env_str("LLMASH_KV", "f16");
    c.load_mode   = env_str("LLMASH_LOAD_MODE", "dio");
    c.keep_alive  = env_str("OLLAMA_KEEP_ALIVE", "15m");

    if (c.models_root.empty()) {
        const std::string home = env_str("USERPROFILE", env_str("HOME"));
        if (!home.empty()) {
            c.models_root = (fs::path(home) / ".ollama" / "models").string();
        }
    }
    return c;
}

} // namespace llmash
