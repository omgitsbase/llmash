#include "config.h"
#include "platform.h"

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
    // /proc/self/exe resolves the symlink an install puts on PATH, so this is
    // the real install directory and not wherever the caller happened to be.
    std::error_code ec;
    const fs::path self = fs::read_symlink("/proc/self/exe", ec);
    if (!ec && !self.empty()) {
        return self.parent_path().string();
    }
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

// LLAMA_BIN, local.json, the install's runtime folder, PATH.
std::string find_llama_bin(const std::string & root, const std::string & from_local) {
    const std::string env = env_str("LLAMA_BIN");
    if (!env.empty()) {
        return env;
    }
    if (!from_local.empty()) {
        return from_local;
    }
    std::vector<std::string> cands;
    for (const char * v : {"ProgramData", "LOCALAPPDATA"}) {
        const std::string base = env_str(v);
        if (!base.empty()) {
            cands.push_back((fs::path(base) / "llmash" / "runtime" / llama_server_exe()).string());
        }
    }
    cands.push_back((fs::path(root) / "runtime" / llama_server_exe()).string());
    cands.push_back((fs::path(root) / "llama.cpp" / llama_server_exe()).string());
    for (const std::string & c : cands) {
        std::error_code ec;
        if (fs::is_regular_file(c, ec)) {
            return c;
        }
    }
    // and finally whatever is on PATH
    const std::string exe  = llama_server_exe();
    const std::string path = env_str("PATH");
#ifdef _WIN32
    const char sep = ';';
#else
    const char sep = ':';
#endif
    size_t start = 0;
    while (start <= path.size()) {
        const size_t at  = path.find(sep, start);
        const std::string dir = path.substr(start, at == std::string::npos ? std::string::npos : at - start);
        if (!dir.empty()) {
            std::error_code ec;
            const fs::path cand = fs::path(dir) / exe;
            if (fs::is_regular_file(cand, ec)) {
                return cand.string();
            }
        }
        if (at == std::string::npos) {
            break;
        }
        start = at + 1;
    }
    return "";
}

std::string install_root(const std::string & exe_directory) {
    const fs::path  dir = exe_directory;
    std::error_code ec;
    std::string     name = dir.filename().string();
    std::transform(name.begin(), name.end(), name.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    // The installer puts a second copy of the program in <root>/bin so one
    // shim is on PATH. Run from there, the install is the folder above: the
    // tray binary, local.json, the logs and the cache all live in it.
    if (name == "bin" && fs::is_regular_file(dir.parent_path() / llmash_exe(), ec)) {
        return dir.parent_path().string();
    }
    return exe_directory;
}

Config load_config() {
    Config c;
    c.root = install_root(exe_dir());

    const json local = read_local(c.root);

    c.models_root = env_str("OLLAMA_MODELS", j_str(local, "models_root"));
    c.gguf_dir    = env_str("LLMASH_GGUF", j_str(local, "gguf_dir"));
    c.llama_bin   = find_llama_bin(c.root, j_str(local, "llama_bin"));

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

    for (const char * key : {"ctx_override", "ctx_max"}) {
        const auto it = local.find(key);
        if (it == local.end() || !it->is_object()) {
            continue;
        }
        auto & dest = std::string(key) == "ctx_max" ? c.ctx_max : c.ctx_override;
        for (const auto & [k, v] : it->items()) {
            if (v.is_number()) {
                std::string low = k;
                std::transform(low.begin(), low.end(), low.begin(), [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
                dest[low] = v.get<int>();
            }
        }
    }

    if (const auto it = local.find("launch_extra"); it != local.end() && it->is_object()) {
        for (const auto & [k, v] : it->items()) {
            std::vector<std::string> flags;
            if (v.is_array()) {
                for (const auto & e : v) {
                    if (e.is_string()) {
                        flags.push_back(e.get<std::string>());
                    }
                }
            }
            if (!flags.empty()) {
                std::string low = k;
                std::transform(low.begin(), low.end(), low.begin(),
                               [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
                c.launch_extra[low] = flags;
            }
        }
    }
    if (const auto it = local.find("no_mmproj"); it != local.end() && it->is_array()) {
        for (const auto & e : *it) {
            if (e.is_string()) {
                c.no_mmproj.push_back(e.get<std::string>());
            }
        }
    }

    const std::string pin_env = env_str("LLMASH_PIN");
    if (!pin_env.empty()) {
        size_t start = 0;
        while (start <= pin_env.size()) {
            const size_t comma = pin_env.find(',', start);
            std::string  one   = pin_env.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
            while (!one.empty() && std::isspace(static_cast<unsigned char>(one.front()))) one.erase(one.begin());
            while (!one.empty() && std::isspace(static_cast<unsigned char>(one.back()))) one.pop_back();
            if (!one.empty()) {
                c.pin.push_back(one);
            }
            if (comma == std::string::npos) {
                break;
            }
            start = comma + 1;
        }
    } else if (const auto it = local.find("pin"); it != local.end() && it->is_array()) {
        for (const auto & e : *it) {
            if (e.is_string()) {
                c.pin.push_back(e.get<std::string>());
            }
        }
    }

    if (c.models_root.empty()) {
        const std::string home = env_str("USERPROFILE", env_str("HOME"));
        if (!home.empty()) {
            c.models_root = (fs::path(home) / ".ollama" / "models").string();
        }
    }
    return c;
}


namespace {
int match_key(const std::map<std::string, int> & table, const std::string & name) {
    std::string low = name;
    std::transform(low.begin(), low.end(), low.begin(), [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    for (const auto & [k, v] : table) {
        if (low.find(k) != std::string::npos) {
            return v;
        }
    }
    return 0;
}
} // namespace

int ctx_target(const Config & cfg, const std::string & name) { return match_key(cfg.ctx_override, name); }

int ctx_ceiling(const Config & cfg, const std::string & name, int native) {
    const int v = match_key(cfg.ctx_max, name);
    return v > native ? v : native;
}

std::vector<std::string> launch_extra_for(const Config & cfg, const std::string & name) {
    std::string low = name;
    std::transform(low.begin(), low.end(), low.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    std::vector<std::string> out;
    for (const auto & [k, flags] : cfg.launch_extra) {
        if (low.find(k) != std::string::npos) {
            out.insert(out.end(), flags.begin(), flags.end());
        }
    }
    return out;
}

bool mmproj_blocked(const Config & cfg, const std::string & name) {
    std::string low = name;
    std::transform(low.begin(), low.end(), low.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    for (const std::string & k : cfg.no_mmproj) {
        std::string lk = k;
        std::transform(lk.begin(), lk.end(), lk.begin(),
                       [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
        if (!lk.empty() && low.find(lk) != std::string::npos) {
            return true;
        }
    }
    return false;
}

} // namespace llmash
