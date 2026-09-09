#include "cli_win.h"
#include "cmd_models.h"

#include "api_logic.h"

#include "cli_util.h"
#include "config.h"
#include "gguf.h"
#include "registry.h"

#include <httplib.h>
#include <nlohmann/json.hpp>
#include <subprocess.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <regex>
#include <thread>

namespace fs  = std::filesystem;
using    json = nlohmann::json;
using namespace llmash::clidoc;

namespace llmash {

namespace {

[[noreturn]] void die(const std::string & msg) {
    std::fprintf(stderr, "%s\n", msg.c_str());
    std::exit(1);
}

template <typename... Args> [[noreturn]] void dief(const char * fmt, Args... args) {
    char buf[2048];
    std::snprintf(buf, sizeof(buf), fmt, args...);
    die(buf);
}

std::string trim_copy(std::string s) {
    const auto notspace = [](unsigned char c) { return !std::isspace(c); };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), notspace));
    s.erase(std::find_if(s.rbegin(), s.rend(), notspace).base(), s.end());
    return s;
}

std::string env_str(const char * name) {
    const char * v = std::getenv(name);
    return v ? v : "";
}

std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

// registry.cpp classifies exactly these files out of a folder scan (they are
// drafters, projectors, or a shard after the first) but keeps that logic to
// itself; there is no public accessor for it, so the two regexes and the tag
// list are repeated here, only for the "N models, M skipped" line.
bool is_sidecar(const std::string & stem) {
    const std::string s = lower(stem);
    for (const char * tag : {".mtp", ".eagle3", ".dspark", ".draft", ".mmproj", "-mmproj", "eagle3", "dspark",
                             "draftmodel", "speculator"}) {
        if (s.find(tag) != std::string::npos) {
            return true;
        }
    }
    return s.rfind("mmproj", 0) == 0;
}

bool is_later_shard(const std::string & stem) {
    static const std::regex shard(R"(-\d{5}-of-\d{5}$)");
    static const std::regex first(R"(-00001-of-\d{5}$)");
    return std::regex_search(stem, shard) && !std::regex_search(stem, first);
}

bool contains_dir(const std::vector<std::string> & xs, const std::string & dir) {
    return std::any_of(xs.begin(), xs.end(), [&](const std::string & x) { return same_dir(x, dir); });
}

int count_manifests(const std::string & store) {
    int             n = 0;
    std::error_code ec;
    const fs::path  manifests = fs::path(store) / "manifests";
    if (!fs::is_directory(manifests, ec)) {
        return 0;
    }
    for (auto it = fs::recursive_directory_iterator(manifests, fs::directory_options::skip_permission_denied, ec);
        it != fs::recursive_directory_iterator(); it.increment(ec)) {
        if (ec) {
            ec.clear();
            continue;
        }
        if (it->is_regular_file(ec)) {
            n++;
        }
    }
    return n;
}

// registry.h's is_ollama_store plus its own public walk_gguf: no folder-type
// detection is reimplemented here, only composed.
bool is_gguf_folder(Registry & reg, const std::string & dir) {
    return dir_exists(dir) && !reg.is_ollama_store(dir) && !walk_gguf(dir).empty();
}

std::vector<std::string> local_string_list(const json & cfg, const char * key) {
    std::vector<std::string> out;
    const auto               it = cfg.find(key);
    if (it != cfg.end() && it->is_array()) {
        for (const auto & v : *it) {
            if (v.is_string()) {
                std::string s = trim_copy(v.get<std::string>());
                if (!s.empty()) {
                    out.push_back(s);
                }
            }
        }
    }
    return out;
}

// tray.go/models.go's reloadPaths and restartForPaths: models set only ever
// calls these when a server is actually up.
void reload_paths() {
    if (!server_up()) {
        return;
    }
    json        out;
    std::string err;
    const json  empty = json::object();
    if (!call_json("POST", "/api/paths", &empty, 30, out, err)) {
        std::printf("the running server could not reload (%s); restart it from the tray\n", err.c_str());
    }
}


bool tray_server_up() {
    httplib::Client cli(server_host());
    cli.set_connection_timeout(1, 500000);
    cli.set_read_timeout(1, 500000);
    return static_cast<bool>(cli.Get("/"));
}

void stop_server_process(const std::string & root) {
    const std::string r    = ps_quote(root);
    const std::string mine = "($_.CommandLine -like '*" + r + "\\llmashw.exe*' -or $_.CommandLine -like '*" + r +
                            "\\llmash.exe*') -and $_.CommandLine -like '* serve*'";
    const std::string script =
        "$s = @(Get-CimInstance Win32_Process -Filter \"Name='llmashw.exe' OR Name='llmash.exe'\" "
        "| Where-Object { " + mine + " }); "
        "$ids = @($s | ForEach-Object { $_.ProcessId }); "
        "if ($ids.Count) { Get-CimInstance Win32_Process -Filter \"Name='llama-server.exe'\" "
        "| Where-Object { $ids -contains $_.ParentProcessId } "
        "| ForEach-Object { Stop-Process -Id $_.ProcessId -Force } }; "
        "$s | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }";
    run_with_timeout("powershell", {"-NoProfile", "-Command", script}, 15000);
    for (int waited = 0; waited < 15000; waited += 500) {
        if (!tray_server_up()) {
            return;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
}

bool start_server_process(const std::string & root) {
    if (tray_server_up()) {
        return true;
    }
    const fs::path exe = fs::path(root) / "llmashw.exe";
    if (!file_exists(exe.string())) {
        return false;
    }
    const std::string exe_str = exe.string();
    const char *      argv[]  = {exe_str.c_str(), "serve", nullptr};
    subprocess_s      proc{};
    if (subprocess_create(argv, subprocess_option_no_window | subprocess_option_inherit_environment, &proc) == 0) {
        subprocess_destroy(&proc); // fire-and-forget: closes our handles, not the child
    }
    for (int waited = 0; waited < 30000; waited += 400) {
        if (tray_server_up()) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(400));
    }
    return false;
}

void restart_for_paths(const Config & cfg) {
    if (!server_up()) {
        return;
    }
    std::printf("restarting the server ... ");
    std::fflush(stdout);
    stop_server_process(cfg.root);
    if (start_server_process(cfg.root)) {
        std::printf("done\n");
    } else {
        std::printf("\nit did not come back; start it with:  %s tray\n", prog_name().c_str());
    }
}

void show_model_dirs(const Config & cfg, Registry & reg) {
    struct Row {
        std::string kind, dir, note;
    };
    std::vector<Row> rows;

    char note[256];
    std::snprintf(note, sizeof(note), "%d models; pulls go here", count_manifests(cfg.models_root));
    rows.push_back({"store", cfg.models_root, note});

    const std::string loose = loose_dir(cfg);
    if (dir_exists(loose)) {
        std::snprintf(note, sizeof(note), "%d models; HuggingFace pulls go here", count_library(loose).first);
        rows.push_back({"loose", loose, note});
    }

    for (const auto & e : cfg.extra_roots) {
        std::snprintf(note, sizeof(note), "%d models; an older store, still read", count_manifests(e));
        rows.push_back({"also", e, note});
    }

    // models.go's libraryDirs(): the model store, and only when it is a
    // folder of GGUFs rather than an Ollama store.
    std::vector<std::string> folders;
    if (!cfg.models_root.empty() && !reg.is_ollama_store(cfg.models_root) && !same_dir(cfg.models_root, loose)) {
        folders.push_back(cfg.models_root);
    }
    for (const auto & l : folders) {
        if (!dir_exists(l)) {
            rows.push_back({"folder", l, "missing"});
            continue;
        }
        const auto  clean_skipped = count_library(l);
        std::string n             = std::to_string(clean_skipped.first) + " models, read in place";
        if (clean_skipped.second > 0) {
            n += ", " + std::to_string(clean_skipped.second) + " files skipped";
        }
        rows.push_back({"folder", l, n});
    }

    size_t width = 0;
    for (const auto & r : rows) {
        width = std::max(width, r.dir.size());
    }
    for (const auto & r : rows) {
        std::printf("  %-7s %-*s  %s\n", r.kind.c_str(), static_cast<int>(width), r.dir.c_str(), r.note.c_str());
    }

    std::vector<std::string> guesses;
    std::string              v = env_str("OLLAMA_MODELS");
    if (v.empty()) {
        v = env_str("LLMASH_MODELS");
    }
    if (!v.empty()) {
        guesses.push_back(v);
    }
    const std::string home = env_str("USERPROFILE");
    if (!home.empty()) {
        guesses.push_back((fs::path(home) / ".ollama" / "models").string());
    }
    const std::string local_appdata = env_str("LOCALAPPDATA");
    if (!local_appdata.empty()) {
        guesses.push_back((fs::path(local_appdata) / "Ollama" / "models").string());
    }
    for (const auto & g : guesses) {
        if (reg.is_ollama_store(g) && !same_dir(g, cfg.models_root) && !contains_dir(cfg.extra_roots, g)) {
            std::printf("\nOllama's models at %s are not being read:  %s models set %s\n", g.c_str(),
                       prog_name().c_str(), g.c_str());
        }
    }
}

void set_model_dir(Config & cfg, Registry & reg, const std::string & dir) {
    for (const char * name : {"OLLAMA_MODELS", "LLMASH_MODELS"}) {
        const std::string v = trim_copy(env_str(name));
        if (!v.empty() && !same_dir(v, dir)) {
            dief("%s is set to %s in your environment, and that wins.\nChange it there, or unset it:  setx %s \"\"",
                name, v.c_str(), name);
        }
    }

    json local = read_local_json(cfg.root);

    if (reg.is_ollama_store(dir)) {
        const std::string old = cfg.models_root;
        if (same_dir(dir, old)) {
            std::printf("%s is already the model store (%d models)\n", dir.c_str(), count_manifests(dir));
            return;
        }
        std::vector<std::string> extras = local_string_list(local, "extra_roots");
        if (!old.empty() && reg.is_ollama_store(old) && !contains_dir(extras, old)) {
            extras.push_back(old);
        }
        local["models_root"] = dir;
        local["extra_roots"] = extras;
        std::string err;
        if (!write_local_json(cfg.root, local, err)) {
            dief("Error: %s", err.c_str());
        }
        _putenv_s("OLLAMA_MODELS", dir.c_str());
        std::printf("model store is now %s (%d models; pulls go here)\n", dir.c_str(), count_manifests(dir));
        restart_for_paths(cfg);
        return;
    }

    if (!is_gguf_folder(reg, dir)) {
        dief("%s has no GGUF in it and is not an Ollama store, so there is nothing to read", dir.c_str());
    }
    local["models_root"] = dir;
    std::string err;
    if (!write_local_json(cfg.root, local, err)) {
        dief("Error: %s", err.c_str());
    }
    const auto  clean_skipped = count_library(dir);
    std::string line          = "reading " + dir + " (" + std::to_string(clean_skipped.first) + " models";
    if (clean_skipped.second > 0) {
        line += ", " + std::to_string(clean_skipped.second) + " files skipped";
    }
    line += ")";
    std::printf("%s\n", line.c_str());
    reload_paths();
}

} // namespace

std::pair<int, int> count_library(const std::string & dir) {
    int clean = 0, skipped = 0;
    for (const auto & p : walk_gguf(dir)) {
        const std::string stem = fs::path(p).stem().string();
        if (is_sidecar(stem) || is_later_shard(stem)) {
            continue;
        }
        if (read_gguf(p).arch.empty()) {
            skipped++;
        } else {
            clean++;
        }
    }
    return {clean, skipped};
}

void cmd_models(const std::vector<std::string> & args) {
    std::vector<std::string> pos;
    for (const auto & a : args) {
        if (a.size() > 1 && a[0] == '-') {
            dief("Error: unknown flag: '%s'", a.c_str());
        }
        pos.push_back(a);
    }

    Config   cfg = load_config();
    Registry reg(cfg);

    if (pos.empty()) {
        show_model_dirs(cfg, reg);
        return;
    }
    const std::string dir = pos.back();
    if (pos[0] == "set" && pos.size() < 2) {
        dief("Error: %s models set needs a folder", prog_name().c_str());
    }
    if (pos[0] != "set" && pos.size() > 1) {
        dief("Error: unknown command \"%s\" for \"%s models\"", pos[0].c_str(), prog_name().c_str());
    }

    std::error_code ec;
    const fs::path  abs = fs::absolute(dir, ec);
    if (ec) {
        dief("Error: %s", ec.message().c_str());
    }
    const std::string abs_str = abs.lexically_normal().string();
    if (!dir_exists(abs_str)) {
        dief("Error: %s is not a folder", abs_str.c_str());
    }
    set_model_dir(cfg, reg, abs_str);
}

} // namespace llmash
