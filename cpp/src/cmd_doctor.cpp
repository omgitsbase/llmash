#include "cmd_doctor.h"

#include "manager.h"
#include "platform.h"
#include "winproc.h"

#include "version.h"

#include "cli_util.h"
#include "cmd_models.h" // count_library, shared with `models`
#include "config.h"
#include "registry.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace fs  = std::filesystem;
using    json = nlohmann::json;
using namespace llmash::clidoc;

namespace llmash {

namespace {

enum State { ST_OK, ST_WARN, ST_FAIL };

struct Check {
    State       state;
    std::string name;
    std::string detail;
};

struct Report {
    std::vector<Check> rows;

    template <typename... Args> void add(State st, const std::string & name, const char * fmt, Args... args) {
        char buf[4096];
        std::snprintf(buf, sizeof(buf), fmt, args...);
        rows.push_back({st, name, buf});
    }
};

// cmds.go/doctor.go both read install.json the same way: origin defaults to
// "dev" and any directory not written by the installer (dev == true) is
// treated as a checkout, not an install.
struct Install {
    std::string origin = "dev";
    bool        dev     = true;
};

Install read_install_json(const std::string & root) {
    Install info;
    std::ifstream in(fs::path(root) / "install.json", std::ios::binary);
    if (!in) {
        return info;
    }
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (text.size() >= 3 && static_cast<unsigned char>(text[0]) == 0xEF) {
        text.erase(0, 3); // a BOM PowerShell may have written
    }
    const json j = json::parse(text, nullptr, false);
    if (j.is_discarded() || !j.is_object()) {
        return info;
    }
    if (j.contains("origin") && j["origin"].is_string() && !j["origin"].get<std::string>().empty()) {
        info.origin = j["origin"].get<std::string>();
    }
    if (j.contains("dev") && j["dev"].is_boolean()) {
        info.dev = j["dev"].get<bool>();
    }
    return info;
}

bool is_dev_install(const Install & inst) { return inst.dev || inst.origin == "dev"; }

std::vector<std::string> split(const std::string & s, const std::string & sep) {
    std::vector<std::string> out;
    size_t                    start = 0;
    for (;;) {
        const size_t pos = s.find(sep, start);
        out.push_back(s.substr(start, pos == std::string::npos ? std::string::npos : pos - start));
        if (pos == std::string::npos) {
            break;
        }
        start = pos + sep.size();
    }
    return out;
}

} // namespace

void cmd_doctor() {
    const Config cfg = load_config();
    Registry     reg(cfg);
    const std::string prog = prog_name();

    Report d;

    // ---- this install ------------------------------------------------
    d.add(ST_OK, "version", "%s %s", prog.c_str(), version_string(cfg).c_str());
    const Install inst = read_install_json(cfg.root);
    std::string   kind = "installed from " + inst.origin;
    if (is_dev_install(inst)) {
        kind = "source checkout";
    }
    d.add(ST_OK, "install", "%s  (%s)", cfg.root.c_str(), kind.c_str());

    // ---- the server ----------------------------------------------------
    {
        json        ver_json;
        std::string err;
        if (call_json("GET", "/api/version", nullptr, 3, ver_json, err)) {
            const std::string server_ver = ver_json.value("version", std::string());
            d.add(ST_OK, "server", "answering at %s, version %s", server_host().c_str(), server_ver.c_str());
            if (server_ver != version_string(cfg)) {
                d.add(ST_WARN, "server version", "server %s but this command is %s; restart the server",
                     server_ver.c_str(), version_string(cfg).c_str());
            }
        } else {
            d.add(ST_FAIL, "server", "not answering at %s (%s)", server_host().c_str(), err.c_str());
        }
    }

    // ---- processes -------------------------------------------------------
    {
        const unsigned long server_pid = read_pid_file(cfg.root);
        bool                tray_up    = false;
        bool                serve_up   = pid_alive(server_pid);
        {
            for (const RunningProcess & p : processes_under(cfg.root)) {
                if (p.name == llmash_daemon_exe() && p.pid != server_pid) {
                    tray_up = true;
                }
            }
        }
        if (tray_up && serve_up) {
            d.add(ST_OK, "processes", "tray and server are running");
        } else if (serve_up) {
            d.add(ST_WARN, "processes", "server is running without the tray");
        } else {
            d.add(ST_WARN, "processes", "no llmashw.exe running from this install");
        }
    }

    // ---- the runtime -----------------------------------------------------
    if (file_exists(cfg.llama_bin)) {
        std::string ver = "version unknown";
        const auto [out, ok] = run_with_timeout(cfg.llama_bin, {"--version"}, 8000);
        if (ok) {
            for (const auto & line : split(out, "\n")) {
                if (line.find("build") != std::string::npos || line.find("version") != std::string::npos) {
                    ver = line;
                    while (!ver.empty() && std::isspace(static_cast<unsigned char>(ver.back()))) {
                        ver.pop_back();
                    }
                    break;
                }
            }
        }
        d.add(ST_OK, "llama-server", "%s  (%s)", cfg.llama_bin.c_str(), ver.c_str());
    } else {
        d.add(ST_FAIL, "llama-server", "missing at %s; reinstall or set LLAMA_BIN", cfg.llama_bin.c_str());
    }

    // ---- models ------------------------------------------------------
    {
        json        tags;
        std::string err;
        if (call_json("GET", "/api/tags", nullptr, 20, tags, err)) {
            double n_models = 0;
            double bytes    = 0;
            if (tags.contains("models") && tags["models"].is_array()) {
                n_models = static_cast<double>(tags["models"].size());
                for (const auto & m : tags["models"]) {
                    bytes += m.value("size", 0.0);
                }
            }
            d.add(ST_OK, "models", "%.0f, %s total", n_models, human_bytes(static_cast<uint64_t>(bytes)).c_str());
            if (n_models == 0) {
                d.add(ST_WARN, "models", "the store is empty; pull one with `%s pull <name>`", prog.c_str());
            }
        }
    }

    // ---- the model store, as the live server sees it ----------------
    {
        json        paths;
        std::string err;
        if (call_json("GET", "/api/paths", nullptr, 5, paths, err)) {
            const std::string store   = paths.value("models", std::string());
            const double      free_gb = free_disk_gb(store);
            const State        state   = (free_gb > 0 && free_gb < 20) ? ST_WARN : ST_OK;
            d.add(state, "model store", "%s, %.0f GB free", store.c_str(), free_gb);
            if (paths.contains("library") && paths["library"].is_array()) {
                for (const auto & lv : paths["library"]) {
                    const std::string dir = lv.is_string() ? lv.get<std::string>() : std::string();
                    if (dir.empty()) {
                        continue;
                    }
                    const auto clean_skipped = count_library(dir);
                    const int  clean = clean_skipped.first, skipped = clean_skipped.second;
                    if (clean == 0) {
                        d.add(ST_WARN, "models folder", "%s yields no models; `%s models set` picks another",
                             dir.c_str(), prog.c_str());
                    } else if (skipped > 0) {
                        d.add(ST_WARN, "models folder", "%s, %d models read in place, %d files skipped", dir.c_str(),
                             clean, skipped);
                    } else {
                        d.add(ST_OK, "models folder", "%s, %d models read in place", dir.c_str(), clean);
                    }
                }
            }
        }
    }
    if (!cfg.models_root.empty() && !reg.is_ollama_store(cfg.models_root) && !dir_exists(cfg.models_root)) {
        d.add(ST_WARN, "models folder", "%s is missing; `%s models set` picks another", cfg.models_root.c_str(),
              prog.c_str());
    }

    // ---- the card ----------------------------------------------------
    std::vector<std::string> gpu_fields;
    {
        const auto [out, ok] = run_with_timeout(
            "nvidia-smi", {"--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu",
                          "--format=csv,noheader,nounits"},
            6000);
        if (ok && !out.empty()) {
            gpu_fields = split(out, ", ");
        }
        if (gpu_fields.size() >= 5) {
            d.add(ST_OK, "gpu", "%s, driver %s, %s of %s MiB used, %s%% busy", gpu_fields[0].c_str(),
                 gpu_fields[1].c_str(), gpu_fields[3].c_str(), gpu_fields[2].c_str(), gpu_fields[4].c_str());
        } else {
            d.add(ST_WARN, "gpu", "nvidia-smi did not answer; running on CPU or the driver is busy");
        }
    }

    // manager.h does not expose a VRAM query (that is where Go's freeVRAMGB
    // lives, via NVML/DXGI); nvidia-smi's own totals, already fetched above,
    // stand in for it here.
    double free_vram_gb = 0;
    if (gpu_fields.size() >= 4) {
        try {
            const double total_mib = std::stod(gpu_fields[2]);
            const double used_mib  = std::stod(gpu_fields[3]);
            free_vram_gb           = (total_mib - used_mib) / 1024.0;
        } catch (const std::exception &) {
            free_vram_gb = 0;
        }
    }
    const double free_ram = free_ram_gb();
    d.add(free_vram_gb < 4 ? ST_WARN : ST_OK, "headroom", "%.1f GB VRAM free, %.1f GB RAM free", free_vram_gb,
         free_ram);

    // ---- what is loaded --------------------------------------------------
    {
        json        ps;
        std::string err;
        if (call_json("GET", "/api/ps", nullptr, 5, ps, err)) {
            const bool has_rows = ps.contains("models") && ps["models"].is_array() && !ps["models"].empty();
            if (!has_rows) {
                d.add(ST_OK, "loaded", "nothing resident");
            } else {
                for (const auto & m : ps["models"]) {
                    const std::string name    = m.value("name", std::string());
                    const uint64_t     size_v  = static_cast<uint64_t>(m.value("size_vram", 0.0));
                    const std::string expires = m.value("expires_at", std::string());
                    d.add(ST_OK, "loaded", "%s, %s, until %s", name.c_str(), human_bytes(size_v).c_str(),
                         human_time_iso(expires, "never").c_str());
                }
            }
        }
    }

    if (const Tuning tuning = auto_tune(); !tuning.flags.empty()) {
        d.add(ST_OK, "auto-tuning", "%s", tuning.why.c_str());
    }

    // ---- speculation ---------------------------------------------------
    const std::string spec_fallback = env_str("LLMASH_SPEC_FALLBACK", "ngram-mod");
    if (!spec_fallback.empty() && spec_fallback != "none") {
        d.add(ST_OK, "speculation", "models without a drafter of their own use %s", spec_fallback.c_str());
    } else {
        d.add(ST_WARN, "speculation", "no fallback drafter; set LLMASH_SPEC_FALLBACK=ngram-mod");
    }

    // fast backends: doctor.go walks remote.go's fastRoutes, which has no
    // C++ counterpart yet (no route config lives in Config); skipped, see
    // the job report.

    // ---- reachability of the commands ------------------------------------
    for (const char * n : {"llmash", "ollama", "llamash"}) {
        const std::string p = which(n);
        if (p.empty()) {
            if (std::string(n) != "ollama") {
                d.add(ST_WARN, std::string("command ") + n, "not on PATH");
            }
            continue;
        }
        const fs::path pp = p;
        const bool     under_root =
            same_dir(pp.parent_path().parent_path().string(), cfg.root) || same_dir(pp.parent_path().string(), cfg.root);
        if (under_root) {
            d.add(ST_OK, std::string("command ") + n, "%s", p.c_str());
        } else if (is_dev_install(inst)) {
            d.add(ST_OK, std::string("command ") + n, "%s (the installed copy, not this checkout)", p.c_str());
        } else {
            d.add(ST_WARN, std::string("command ") + n, "%s points at a different install", p.c_str());
        }
    }

    // ---- updates -----------------------------------------------------
    if (is_dev_install(inst)) {
        d.add(ST_OK, "updates", "source checkout; rebuild with python build.py --here");
    } else {
        Release      rel;
        std::string  err;
        const std::string slug = repo_slug();
        if (!latest_release(slug, rel, err)) {
            d.add(ST_WARN, "updates", "%s", err.c_str());
        } else {
            const std::string rel_ver = release_version(rel.tag);
            if (rel_ver == version_string(cfg)) {
                d.add(ST_OK, "updates", "%s has %s; up to date", slug.c_str(), rel_ver.c_str());
            } else {
                d.add(ST_WARN, "updates", "%s has %s; run `%s update`", slug.c_str(), rel_ver.c_str(), prog.c_str());
            }
        }
    }

    // ---- print -------------------------------------------------------
    size_t width = 0;
    for (const auto & r : d.rows) {
        width = std::max(width, r.name.size());
    }
    int bad = 0;
    for (const auto & r : d.rows) {
        const char * mark  = "ok  ";
        const char * color = "\x1b[38;5;42m"; // green
        if (r.state == ST_WARN) {
            mark  = "warn";
            color = "\x1b[38;5;208m"; // orange
            bad++;
        } else if (r.state == ST_FAIL) {
            mark  = "FAIL";
            color = "\x1b[38;5;203m";
            bad++;
        }
        std::printf("%s%s\x1b[0m  %-*s  %s\n", color, mark, static_cast<int>(width), r.name.c_str(), r.detail.c_str());
    }
    std::printf("\n");
    if (bad == 0) {
        std::printf("everything checks out\n");
    } else {
        std::printf("%d thing(s) worth a look\n", bad);
    }
    for (const auto & r : d.rows) {
        if (r.state == ST_FAIL) {
            std::exit(1);
        }
    }
}

} // namespace llmash
