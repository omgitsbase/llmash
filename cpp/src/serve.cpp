#include "serve.h"

#include "api.h"
#include "api_logic.h"
#include "cli_util.h"
#include "cli_win.h"
#include "config.h"
#include "log.h"
#include "manager.h"
#include "registry.h"
#include "remote.h"
#include "winproc.h"

#include <httplib.h>
#include <nlohmann/json.hpp>
#include <windows.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

namespace llmash {
namespace {

std::atomic<bool> g_stop{false};
httplib::Server * g_servers[2] = {nullptr, nullptr};

BOOL WINAPI on_console_ctrl(DWORD) {
    g_stop = true;
    for (httplib::Server * s : g_servers) {
        if (s != nullptr) {
            s->stop();
        }
    }
    return TRUE;
}

// LLMASH_LOGFILE, else llmash.log beside the program when there is no
// console, else stdout.
void setup_logging(const Config & cfg) {
    const std::string lf = env_str("LLMASH_LOGFILE");
    if (!lf.empty() && set_log_file(lf)) {
        return;
    }
    if (GetConsoleWindow() == nullptr) {
        if (set_log_file((fs::path(cfg.root) / "llmash.log").string())) {
            return;
        }
    }
    set_log_stream(stdout);
}

// Orphans from a previous run: a llama-server whose parent is gone.
int reap_orphans(const Config & cfg) {
    int killed = 0;
    const std::string runtime = fs::path(cfg.llama_bin).parent_path().string();
    for (const RunningProcess & p : processes_under(runtime)) {
        if (p.name == "llama-server.exe" && kill_pid(p.pid)) {
            killed++;
        }
    }
    if (killed > 0) {
        logf("reaped %d orphaned llama-server process(es) from a previous run", killed);
    }
    return killed;
}

void sleep_unless_stopped(int ms) {
    for (int t = 0; t < ms && !g_stop; t += 100) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

void manager_reaper(Manager & mgr) {
    while (!g_stop) {
        sleep_unless_stopped(5000);
        if (!g_stop) {
            mgr.reap_idle();
        }
    }
}

std::string dir_stamp(const Config & cfg) {
    std::string out;
    for (const std::string & d : {(fs::path(cfg.models_root) / "manifests").string(),
                                  (fs::path(cfg.models_root) / "blobs").string(), loose_dir(cfg)}) {
        std::error_code ec;
        const auto      t = fs::last_write_time(d, ec);
        out += ec ? "0;" : std::to_string(t.time_since_epoch().count()) + ";";
    }
    return out;
}

void write_atomic(const fs::path & dest, const std::string & text) {
    const fs::path tmp = dest.string() + ".tmp";
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!out) {
            return;
        }
        out << text;
    }
    std::error_code ec;
    fs::rename(tmp, dest, ec);
}

// The finished `list` and `ps` tables beside a heartbeat, so the CLI can
// print them without a round trip.
void cli_cache_tick(Config & cfg, Manager & mgr, Registry & reg) {
    const fs::path dir        = fs::path(cfg.root) / "cache";
    std::string    last_stamp;
    bool           have_list = false;
    while (!g_stop) {
        std::error_code ec;
        fs::create_directories(dir, ec);
        {
            std::ofstream alive(dir / "alive", std::ios::binary | std::ios::trunc);
        }
        const std::string stamp = dir_stamp(cfg);
        if (stamp != last_stamp || !have_list) {
            if (have_list) {
                reg.invalidate();
            }
            write_atomic(dir / "list.txt", cli_table("list", cfg, mgr, reg));
            last_stamp = stamp;
            have_list  = true;
        }
        write_atomic(dir / "ps.txt", cli_table("ps", cfg, mgr, reg));
        sleep_unless_stopped(2000);
    }
}

void tune_server(httplib::Server & srv) {
    srv.set_keep_alive_timeout(300);
    srv.set_read_timeout(300);
    srv.set_write_timeout(300);
    srv.set_payload_max_length(static_cast<size_t>(512) * 1024 * 1024);
}

} // namespace

int cmd_serve(const std::vector<std::string> & args) {
    Config cfg = load_config();
    setup_logging(cfg);

    int         port = cfg.port;
    std::string host = "127.0.0.1";
    for (size_t i = 0; i < args.size(); i++) {
        if (args[i] == "--port" && i + 1 < args.size()) {
            port = std::atoi(args[++i].c_str());
        } else if (args[i] == "--host" && i + 1 < args.size()) {
            host = args[++i];
        }
    }
    cfg.port = port;
    if (cfg.public_port == port) {
        cfg.public_port = 0;
        logf("public port %d is the main port; public API disabled", port);
    }
    std::error_code ec;
    if (cfg.llama_bin.empty() || !fs::is_regular_file(cfg.llama_bin, ec)) {
        logf("!! llama-server not found at %s", cfg.llama_bin.c_str());
        logf("   put a llama.cpp build in that runtime folder, set LLAMA_BIN, or re-run the installer, which "
             "downloads one for this machine");
        return 1;
    }

    // config.go: LLMASH_PIN, else local.json's "pin" list. The manager reads
    // the variable, so the file's list is handed over through it.
    if (env_str("LLMASH_PIN").empty()) {
        const nlohmann::json local = clidoc::read_local_json(cfg.root);
        std::string          pins;
        if (local.contains("pin") && local["pin"].is_array()) {
            for (const auto & p : local["pin"]) {
                if (p.is_string() && !p.get<std::string>().empty()) {
                    pins += (pins.empty() ? "" : ",") + p.get<std::string>();
                }
            }
        }
        if (!pins.empty()) {
            SetEnvironmentVariableA("LLMASH_PIN", pins.c_str());
        }
    }

    Registry     reg(cfg);
    Manager      mgr(cfg, &reg);
    RemoteRouter router(cfg, &reg);
    router.load_routes();

    if (env_str("LLMASH_NO_REAP").empty()) {
        reap_orphans(cfg);
        router.reap_orphans();
    }

    logf("models from %s", cfg.models_root.empty() ? loose_dir(cfg).c_str() : cfg.models_root.c_str());
    logf("llama-server %s", cfg.llama_bin.c_str());
    if (can_offload()) {
        logf("kv cache %s, GPU offload available", cfg.kv_type.c_str());
    } else {
        logf("kv cache %s, no GPU found: models will load into system RAM", cfg.kv_type.c_str());
    }
    if (cfg.public_port != 0) {
        logf("public API on :%d (key required), expose with `llmash link`", cfg.public_port);
    }

    httplib::Server main_srv;
    httplib::Server pub_srv;
    tune_server(main_srv);
    tune_server(pub_srv);
    register_routes(main_srv, cfg, mgr, reg);
    if (cfg.public_port != 0) {
        register_routes(pub_srv, cfg, mgr, reg);
    }

    if (!main_srv.bind_to_port(host, port)) {
        logf("!! %s:%d: could not listen (is another server on that port?)", host.c_str(), port);
        return 1;
    }
    std::thread pub_thread;
    if (cfg.public_port != 0) {
        if (pub_srv.bind_to_port("127.0.0.1", cfg.public_port)) {
            g_servers[1] = &pub_srv;
            pub_thread   = std::thread([&pub_srv] { pub_srv.listen_after_bind(); });
        } else {
            logf("!! 127.0.0.1:%d: could not listen; public API disabled", cfg.public_port);
        }
    }
    g_servers[0] = &main_srv;
    SetConsoleCtrlHandler(on_console_ctrl, TRUE);

    // The tray checks this instead of enumerating processes. Spawning a
    // PowerShell every five seconds to ask whether we are running cost a few
    // percent of a laptop's CPU for as long as the tray was up.
    write_pid_file(cfg.root, GetCurrentProcessId());

    std::thread reaper([&mgr] { manager_reaper(mgr); });
    std::thread remote_reaper([&router] { router.reaper_loop(g_stop); });
    std::thread cache([&cfg, &mgr, &reg] { cli_cache_tick(cfg, mgr, reg); });

    logf("listening on %s:%d", host.c_str(), port);
    main_srv.listen_after_bind();

    logf("shutting down");
    g_stop = true;
    pub_srv.stop();
    if (pub_thread.joinable()) {
        pub_thread.join();
    }
    reaper.join();
    remote_reaper.join();
    cache.join();
    remove_pid_file(cfg.root);
    mgr.shutdown();
    return 0;
}

} // namespace llmash
