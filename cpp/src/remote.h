#pragma once

#include "config.h"
#include "registry.h"

#include <atomic>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace llmash {

// A fast route: a model-name substring naming an OpenAI-compatible server
// that answers for it, brought up on demand as a Docker container or a plain
// executable. Fields mirror routes.json (or its older name, vllm_routes.json)
// directly.
struct Route {
    std::string match;
    std::string url;
    std::string container;
    std::string exe;
    std::string path;              // extra PATH entry, exe routes only
    std::vector<std::string> args; // exe launch arguments
};

struct RouteLoad {
    double   pct     = 0; // 0-99.9
    uint64_t total   = 0; // bytes, once known
    double   elapsed = 0; // seconds since the backend was asked to start
};

// Routes a model name to a remote backend: matches it against routes.json,
// probes whether the backend already answers, and starts/stops it on demand,
// falling back to the local llama.cpp path when it will not come up. One
// instance per server process.
class RemoteRouter {
public:
    RemoteRouter(Config cfg, Registry * reg);
    ~RemoteRouter();
    RemoteRouter(const RemoteRouter &)             = delete;
    RemoteRouter & operator=(const RemoteRouter &) = delete;

    // Reads <root>/routes.json, falling back to <root>/vllm_routes.json.
    // Never throws: a missing or unreadable file leaves the route list empty.
    void load_routes();
    std::vector<std::string> route_match_names() const;

    // First route whose "match" is a case-insensitive substring of name.
    const Route * match(const std::string & name) const;
    std::string   url_of(const Route & r) const;
    std::string   key_of(const Route & r) const; // identifies its lifetime: container name, or "exe " + url

    // Cached reachability probe: GET <url>/models with a short timeout.
    bool is_up(const Route & r);

    // Starts the backend if it is not already answering and waits for it.
    bool ensure_up(const Route & r);
    bool stop_for(const std::string & model_name);

    void   set_keep_alive(const std::string & key, double idle_seconds);
    double expiry(const std::string & key) const; // +inf when pinned or idle forever

    std::optional<RouteLoad> load_progress(const std::string & key) const;
    std::vector<std::string> loading_keys() const;
    std::vector<std::string> models_on(const std::string & key) const;

    void reap_orphans(); // kill an exe backend left running by an earlier llmash
    void reap_idle();    // one idle-timeout pass

    // Runs reap_idle every 30s until stop is set; run this on its own thread,
    // the way serve.go runs remoteReaper on its own goroutine.
    void reaper_loop(const std::atomic<bool> & stop);

private:
    struct LoadState;
    struct ExeProc;

    Config     cfg_;
    Registry * reg_;
    std::vector<Route> routes_;
    std::string default_container_;
    double remote_idle_       = 900;
    double remote_boot_wait_  = 45;
    double remote_load_guess_ = 20;

    mutable std::mutex mu_;
    std::map<std::string, std::pair<double, bool>> up_cache_;      // url -> (checked_at, ok)
    std::map<std::string, std::pair<double, bool>> running_cache_; // container -> (checked_at, running)
    std::map<std::string, double> last_used_;
    std::map<std::string, double> keep_;
    std::map<std::string, std::unique_ptr<std::mutex>> locks_;
    std::map<std::string, std::unique_ptr<LoadState>>  loads_;
    std::map<std::string, double> expect_;
    std::map<std::string, std::unique_ptr<ExeProc>> exe_procs_;
    mutable std::map<std::string, std::vector<std::string>> names_cache_;

    std::string  state_path() const;
    void         load_expect();
    void         save_expect() const;
    std::mutex & lock_for(const std::string & key);
    void         drop_up(const std::string & url);
    double       idle_for(const std::string & key) const;

    bool docker_running(const std::string & container);
    bool port_published(const std::string & container, const Route & r);

    bool start_exe(const Route & r, const std::string & key);
    void stop_exe(const Route & r, const std::string & key);
    bool exe_running(const Route & r, const std::string & key);

    void load_begin_docker(const std::string & container);
    void load_begin_exe(const std::string & key, ExeProc & proc);
    void load_end(const std::string & key, bool ok);
};

} // namespace llmash
