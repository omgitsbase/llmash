#pragma once

#include "config.h"
#include "registry.h"

#include <map>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace llmash {

// How a client wants the cache: its type, and whether it sits on the card.
struct LoadPrefs {
    std::string kv_type;         // empty: the server's default
    bool        kv_on_gpu = true;
};

class Instance {
public:
    Instance(Model m, int ctx, bool vision, Config * cfg);

    std::string url() const;
    void        touch();
    void        set_keep_alive(double ka);
    bool        ready() const;
    bool        on_gpu() const;
    double      vram_gb() const;
    bool        alive() const;
    std::string tail_log(size_t n) const;
    double      progress() const;
    std::string start();
    void        mark_loaded();
    void        stop();

    Model  model;
    int    ctx = 0;
    bool   vision = false;
    std::string kv_type;         // the cache type it was started with
    bool        kv_on_gpu = true;
    int    port = 0;
    double last_used = 0;
    double expires_at = 0;
    double keep_alive = 0;
    std::string err;
    std::string load_mode;
    std::string tune_note;
    std::string spec_note;
    std::string logfile;
    bool        stalled = false; // gave up on a load that stopped moving

private:
    Config * cfg_;
    mutable std::mutex mu_;
    unsigned long pid_ = 0;
    bool ready_ = false;
    bool on_gpu_ = false;
    bool plain_args_ = false;
    int exit_code_ = 0;
    bool exited_ = false;

    std::vector<std::string> args();
};

class Manager {
public:
    explicit Manager(Config cfg, Registry * reg);

    std::vector<Instance *> loaded(); // ready ones only
    std::vector<Instance *> live();   // every instance, still loading included
    Instance *              get(const std::string & name, int ctx, double keep_alive, bool vision, std::string & err,
                                const LoadPrefs & prefs = {});

    // What a context costs on this card, from the model's header.
    struct Fit {
        double weights_gb = 0, state_gb = 0, compute_gb = 0, free_gb = 0, total_gb = 0;
        double ram_gb     = 0;  // weights llama.cpp keeps in RAM: the input layer
        double per_tok_gb = 0;  // an f16 cache, per token
        int    native     = 0;
    };
    Fit            fit_report(const Model & m);
    double         fit_room(const Fit & f) const;
    int            fit_at(const Fit & f, int ctx, double kv_scale) const;
    nlohmann::json fit_json(const Model & m, int ctx);

    // The context an OpenAI-API request loads, and how: a forced or kept
    // choice, else the model's window fitted to the card. Advertised as such.
    int       v1_ctx(const Model & m);
    LoadPrefs v1_prefs(const Model & m) const;

    // local.json changed underneath a running server
    void set_config(const Config & c);
    bool                    unload(const std::string & name);
    Instance *              find(const std::string & name);
    void                    shutdown();
    void                    reap_idle(); // one keep_alive pass

private:
    Config      cfg_;
    Registry *  reg_;
    std::mutex  mu_;
    std::vector<std::unique_ptr<Instance>> live_;
    std::map<std::string, int> stalls_; // loads per model that stopped moving

    void   evict_for(double need_gb, const std::string & keep);
    void   drop_dead();
    int    fit_ctx(const Model & m, int ctx, const LoadPrefs & prefs);
};

struct Tuning {
    std::vector<std::string> flags;
    std::string              why;
};

// tune.go's autoTune: the flags to add and a one-line account of why.
Tuning auto_tune(int ctx);
// the micro-batch a context can afford, and the scratch it costs in GB
int    ubatch_for(int ctx);
double scratch_gb(int ctx);

int  free_port();
bool can_offload();
std::pair<int, uint64_t> cpu_threads_and_mask();
bool has_thinking_block(const std::string & chat_template);

} // namespace llmash
