#pragma once

#include "config.h"
#include "registry.h"

#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace llmash {

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
    int    port = 0;
    double last_used = 0;
    double expires_at = 0;
    double keep_alive = 0;
    std::string err;
    std::string load_mode;
    std::string tune_note;
    std::string spec_note;
    std::string logfile;

private:
    Config * cfg_;
    mutable std::mutex mu_;
    unsigned long pid_ = 0;
    bool ready_ = false;
    bool on_gpu_ = false;
    bool plain_args_ = false;
    int exit_code_ = 0;
    bool exited_ = false;

    std::vector<std::string> args() const;
};

class Manager {
public:
    explicit Manager(Config cfg, Registry * reg);

    std::vector<Instance *> loaded();
    Instance *              get(const std::string & name, int ctx, double keep_alive, bool vision, std::string & err);
    bool                    unload(const std::string & name);
    Instance *              find(const std::string & name);
    void                    shutdown();
    void                    reap_idle(); // one keep_alive pass

private:
    Config      cfg_;
    Registry *  reg_;
    std::mutex  mu_;
    std::vector<std::unique_ptr<Instance>> live_;

    void   evict_for(double need_gb, const std::string & keep);
    void   drop_dead();
    int    fit_ctx(const Model & m, int ctx);
};

struct Tuning {
    std::vector<std::string> flags;
    std::string              why;
};

// tune.go's autoTune: the flags to add and a one-line account of why.
Tuning auto_tune();

int  free_port();
bool can_offload();
std::pair<int, uint64_t> cpu_threads_and_mask();

} // namespace llmash
