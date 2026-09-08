#pragma once

// The testable core of the API layer: JSON shaping, the API-key check, and
// the file-system work behind /api/create, /api/copy and /api/delete. Kept
// free of httplib and manager.h so it links and runs standalone, without a
// live server or a real Instance (whose methods are defined in manager.cpp,
// not yet written). api.cpp adapts httplib::Request/Response to these calls.

#include "config.h"
#include "registry.h"

#include <nlohmann/json.hpp>

#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

struct subprocess_s; // sheredom/subprocess.h; kept out of this header

namespace llmash {

// A snapshot of the fields ps() needs from a live Instance, read by the
// caller through manager.h's public API (vram_gb(), on_gpu(), ready(), the
// public last_used/expires_at/keep_alive/ctx fields) since Instance itself
// cannot be constructed or linked here.
struct InstanceView {
    Model  model;
    double vram_gb   = 0;
    bool   on_gpu    = false;
    bool   ready     = true;
    double last_used = 0;
    double expires_at = 0;
    double keep_alive = 0;
    int    ctx = 0;
};

// ------------------------------------------------------------ formatting

// Go's `05.999999+00:00`: microseconds shown with trailing zeros trimmed,
// and the fractional part dropped entirely when it is zero.
std::string iso(double unix_seconds);
double      now_unix();
std::string human_bytes(double n);

// ------------------------------------------------------------ model JSON

// Architectures llama.cpp cannot load; kept in sync with Go's badArch. A
// per-model runtime "broken" flag also gates this in Go (set after a load
// fails) but has no home in the given Manager/Registry contracts yet.
bool loadable(const Model & m);

// A default context length. Go's advertisedCtx() additionally applies
// per-model overrides and a remote-backend ceiling that live in tune.go /
// remote.go, neither of which this Config carries yet.
int advertised_ctx(const Config & cfg);

nlohmann::json tag_entry_json(const Model & m, const Config & cfg);
nlohmann::json tags_json(const std::vector<Model> & models, const Config & cfg);
nlohmann::json v1_entry_json(const Model & m, const Config & cfg);
nlohmann::json v1_models_json(const std::vector<Model> & models, const Config & cfg);
nlohmann::json show_json(const Model & m, const Config & cfg);
nlohmann::json openai_error_json(const std::string & message, const std::string & type, const std::string & code);

nlohmann::json ps_entry_json(const InstanceView & v, const Config & cfg);
nlohmann::json ps_json(const std::vector<InstanceView> & live, const Config & cfg);

// A cheap, non-cryptographic stand-in: registry.h's Model carries no digest
// (an ollama-store model's real one lives in its manifest, which Registry
// does not expose either). Stable for a given name/size/quant/arch, not a
// content hash.
std::string digest_of(const Model & m);

// ---------------------------------------------------------------- auth

// Mirrors guarded(): a Bearer token takes precedence over X-API-Key, both
// compared to expected_key in constant time, and an empty expected_key
// always rejects.
bool check_api_key(const std::string & authorization_header,
                    const std::string & x_api_key_header,
                    const std::string & expected_key);

// The key gating cfg.public_port: LLMASH_LINK_KEY, else <root>/link.json,
// else a freshly minted one persisted there. Same source of truth the `link`
// CLI command reads in Go (cmds.go); reimplemented here since it is small,
// self-contained, and nothing declared in config.h/registry.h/manager.h
// currently carries it.
std::string link_key(const Config & cfg);

// -------------------------------------------------------- keep_alive

// Ollama accepts 5m / 1h / seconds / -1 (forever) / 0 (now).
double parse_keep_alive(const nlohmann::json & v, const std::string & default_keep);
// JSON has no infinity: a pinned model reports Ollama's -1.
double keep_alive_out(double ka);

// ----------------------------------------------------- create / copy

// LLMASH_GGUF, or <models_root>/gguf: where a pulled or imported model is
// written. Registry (its owner in Go) does not expose this, so it is
// recomputed here from the same Config fields Registry itself reads.
std::string loose_dir(const Config & cfg);
std::string safe_model_name(const std::string & name);

// A GGUF's vision projector sitting beside it, matched by stem + "mmproj";
// empty if none. Registry does not track this (no Projector field on Model).
std::string find_projector(const std::string & gguf_path);

void copy_file_preserving_mtime(const std::string & src, const std::string & dst);

// Runs a llama-quantize.exe next to cfg.llama_bin. Throws std::runtime_error
// with a message meant for the ndjson stream on any failure.
void quantize_gguf(const Config & cfg, const std::string & src, const std::string & dst, const std::string & level);

using Emit = std::function<void(const nlohmann::json &)>;

// Each ndjson line apiCreate/apiCopy would have streamed, in order, via
// emit. Never throws: every failure is reported through emit instead, the
// way the Go handlers report to their ndjson writer instead of erroring the
// HTTP response.
void run_create(const Config & cfg, const std::string & name, const std::string & from,
                const std::string & quantize, const std::string & draft_quantize, const Emit & emit);
void run_copy(const Config & cfg, const Model & source, const std::string & destination, const Emit & emit);

// ------------------------------------------------------------- delete

struct DeleteOutcome {
    int            status = 200;
    nlohmann::json body;
};

// The file-system half of apiDelete: given the model apiDelete already
// resolved (404 before this point if it did not), removes its file and any
// shard siblings, or reports why it would not. Unloading the running
// instance is the caller's job (Manager is not linkable here).
DeleteOutcome run_delete(const Model & m);

// -------------------------------------------------------------- cli cache

// Go's cliCached()/cliBuild(): serve a build no older than ttl, otherwise
// rebuild synchronously. Go also warms this from a 2s background ticker
// (cliCacheTick, started in serve.go); register_routes() has no lifecycle
// hook to stop such a thread without leaking it, so this cache is
// rebuild-on-read only. Thread-safe; one instance per cached "kind".
class CliTextCache {
public:
    std::string get(const std::string & kind, double ttl_seconds, const std::function<std::string()> & build);
    void        invalidate();

private:
    std::mutex mu_;
    struct Entry {
        double      at = 0;
        std::string text;
    };
    std::unordered_map<std::string, Entry> cache_;
};

// -------------------------------------------------------------- process

// RAII over sheredom/subprocess.h's C handle: run() always leaves the
// process joined and its handles closed, even if the caller never calls
// join() explicitly.
class Subprocess {
public:
    Subprocess();
    ~Subprocess();
    Subprocess(const Subprocess &) = delete;
    Subprocess & operator=(const Subprocess &) = delete;

    bool start(const std::vector<std::string> & args);
    // Blocks until exit; -1 if the process was never started or the wait failed.
    int  join();

private:
    std::unique_ptr<subprocess_s> proc_;
    bool started_ = false;
};

} // namespace llmash
