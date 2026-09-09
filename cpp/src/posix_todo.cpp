// What the portable half calls but has not been ported yet: the process
// supervisor, the downloader and the CLI table renderer. Building these for
// real on macOS is the rest of the job; until then they stand in so the
// portable code compiles and its tests run on a mac.
#ifndef _WIN32

#include "manager.h"
#include "pull.h"
#include "table.h"

#include <limits>
#include <stdexcept>

namespace llmash {

static std::string not_ported(const char * what) { return std::string(what) + " is not ported to macOS yet"; }

Instance::Instance(Model m, int ctx_in, bool vision_in, Config * cfg)
    : model(std::move(m)), ctx(ctx_in), vision(vision_in), cfg_(cfg) {}

std::string Instance::url() const { return ""; }
void        Instance::touch() {}
void        Instance::set_keep_alive(double ka) { keep_alive = ka; }
bool        Instance::ready() const { return false; }
bool        Instance::on_gpu() const { return false; }
double      Instance::vram_gb() const { return 0.0; }
bool        Instance::alive() const { return false; }
std::string Instance::tail_log(size_t) const { return ""; }
double      Instance::progress() const { return 0.0; }
std::string Instance::start() { return not_ported("starting a model"); }
void        Instance::mark_loaded() {}
void        Instance::stop() {}
std::vector<std::string> Instance::args() const { return {}; }

Manager::Manager(Config cfg, Registry * reg) : cfg_(std::move(cfg)), reg_(reg) {}

std::vector<Instance *> Manager::loaded() { return {}; }

Instance * Manager::get(const std::string &, int, double, bool, std::string & err) {
    err = not_ported("loading a model");
    return nullptr;
}

bool       Manager::unload(const std::string &) { return false; }
Instance * Manager::find(const std::string &) { return nullptr; }
void       Manager::shutdown() {}
void       Manager::reap_idle() {}
void       Manager::evict_for(double, const std::string &) {}
void       Manager::drop_dead() {}
int        Manager::fit_ctx(const Model &, int ctx) { return ctx; }

int  free_port() { return 0; }
bool can_offload() { return false; }
std::pair<int, uint64_t> cpu_threads_and_mask() { return {0, 0}; }
Tuning auto_tune() { return {}; }

HttpResult http_request(const std::string &, const std::string &, const std::string &,
                        const std::vector<std::string> &, int) {
    HttpResult r;
    r.error = not_ported("downloading");
    return r;
}

void handle_pull(const httplib::Request &, httplib::Response & res, Config &, Registry &) {
    res.status = 501;
    res.set_content("{\"error\":\"pull is not ported to macOS yet\"}", "application/json");
}

void handle_quants(const httplib::Request &, httplib::Response & res, Config &, Registry &) {
    res.status = 501;
    res.set_content("{\"error\":\"quants is not ported to macOS yet\"}", "application/json");
}

void handle_resolve(const httplib::Request &, httplib::Response & res, Config &, Registry &) {
    res.status = 501;
    res.set_content("{\"error\":\"resolve is not ported to macOS yet\"}", "application/json");
}

std::string render_list(const std::vector<nlohmann::json> &) { return ""; }
std::string render_ps(const std::vector<nlohmann::json> &) { return ""; }

std::string loose_dir(const Config & cfg) { return cfg.gguf_dir; }

} // namespace llmash

#endif // !_WIN32
