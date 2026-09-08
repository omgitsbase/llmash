#include "api.h"

#include "api_logic.h"
#include "chat.h"
#include "pull.h"
#include "table.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace llmash {
namespace {

using json = nlohmann::json;
using httplib::Request;
using httplib::Response;

// config.go's serverVersion/serverBuild. Nothing in config.h carries these
// yet; main.cpp prints the same literal for `llmash --version`.
constexpr const char * kServerVersion = "0.4.0";
constexpr const char * kServerBuild   = "llmash";

// State the routes share for the life of the server. register_routes() owns
// nothing per api.h, so it lives in a shared_ptr every handler captures.
struct State {
    CliTextCache cli;
    std::string  link_key;
    std::mutex   reload_mu;
};

void write_json(Response & res, int code, const json & v) {
    res.status = code;
    res.set_content(v.dump(), "application/json");
}

void write_text(Response & res, int code, const std::string & s) {
    res.status = code;
    res.set_content(s, "text/plain; charset=utf-8");
}

json error_obj(const std::string & msg) { return json{{"error", msg}}; }

// An unparseable or non-object body is an empty object, the way Go's
// readBody hands back the map it was decoding into and the callers ignore
// its error.
json read_body(const Request & req) {
    json d = json::parse(req.body, nullptr, false);
    if (!d.is_object()) {
        return json::object();
    }
    return d;
}

// http.go's str(): a missing or null key is "", a non-string is printed.
std::string jstr(const json & o, const char * k) {
    if (!o.is_object()) {
        return "";
    }
    auto it = o.find(k);
    if (it == o.end() || it->is_null()) {
        return "";
    }
    return it->is_string() ? it->get<std::string>() : it->dump();
}

std::string first_of(const std::string & a, const std::string & b) { return a.empty() ? b : a; }

// Go's float64(int(x*10))/10: truncation, not rounding.
double tenths(double x) { return std::trunc(x * 10.0) / 10.0; }

// Go's mux dispatches on path only; every handler there ignores the method.
void mount(httplib::Server & srv, const std::string & pattern, httplib::Server::Handler h) {
    srv.Get(pattern, h);
    srv.Post(pattern, h);
    srv.Put(pattern, h);
    srv.Delete(pattern, h);
    srv.Patch(pattern, h);
}

// newNDJSON: one JSON object per line, flushed as it is produced.
void stream_ndjson(Response & res, std::function<void(const Emit &)> work) {
    res.set_chunked_content_provider(
        "application/x-ndjson", [work = std::move(work)](size_t, httplib::DataSink & sink) {
            Emit emit = [&sink](const json & v) {
                std::string line = v.dump();
                line.push_back('\n');
                sink.write(line.data(), line.size());
            };
            work(emit);
            sink.done();
            return true;
        });
}

bool is_public(const Request & req, const Config & cfg) {
    return cfg.public_port != 0 && req.local_port == cfg.public_port;
}

std::vector<InstanceView> live_views(Manager & mgr) {
    std::vector<InstanceView> out;
    for (Instance * in : mgr.loaded()) {
        InstanceView v;
        v.model      = in->model;
        v.vram_gb    = in->vram_gb();
        v.on_gpu     = in->on_gpu();
        v.ready      = in->ready();
        v.last_used  = in->last_used;
        v.expires_at = in->expires_at;
        v.keep_alive = in->keep_alive;
        v.ctx        = in->ctx;
        out.push_back(std::move(v));
    }
    return out;
}

} // namespace

void register_routes(httplib::Server & srv, Config & cfg, Manager & mgr, Registry & reg) {
    auto st = std::make_shared<State>();
    if (cfg.public_port != 0) {
        st->link_key = link_key(cfg);
    }

    srv.set_pre_routing_handler([&cfg, st](const Request & req, Response & res) {
        // Ollama ships permissive CORS on its local API and browser clients rely on it.
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Methods", "*");
        res.set_header("Access-Control-Allow-Headers", "*");
        if (req.method == "OPTIONS") {
            res.status = 200;
            return httplib::Server::HandlerResponse::Handled;
        }
        if (is_public(req, cfg) &&
            !check_api_key(req.get_header_value("Authorization"), req.get_header_value("X-API-Key"),
                            st->link_key)) {
            write_json(res, 401, error_obj("unauthorized: missing or invalid API key"));
            return httplib::Server::HandlerResponse::Handled;
        }
        return httplib::Server::HandlerResponse::Unhandled;
    });

    // Go's mux sends every unmatched path to its "/" handler, which 404s
    // anything that is not exactly "/". httplib has no catch-all pattern, so
    // the same answer comes from the error hook, and only when no handler
    // wrote a body of its own.
    srv.set_error_handler([](const Request &, Response & res) {
        if (res.status == 404 && res.body.empty() && !res.content_provider_) {
            write_json(res, 404, error_obj("not found"));
        }
    });

    mount(srv, "/", [](const Request &, Response & res) { write_text(res, 200, "Ollama is running"); });

    mount(srv, "/api/version", [](const Request &, Response & res) {
        write_json(res, 200, json{{"version", kServerVersion}, {"build", kServerBuild}});
    });

    // ------------------------------------------------------------ listing

    mount(srv, "/api/tags", [&cfg, &reg](const Request &, Response & res) {
        write_json(res, 200, tags_json(reg.all(), cfg));
    });

    mount(srv, "/api/ps", [&cfg, &mgr](const Request &, Response & res) {
        write_json(res, 200, ps_json(live_views(mgr), cfg));
    });

    mount(srv, "/v1/models", [&cfg, &reg](const Request &, Response & res) {
        write_json(res, 200, v1_models_json(reg.all(), cfg));
    });

    // A model name can hold both ':' and '/', so the id is everything after
    // the prefix, the way Go's TrimPrefix takes it.
    mount(srv, R"(/v1/models/(.*))", [&cfg, &reg](const Request & req, Response & res) {
        const std::string name = req.matches.size() > 1 ? req.matches[1].str() : std::string();
        const Model *     m    = reg.find(name);
        if (m == nullptr || !std::filesystem::exists(m->path)) {
            write_json(res, 404,
                       openai_error_json("model '" + name + "' not found", "invalid_request_error",
                                          "model_not_found"));
            return;
        }
        write_json(res, 200, v1_entry_json(*m, cfg));
    });

    mount(srv, "/api/show", [&cfg, &reg](const Request & req, Response & res) {
        const json  body = read_body(req);
        std::string name = first_of(jstr(body, "model"), jstr(body, "name"));
        const Model * m  = reg.find(name);
        if (m == nullptr) {
            write_json(res, 404, error_obj("model not found"));
            return;
        }
        write_json(res, 200, show_json(*m, cfg));
    });

    mount(srv, "/api/loading", [&mgr](const Request &, Response & res) {
        json out = json::array();
        // manager.h exposes only instances that are already ready, so a model
        // still loading is invisible from here; Go reads mgr.Live().
        for (Instance * in : mgr.loaded()) {
            if (in->ready()) {
                continue;
            }
            out.push_back(json{{"name", in->model.name},
                               {"pct", tenths(in->progress() * 100.0)},
                               {"size", in->model.size},
                               {"ctx", in->ctx},
                               {"elapsed", tenths(now_unix() - in->last_used)},
                               {"backend", "llama.cpp"}});
        }
        write_json(res, 200, json{{"loading", out}});
    });

    // --------------------------------------------------------- cli cache

    mount(srv, "/cli/list", [&cfg, &reg, st](const Request &, Response & res) {
        write_text(res, 200, st->cli.get("list", 1.0, [&cfg, &reg] {
            return render_list(tags_json(reg.all(), cfg).at("models"));
        }));
    });

    mount(srv, "/cli/ps", [&cfg, &mgr, st](const Request &, Response & res) {
        write_text(res, 200, st->cli.get("ps", 1.0, [&cfg, &mgr] {
            return render_ps(ps_json(live_views(mgr), cfg).at("models"));
        }));
    });

    // ------------------------------------------------------------- paths

    // A POST re-reads local.json first, which is how `llmash models` applies
    // a new library folder to a server that is already up.
    mount(srv, "/api/paths", [&cfg, &reg, st](const Request & req, Response & res) {
        if (req.method == "POST") {
            {
                std::lock_guard<std::mutex> lock(st->reload_mu);
                cfg = load_config();
            }
            reg.invalidate();
            st->cli.invalidate();
        }
        write_json(res, 200,
                   json{{"root", cfg.root},
                        {"loose_dir", loose_dir(cfg)},
                        {"models", cfg.models_root},
                        {"extra_roots", cfg.extra_roots},
                        {"library", reg.library_dirs()}});
    });

    // ------------------------------------------------------------ delete

    mount(srv, "/api/delete", [&mgr, &reg, st](const Request & req, Response & res) {
        const json    body = read_body(req);
        std::string   name = first_of(jstr(body, "model"), jstr(body, "name"));
        const Model * found = reg.find(name);
        if (found == nullptr) {
            write_json(res, 404, error_obj("model not found"));
            return;
        }
        const Model m = *found; // the invalidate below drops what `found` points at
        mgr.unload(m.name);
        DeleteOutcome out = run_delete(m);
        reg.invalidate();
        st->cli.invalidate();
        write_json(res, out.status, out.body);
    });

    // -------------------------------------------------------- keep_alive

    mount(srv, "/api/keep_alive", [&cfg, &mgr](const Request & req, Response & res) {
        const json        body = read_body(req);
        const std::string name = jstr(body, "model");
        const auto        it   = body.find("keep_alive");
        const double      ka   = parse_keep_alive(it == body.end() ? json() : *it, cfg.keep_alive);

        Instance * in = mgr.find(name);
        if (in == nullptr) {
            // Go also answers here for a model served by a fast backend,
            // whose idle clock lives in remote.h's RemoteRouter;
            // register_routes is not given one, so that branch has no home
            // yet.
            write_json(res, 404, error_obj("model '" + name + "' is not loaded"));
            return;
        }
        if (ka == 0) {
            const std::string loaded = in->model.name;
            mgr.unload(loaded);
            write_json(res, 200, json{{"model", loaded}, {"done_reason", "unload"}});
            return;
        }
        in->set_keep_alive(ka);
        const double exp = in->expires_at;
        write_json(res, 200,
                   json{{"model", in->model.name},
                        {"keep_alive", keep_alive_out(ka)},
                        {"expires_at", exp < 1e15 ? json(iso(exp)) : json()}});
    });

    // ------------------------------------------------------- create / copy

    mount(srv, "/api/create", [&cfg, &reg, st](const Request & req, Response & res) {
        const json        body  = read_body(req);
        const std::string name  = first_of(jstr(body, "model"), jstr(body, "name"));
        const std::string from  = jstr(body, "from");
        const std::string quant = jstr(body, "quantize");
        const std::string draft = jstr(body, "draft_quantize");
        stream_ndjson(res, [&cfg, &reg, st, name, from, quant, draft](const Emit & emit) {
            run_create(cfg, name, from, quant, draft, emit);
            reg.invalidate();
            st->cli.invalidate();
        });
    });

    mount(srv, "/api/copy", [&cfg, &reg, st](const Request & req, Response & res) {
        const json        body        = read_body(req);
        const std::string source      = jstr(body, "source");
        const std::string destination = jstr(body, "destination");

        std::optional<Model> m;
        if (const Model * found = reg.find(source)) {
            m = *found;
        }
        stream_ndjson(res, [&cfg, &reg, st, m, source, destination](const Emit & emit) {
            if (!m) {
                emit(error_obj(source + ": model not found"));
                return;
            }
            run_copy(cfg, *m, destination, emit);
            reg.invalidate();
            st->cli.invalidate();
        });
    });

    // -------------------------------------------------------------- pull

    mount(srv, "/api/pull", [&cfg, &reg](const Request & req, Response & res) {
        handle_pull(req, res, cfg, reg);
    });
    mount(srv, "/api/quants", [&cfg, &reg](const Request & req, Response & res) {
        handle_quants(req, res, cfg, reg);
    });
    mount(srv, "/api/resolve", [&cfg, &reg](const Request & req, Response & res) {
        handle_resolve(req, res, cfg, reg);
    });

    // -------------------------------------------------------- generation

    mount(srv, "/api/chat", [&cfg, &mgr, &reg](const Request & req, Response & res) {
        handle_chat(req, res, cfg, mgr, reg);
    });
    mount(srv, "/api/generate", [&cfg, &mgr, &reg](const Request & req, Response & res) {
        handle_generate(req, res, cfg, mgr, reg);
    });
    mount(srv, "/api/embed", [&cfg, &mgr, &reg](const Request & req, Response & res) {
        handle_embed(req, res, cfg, mgr, reg);
    });
    mount(srv, "/api/embeddings", [&cfg, &mgr, &reg](const Request & req, Response & res) {
        handle_embed(req, res, cfg, mgr, reg);
    });
    mount(srv, "/v1/chat/completions", [&cfg, &mgr, &reg](const Request & req, Response & res) {
        handle_v1_chat_completions(req, res, cfg, mgr, reg);
    });
    mount(srv, "/v1/completions", [&cfg, &mgr, &reg](const Request & req, Response & res) {
        handle_v1_completions(req, res, cfg, mgr, reg);
    });
}

} // namespace llmash
