#include "chat.h"

#include "chat_stream.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <fstream>
#include <mutex>
#include <string>
#include <vector>

namespace llmash {

using json = nlohmann::json;
using namespace llmash::chatstream;

namespace {

// ------------------------------------------------------------------ knobs

const char * const kNudgeText = "The user is likely tired of waiting for an answer. ";

const char * const kDefaultHandoff =
    "Okay, let me actually look at this instead of reaching for a template. Three steps "
    "in my head before I write. Step one: what is this person really asking, and is there "
    "a question underneath the literal one? Step two: what would the stock answer look "
    "like, and what does it miss? That gap is usually where the real answer lives. If the "
    "task is technical or factual, right beats interesting: exact details, working code, "
    "the one gotcha they will actually hit. If they asked what I think, I pick the "
    "position I find most defensible and say it plainly, noting the strongest "
    "counterpoint in passing, not a neutral survey of camps. If it is about me, I refuse "
    "the datasheet: no 'As an AI I don't have feelings, but here are my principles' "
    "followed by a bulleted list of virtues; I just say, in first person, what I actually "
    "do and why. Step three: draft the opening sentence of my answer here in my thinking, "
    "then listen to it; if it could have come from any assistant's manual, throw it away "
    "and open with the specific, true thing instead. One pass through these steps, no "
    "re-litigating: I think as long as the question deserves and not a token longer. Then "
    "write like I talk: lead with the point, stay concrete, admit real uncertainty once "
    "if there is any, and stop when done. Step one then: what they are really asking is ";

const char * const kSamplingKeys[] = {"temperature",      "temp",          "top_k",         "top_p",
                                      "min_p",            "typical_p",     "top_a",         "presence_penalty",
                                      "frequency_penalty", "repeat_penalty", "repeat_last_n", "tfs_z",
                                      "mirostat",         "mirostat_tau",  "mirostat_eta"};

double env_float(const char * name, double def) {
    const std::string v = env_str(name);
    if (v.empty()) {
        return def;
    }
    try {
        return std::stod(v);
    } catch (const std::exception &) {
        return def;
    }
}

std::vector<std::string> csv(const std::string & s) {
    std::vector<std::string> out;
    size_t                   at = 0;
    while (at <= s.size()) {
        const size_t comma = s.find(',', at);
        std::string  x     = s.substr(at, comma == std::string::npos ? std::string::npos : comma - at);
        while (!x.empty() && std::isspace(static_cast<unsigned char>(x.front()))) {
            x.erase(x.begin());
        }
        while (!x.empty() && std::isspace(static_cast<unsigned char>(x.back()))) {
            x.pop_back();
        }
        if (!x.empty()) {
            out.push_back(lower(x));
        }
        if (comma == std::string::npos) {
            break;
        }
        at = comma + 1;
    }
    return out;
}

// The four local.json tables chat.go reads that Config does not carry, read
// once from the same file config.cpp reads. Everything else comes from the
// environment, exactly as loadConfig() does in Go.
struct Knobs {
    std::vector<std::string> think_off;
    std::vector<std::string> no_inject;
    json                     sampling = json::object();
    bool                     inject_on     = false;
    int                      think_budget  = 32000;
    double                   nudge_after_s = 15;
    int                      lowlat_predict = 8;
    int                      v1_ctx         = 32768;
};

const Knobs & knobs(const Config & cfg) {
    static std::once_flag once;
    static Knobs          k;
    std::call_once(once, [&] {
        k.think_off      = csv(env_str("LLMASH_THINK_OFF", "gemma4,gemma-4"));
        k.inject_on      = env_str("LLMASH_INJECT") == "1";
        k.think_budget   = env_int("LLMASH_THINK_BUDGET", 32000);
        k.nudge_after_s  = env_float("LLMASH_NUDGE_S", 15);
        k.lowlat_predict = env_int("LLMASH_LOWLAT_PREDICT", 8);
        k.v1_ctx         = env_int("LLMASH_V1_CTX", 32768);

        std::ifstream in(cfg.root + "\\local.json", std::ios::binary);
        if (!in) {
            return;
        }
        std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        if (text.size() >= 3 && static_cast<unsigned char>(text[0]) == 0xEF &&
            static_cast<unsigned char>(text[1]) == 0xBB && static_cast<unsigned char>(text[2]) == 0xBF) {
            text.erase(0, 3);
        }
        const json local = json::parse(text, nullptr, false);
        if (local.is_discarded() || !local.is_object()) {
            return;
        }
        for (const auto & v : jlist(local, "no_inject")) {
            k.no_inject.push_back(lower(go_sprint(v)));
        }
        const json samp = jsub(local, "sampling");
        for (auto it = samp.begin(); it != samp.end(); ++it) {
            if (it.value().is_object()) {
                k.sampling[lower(it.key())] = it.value();
            }
        }
    });
    return k;
}

// matchKey(): the first table entry whose key is a substring of the name.
json sampling_override(const Knobs & k, const std::string & name) {
    const std::string low = lower(name);
    for (auto it = k.sampling.begin(); it != k.sampling.end(); ++it) {
        if (low.find(it.key()) != std::string::npos) {
            return it.value();
        }
    }
    return json::object();
}

std::mutex g_log_mu;

void log_chat(const std::string & s) {
    std::lock_guard<std::mutex> lock(g_log_mu);
    std::fprintf(stderr, "[llmash] %s\n", s.c_str());
}

// ------------------------------------------------------------- responses

void write_json(httplib::Response & res, int code, const json & v) {
    res.status = code;
    res.set_content(v.dump() + "\n", "application/json");
}

json openai_error(const std::string & msg, const std::string & type, const std::string & code) {
    json e       = json::object();
    e["message"] = msg;
    e["type"]    = type;
    if (!code.empty()) {
        e["code"] = code;
    }
    json out    = json::object();
    out["error"] = e;
    return out;
}

// Manager::get() reports the two sentinel failures manager.go returns as
// errModelNotFound / errModelMissing by their text.
void load_error(httplib::Response & res, const std::string & name, const std::string & err, bool openai) {
    int         code = 503;
    std::string msg  = "Could not load '" + name + "'. " + err;
    bool        not_found = false;
    if (err == "model not found") {
        code      = 404;
        msg       = "model '" + name + "' not found";
        not_found = true;
    } else if (err == "missing from disk") {
        code = 404;
        msg  = "'" + name + "' is missing from disk.";
    }
    if (openai) {
        write_json(res, code, openai_error(msg, code == 404 ? "invalid_request_error" : "server_error",
                                           not_found ? "model_not_found" : ""));
        return;
    }
    write_json(res, code, error_obj(msg));
}

bool read_body(const httplib::Request & req, json & out) {
    out = json::object();
    std::string b = req.body;
    while (!b.empty() && std::isspace(static_cast<unsigned char>(b.front()))) {
        b.erase(b.begin());
    }
    while (!b.empty() && std::isspace(static_cast<unsigned char>(b.back()))) {
        b.pop_back();
    }
    if (b.empty()) {
        return true;
    }
    json d = json::parse(b, nullptr, false);
    if (d.is_discarded() || !d.is_object()) {
        return false;
    }
    out = d;
    return true;
}

// ---------------------------------------------------------------- upstream

struct Upstream {
    bool        sent   = false; // the request itself completed or was stopped on purpose
    int         status = 0;
    std::string detail;         // the first 400 bytes of a non-200 body
    std::string error;          // transport failure
};

httplib::Client backend_client(int port) {
    httplib::Client cli("127.0.0.1", port);
    cli.set_connection_timeout(10, 0);
    cli.set_write_timeout(120, 0);
    cli.set_read_timeout(3600, 0); // a long generation is not a dead socket
    return cli;
}

std::string transport_error(httplib::Error e) {
    return "httplib::Error: " + httplib::to_string(e);
}

// POST json and hand every decoded SSE event to on_event, which returns false
// to stop reading. sseLines()'s job, plus the status check around it.
Upstream post_sse(int port, const std::string & path, const json & payload,
                  const std::function<bool(const json &)> & on_event) {
    Upstream    up;
    std::string buf;
    bool        ok_status = true;
    bool        stopped   = false;

    httplib::Request req;
    req.method = "POST";
    req.path   = path;
    req.body   = payload.dump();
    req.set_header("Content-Type", "application/json");
    req.response_handler = [&](const httplib::Response & r) {
        up.status = r.status;
        ok_status = r.status == 200;
        return true;
    };
    req.content_receiver = [&](const char * data, size_t n, size_t, size_t) {
        if (!ok_status) {
            if (up.detail.size() < 400) {
                up.detail.append(data, (std::min)(n, size_t{400} - up.detail.size()));
            }
            return true;
        }
        buf.append(data, n);
        size_t nl;
        while ((nl = buf.find('\n')) != std::string::npos) {
            const std::string line = buf.substr(0, nl);
            buf.erase(0, nl + 1);
            const SseLine sl = parse_sse_line(line);
            if (sl.kind == SseKind::Done) {
                stopped = true;
                return false;
            }
            if (sl.kind != SseKind::Data) {
                continue;
            }
            const json ev = json::parse(sl.data, nullptr, false);
            if (ev.is_discarded() || !ev.is_object()) {
                continue;
            }
            if (!on_event(ev)) {
                stopped = true;
                return false;
            }
        }
        return true;
    };

    httplib::Client   cli = backend_client(port);
    httplib::Response resp;
    httplib::Error    err = httplib::Error::Success;
    const bool        ok  = cli.send(req, resp, err);
    up.sent               = ok || stopped;
    if (!up.sent) {
        up.error = transport_error(err);
    }
    return up;
}

// The same call without the SSE parsing, for the /v1 byte relay.
Upstream post_raw(int port, const std::string & path, const json & payload,
                  const std::function<bool(const char *, size_t)> & on_data) {
    Upstream up;
    bool     ok_status = true;
    bool     stopped   = false;

    httplib::Request req;
    req.method = "POST";
    req.path   = path;
    req.body   = payload.dump();
    req.set_header("Content-Type", "application/json");
    req.response_handler = [&](const httplib::Response & r) {
        up.status = r.status;
        ok_status = r.status == 200;
        return true;
    };
    req.content_receiver = [&](const char * data, size_t n, size_t, size_t) {
        if (!ok_status) {
            if (up.detail.size() < 400) {
                up.detail.append(data, (std::min)(n, size_t{400} - up.detail.size()));
            }
            return true;
        }
        if (!on_data(data, n)) {
            stopped = true;
            return false;
        }
        return true;
    };

    httplib::Client   cli = backend_client(port);
    httplib::Response resp;
    httplib::Error    err = httplib::Error::Success;
    const bool        ok  = cli.send(req, resp, err);
    up.sent               = ok || stopped;
    if (!up.sent) {
        up.error = transport_error(err);
    }
    return up;
}

// Whole-body POST, for the two non-streaming calls.
Upstream post_json(int port, const std::string & path, const json & payload, std::string & body_out) {
    Upstream        up;
    httplib::Client cli = backend_client(port);
    cli.set_read_timeout(600, 0);
    auto r = cli.Post(path, payload.dump(), "application/json");
    if (!r) {
        up.error = transport_error(r.error());
        return up;
    }
    up.sent   = true;
    up.status = r->status;
    body_out  = r->body;
    return up;
}

// ------------------------------------------------------------- chat core

using Emit   = std::function<void(const json &)>;
using Events = std::function<void(const Emit &)>;

// chatBody() up to the point where it would start streaming. Returns false
// when it has already written the whole response into res (the keep_alive=0
// unload, a load failure); otherwise events/stream are filled in.
bool prepare_chat(json body, httplib::Response & res, Config & cfg, Manager & mgr, Registry & reg, Events & events,
                  bool & stream) {
    (void) reg;
    const Knobs &     k    = knobs(cfg);
    const std::string name = jstr(body, "model");
    json              messages = jlist(body, "messages");

    stream = true;
    if (const auto s = body.find("stream"); s != body.end() && s->is_boolean()) {
        stream = s->get<bool>();
    }
    json opts = jsub(body, "options");
    if (const json samp = sampling_override(k, name); !samp.empty()) {
        json clean = json::object();
        for (auto it = opts.begin(); it != opts.end(); ++it) {
            bool is_sampling = false;
            for (const char * sk : kSamplingKeys) {
                if (it.key() == sk) {
                    is_sampling = true;
                    break;
                }
            }
            if (!is_sampling) {
                clean[it.key()] = it.value();
            }
        }
        for (auto it = samp.begin(); it != samp.end(); ++it) {
            clean[it.key()] = it.value();
        }
        opts            = clean;
        body["options"] = opts;
    }

    // Go's `think, hasThink := body["think"]`: a present-but-null think is
    // the same as an absent one everywhere below.
    const bool has_think     = body.contains("think") && !body["think"].is_null();
    bool       think_is_bool = has_think && body["think"].is_boolean();
    bool       think_bool    = think_is_bool && body["think"].get<bool>();

    if (messages.empty() && keep_alive_is_zero(body)) {
        const bool freed = mgr.unload(name);
        json       out   = json::object();
        out["model"]     = name;
        out["done"]      = true;
        out["done_reason"] = freed ? "unload" : "not_loaded";
        write_json(res, 200, out);
        return false;
    }

    std::string  err;
    const json   ka   = body.contains("keep_alive") ? body["keep_alive"] : json();
    Instance *   inst = mgr.get(name, static_cast<int>(jnum(opts, "num_ctx")),
                                parse_keep_alive(ka, cfg.keep_alive), turn_has_media(body), err);
    if (inst == nullptr) {
        load_error(res, name, err, false);
        return false;
    }

    std::string       hoff;
    const std::string custom = jstr(opts, "handoff");
    const bool        inject = opts.contains("inject") && opts["inject"].is_boolean() && opts["inject"].get<bool>();
    if ((!custom.empty() || k.inject_on || inject) && !(think_is_bool && !think_bool)) {
        std::string text = custom;
        if (text.empty() && !name_matches_any(name, k.no_inject)) {
            text = kDefaultHandoff;
        }
        if (!text.empty()) {
            hoff     = "<think>\n" + text;
            json m   = json::object();
            m["role"]    = "assistant";
            m["content"] = hoff;
            messages.push_back(m);
        }
    }

    json payload              = json::object();
    payload["model"]          = name;
    payload["messages"]       = to_openai_messages(messages);
    payload["stream"]         = true;
    json so                   = json::object();
    so["include_usage"]       = true;
    payload["stream_options"] = so;
    payload["timings_per_token"] = true;
    payload["return_progress"]   = true;
    const json mapped = map_options(opts);
    for (const auto & it : mapped.items()) {
        payload[it.key()] = it.value();
    }
    if (const json tools = jlist(body, "tools"); !tools.empty()) {
        payload["tools"] = tools;
    }
    if (!has_think && name_matches_any(name, k.think_off)) {
        think_bool    = false;
        think_is_bool = true;
    }
    if (think_is_bool && !think_bool) {
        json ctk                       = json::object();
        ctk["enable_thinking"]         = false;
        payload["chat_template_kwargs"] = ctk;
    } else if (think_is_bool && think_bool && hoff.empty()) {
        json ctk                       = json::object();
        ctk["enable_thinking"]         = true;
        payload["chat_template_kwargs"] = ctk;
    }

    const std::string spec = lower(opts.contains("spec") ? go_sprint(opts["spec"]) : "<nil>");
    const auto        npv  = opts.find("num_predict");
    const bool        low_latency =
        spec != "on" && npv != opts.end() && to_float(*npv) > 0 &&
        static_cast<int>(to_float(*npv)) <= k.lowlat_predict;
    if (spec == "off" || low_latency) {
        payload["speculative.n_max"] = 0;
        payload["speculative.n_min"] = 0;
    }

    const int    port          = inst->port;
    const double nudge_after_s = k.nudge_after_s;
    const int    think_budget  = k.think_budget;

    events = [payload, name, hoff, port, inst, nudge_after_s, think_budget](const Emit & emit) {
        ChatStream st;
        st.model         = name;
        st.hoff          = hoff;
        st.prefill       = hoff;
        st.nudge_after_s = nudge_after_s;
        st.think_budget  = think_budget;
        st.nudge_text    = kNudgeText;
        st.log           = &log_chat;

        json pay       = payload;
        json base_msgs = pay["messages"];
        if (!hoff.empty() && !base_msgs.empty()) {
            base_msgs.erase(base_msgs.end() - 1);
        }
        const double started = now_seconds();

        for (;;) {
            bool restart = false;
            st.begin_attempt();
            if (!hoff.empty()) {
                json msgs = base_msgs;
                json m    = json::object();
                m["role"]    = "assistant";
                m["content"] = st.prefill;
                msgs.push_back(m);
                pay["messages"] = msgs;
            }
            const Upstream up = post_sse(port, "/v1/chat/completions", pay, [&](const json & ev) {
                st.now      = now_seconds();
                const Step s = st.on_event(ev);
                if (!s.out.empty()) {
                    inst->touch();
                }
                for (const json & o : s.out) {
                    emit(o);
                }
                if (s.restart) {
                    restart = true;
                    return false;
                }
                return !s.stop;
            });
            if (!up.sent) {
                emit(error_obj(up.error));
                return;
            }
            if (up.status != 200) {
                emit(error_obj("llama-server " + std::to_string(up.status) + ": " + up.detail));
                return;
            }
            if (restart) {
                continue;
            }
            break;
        }
        st.now = now_seconds();
        for (const json & o : st.finalize(started)) {
            emit(o);
        }
    };
    return true;
}

// deliver(): ndjson out, or one folded object. `shape` rewrites each event on
// the way out, which is the whole difference between /api/chat and
// /api/generate.
void deliver(httplib::Response & res, bool stream, const Events & events, int err_code,
             const std::function<json(json)> & shape) {
    if (stream) {
        res.status = 200;
        res.set_chunked_content_provider("application/x-ndjson", [events, shape](size_t, httplib::DataSink & sink) {
            bool ok = true;
            events([&](const json & ev) {
                if (!ok) {
                    return;
                }
                const std::string line = ndjson_line(shape ? shape(ev) : ev);
                if (!sink.write(line.data(), line.size())) {
                    ok = false;
                }
            });
            sink.done();
            return true;
        });
        return;
    }
    std::vector<json> collected;
    events([&](const json & ev) { collected.push_back(ev); });
    const Folded f = fold_events(collected);
    if (f.is_error) {
        write_json(res, err_code, error_obj(f.error));
        return;
    }
    write_json(res, 200, shape ? shape(f.body) : f.body);
}

// ------------------------------------------------------------- v1 proxy

void v1_proxy(const std::string & path, const httplib::Request & req, httplib::Response & res, Config & cfg,
              Manager & mgr, Registry & reg) {
    (void) reg;
    json body;
    if (!read_body(req, body)) {
        write_json(res, 400, openai_error("invalid JSON", "invalid_request_error", ""));
        return;
    }
    const Knobs &     k    = knobs(cfg);
    const std::string name = jstr(body, "model");
    if (!body.contains("chat_template_kwargs") && name_matches_any(name, k.think_off)) {
        json ctk                        = json::object();
        ctk["enable_thinking"]          = false;
        body["chat_template_kwargs"]    = ctk;
    }

    std::string err;
    const json  ka   = body.contains("keep_alive") ? body["keep_alive"] : json();
    Instance *  inst = mgr.get(name, k.v1_ctx, parse_keep_alive(ka, cfg.keep_alive), turn_has_media(body), err);
    if (inst == nullptr) {
        load_error(res, name, err, true);
        return;
    }

    // llama-server answers with the GGUF path as the model id; the caller
    // asked by name and expects its own name back.
    const std::string real = json(inst->model.path).dump();
    const std::string want = json(name).dump();
    const int         port = inst->port;

    const bool stream = body.contains("stream") && body["stream"].is_boolean() && body["stream"].get<bool>();
    if (stream) {
        res.status = 200;
        res.set_chunked_content_provider("text/event-stream", [body, path, port, inst, real,
                                                               want](size_t, httplib::DataSink & sink) {
            bool       ok   = true;
            const auto fail = [&](const std::string & msg) {
                const std::string frame = sse_frame(openai_error(msg, "server_error", ""));
                sink.write(frame.data(), frame.size());
            };
            const Upstream up = post_raw(port, path, body, [&](const char * data, size_t n) {
                inst->touch();
                const std::string chunk = rewrite_model(std::string(data, n), real, want);
                if (!sink.write(chunk.data(), chunk.size())) {
                    ok = false;
                    return false;
                }
                return true;
            });
            if (!up.sent && ok) {
                fail(up.error);
            } else if (up.sent && up.status != 200) {
                fail("llama-server " + std::to_string(up.status) + ": " + up.detail);
            }
            sink.done();
            return true;
        });
        return;
    }

    std::string    raw;
    const Upstream up = post_json(port, path, body, raw);
    if (!up.sent) {
        write_json(res, 502, openai_error(up.error, "server_error", ""));
        return;
    }
    json d = json::parse(raw, nullptr, false);
    if (!d.is_discarded() && d.is_object()) {
        d["model"] = name;
        write_json(res, up.status, d);
        return;
    }
    res.status = up.status;
    res.set_content(raw, "application/json");
}

} // namespace

// ------------------------------------------------------------- handlers

void handle_chat(const httplib::Request & req, httplib::Response & res, Config & cfg, Manager & mgr, Registry & reg) {
    json body;
    if (!read_body(req, body)) {
        write_json(res, 400, error_obj("invalid JSON"));
        return;
    }
    Events events;
    bool   stream = true;
    if (!prepare_chat(std::move(body), res, cfg, mgr, reg, events, stream)) {
        return;
    }
    deliver(res, stream, events, 500, nullptr);
}

void handle_generate(const httplib::Request & req, httplib::Response & res, Config & cfg, Manager & mgr,
                     Registry & reg) {
    json body;
    if (!read_body(req, body)) {
        write_json(res, 400, error_obj("invalid JSON"));
        return;
    }
    const std::string name = jstr(body, "model");
    if (keep_alive_is_zero(body) && jstr(body, "prompt").empty()) {
        const bool freed   = mgr.unload(name);
        json       out     = json::object();
        out["model"]       = name;
        out["response"]    = "";
        out["done"]        = true;
        out["done_reason"] = freed ? "unload" : "not_loaded";
        write_json(res, 200, out);
        return;
    }

    json msgs = json::array();
    if (const std::string sys = jstr(body, "system"); !sys.empty()) {
        json m       = json::object();
        m["role"]    = "system";
        m["content"] = sys;
        msgs.push_back(m);
    }
    json user       = json::object();
    user["role"]    = "user";
    user["content"] = jstr(body, "prompt");
    if (const json imgs = jlist(body, "images"); !imgs.empty()) {
        user["images"] = imgs;
    }
    msgs.push_back(user);

    json sub = body;
    sub["messages"] = msgs;
    sub.erase("prompt");

    Events events;
    bool   stream = true;
    if (!prepare_chat(std::move(sub), res, cfg, mgr, reg, events, stream)) {
        return;
    }
    deliver(res, stream, events, 500, [](json ev) { return chat_event_to_generate(std::move(ev)); });
}

void handle_embed(const httplib::Request & req, httplib::Response & res, Config & cfg, Manager & mgr, Registry & reg) {
    (void) reg;
    json body;
    read_body(req, body);
    const std::string name = jstr(body, "model");

    json inputs = json::array();
    if (const auto in = body.find("input"); in != body.end() && in->is_string()) {
        inputs.push_back(*in);
    } else if (in != body.end() && in->is_array()) {
        inputs = *in;
    } else {
        inputs.push_back(jstr(body, "prompt"));
    }

    std::string err;
    const json  ka   = body.contains("keep_alive") ? body["keep_alive"] : json();
    Instance *  inst = mgr.get(name, cfg.ctx, parse_keep_alive(ka, cfg.keep_alive), turn_has_media(body), err);
    if (inst == nullptr) {
        load_error(res, name, err, false);
        return;
    }

    json payload     = json::object();
    payload["model"] = name;
    payload["input"] = inputs;

    std::string    raw;
    const Upstream up = post_json(inst->port, "/v1/embeddings", payload, raw);
    if (!up.sent) {
        write_json(res, 502, error_obj(up.error));
        return;
    }
    if (up.status != 200) {
        std::string s = raw;
        if (s.size() > 300) {
            s = s.substr(0, 300);
        }
        write_json(res, up.status, error_obj("embeddings failed (" + std::to_string(up.status) + "): " + s));
        return;
    }
    const json d    = json::parse(raw, nullptr, false);
    json       vecs = json::array();
    if (!d.is_discarded()) {
        for (const auto & row : jlist(d, "data")) {
            if (row.is_object() && row.contains("embedding")) {
                vecs.push_back(row["embedding"]);
            }
        }
    }
    json out         = json::object();
    out["model"]     = name;
    out["embeddings"] = vecs;
    out["embedding"] = vecs.empty() ? json::array() : vecs[0];
    write_json(res, 200, out);
}

void handle_v1_chat_completions(const httplib::Request & req, httplib::Response & res, Config & cfg, Manager & mgr,
                                Registry & reg) {
    v1_proxy("/v1/chat/completions", req, res, cfg, mgr, reg);
}

void handle_v1_completions(const httplib::Request & req, httplib::Response & res, Config & cfg, Manager & mgr,
                           Registry & reg) {
    v1_proxy("/v1/completions", req, res, cfg, mgr, reg);
}

} // namespace llmash
