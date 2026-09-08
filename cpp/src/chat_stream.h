#pragma once

// The testable core of chat.cpp: every translation between llama.cpp's
// OpenAI-shaped SSE and Ollama's newline-delimited JSON, with no httplib and
// no Manager in sight, so chat_test.cpp can drive it on recorded lines. The
// handlers in chat.cpp are the I/O shell around this.
//
// Names live in llmash::chatstream, not llmash, because api_logic.h declares
// its own iso()/now_unix() for the same clock and only one of them may win at
// link time.

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <functional>
#include <limits>
#include <map>
#include <string>
#include <vector>

namespace llmash {
namespace chatstream {

using json = nlohmann::json;

// ------------------------------------------------------- Go's map helpers

// Go's str(): a string comes back as itself, anything else through
// fmt.Sprint, and a missing or null key as "".
inline std::string go_sprint(const json & v) {
    if (v.is_null()) {
        return "<nil>";
    }
    if (v.is_string()) {
        return v.get<std::string>();
    }
    if (v.is_boolean()) {
        return v.get<bool>() ? "true" : "false";
    }
    if (v.is_number_integer()) {
        return std::to_string(v.get<long long>());
    }
    if (v.is_number_unsigned()) {
        return std::to_string(v.get<unsigned long long>());
    }
    if (v.is_number_float()) {
        const double d = v.get<double>();
        if (d == std::floor(d) && std::fabs(d) < 1e15) {
            return std::to_string(static_cast<long long>(d));
        }
        char buf[40];
        std::snprintf(buf, sizeof(buf), "%g", d);
        return buf;
    }
    return v.dump();
}

inline std::string jstr(const json & m, const char * k) {
    if (!m.is_object()) {
        return "";
    }
    const auto it = m.find(k);
    if (it == m.end() || it->is_null()) {
        return "";
    }
    return go_sprint(*it);
}

inline double to_float(const json & v) {
    if (v.is_number()) {
        return v.get<double>();
    }
    if (v.is_string()) {
        try {
            return std::stod(v.get<std::string>());
        } catch (const std::exception &) {
            return 0;
        }
    }
    return 0;
}

inline double jnum(const json & m, const char * k) {
    if (!m.is_object()) {
        return 0;
    }
    const auto it = m.find(k);
    if (it == m.end() || !it->is_number()) {
        return 0;
    }
    return it->get<double>();
}

inline json jsub(const json & m, const char * k) {
    if (m.is_object()) {
        const auto it = m.find(k);
        if (it != m.end() && it->is_object()) {
            return *it;
        }
    }
    return json::object();
}

inline json jlist(const json & m, const char * k) {
    if (m.is_object()) {
        const auto it = m.find(k);
        if (it != m.end() && it->is_array()) {
            return *it;
        }
    }
    return json::array();
}

inline std::string jfirst(const std::string & a, const std::string & b) { return a.empty() ? b : a; }

inline std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

inline bool starts_with(const std::string & s, const std::string & p) {
    return s.size() >= p.size() && s.compare(0, p.size(), p) == 0;
}

// ------------------------------------------------------------------ time

inline double now_seconds() {
    return static_cast<double>(
               std::chrono::duration_cast<std::chrono::nanoseconds>(
                   std::chrono::system_clock::now().time_since_epoch())
                   .count()) /
           1e9;
}

// Go's "2006-01-02T15:04:05.999999+00:00": microseconds, trailing zeros
// trimmed, the fraction dropped entirely when it is zero.
inline std::string iso_time(double ts) {
    const long long us   = static_cast<long long>(std::llround(ts * 1e6));
    long long       secs = us / 1000000;
    long long       frac = us % 1000000;
    if (frac < 0) {
        frac += 1000000;
        secs -= 1;
    }
    std::time_t t = static_cast<std::time_t>(secs);
    std::tm     g{};
#ifdef _WIN32
    gmtime_s(&g, &t);
#else
    gmtime_r(&t, &g);
#endif
    char head[32];
    std::snprintf(head, sizeof(head), "%04d-%02d-%02dT%02d:%02d:%02d", g.tm_year + 1900, g.tm_mon + 1, g.tm_mday,
                  g.tm_hour, g.tm_min, g.tm_sec);
    std::string out = head;
    if (frac != 0) {
        char f[16];
        std::snprintf(f, sizeof(f), "%06lld", frac);
        std::string fs = f;
        while (!fs.empty() && fs.back() == '0') {
            fs.pop_back();
        }
        out += "." + fs;
    }
    return out + "+00:00";
}

// ------------------------------------------------------------ SSE / ndjson

enum class SseKind { Ignore, Data, Done };

struct SseLine {
    SseKind     kind = SseKind::Ignore;
    std::string data;
};

// One raw line off the wire. Only `data:` lines carry anything; "[DONE]"
// ends the stream.
inline SseLine parse_sse_line(const std::string & raw) {
    std::string line = raw;
    const auto  ws   = [](unsigned char c) { return std::isspace(c) != 0; };
    while (!line.empty() && ws(static_cast<unsigned char>(line.front()))) {
        line.erase(line.begin());
    }
    while (!line.empty() && ws(static_cast<unsigned char>(line.back()))) {
        line.pop_back();
    }
    SseLine out;
    if (line.rfind("data:", 0) != 0) {
        return out;
    }
    std::string data = line.substr(5);
    while (!data.empty() && ws(static_cast<unsigned char>(data.front()))) {
        data.erase(data.begin());
    }
    while (!data.empty() && ws(static_cast<unsigned char>(data.back()))) {
        data.pop_back();
    }
    if (data == "[DONE]") {
        out.kind = SseKind::Done;
        return out;
    }
    out.kind = SseKind::Data;
    out.data = data;
    return out;
}

inline std::string sse_frame(const json & v) { return "data: " + v.dump() + "\n\n"; }
inline std::string ndjson_line(const json & v) { return v.dump() + "\n"; }

inline json error_obj(const std::string & msg) {
    json j = json::object();
    j["error"] = msg;
    return j;
}

// The requested model name put back over the real one llama-server echoes.
// Both arguments are already JSON-quoted, so a name is never confused with a
// substring of the surrounding text.
inline std::string rewrite_model(std::string chunk, const std::string & real, const std::string & want) {
    if (real.empty() || real == want) {
        return chunk;
    }
    size_t at = 0;
    while ((at = chunk.find(real, at)) != std::string::npos) {
        chunk.replace(at, real.size(), want);
        at += want.size();
    }
    return chunk;
}

// -------------------------------------------------------------- events

inline json assistant_msg(const json & fields) {
    json m    = json::object();
    m["role"] = "assistant";
    m["content"] = "";
    if (fields.is_object()) {
        for (auto it = fields.begin(); it != fields.end(); ++it) {
            m[it.key()] = it.value();
        }
    }
    return m;
}

inline json ev_msg(const std::string & name, const json & msg, const std::string & created_at) {
    json e         = json::object();
    e["model"]     = name;
    e["created_at"] = created_at;
    e["message"]   = msg;
    e["done"]      = false;
    return e;
}

inline json final_event(const std::string & name, const std::string & finish, double total, double load, int n_in,
                        int n_out, const std::string & created_at) {
    if (load < 0) {
        load = 0;
    }
    double eval_dur = total - load;
    if (eval_dur < 0.001) {
        eval_dur = 0.001;
    }
    json m       = json::object();
    m["role"]    = "assistant";
    m["content"] = "";

    json e                     = json::object();
    e["model"]                 = name;
    e["created_at"]            = created_at;
    e["message"]               = m;
    e["done"]                  = true;
    e["done_reason"]           = finish;
    e["total_duration"]        = static_cast<int64_t>(total * 1e9);
    e["load_duration"]         = static_cast<int64_t>(load * 1e9);
    e["prompt_eval_count"]     = n_in;
    e["prompt_eval_duration"]  = static_cast<int64_t>(load * 1e9);
    e["eval_count"]            = n_out;
    e["eval_duration"]         = static_cast<int64_t>(eval_dur * 1e9);
    return e;
}

struct ToolSlot {
    std::string name;
    std::string args;
    std::string id;
    bool        announced = false;
    std::string buf;
    int         n       = 0;
    double      flushed = 0;
};

inline json finish_tool_calls(const std::map<int, ToolSlot> & slots, const std::vector<int> & order) {
    json calls = json::array();
    for (size_t i = 0; i < order.size(); ++i) {
        const auto it = slots.find(order[i]);
        if (it == slots.end()) {
            continue;
        }
        const ToolSlot & s    = it->second;
        std::string      trim = s.args;
        while (!trim.empty() && std::isspace(static_cast<unsigned char>(trim.front()))) {
            trim.erase(trim.begin());
        }
        while (!trim.empty() && std::isspace(static_cast<unsigned char>(trim.back()))) {
            trim.pop_back();
        }
        json args;
        if (trim.empty()) {
            args = json::object();
        } else {
            args = json::parse(s.args, nullptr, false);
            if (args.is_discarded()) {
                args         = json::object();
                args["_raw"] = s.args;
            }
        }
        json fn      = json::object();
        fn["name"]      = s.name;
        fn["arguments"] = args;

        json c        = json::object();
        c["id"]       = s.id.empty() ? ("call_" + std::to_string(i)) : s.id;
        c["type"]     = "function";
        c["function"] = fn;
        calls.push_back(c);
    }
    return calls;
}

// --------------------------------------------------- upstream translation

struct Step {
    std::vector<json> out;
    bool              restart = false; // reissue the request with the grown prefill
    bool              stop    = false; // stop reading this response
};

// One /v1/chat/completions SSE event in, the Ollama /api/chat events it
// becomes out. The caller sets `now` before each call, which is what makes
// the tool-argument flush interval and the think nudge testable.
class ChatStream {
public:
    std::string model;

    // The injected assistant prefill, empty when nothing was injected. With
    // it set, llama.cpp replays the prefill in `content`, so the visible text
    // has to be recovered by subtraction and split on </think> by hand.
    std::string hoff;
    std::string prefill;
    double      nudge_after_s = 15;
    int         think_budget  = 32000;
    std::string nudge_text;

    double now = 0;

    std::string finish   = "stop";
    int         n_in     = 0;
    int         n_out    = 0;
    int         n_reproc = -1;
    double      first_token_at = 0;

    std::function<void(const std::string &)> log;

    void begin_attempt() { acc_.clear(); }

    Step on_event(const json & ev) {
        Step e;
        const auto pp = ev.is_object() ? ev.find("prompt_progress") : ev.end();
        if (ev.is_object() && pp != ev.end() && !pp->is_null()) {
            json p            = json::object();
            p["model"]        = model;
            p["created_at"]   = iso_time(now);
            p["prompt_progress"] = *pp;
            p["done"]         = false;
            e.out.push_back(p);
        }
        const json choices = jlist(ev, "choices");
        json       ch      = json::object();
        if (!choices.empty() && choices[0].is_object()) {
            ch = choices[0];
        }
        const json delta = jsub(ch, "delta");
        if (const std::string f = jstr(ch, "finish_reason"); !f.empty()) {
            finish = f;
        }
        const json usage = jsub(ev, "usage");
        if (!usage.empty()) {
            if (usage.contains("prompt_tokens")) {
                n_in = static_cast<int>(to_float(usage["prompt_tokens"]));
            }
            if (usage.contains("completion_tokens")) {
                n_out = static_cast<int>(to_float(usage["completion_tokens"]));
            }
        }
        const json tm    = jsub(ev, "timings");
        bool       has_gen = tm.contains("predicted_n");
        int        gen_n   = has_gen ? static_cast<int>(to_float(tm["predicted_n"])) : 0;
        if (tm.contains("prompt_n") && !tm["prompt_n"].is_null()) {
            const int pn = static_cast<int>(to_float(tm["prompt_n"]));
            if (pn != n_reproc && log) {
                const int cached = static_cast<int>(jnum(tm, "cache_n"));
                char      buf[160];
                std::snprintf(buf, sizeof(buf), "prompt %d: %d cached, %d reprocessed (%.0fms)", pn + cached, cached,
                              pn, jnum(tm, "prompt_ms"));
                log(buf);
            }
            n_reproc = pn;
        }

        json msg       = json::object();
        msg["role"]    = "assistant";
        msg["content"] = "";
        if (has_gen) {
            msg["gen_n"] = gen_n;
        }
        bool emit_it = false;

        if (const std::string reason = jfirst(jstr(delta, "reasoning_content"), jstr(delta, "reasoning"));
            !reason.empty()) {
            msg["thinking"] = reason;
            emit_it         = true;
            if (first_token_at == 0) {
                first_token_at = now;
            }
        }
        const std::string content = jstr(delta, "content");
        if (!content.empty() && hoff.empty()) {
            msg["content"] = content;
            emit_it        = true;
            if (first_token_at == 0) {
                first_token_at = now;
            }
        } else if (!content.empty()) {
            prefill_step(content, e);
            if (e.restart) {
                return e;
            }
        }

        for (const auto & t : jlist(delta, "tool_calls")) {
            if (!t.is_object()) {
                continue;
            }
            const int idx = static_cast<int>(jnum(t, "index"));
            if (slots.find(idx) == slots.end()) {
                slots[idx] = ToolSlot{};
                order.push_back(idx);
            }
            ToolSlot & s  = slots[idx];
            const json fn = jsub(t, "function");
            if (const std::string id = jstr(t, "id"); !id.empty()) {
                s.id = id;
            }
            if (const std::string fname = jstr(fn, "name"); !fname.empty()) {
                s.name = fname;
                if (!s.announced) {
                    s.announced = true;
                    json f      = json::object();
                    f["tool_pending"] = fname;
                    e.out.push_back(ev_msg(model, assistant_msg(f), iso_time(now)));
                }
            }
            if (const std::string a = jstr(fn, "arguments"); !a.empty()) {
                s.args += a;
                s.buf += a;
                s.n++;
                if (now - s.flushed >= 0.05) {
                    s.flushed             = now;
                    const std::string chunk = s.buf;
                    const int         nn    = s.n;
                    s.buf.clear();
                    s.n = 0;
                    e.out.push_back(tool_args_event(idx, s.name, nn, chunk));
                }
            }
        }

        if (emit_it) {
            e.out.push_back(ev_msg(model, msg, iso_time(now)));
        }
        return e;
    }

    // Everything owed after the upstream stream ends: the tail of an
    // unterminated <think>, any unflushed tool arguments, the assembled tool
    // calls, and the done frame.
    std::vector<json> finalize(double started) {
        std::vector<json> out;
        if (!hoff.empty() && !open_th_.empty() && open_th_.size() > th_emitted_) {
            json f        = json::object();
            f["thinking"] = open_th_.substr(th_emitted_);
            out.push_back(ev_msg(model, assistant_msg(f), iso_time(now)));
        }
        if (!slots.empty()) {
            for (const int idx : order) {
                ToolSlot & s = slots[idx];
                if (!s.buf.empty()) {
                    out.push_back(tool_args_event(idx, s.name, s.n, s.buf));
                    s.buf.clear();
                    s.n = 0;
                }
            }
            json f          = json::object();
            f["tool_calls"] = finish_tool_calls(slots, order);
            out.push_back(ev_msg(model, assistant_msg(f), iso_time(now)));
            if (finish == "stop") {
                finish = "tool_calls";
            }
        }
        const double total = now - started;
        double       load  = total;
        if (first_token_at > 0) {
            load = first_token_at - started;
        }
        int prompt_eval = n_in;
        if (n_reproc >= 0) {
            prompt_eval = n_reproc;
        }
        out.push_back(final_event(model, finish, total, load, prompt_eval, n_out, iso_time(now)));
        return out;
    }

    std::map<int, ToolSlot> slots;
    std::vector<int>        order;

    // exposed for the tests, which assert on how far the prefill has grown
    const std::string & open_think() const { return open_th_; }
    size_t              thinking_emitted() const { return th_emitted_; }
    bool                closed() const { return closed_mode_; }

private:
    std::string acc_;
    std::string open_th_;
    size_t      th_emitted_  = 0;
    size_t      ans_emitted_ = 0;
    bool        nudged_      = false;
    bool        closed_mode_ = false;
    double      t_think0_    = 0;
    long long   watch_from_  = -1;

    json tool_args_event(int idx, const std::string & name, int n, const std::string & delta) const {
        json a     = json::object();
        a["index"] = idx;
        a["name"]  = name;
        a["n"]     = n;
        a["delta"] = delta;
        json f          = json::object();
        f["tool_args"]  = a;
        return ev_msg(model, assistant_msg(f), iso_time(now));
    }

    json out_fields(const char * key, const std::string & value) const {
        json f = json::object();
        f[key] = value;
        return ev_msg(model, assistant_msg(f), iso_time(now));
    }

    void prefill_step(const std::string & content, Step & e) {
        acc_ += content;
        std::string vis;
        if (starts_with(acc_, prefill)) {
            vis = acc_.substr(prefill.size());
        } else if (starts_with(prefill, acc_)) {
            vis = "";
        } else {
            vis = acc_;
        }
        const std::string carry = prefill.substr(hoff.size());
        if (!vis.empty() && closed_mode_) {
            if (vis.size() > ans_emitted_) {
                if (first_token_at == 0) {
                    first_token_at = now;
                }
                e.out.push_back(out_fields("content", vis.substr(ans_emitted_)));
                ans_emitted_ = vis.size();
            }
            return;
        }
        if (vis.empty()) {
            return;
        }

        static const std::string kClose = "</think>";
        std::string              th;
        std::string              ans;
        const size_t             i = vis.find(kClose);
        if (i != std::string::npos) {
            th = vis.substr(0, i);
            ans = vis.substr(i + kClose.size());
            open_th_.clear();
        } else {
            // the last 8 bytes could still be a half-written </think>
            if (vis.size() > kClose.size()) {
                th = vis.substr(0, vis.size() - kClose.size());
            }
            open_th_ = carry + vis;
        }
        if (!th.empty() && t_think0_ == 0) {
            t_think0_ = now;
        }
        const std::string full_th = carry + th;
        if (full_th.size() > th_emitted_) {
            e.out.push_back(out_fields("thinking", full_th.substr(th_emitted_)));
            th_emitted_ = full_th.size();
        }
        if (!ans.empty() && ans.size() > ans_emitted_) {
            if (first_token_at == 0) {
                first_token_at = now;
            }
            e.out.push_back(out_fields("content", ans.substr(ans_emitted_)));
            ans_emitted_ = ans.size();
        }
        if (i != std::string::npos) {
            return;
        }

        const std::string whole_th = carry + vis;
        if (!nudged_ && nudge_after_s > 0 && t_think0_ > 0 && now - t_think0_ > nudge_after_s) {
            if (watch_from_ < 0) {
                watch_from_ = static_cast<long long>(whole_th.size());
            }
            size_t from = watch_from_ > 0 ? static_cast<size_t>(watch_from_ - 1) : 0;
            if (from < carry.size()) {
                from = carry.size();
            }
            long long cut_at = -1;
            if (from <= whole_th.size()) {
                if (const size_t j = whole_th.find("\n\n", from); j != std::string::npos) {
                    cut_at = static_cast<long long>(j + 2);
                }
            }
            if (cut_at < 0 && static_cast<long long>(whole_th.size()) - watch_from_ > 300) {
                const size_t w = static_cast<size_t>(watch_from_);
                if (w <= whole_th.size()) {
                    if (const size_t k = whole_th.find(". ", w); k != std::string::npos) {
                        cut_at = static_cast<long long>(k + 2);
                    }
                }
            }
            if (cut_at >= 0) {
                const std::string cut = whole_th.substr(0, static_cast<size_t>(cut_at));
                if (th_emitted_ > cut.size()) {
                    th_emitted_ = cut.size();
                }
                e.out.push_back(out_fields("thinking", cut.substr(th_emitted_) + nudge_text));
                th_emitted_ = cut.size() + nudge_text.size();
                prefill     = hoff + cut + nudge_text;
                nudged_     = true;
                e.restart   = true;
                e.stop      = true;
                return;
            }
        }
        if (think_budget > 0 && static_cast<int>(whole_th.size()) > think_budget) {
            if (whole_th.size() > th_emitted_) {
                e.out.push_back(out_fields("thinking", whole_th.substr(th_emitted_)));
                th_emitted_ = whole_th.size();
            }
            prefill      = hoff + whole_th + "\n</think>\n\n";
            closed_mode_ = true;
            open_th_.clear();
            e.restart = true;
            e.stop    = true;
        }
    }
};

// ------------------------------------------------------ non-streaming fold

struct Folded {
    bool        is_error = false;
    std::string error;
    json        body;
};

// deliver()'s non-streaming half: the same events folded into one object.
inline Folded fold_events(const std::vector<json> & evs) {
    Folded      f;
    std::string content;
    std::string thinking;
    json        tools;
    json        tail;
    for (const json & ev : evs) {
        if (f.is_error) {
            break;
        }
        if (const std::string e = jstr(ev, "error"); !e.empty()) {
            f.is_error = true;
            f.error    = e;
            continue;
        }
        const json m = jsub(ev, "message");
        content += jstr(m, "content");
        thinking += jstr(m, "thinking");
        if (const json tc = jlist(m, "tool_calls"); !tc.empty()) {
            tools = tc;
        }
        if (ev.is_object()) {
            const auto d = ev.find("done");
            if (d != ev.end() && d->is_boolean() && d->get<bool>()) {
                tail = ev;
            }
        }
    }
    if (f.is_error) {
        return f;
    }
    if (tail.is_null()) {
        tail = json::object();
    }
    json msg       = json::object();
    msg["role"]    = "assistant";
    msg["content"] = content;
    if (!thinking.empty()) {
        msg["thinking"] = thinking;
    }
    if (!tools.is_null()) {
        msg["tool_calls"] = tools;
    }
    tail["message"] = msg;
    tail["done"]    = true;
    f.body          = tail;
    return f;
}

// /api/generate speaks response/thinking where /api/chat speaks message.
inline json chat_event_to_generate(json ev) {
    const json m  = jsub(ev, "message");
    ev["response"] = jstr(m, "content");
    if (const std::string t = jstr(m, "thinking"); !t.empty()) {
        ev["thinking"] = t;
    }
    ev.erase("message");
    return ev;
}

// ------------------------------------------------------- request shaping

inline bool turn_has_media(const json & body) {
    if (!jlist(body, "images").empty()) {
        return true;
    }
    for (const auto & m : jlist(body, "messages")) {
        if (!m.is_object()) {
            continue;
        }
        if (!jlist(m, "images").empty() || !jlist(m, "audio").empty()) {
            return true;
        }
        const auto c = m.find("content");
        if (c == m.end() || !c->is_array()) {
            continue;
        }
        for (const auto & p : *c) {
            if (!p.is_object()) {
                continue;
            }
            const std::string t = jstr(p, "type");
            if (t == "image_url" || t == "image" || t == "audio" || t == "audio_url" || t == "input_audio" ||
                t == "video" || t == "video_url") {
                return true;
            }
        }
    }
    return false;
}

inline json to_openai_messages(const json & messages) {
    json out = json::array();
    for (const auto & m : messages) {
        if (!m.is_object()) {
            continue;
        }
        std::string role = jstr(m, "role");
        if (role.empty()) {
            role = "user";
        }
        const json  content     = m.contains("content") ? m["content"] : json();
        std::string content_str = content.is_string() ? content.get<std::string>() : std::string();
        const json  images      = jlist(m, "images");
        const json  audio       = jlist(m, "audio");
        if (!images.empty() || !audio.empty()) {
            json parts = json::array();
            if (!content_str.empty()) {
                json p    = json::object();
                p["type"] = "text";
                p["text"] = content_str;
                parts.push_back(p);
            }
            for (const auto & a : audio) {
                std::string data;
                std::string format = "wav";
                if (a.is_string()) {
                    data = a.get<std::string>();
                } else if (a.is_object()) {
                    data   = jstr(a, "data");
                    format = lower(jstr(a, "format"));
                    if (format.empty()) {
                        format = "wav";
                    }
                }
                json ia      = json::object();
                ia["data"]   = data;
                ia["format"] = format;
                json p            = json::object();
                p["type"]         = "input_audio";
                p["input_audio"]  = ia;
                parts.push_back(p);
            }
            for (const auto & b64 : images) {
                json u   = json::object();
                u["url"] = "data:image/png;base64," + go_sprint(b64);
                json p        = json::object();
                p["type"]     = "image_url";
                p["image_url"] = u;
                parts.push_back(p);
            }
            json msg      = json::object();
            msg["role"]   = role;
            msg["content"] = parts;
            out.push_back(msg);
            continue;
        }
        if (role == "tool") {
            std::string id = jstr(m, "tool_call_id");
            if (id.empty()) {
                id = jstr(m, "tool_name");
            }
            if (id.empty()) {
                id = jstr(m, "name");
            }
            if (id.empty()) {
                id = "call";
            }
            json msg            = json::object();
            msg["role"]         = "tool";
            msg["content"]      = content;
            msg["tool_call_id"] = id;
            out.push_back(msg);
            continue;
        }
        json msg       = json::object();
        msg["role"]    = role;
        msg["content"] = content;
        const json tcs = jlist(m, "tool_calls");
        if (!tcs.empty()) {
            std::string trimmed = content_str;
            while (!trimmed.empty() && std::isspace(static_cast<unsigned char>(trimmed.front()))) {
                trimmed.erase(trimmed.begin());
            }
            while (!trimmed.empty() && std::isspace(static_cast<unsigned char>(trimmed.back()))) {
                trimmed.pop_back();
            }
            if (trimmed.empty()) {
                msg["content"] = nullptr;
            }
            json calls = json::array();
            for (size_t i = 0; i < tcs.size(); ++i) {
                const json  tc  = tcs[i].is_object() ? tcs[i] : json::object();
                const json  fn  = jsub(tc, "function");
                const auto  a   = fn.find("arguments");
                std::string arg_str;
                if (a != fn.end() && a->is_string()) {
                    arg_str = a->get<std::string>();
                } else if (a == fn.end() || a->is_null()) {
                    arg_str = "{}";
                } else {
                    arg_str = a->dump();
                }
                std::string id = jstr(tc, "id");
                if (id.empty()) {
                    id = "call_" + std::to_string(i);
                }
                json f          = json::object();
                f["name"]       = jstr(fn, "name");
                f["arguments"]  = arg_str;
                json c          = json::object();
                c["id"]         = id;
                c["type"]       = "function";
                c["function"]   = f;
                calls.push_back(c);
            }
            msg["tool_calls"] = calls;
        }
        out.push_back(msg);
    }
    return out;
}

inline json map_options(const json & opts) {
    json m = json::object();
    if (!opts.is_object()) {
        return m;
    }
    static const char * kPass[] = {"temperature",      "top_p",  "top_k",         "seed",
                                   "repeat_penalty",   "presence_penalty", "min_p", "frequency_penalty"};
    for (const char * k : kPass) {
        const auto it = opts.find(k);
        if (it != opts.end() && !it->is_null()) {
            m[k] = *it;
        }
    }
    if (const int np = static_cast<int>(jnum(opts, "num_predict")); np > 0) {
        m["max_tokens"] = np;
    }
    const auto s = opts.find("stop");
    if (s != opts.end() && !s->is_null()) {
        if (!s->is_array() || !s->empty()) {
            m["stop"] = *s;
        }
    }
    return m;
}

inline bool name_matches_any(const std::string & name, const std::vector<std::string> & keys) {
    const std::string low = lower(name);
    for (const std::string & k : keys) {
        if (!k.empty() && low.find(k) != std::string::npos) {
            return true;
        }
    }
    return false;
}

// Ollama accepts 5m / 1h / bare seconds / -1 (forever) / 0 (now).
inline double parse_keep_alive(const json & v, const std::string & fallback) {
    const double inf = std::numeric_limits<double>::infinity();
    std::string  s;
    if (v.is_null()) {
        s = fallback;
    } else if (v.is_number()) {
        const double d = v.get<double>();
        return d < 0 ? inf : d;
    } else {
        s = go_sprint(v);
    }
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front()))) {
        s.erase(s.begin());
    }
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) {
        s.pop_back();
    }
    if (s == "-1" || s == "-1s") {
        return inf;
    }
    const auto parse = [](const std::string & x, double & out) {
        try {
            size_t used = 0;
            out         = std::stod(x, &used);
            return used == x.size();
        } catch (const std::exception &) {
            return false;
        }
    };
    double f = 0;
    if (s.size() > 2 && s.compare(s.size() - 2, 2, "ms") == 0) {
        if (parse(s.substr(0, s.size() - 2), f)) {
            return f / 1000;
        }
    } else if (!s.empty() && s.back() == 's') {
        if (parse(s.substr(0, s.size() - 1), f)) {
            return f;
        }
    } else if (!s.empty() && s.back() == 'm') {
        if (parse(s.substr(0, s.size() - 1), f)) {
            return f * 60;
        }
    } else if (!s.empty() && s.back() == 'h') {
        if (parse(s.substr(0, s.size() - 1), f)) {
            return f * 3600;
        }
    } else if (parse(s, f)) {
        return f < 0 ? inf : f;
    }
    return 300;
}

// The two spellings Ollama uses to mean "drop it now".
inline bool keep_alive_is_zero(const json & body) {
    const auto it = body.find("keep_alive");
    if (it == body.end()) {
        return false;
    }
    const std::string s = go_sprint(*it);
    return s == "0" || s == "0s";
}

} // namespace chatstream
} // namespace llmash
