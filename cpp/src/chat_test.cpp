// Drives chat.cpp's translation layer on recorded llama.cpp SSE and recorded
// Ollama ndjson, with no server and no model.

#include "chat_stream.h"

#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

using namespace llmash::chatstream;
using json = nlohmann::json;

namespace {

int g_failed = 0;
int g_checks = 0;

void eq(const std::string & what, const std::string & got, const std::string & want) {
    ++g_checks;
    if (got == want) {
        return;
    }
    ++g_failed;
    std::cout << "FAIL " << what << "\n  want: " << want << "\n  got:  " << got << "\n";
}

void eq(const std::string & what, double got, double want) {
    ++g_checks;
    if (got == want) {
        return;
    }
    ++g_failed;
    std::cout << "FAIL " << what << "\n  want: " << want << "\n  got:  " << got << "\n";
}

void ok(const std::string & what, bool cond) {
    ++g_checks;
    if (cond) {
        return;
    }
    ++g_failed;
    std::cout << "FAIL " << what << "\n";
}

json ev_of(const std::string & sse_line) {
    const SseLine sl = parse_sse_line(sse_line);
    if (sl.kind != SseKind::Data) {
        return json();
    }
    return json::parse(sl.data, nullptr, false);
}

// The fixed clock every expectation below is written against.
const double kT0 = 1700000000.0; // 2023-11-14T22:13:20Z

// ------------------------------------------------------------------ tests

void test_sse_framing() {
    const SseLine a = parse_sse_line(
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hi\"}}]}\r");
    ok("data line is Data", a.kind == SseKind::Data);
    eq("data payload", a.data, "{\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hi\"}}]}");

    ok("[DONE] ends the stream", parse_sse_line("data: [DONE]").kind == SseKind::Done);
    ok("blank line ignored", parse_sse_line("").kind == SseKind::Ignore);
    ok("event: line ignored", parse_sse_line("event: message").kind == SseKind::Ignore);
    ok("comment ignored", parse_sse_line(": ping").kind == SseKind::Ignore);

    // the other direction: an object back out as an SSE frame
    json e            = json::object();
    e["done"]         = true;
    e["done_reason"]  = "stop";
    eq("sse_frame", sse_frame(e), "data: {\"done\":true,\"done_reason\":\"stop\"}\n\n");
    eq("frame round trips", parse_sse_line("data: {\"done\":true,\"done_reason\":\"stop\"}").data,
       "{\"done\":true,\"done_reason\":\"stop\"}");

    eq("ndjson_line", ndjson_line(e), "{\"done\":true,\"done_reason\":\"stop\"}\n");
}

void test_iso() {
    eq("iso whole second", iso_time(kT0), "2023-11-14T22:13:20+00:00");
    eq("iso trims zeros", iso_time(kT0 + 0.5), "2023-11-14T22:13:20.5+00:00");
    eq("iso keeps digits", iso_time(kT0 + 0.125), "2023-11-14T22:13:20.125+00:00");
}

// A real /api/chat turn: role frame, a reasoning_content frame with timings,
// a content frame, the finish frame, the usage frame.
void test_sse_to_ndjson() {
    ChatStream st;
    st.model = "qwen3:8b";
    st.now   = kT0;

    std::vector<std::string> log;
    st.log = [&](const std::string & s) { log.push_back(s); };

    std::vector<json> out;
    const auto        feed = [&](const std::string & line) {
        const Step s = st.on_event(ev_of(line));
        for (const json & o : s.out) {
            out.push_back(o);
        }
        return s;
    };

    // the opening role-only delta emits nothing
    feed("data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":\"\"},\"finish_reason\":null,\"index\":0}],"
         "\"created\":1700000000,\"id\":\"chatcmpl-1\",\"model\":\"B:/gguf/qwen3-8b.gguf\","
         "\"object\":\"chat.completion.chunk\"}");
    eq("role frame emits nothing", static_cast<double>(out.size()), 0.0);

    // prompt_progress rides its own frame
    feed("data: {\"prompt_progress\":{\"cache\":12,\"processed\":20,\"time_ms\":80,\"total\":34},\"choices\":[]}");
    eq("prompt_progress passthrough", out.at(0).dump(),
       "{\"created_at\":\"2023-11-14T22:13:20+00:00\",\"done\":false,\"model\":\"qwen3:8b\","
       "\"prompt_progress\":{\"cache\":12,\"processed\":20,\"time_ms\":80,\"total\":34}}");

    // reasoning_content becomes Ollama's `thinking`
    feed("data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"The user wants \"},\"finish_reason\":null,"
         "\"index\":0}],\"timings\":{\"cache_n\":12,\"predicted_n\":1,\"prompt_ms\":118.5,\"prompt_n\":34}}");
    eq("reasoning_content frame", out.at(1).dump(),
       "{\"created_at\":\"2023-11-14T22:13:20+00:00\",\"done\":false,"
       "\"message\":{\"content\":\"\",\"gen_n\":1,\"role\":\"assistant\",\"thinking\":\"The user wants \"},"
       "\"model\":\"qwen3:8b\"}");
    eq("prompt log line", log.at(0), "prompt 46: 12 cached, 34 reprocessed (118ms)");

    // `reasoning` is the other spelling of the same field
    feed("data: {\"choices\":[{\"delta\":{\"reasoning\":\"a greeting.\"},\"index\":0}],"
         "\"timings\":{\"predicted_n\":4,\"prompt_n\":34}}");
    eq("reasoning frame", out.at(2).dump(),
       "{\"created_at\":\"2023-11-14T22:13:20+00:00\",\"done\":false,"
       "\"message\":{\"content\":\"\",\"gen_n\":4,\"role\":\"assistant\",\"thinking\":\"a greeting.\"},"
       "\"model\":\"qwen3:8b\"}");
    eq("prompt_n logged once", static_cast<double>(log.size()), 1.0);

    st.now = kT0 + 0.5;
    feed("data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"},\"finish_reason\":null,\"index\":0}],"
         "\"timings\":{\"predicted_n\":7,\"prompt_n\":34}}");
    eq("content frame", out.at(3).dump(),
       "{\"created_at\":\"2023-11-14T22:13:20.5+00:00\",\"done\":false,"
       "\"message\":{\"content\":\"Hello\",\"gen_n\":7,\"role\":\"assistant\"},\"model\":\"qwen3:8b\"}");

    feed("data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\",\"index\":0}]}");
    feed("data: {\"choices\":[],\"usage\":{\"completion_tokens\":9,\"prompt_tokens\":34,\"total_tokens\":43}}");
    eq("finish and usage emit nothing", static_cast<double>(out.size()), 4.0);
    eq("finish_reason captured", st.finish, "stop");
    eq("completion tokens", static_cast<double>(st.n_out), 9.0);

    // the done frame the client waits on; the request went out half a second
    // before the first token, so that half second is the load half
    st.now                       = kT0 + 2.0;
    const std::vector<json> tail = st.finalize(kT0 - 0.5);
    eq("one tail event", static_cast<double>(tail.size()), 1.0);
    eq("done frame", tail.at(0).dump(),
       "{\"created_at\":\"2023-11-14T22:13:22+00:00\",\"done\":true,\"done_reason\":\"stop\","
       "\"eval_count\":9,\"eval_duration\":2000000000,\"load_duration\":500000000,"
       "\"message\":{\"content\":\"\",\"role\":\"assistant\"},\"model\":\"qwen3:8b\","
       "\"prompt_eval_count\":34,\"prompt_eval_duration\":500000000,\"total_duration\":2500000000}");
}

void test_tool_calls() {
    ChatStream st;
    st.model = "qwen3:8b";
    st.now   = kT0;

    std::vector<json> out;
    const auto        feed = [&](const std::string & line) {
        for (const json & o : st.on_event(ev_of(line)).out) {
            out.push_back(o);
        }
    };

    feed("data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"function\":{\"arguments\":\"\","
         "\"name\":\"get_weather\"},\"id\":\"call_abc\",\"index\":0,\"type\":\"function\"}]},"
         "\"finish_reason\":null,\"index\":0}]}");
    eq("tool_pending announced", out.at(0).dump(),
       "{\"created_at\":\"2023-11-14T22:13:20+00:00\",\"done\":false,"
       "\"message\":{\"content\":\"\",\"role\":\"assistant\",\"tool_pending\":\"get_weather\"},"
       "\"model\":\"qwen3:8b\"}");

    feed("data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"function\":{\"arguments\":\"{\\\"city\\\":\"},"
         "\"index\":0}]},\"index\":0}]}");
    eq("first args flush", out.at(1).dump(),
       "{\"created_at\":\"2023-11-14T22:13:20+00:00\",\"done\":false,"
       "\"message\":{\"content\":\"\",\"role\":\"assistant\","
       "\"tool_args\":{\"delta\":\"{\\\"city\\\":\",\"index\":0,\"n\":1,\"name\":\"get_weather\"}},"
       "\"model\":\"qwen3:8b\"}");

    // same instant: inside the 50 ms flush window, so this one buffers
    feed("data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"function\":{\"arguments\":\"\\\"Paris\\\"}\"},"
         "\"index\":0}]},\"index\":0}]}");
    eq("second chunk buffered", static_cast<double>(out.size()), 2.0);

    feed("data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\",\"index\":0}]}");

    st.now                       = kT0 + 1.0;
    const std::vector<json> tail = st.finalize(kT0);
    eq("tail events", static_cast<double>(tail.size()), 3.0);
    eq("buffered args flushed", tail.at(0).dump(),
       "{\"created_at\":\"2023-11-14T22:13:21+00:00\",\"done\":false,"
       "\"message\":{\"content\":\"\",\"role\":\"assistant\","
       "\"tool_args\":{\"delta\":\"\\\"Paris\\\"}\",\"index\":0,\"n\":1,\"name\":\"get_weather\"}},"
       "\"model\":\"qwen3:8b\"}");
    eq("assembled tool_calls", tail.at(1).dump(),
       "{\"created_at\":\"2023-11-14T22:13:21+00:00\",\"done\":false,"
       "\"message\":{\"content\":\"\",\"role\":\"assistant\",\"tool_calls\":[{\"function\":"
       "{\"arguments\":{\"city\":\"Paris\"},\"name\":\"get_weather\"},\"id\":\"call_abc\","
       "\"type\":\"function\"}]},\"model\":\"qwen3:8b\"}");
    eq("stop promoted to tool_calls", tail.at(2)["done_reason"].get<std::string>(), "tool_calls");
}

void test_unparsable_tool_args() {
    std::map<int, ToolSlot> slots;
    std::vector<int>        order{0};
    ToolSlot                s;
    s.name  = "run";
    s.args  = "{not json";
    slots[0] = s;
    eq("raw args preserved", finish_tool_calls(slots, order).dump(),
       "[{\"function\":{\"arguments\":{\"_raw\":\"{not json\"},\"name\":\"run\"},\"id\":\"call_0\","
       "\"type\":\"function\"}]");

    slots[0].args = "   ";
    slots[0].id   = "";
    eq("blank args become {}", finish_tool_calls(slots, order).dump(),
       "[{\"function\":{\"arguments\":{},\"name\":\"run\"},\"id\":\"call_0\",\"type\":\"function\"}]");
}

// The prefill path: with an injected <think> handoff llama.cpp replays the
// prefill in `content`, so the visible text is recovered by subtraction and
// split on </think> by hand.
void test_injected_prefill() {
    ChatStream st;
    st.model         = "qwen3:8b";
    st.hoff          = "<think>\nLet me think. ";
    st.prefill       = st.hoff;
    st.nudge_after_s = 0; // off for this case
    st.think_budget  = 0;
    st.now           = kT0;
    st.begin_attempt();

    std::vector<json> out;
    const auto        feed = [&](const json & delta_content) {
        json d       = json::object();
        d["content"] = delta_content;
        json ch      = json::object();
        ch["delta"]  = d;
        ch["index"]  = 0;
        json ev        = json::object();
        ev["choices"]  = json::array({ch});
        for (const json & o : st.on_event(ev).out) {
            out.push_back(o);
        }
    };

    feed("<think>\nLet me think. First I check");
    eq("only the settled prefix is thinking", out.at(0).dump(),
       "{\"created_at\":\"2023-11-14T22:13:20+00:00\",\"done\":false,"
       "\"message\":{\"content\":\"\",\"role\":\"assistant\",\"thinking\":\"First\"},\"model\":\"qwen3:8b\"}");

    feed(" the docs.\n</think>\n\nHere it is.");
    eq("thinking completed", out.at(1)["message"]["thinking"].get<std::string>(), " I check the docs.\n");
    eq("answer after the close tag", out.at(2)["message"]["content"].get<std::string>(), "\n\nHere it is.");
    eq("three prefill events", static_cast<double>(out.size()), 3.0);
    ok("open think cleared", st.open_think().empty());
}

void test_think_budget_restart() {
    ChatStream st;
    st.model         = "qwen3:8b";
    st.hoff          = "<think>\n";
    st.prefill       = st.hoff;
    st.nudge_after_s = 0;
    st.think_budget  = 16;
    st.now           = kT0;
    st.begin_attempt();

    json d       = json::object();
    d["content"] = "<think>\nrambling on and on";
    json ch      = json::object();
    ch["delta"]  = d;
    json ev       = json::object();
    ev["choices"] = json::array({ch});

    const Step s = st.on_event(ev);
    ok("budget forces a restart", s.restart && s.stop);
    eq("prefill closed off", st.prefill, "<think>\nrambling on and on\n</think>\n\n");
    ok("closed mode latched", st.closed());
}

// Ollama's /api/chat ndjson turned into /api/generate's response/thinking.
void test_ndjson_to_generate() {
    const std::string chat_line =
        "{\"created_at\":\"2023-11-14T22:13:20+00:00\",\"done\":false,"
        "\"message\":{\"content\":\"Hello\",\"gen_n\":7,\"role\":\"assistant\"},\"model\":\"qwen3:8b\"}";
    eq("content becomes response", chat_event_to_generate(json::parse(chat_line)).dump(),
       "{\"created_at\":\"2023-11-14T22:13:20+00:00\",\"done\":false,\"model\":\"qwen3:8b\","
       "\"response\":\"Hello\"}");

    const std::string think_line =
        "{\"created_at\":\"2023-11-14T22:13:20+00:00\",\"done\":false,"
        "\"message\":{\"content\":\"\",\"role\":\"assistant\",\"thinking\":\"The user wants \"},"
        "\"model\":\"qwen3:8b\"}";
    eq("thinking is lifted out", chat_event_to_generate(json::parse(think_line)).dump(),
       "{\"created_at\":\"2023-11-14T22:13:20+00:00\",\"done\":false,\"model\":\"qwen3:8b\","
       "\"response\":\"\",\"thinking\":\"The user wants \"}");

    const std::string done_line =
        "{\"created_at\":\"2023-11-14T22:13:22+00:00\",\"done\":true,\"done_reason\":\"stop\","
        "\"eval_count\":9,\"eval_duration\":1500000000,\"load_duration\":500000000,"
        "\"message\":{\"content\":\"\",\"role\":\"assistant\"},\"model\":\"qwen3:8b\","
        "\"prompt_eval_count\":34,\"prompt_eval_duration\":500000000,\"total_duration\":2000000000}";
    eq("done frame keeps its counters", chat_event_to_generate(json::parse(done_line)).dump(),
       "{\"created_at\":\"2023-11-14T22:13:22+00:00\",\"done\":true,\"done_reason\":\"stop\","
       "\"eval_count\":9,\"eval_duration\":1500000000,\"load_duration\":500000000,\"model\":\"qwen3:8b\","
       "\"prompt_eval_count\":34,\"prompt_eval_duration\":500000000,\"response\":\"\","
       "\"total_duration\":2000000000}");

    const std::string err_line = "{\"error\":\"llama-server 500: out of memory\"}";
    eq("an error frame survives", chat_event_to_generate(json::parse(err_line)).dump(),
       "{\"error\":\"llama-server 500: out of memory\",\"response\":\"\"}");
}

void test_fold_events() {
    std::vector<json> evs;
    evs.push_back(json::parse("{\"done\":false,\"message\":{\"content\":\"\",\"role\":\"assistant\","
                              "\"thinking\":\"Think \"},\"model\":\"m\"}"));
    evs.push_back(json::parse("{\"done\":false,\"message\":{\"content\":\"\",\"role\":\"assistant\","
                              "\"thinking\":\"harder.\"},\"model\":\"m\"}"));
    evs.push_back(json::parse("{\"done\":false,\"message\":{\"content\":\"Hel\",\"role\":\"assistant\"},"
                              "\"model\":\"m\"}"));
    evs.push_back(json::parse("{\"done\":false,\"message\":{\"content\":\"lo\",\"role\":\"assistant\"},"
                              "\"model\":\"m\"}"));
    evs.push_back(json::parse("{\"done\":true,\"done_reason\":\"stop\",\"eval_count\":4,"
                              "\"message\":{\"content\":\"\",\"role\":\"assistant\"},\"model\":\"m\"}"));
    const Folded f = fold_events(evs);
    ok("no error", !f.is_error);
    eq("folded body", f.body.dump(),
       "{\"done\":true,\"done_reason\":\"stop\",\"eval_count\":4,"
       "\"message\":{\"content\":\"Hello\",\"role\":\"assistant\",\"thinking\":\"Think harder.\"},"
       "\"model\":\"m\"}");

    std::vector<json> bad;
    bad.push_back(json::parse("{\"done\":false,\"message\":{\"content\":\"partial\",\"role\":\"assistant\"}}"));
    bad.push_back(error_obj("llama-server 503: loading model"));
    bad.push_back(json::parse("{\"done\":true,\"done_reason\":\"stop\"}"));
    const Folded e = fold_events(bad);
    ok("error detected", e.is_error);
    eq("error text", e.error, "llama-server 503: loading model");

    std::vector<json> tools;
    tools.push_back(json::parse("{\"done\":false,\"message\":{\"content\":\"\",\"role\":\"assistant\","
                                "\"tool_calls\":[{\"function\":{\"arguments\":{\"city\":\"Paris\"},"
                                "\"name\":\"get_weather\"},\"id\":\"call_abc\",\"type\":\"function\"}]}}"));
    tools.push_back(json::parse("{\"done\":true,\"done_reason\":\"tool_calls\"}"));
    eq("tool calls survive the fold", fold_events(tools).body.dump(),
       "{\"done\":true,\"done_reason\":\"tool_calls\",\"message\":{\"content\":\"\",\"role\":\"assistant\","
       "\"tool_calls\":[{\"function\":{\"arguments\":{\"city\":\"Paris\"},\"name\":\"get_weather\"},"
       "\"id\":\"call_abc\",\"type\":\"function\"}]}}");
}

void test_v1_model_rewrite() {
    const std::string real = json("B:\\gguf\\qwen3-8b-q4.gguf").dump();
    const std::string want = json("qwen3:8b").dump();
    const std::string chunk =
        "data: {\"id\":\"chatcmpl-1\",\"model\":\"B:\\\\gguf\\\\qwen3-8b-q4.gguf\",\"object\":"
        "\"chat.completion.chunk\",\"choices\":[{\"delta\":{\"content\":\"Hi\"}}]}\n\n";
    eq("gguf path swapped for the asked-for name", rewrite_model(chunk, real, want),
       "data: {\"id\":\"chatcmpl-1\",\"model\":\"qwen3:8b\",\"object\":"
       "\"chat.completion.chunk\",\"choices\":[{\"delta\":{\"content\":\"Hi\"}}]}\n\n");
    eq("nothing to swap", rewrite_model("data: [DONE]\n\n", real, want), "data: [DONE]\n\n");
}

void test_request_shaping() {
    const json body = json::parse(
        "{\"messages\":[{\"content\":\"look\",\"images\":[\"QUJD\"],\"role\":\"user\"},"
        "{\"content\":\"\",\"role\":\"assistant\",\"tool_calls\":[{\"function\":{\"arguments\":"
        "{\"city\":\"Paris\"},\"name\":\"get_weather\"}}]},"
        "{\"content\":\"18C\",\"role\":\"tool\",\"tool_name\":\"get_weather\"}],\"model\":\"m\"}");
    ok("media turn detected", turn_has_media(body));
    eq("openai messages", to_openai_messages(jlist(body, "messages")).dump(),
       "[{\"content\":[{\"text\":\"look\",\"type\":\"text\"},{\"image_url\":"
       "{\"url\":\"data:image/png;base64,QUJD\"},\"type\":\"image_url\"}],\"role\":\"user\"},"
       "{\"content\":null,\"role\":\"assistant\",\"tool_calls\":[{\"function\":"
       "{\"arguments\":\"{\\\"city\\\":\\\"Paris\\\"}\",\"name\":\"get_weather\"},\"id\":\"call_0\","
       "\"type\":\"function\"}]},"
       "{\"content\":\"18C\",\"role\":\"tool\",\"tool_call_id\":\"get_weather\"}]");

    ok("plain turn has no media", !turn_has_media(json::parse(
                                      "{\"messages\":[{\"content\":\"hi\",\"role\":\"user\"}]}")));

    const json opts = json::parse("{\"num_ctx\":8192,\"num_predict\":128,\"stop\":[\"</s>\"],"
                                  "\"temperature\":0.7,\"top_k\":40,\"unknown\":1}");
    eq("options mapped", map_options(opts).dump(),
       "{\"max_tokens\":128,\"stop\":[\"</s>\"],\"temperature\":0.7,\"top_k\":40}");
    eq("empty stop list dropped", map_options(json::parse("{\"stop\":[]}")).dump(), "{}");
}

void test_keep_alive() {
    eq("default", parse_keep_alive(json(), "15m"), 900.0);
    eq("minutes", parse_keep_alive(json("5m"), "15m"), 300.0);
    eq("hours", parse_keep_alive(json("1h"), "15m"), 3600.0);
    eq("seconds suffix", parse_keep_alive(json("30s"), "15m"), 30.0);
    eq("milliseconds", parse_keep_alive(json("1500ms"), "15m"), 1.5);
    eq("bare number", parse_keep_alive(json(120), "15m"), 120.0);
    eq("zero", parse_keep_alive(json(0), "15m"), 0.0);
    ok("-1 pins forever", std::isinf(parse_keep_alive(json(-1), "15m")));
    ok("\"-1\" pins forever", std::isinf(parse_keep_alive(json("-1"), "15m")));
    eq("garbage falls back", parse_keep_alive(json("soon"), "15m"), 300.0);

    ok("0 unloads", keep_alive_is_zero(json::parse("{\"keep_alive\":0}")));
    ok("0s unloads", keep_alive_is_zero(json::parse("{\"keep_alive\":\"0s\"}")));
    ok("absent does not", !keep_alive_is_zero(json::parse("{}")));
    ok("5m does not", !keep_alive_is_zero(json::parse("{\"keep_alive\":\"5m\"}")));
}

void test_think_off() {
    const std::vector<std::string> keys{"gemma4", "gemma-4"};
    ok("gemma4 matches", name_matches_any("Gemma4:e2b", keys));
    ok("gemma-4 matches", name_matches_any("google/gemma-4-9b", keys));
    ok("qwen does not", !name_matches_any("qwen3:8b", keys));
}

} // namespace

int main() {
    test_sse_framing();
    test_iso();
    test_sse_to_ndjson();
    test_tool_calls();
    test_unparsable_tool_args();
    test_injected_prefill();
    test_think_budget_restart();
    test_ndjson_to_generate();
    test_fold_events();
    test_v1_model_rewrite();
    test_request_shaping();
    test_keep_alive();
    test_think_off();

    std::printf("chat_test: %d checks, %d failed\n", g_checks, g_failed);
    return g_failed == 0 ? 0 : 1;
}
