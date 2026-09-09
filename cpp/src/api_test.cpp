// Exercises api.cpp against a real httplib server on an ephemeral port with
// a real httplib client: route mounting, the method-agnostic dispatch Go's
// mux gives every path, the catch-all 404, CORS/OPTIONS, the public-port key
// gate, the ndjson framing of /api/create and /api/copy, and every response
// api.cpp shapes on its own.

#include "api.h"
#include "platform.h"
#include "api_logic.h"
#include "chat.h"
#include "pull.h"
#include "table.h"

#include <atomic>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
using namespace llmash;
using json = nlohmann::json;

// --------------------------------------------------------------- STUBS
// Stand-ins for the modules landing alongside this one. Each records what it
// was handed so the test can assert api.cpp passed the right thing, and
// returns a sentinel that is never mistaken for the real module's output.

namespace llmash {

namespace stub {
std::string  auth_seen, xkey_seen, expected_seen;
std::string  create_name, create_from, create_quant, create_draft;
std::string  copy_dest;
bool         copy_called   = false;
int          tags_calls    = 0;
int          ps_calls      = 0;
std::string  v1_id_seen;
std::string  keep_default_seen;
double       keep_returns  = 900;
} // namespace stub

std::string iso(double unix_seconds) { return "iso(" + std::to_string(unix_seconds) + ")"; } // STUB
double      now_unix() { return 1000.0; }                                                    // STUB
json tags_json(const std::vector<Model> & models, const Config &) {                     // STUB
    stub::tags_calls++;
    return json{{"models", json::array({json{{"name", "stub_models"}, {"count", models.size()}}})}};
}
json v1_entry_json(const Model & m, const Config &) { return json{{"id", m.name}}; } // STUB
json v1_models_json(const std::vector<Model> &, const Config &) {                    // STUB
    return json{{"object", "list"}, {"data", json::array()}};
}
json show_json(const Model & m, const Config &) { return json{{"modelfile", "FROM " + m.path}}; } // STUB
json openai_error_json(const std::string & message, const std::string & type,                    // STUB
                       const std::string & code) {
    return json{{"error", json{{"message", message}, {"type", type}, {"code", code}}}};
}
json ps_json(const std::vector<InstanceView> & live, const Config &) {                              // STUB
    stub::ps_calls++;
    return json{{"models", json::array({json{{"name", "stub_live"}, {"count", live.size()}}})}};
}
// STUB, implementing the contract api_logic.h documents so the gate wiring
// around it means something: a Bearer token wins over X-API-Key, and an
// empty expected key always rejects.
bool check_api_key(const std::string & authorization_header, const std::string & x_api_key_header,
                    const std::string & expected_key) {
    stub::auth_seen     = authorization_header;
    stub::xkey_seen     = x_api_key_header;
    stub::expected_seen = expected_key;
    std::string key     = x_api_key_header;
    if (authorization_header.size() > 7 && starts_with_ci(authorization_header, "bearer ")) {
        key = authorization_header.substr(7);
        while (!key.empty() && (key.front() == ' ' || key.front() == '\t')) {
            key.erase(key.begin());
        }
    }
    return !expected_key.empty() && key == expected_key;
}

std::string link_key(const Config &) { return "sk-llmash-testkey"; } // STUB

double parse_keep_alive(const json &, const std::string & default_keep) { // STUB
    stub::keep_default_seen = default_keep;
    return stub::keep_returns;
}
double keep_alive_out(double ka) { return ka; } // STUB

std::string loose_dir(const Config & cfg) { return cfg.root + "\\gguf"; } // STUB

void run_create(const Config &, const std::string & name, const std::string & from, // STUB
                const std::string & quantize, const std::string & draft_quantize, const Emit & emit) {
    stub::create_name  = name;
    stub::create_from  = from;
    stub::create_quant = quantize;
    stub::create_draft = draft_quantize;
    emit(json{{"status", "a"}});
    emit(json{{"status", "b"}});
}

void run_copy(const Config &, const Model &, const std::string & destination, const Emit & emit) { // STUB
    stub::copy_called = true;
    stub::copy_dest   = destination;
    emit(json{{"status", "success"}});
}

DeleteOutcome run_delete(const Model &) { return DeleteOutcome{}; } // STUB

std::string CliTextCache::get(const std::string & kind, double, const std::function<std::string()> & build) {
    (void)kind;
    return build();
} // STUB
void CliTextCache::invalidate() {} // STUB


void handle_chat(const httplib::Request &, httplib::Response &, Config &, Manager &, Registry &) {}     // STUB
void handle_generate(const httplib::Request &, httplib::Response &, Config &, Manager &, Registry &) {} // STUB
void handle_embed(const httplib::Request &, httplib::Response &, Config &, Manager &, Registry &) {}    // STUB
void handle_v1_chat_completions(const httplib::Request &, httplib::Response &, Config &, Manager &,     // STUB
                                 Registry &) {}
void handle_v1_completions(const httplib::Request &, httplib::Response &, Config &, Manager &, Registry &) {} // STUB


} // namespace llmash

// ---------------------------------------------------------------- harness

namespace {

int failures = 0;

void check(bool ok, const std::string & what) {
    std::printf("%s  %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) {
        failures++;
    }
}

struct Live {
    httplib::Server srv;
    std::thread     th;
    int             port = 0;

    void start(Config & cfg, Manager & mgr, Registry & reg, bool as_public) {
        port             = srv.bind_to_any_port("127.0.0.1");
        cfg.public_port  = as_public ? port : 0;
        register_routes(srv, cfg, mgr, reg);
        th = std::thread([this] { srv.listen_after_bind(); });
        srv.wait_until_ready();
    }
    ~Live() {
        srv.stop();
        if (th.joinable()) {
            th.join();
        }
    }
};

httplib::Client client_for(int port) {
    httplib::Client c("127.0.0.1", port);
    c.set_read_timeout(5, 0);
    c.set_connection_timeout(5, 0);
    return c;
}

// Never throws on a body that is not an object, so a failed request reads
// as an empty document instead of taking the test process down.
json body_json(const std::string & s) {
    json j = json::parse(s, nullptr, false);
    return j.is_object() ? j : json::object();
}

std::string err_msg(const json & b) { return b.value("error", json::object()).value("message", ""); }
std::string err_code(const json & b) { return b.value("error", json::object()).value("code", ""); }

} // namespace

int main() {
    const fs::path dir = fs::temp_directory_path() / "llmash_api_test";
    fs::remove_all(dir);
    fs::create_directories(dir);

    Config cfg;
    cfg.root        = dir.string();
    cfg.models_root = (dir / "models").string();
    cfg.extra_roots = {"C:\\extra\\one"};
    cfg.keep_alive  = "15m";

    Registry reg(cfg);
    Manager  mgr(cfg, &reg);

    Live local;
    local.start(cfg, mgr, reg, false);
    auto cli = client_for(local.port);

    // ------------------------------------------------------------ root
    {
        auto res = cli.Get("/");
        check(res && res->status == 200, "GET / answers 200");
        check(res && res->body == "Ollama is running", "GET / is ollama's own banner");
        check(res && res->get_header_value("Content-Type") == "text/plain; charset=utf-8",
              "the banner is text/plain; charset=utf-8");
        check(res && res->get_header_value("Access-Control-Allow-Origin") == "*",
              "every answer carries permissive CORS");
    }
    {
        // Go's mux dispatches on path alone; the handlers never look at the method.
        auto res = cli.Post("/", std::string("{}"), "application/json");
        check(res && res->status == 200 && res->body == "Ollama is running", "POST / is the same route");
    }

    // --------------------------------------------------------- version
    {
        auto res = cli.Get("/api/version");
        const json b = res ? body_json(res->body) : json::object();
        check(res && res->status == 200, "GET /api/version answers 200");
        check(res && res->get_header_value("Content-Type") == "application/json",
              "/api/version is application/json");
        check(b.is_object() && b.value("version", "") == LLMASH_VERSION && b.value("build", "") == "llmash",
              "/api/version reports config.go's version and build");
    }

    // ------------------------------------------------------- not found
    {
        auto res = cli.Get("/nope");
        const json b = res ? body_json(res->body) : json::object();
        check(res && res->status == 404, "an unmounted path is 404");
        check(b.is_object() && b.value("error", "") == "not found", "and its body is the mux's error object");
        check(res && res->get_header_value("Access-Control-Allow-Origin") == "*",
              "a 404 still carries CORS");
    }
    {
        auto res = cli.Get("/api/versionx");
        check(res && res->status == 404, "a mounted path is matched exactly, not as a prefix");
    }

    // --------------------------------------------------------- OPTIONS
    {
        auto res = cli.Options("/api/tags");
        check(res && res->status == 200, "a CORS preflight is answered 200");
        check(res && res->get_header_value("Access-Control-Allow-Methods") == "*",
              "the preflight advertises every method");
        check(res && res->body.empty(), "and carries no body");
    }
    {
        auto res = cli.Options("/nope");
        check(res && res->status == 200, "a preflight is answered before routing, so any path works");
    }

    // ------------------------------------------------- model lookups
    {
        auto res = cli.Post("/api/show", std::string(R"({"model":"ghost"})"), "application/json");
        const json b = res ? body_json(res->body) : json::object();
        check(res && res->status == 404 && b.value("error", "") == "model not found",
              "/api/show 404s a model the registry does not have");
    }
    {
        // "name" is ollama's older spelling of the same field.
        auto res = cli.Post("/api/show", std::string(R"({"name":"ghost"})"), "application/json");
        check(res && res->status == 404, "/api/show reads the legacy `name` key too");
    }
    {
        auto res = cli.Post("/api/delete", std::string(R"({"model":"ghost"})"), "application/json");
        const json b = res ? body_json(res->body) : json::object();
        check(res && res->status == 404 && b.value("error", "") == "model not found",
              "/api/delete 404s before it touches the disk");
    }
    {
        auto res = cli.Get("/v1/models/ghost");
        const json b = res ? body_json(res->body) : json::object();
        check(res && res->status == 404, "/v1/models/<id> 404s an unknown model");
        check(err_msg(b) == "model 'ghost' not found" && err_code(b) == "model_not_found",
              "with openai's error shape and the id quoted into the message");
    }
    {
        // A model name holds ':' and '/', so the id is everything after the
        // prefix -- a ":id" path parameter would stop at the first slash.
        auto res = cli.Get("/v1/models/hf.co/owner/repo:Q4_K_M");
        const json b = res ? body_json(res->body) : json::object();
        check(res && res->status == 404 && err_msg(b) == "model 'hf.co/owner/repo:Q4_K_M' not found",
              "the whole tail of /v1/models/ is the model id");
    }

    // -------------------------------------------------------- keep_alive
    {
        auto res = cli.Post("/api/keep_alive", std::string(R"({"model":"ghost"})"), "application/json");
        const json b = res ? body_json(res->body) : json::object();
        check(res && res->status == 404 && b.value("error", "") == "model 'ghost' is not loaded",
              "/api/keep_alive 404s a model that is not running");
        check(stub::keep_default_seen == "15m", "and the configured keep_alive is the default it parses");
    }

    // ---------------------------------------------------------- loading
    {
        auto res = cli.Get("/api/loading");
        const json b = res ? body_json(res->body) : json::object();
        check(res && res->status == 200 && b.value("loading", json()) == json::array(),
              "/api/loading answers an empty array with nothing loading");
    }

    // ------------------------------------------------------------ paths
    {
        auto res = cli.Get("/api/paths");
        const json b = res ? body_json(res->body) : json::object();
        check(res && res->status == 200, "GET /api/paths answers 200");
        check(b.value("root", "") == cfg.root && b.value("models", "") == cfg.models_root,
              "and reports the install root and the model root");
        check(b.contains("loose_dir") && b["extra_roots"] == json::array({"C:\\extra\\one"}) &&
                  b["library"].is_array(),
              "with the loose folder, the extra roots and the library folders");
    }

    // ------------------------------------------------------ ndjson body
    {
        auto res = cli.Post("/api/create",
                            std::string(R"({"name":"my-model","from":"C:\\x\\a.gguf","quantize":"q4_k_m",)"
                                        R"("draft_quantize":"q4_0"})"),
                            "application/json");
        check(res && res->status == 200, "/api/create answers 200");
        check(res && res->get_header_value("Content-Type") == "application/x-ndjson",
              "and streams application/x-ndjson");
        check(res && res->body == "{\"status\":\"a\"}\n{\"status\":\"b\"}\n",
              "one compact JSON object per line, newline terminated");
        check(stub::create_name == "my-model" && stub::create_from == "C:\\x\\a.gguf" &&
                  stub::create_quant == "q4_k_m" && stub::create_draft == "q4_0",
              "every create field reaches the importer");
    }
    {
        stub::copy_called = false;
        auto res = cli.Post("/api/copy", std::string(R"({"source":"ghost","destination":"clone"})"),
                            "application/json");
        check(res && res->status == 200 && res->body == "{\"error\":\"ghost: model not found\"}\n",
              "/api/copy reports an unknown source inside the stream, not as an HTTP error");
        check(!stub::copy_called, "and never starts the copy");
    }

    // ------------------------------------------------- listing wiring
    // The bodies come from api_logic/table; only the mounting, status and
    // content type below are this module's to get right.
    {
        auto res = cli.Get("/api/tags");
        check(res && res->status == 200 && res->get_header_value("Content-Type") == "application/json" &&
                  stub::tags_calls > 0,
              "/api/tags is mounted and answers JSON");
        auto ps = cli.Get("/api/ps");
        check(ps && ps->status == 200 && stub::ps_calls > 0, "/api/ps is mounted and shapes from the manager");
        auto v1 = cli.Get("/v1/models");
        check(v1 && v1->status == 200 && body_json(v1->body).value("object", "") == "list",
              "/v1/models is mounted");
    }
    {
        auto res = cli.Get("/cli/list");
        check(res && res->status == 200 &&
                  res->get_header_value("Content-Type") == "text/plain; charset=utf-8",
              "/cli/list serves rendered text");
        check(res && res->body.rfind("NAME", 0) == 0 && res->body.find("stub_models") != std::string::npos,
              "and renders the rows of tags() as the list table");
        auto ps = cli.Get("/cli/ps");
        check(ps && ps->status == 200 && ps->body.rfind("NAME", 0) == 0 &&
                  ps->body.find("stub_live") != std::string::npos,
              "/cli/ps renders the rows of ps() as the ps table");
    }

    // ---------------------------------------------- the public-port gate
    {
        Config pub = cfg;
        Live   guarded;
        guarded.start(pub, mgr, reg, true);
        auto pcli = client_for(guarded.port);

        auto anon = pcli.Get("/api/version");
        const json b = anon ? body_json(anon->body) : json::object();
        check(anon && anon->status == 401, "the public port rejects a request with no key");
        check(b.value("error", "") == "unauthorized: missing or invalid API key", "with ollama's wording");
        check(stub::expected_seen == "sk-llmash-testkey", "the key checked against is the link key");

        auto bearer = pcli.Get("/api/version", httplib::Headers{{"Authorization", "Bearer sk-llmash-testkey"}});
        check(bearer && bearer->status == 200, "a correct bearer token gets through");
        check(stub::auth_seen == "Bearer sk-llmash-testkey",
              "and the raw Authorization header is what was checked");

        auto xkey = pcli.Get("/api/version", httplib::Headers{{"X-API-Key", "sk-llmash-testkey"}});
        check(xkey && xkey->status == 200 && stub::xkey_seen == "sk-llmash-testkey",
              "X-API-Key is passed through as well");

        auto wrong = pcli.Get("/api/version", httplib::Headers{{"Authorization", "Bearer nope"}});
        check(wrong && wrong->status == 401, "a wrong key is rejected");

        auto pre = pcli.Options("/api/chat");
        check(pre && pre->status == 200, "a preflight is not gated, the way browsers need it");

        auto miss = pcli.Get("/nope");
        check(miss && miss->status == 401, "an unmounted path on the public port is gated too, not 404");
    }
    {
        auto res = cli.Get("/api/version", httplib::Headers{{"Authorization", "Bearer nope"}});
        check(res && res->status == 200, "the local port takes no key at all");
    }

    // --------------------------------------------- reload on POST /api/paths
    // Last: this one replaces cfg from local.json, the way `llmash models`
    // applies a new folder to a running server.
    {
        auto res = cli.Post("/api/paths", std::string("{}"), "application/json");
        const json b = res ? body_json(res->body) : json::object();
        check(res && res->status == 200 && b.contains("root") && b.contains("loose_dir") &&
                  b.contains("models") && b.contains("extra_roots") && b.contains("library"),
              "POST /api/paths re-reads local.json and still answers the same shape");
    }

    fs::remove_all(dir);
    std::printf("\n%s\n", failures == 0 ? "all passed" : "FAILURES");
    return failures == 0 ? 0 : 1;
}
