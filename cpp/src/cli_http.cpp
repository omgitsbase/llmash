#include "cli_http.h"

#include <httplib.h>

#include <algorithm>
#include <chrono>
#include <thread>

namespace llmash {

namespace {

std::string resolve_host(const Config & cfg) {
    std::string h = env_str("OLLAMA_HOST");
    if (h.empty()) {
        h = "http://127.0.0.1:" + std::to_string(cfg.port);
    }
    if (h.rfind("http", 0) != 0) {
        h = "http://" + h;
    }
    while (!h.empty() && h.back() == '/') {
        h.pop_back();
    }
    return h;
}

std::string trim(const std::string & s) {
    const size_t b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) {
        return "";
    }
    const size_t e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

} // namespace

ApiClient::ApiClient(const Config & cfg) : host_(resolve_host(cfg)) {}

ApiResult ApiClient::call(const std::string & method, const std::string & path, const nlohmann::json * body,
                           int timeout_sec) {
    httplib::Client cli(host_);
    cli.set_connection_timeout(timeout_sec);
    cli.set_read_timeout(timeout_sec);
    cli.set_write_timeout(timeout_sec);
    cli.set_keep_alive(false);

    const std::string    payload = body != nullptr ? body->dump() : std::string();
    httplib::Result       res;
    if (method == "GET") {
        res = cli.Get(path);
    } else if (method == "POST") {
        res = body != nullptr ? cli.Post(path, payload, "application/json") : cli.Post(path);
    } else if (method == "DELETE") {
        res = body != nullptr ? cli.Delete(path, payload, "application/json") : cli.Delete(path);
    } else {
        ApiResult r;
        r.error = "unsupported method: " + method;
        return r;
    }

    ApiResult r;
    if (!res) {
        r.ok    = false;
        r.error = httplib::to_string(res.error());
        return r;
    }
    r.ok     = true;
    r.status = res->status;
    r.body   = res->body;
    return r;
}

nlohmann::json ApiClient::call_json(const std::string & method, const std::string & path, const nlohmann::json * body,
                                     int timeout_sec, ApiResult & result) {
    result = call(method, path, body, timeout_sec);
    if (!result.ok) {
        return nlohmann::json::object();
    }
    if (trim(result.body).empty()) {
        return nlohmann::json::object();
    }
    nlohmann::json parsed = nlohmann::json::parse(result.body, nullptr, false);
    if (parsed.is_discarded()) {
        result.ok    = false;
        result.error = "not JSON: " + trim(result.body);
        return nlohmann::json::object();
    }
    return parsed;
}

bool ApiClient::stream(const std::string & path, const nlohmann::json & body,
                        const std::function<bool(const nlohmann::json &)> & fn, std::string & err) {
    httplib::Client cli(host_);
    cli.set_connection_timeout(30);
    cli.set_read_timeout(3600);
    cli.set_write_timeout(60);
    cli.set_keep_alive(false);

    std::string buf;
    bool        stopped_by_caller = false;
    bool        parse_failed      = false;

    const auto receiver = [&](const char * data, size_t len) -> bool {
        buf.append(data, len);
        size_t nl;
        while ((nl = buf.find('\n')) != std::string::npos) {
            std::string line = trim(buf.substr(0, nl));
            buf.erase(0, nl + 1);
            if (line.empty()) {
                continue;
            }
            nlohmann::json ev = nlohmann::json::parse(line, nullptr, false);
            if (ev.is_discarded()) {
                parse_failed = true;
                return false;
            }
            if (!fn(ev)) {
                stopped_by_caller = true;
                return false;
            }
        }
        return true;
    };

    const httplib::Result res = cli.Post(path, httplib::Headers{}, body.dump(), "application/json", receiver);

    if (parse_failed) {
        err = "the server sent a response that isn't NDJSON";
        return false;
    }
    if (stopped_by_caller) {
        return true;
    }
    if (!res) {
        err = httplib::to_string(res.error());
        return false;
    }
    return true;
}

bool ApiClient::up() {
    for (const int wait : {4, 8}) {
        ApiResult r = call("GET", "/api/version", nullptr, wait);
        if (r.ok) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
    }
    return false;
}

} // namespace llmash
