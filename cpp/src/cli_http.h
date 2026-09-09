#pragma once

// How the commands reach the local server: call, call_json and stream,
// plus a two-try health check, against OLLAMA_HOST or the configured port.

#include "config.h"

#include <nlohmann/json.hpp>

#include <functional>
#include <string>

namespace llmash {

struct ApiResult {
    bool        ok     = false; // request + (for call_json) JSON parse both succeeded
    int         status = 0;
    std::string body;
    std::string error; // transport error, or callJSON's "not JSON: ..." text
};

class ApiClient {
public:
    explicit ApiClient(const Config & cfg);

    ApiResult call(const std::string & method, const std::string & path, const nlohmann::json * body, int timeout_sec);

    // On a non-2xx/4xx-but-not-JSON body this still comes back with ok=false
    // and result.error set, the way callJSON's "not JSON: %s" does.
    nlohmann::json call_json(const std::string & method, const std::string & path, const nlohmann::json * body,
                              int timeout_sec, ApiResult & result);

    // One ndjson object per line; stops early (without error) the moment fn
    // returns false, same as stream()'s early return.
    bool stream(const std::string & path, const nlohmann::json & body,
                const std::function<bool(const nlohmann::json &)> & fn, std::string & err);

    bool up();

    const std::string & host() const { return host_; }

private:
    std::string host_;
};

} // namespace llmash
