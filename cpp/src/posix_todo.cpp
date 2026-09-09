// What the portable half calls but has not been ported yet: the downloader.
// Building it for real on macOS is the rest of the job; until then they stand in so the
// portable code compiles and its tests run on a mac.
#ifndef _WIN32

#include "pull.h"

#include <limits>
#include <stdexcept>

namespace llmash {

static std::string not_ported(const char * what) { return std::string(what) + " is not ported to macOS yet"; }

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

} // namespace llmash

#endif // !_WIN32
