// Stands in for what is still Windows-only: the downloader, the tray, and the
// shell integration. Each says so rather than pretending to work.
#ifndef _WIN32

#include "cli_win.h"
#include "pull.h"
#include "tray.h"

#include <cstdlib>
#include <filesystem>
#include <string>

namespace fs = std::filesystem;

namespace llmash {

static std::string not_ported(const char * what) { return std::string(what) + " is not available on Linux yet"; }

HttpResult http_request(const std::string &, const std::string &, const std::string &,
                        const std::vector<std::string> &, int) {
    HttpResult r;
    r.error = not_ported("downloading");
    return r;
}

void handle_pull(const httplib::Request &, httplib::Response & res, Config &, Registry &) {
    res.status = 501;
    res.set_content("{\"error\":\"pull is not available on Linux yet\"}", "application/json");
}

void handle_quants(const httplib::Request &, httplib::Response & res, Config &, Registry &) {
    res.status = 501;
    res.set_content("{\"error\":\"quants is not available on Linux yet\"}", "application/json");
}

void handle_resolve(const httplib::Request &, httplib::Response & res, Config &, Registry &) {
    res.status = 501;
    res.set_content("{\"error\":\"resolve is not available on Linux yet\"}", "application/json");
}

std::string quant_tag(const std::string &) { return ""; }

bool cmd_tray(const Config &, std::string & err) {
    err = not_ported("the tray");
    return false;
}

std::string which_exe(const std::string & exe) {
    const char * path = std::getenv("PATH");
    if (path == nullptr) {
        return "";
    }
    const std::string all(path);
    size_t            start = 0;
    while (start <= all.size()) {
        const size_t      at  = all.find(':', start);
        const std::string dir = all.substr(start, at == std::string::npos ? std::string::npos : at - start);
        if (!dir.empty()) {
            std::error_code ec;
            const fs::path  cand = fs::path(dir) / exe;
            if (fs::is_regular_file(cand, ec)) {
                return cand.string();
            }
        }
        if (at == std::string::npos) {
            break;
        }
        start = at + 1;
    }
    return "";
}

std::string ps_quote(const std::string & s) { return s; }
std::string shortcut_target(const std::string &) { return ""; }
std::string hidden_powershell(const std::string &, bool, int *) { return ""; }
bool        reg_delete_hkcu_key(const std::string &) { return false; }
bool        remove_from_user_path(const std::string &) { return false; }

} // namespace llmash

#endif
