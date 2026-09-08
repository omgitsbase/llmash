#include "cli_util.h"

#include <httplib.h>
#include <subprocess.h>

#include <windows.h>
#include <winhttp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <thread>

#pragma comment(lib, "winhttp.lib")

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace llmash {
namespace clidoc {

namespace {

std::string get_env(const char * name) {
    const char * v = std::getenv(name);
    return v ? v : "";
}

std::string trim(std::string s) {
    const auto notspace = [](unsigned char c) { return !std::isspace(c); };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), notspace));
    s.erase(std::find_if(s.rbegin(), s.rend(), notspace).base(), s.end());
    return s;
}

std::wstring widen(const std::string & s) {
    if (s.empty()) {
        return L"";
    }
    const int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), nullptr, 0);
    std::wstring w(n, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), w.data(), n);
    return w;
}

// A HANDLE-shaped resource wrapper for the handful of Win32 APIs used here
// that don't already come with one (WinHTTP's HINTERNET, in particular).
template <typename Closer> class WinHandle {
public:
    WinHandle() = default;
    WinHandle(HINTERNET h, Closer closer) : h_(h), closer_(closer) {}
    ~WinHandle() {
        if (h_ != nullptr) {
            closer_(h_);
        }
    }
    WinHandle(const WinHandle &)             = delete;
    WinHandle & operator=(const WinHandle &) = delete;
    WinHandle(WinHandle && o) noexcept : h_(o.h_), closer_(o.closer_) { o.h_ = nullptr; }

    operator HINTERNET() const { return h_; }
    bool ok() const { return h_ != nullptr; }

private:
    HINTERNET h_ = nullptr;
    Closer    closer_{};
};

WinHandle<decltype(&WinHttpCloseHandle)> wrap(HINTERNET h) {
    return WinHandle<decltype(&WinHttpCloseHandle)>(h, &WinHttpCloseHandle);
}

} // namespace

// ------------------------------------------------------------- formatting

std::string human_bytes(uint64_t b) {
    constexpr double kb = 1000.0, mb = 1000.0 * 1000, gb = 1000.0 * 1000 * 1000, tb = gb * 1000;
    const double      v = static_cast<double>(b);
    double            value;
    const char *      unit;
    if (v >= tb) {
        value = v / tb;
        unit  = "TB";
    } else if (v >= gb) {
        value = v / gb;
        unit  = "GB";
    } else if (v >= mb) {
        value = v / mb;
        unit  = "MB";
    } else if (v >= kb) {
        value = v / kb;
        unit  = "KB";
    } else {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%llu B", static_cast<unsigned long long>(b));
        return buf;
    }
    char buf[32];
    if (value >= 10 || value == std::trunc(value)) {
        std::snprintf(buf, sizeof(buf), "%d %s", static_cast<int>(value), unit);
    } else {
        std::snprintf(buf, sizeof(buf), "%.1f %s", value, unit);
    }
    return buf;
}

namespace {

std::string human_duration_seconds(double seconds) {
    char buf[64];
    if (seconds < 1) {
        return "Less than a second";
    }
    if (seconds < 1.5) {
        return "1 second";
    }
    if (seconds < 60) {
        std::snprintf(buf, sizeof(buf), "%d seconds", static_cast<int>(seconds));
        return buf;
    }
    const double minutes = seconds / 60.0;
    if (minutes < 1.5) {
        return "About a minute";
    }
    if (minutes < 60) {
        std::snprintf(buf, sizeof(buf), "%d minutes", static_cast<int>(minutes));
        return buf;
    }
    const int hours = static_cast<int>(std::round(seconds / 3600.0));
    if (hours == 1) {
        return "About an hour";
    }
    if (hours < 48) {
        std::snprintf(buf, sizeof(buf), "%d hours", hours);
    } else if (hours < 24 * 7 * 2) {
        std::snprintf(buf, sizeof(buf), "%d days", hours / 24);
    } else if (hours < 24 * 30 * 2) {
        std::snprintf(buf, sizeof(buf), "%d weeks", hours / 24 / 7);
    } else if (hours < 24 * 365 * 2) {
        std::snprintf(buf, sizeof(buf), "%d months", hours / 24 / 30);
    } else {
        std::snprintf(buf, sizeof(buf), "%d years", hours / 24 / 365);
    }
    return buf;
}

// RFC3339(-nano): "YYYY-MM-DDTHH:MM:SS[.frac](Z|+HH:MM|-HH:MM)". Returns -1
// on anything that doesn't parse; good enough for what a JSON API actually
// sends, not a general-purpose parser.
double parse_rfc3339(const std::string & s) {
    if (s.size() < 20) {
        return -1;
    }
    std::tm tm{};
    int      frac_start = 0;
    if (std::sscanf(s.c_str(), "%4d-%2d-%2dT%2d:%2d:%2d%n", &tm.tm_year, &tm.tm_mon, &tm.tm_mday, &tm.tm_hour,
                    &tm.tm_min, &tm.tm_sec, &frac_start) != 6) {
        return -1;
    }
    tm.tm_year -= 1900;
    tm.tm_mon -= 1;
    size_t pos = static_cast<size_t>(frac_start);
    if (pos < s.size() && s[pos] == '.') {
        pos++;
        while (pos < s.size() && std::isdigit(static_cast<unsigned char>(s[pos]))) {
            pos++;
        }
    }
    long offset_min = 0;
    if (pos < s.size() && s[pos] != 'Z') {
        int sign = s[pos] == '-' ? -1 : 1;
        int oh = 0, om = 0;
        if (std::sscanf(s.c_str() + pos, "%*c%2d:%2d", &oh, &om) == 2) {
            offset_min = sign * (oh * 60 + om);
        }
    }
    const time_t utc = _mkgmtime(&tm);
    if (utc == static_cast<time_t>(-1)) {
        return -1;
    }
    return static_cast<double>(utc) - offset_min * 60.0;
}

} // namespace

std::string human_time_iso(const std::string & iso, const std::string & zero_value) {
    const double t = parse_rfc3339(iso);
    if (t < 0) {
        return zero_value;
    }
    const double now   = static_cast<double>(std::time(nullptr));
    const double delta = now - t;
    if (delta / 3600.0 / 24.0 / 365.0 < -20) {
        return "Forever";
    }
    if (delta < 0) {
        return human_duration_seconds(-delta) + " from now";
    }
    return human_duration_seconds(delta) + " ago";
}

// -------------------------------------------------------- filesystem / PATH

bool file_exists(const std::string & path) {
    std::error_code ec;
    return fs::is_regular_file(path, ec) || (fs::exists(path, ec) && !fs::is_directory(path, ec) && !ec);
}

bool dir_exists(const std::string & path) {
    std::error_code ec;
    return fs::is_directory(path, ec);
}

std::string which(const std::string & exe) {
    std::vector<std::string> exts;
    if (fs::path(exe).has_extension()) {
        exts.push_back("");
    } else {
        std::string pathext = get_env("PATHEXT");
        if (pathext.empty()) {
            pathext = ".COM;.EXE;.BAT;.CMD";
        }
        size_t start = 0;
        while (start <= pathext.size()) {
            const size_t semi = pathext.find(';', start);
            std::string  e    = pathext.substr(start, semi == std::string::npos ? std::string::npos : semi - start);
            if (!e.empty()) {
                exts.push_back(e);
            }
            if (semi == std::string::npos) {
                break;
            }
            start = semi + 1;
        }
        exts.push_back("");
    }

    const fs::path given(exe);
    if (given.has_parent_path()) {
        for (const auto & e : exts) {
            const std::string cand = exe + e;
            if (file_exists(cand)) {
                std::error_code ec;
                const auto      abs = fs::absolute(cand, ec);
                return ec ? cand : abs.string();
            }
        }
        return "";
    }

    const std::string path_env = get_env("PATH");
    size_t            start    = 0;
    while (start <= path_env.size()) {
        const size_t semi = path_env.find(';', start);
        const std::string dir = path_env.substr(start, semi == std::string::npos ? std::string::npos : semi - start);
        if (!dir.empty()) {
            for (const auto & e : exts) {
                const fs::path cand = fs::path(dir) / (exe + e);
                if (file_exists(cand.string())) {
                    return cand.string();
                }
            }
        }
        if (semi == std::string::npos) {
            break;
        }
        start = semi + 1;
    }
    return "";
}

double free_disk_gb(const std::string & path) {
    ULARGE_INTEGER free{}, total{}, totalFree{};
    if (!GetDiskFreeSpaceExW(widen(path).c_str(), &free, &total, &totalFree)) {
        return 0;
    }
    return static_cast<double>(free.QuadPart) / static_cast<double>(1ull << 30);
}

double free_ram_gb() {
    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof(ms);
    if (!GlobalMemoryStatusEx(&ms)) {
        return 999;
    }
    return static_cast<double>(ms.ullAvailPhys) / static_cast<double>(1ull << 30);
}

// -------------------------------------------------------------- the server

std::string server_host() {
    std::string h = get_env("OLLAMA_HOST");
    if (h.empty()) {
        h = "http://127.0.0.1:11434";
    }
    if (h.rfind("http", 0) != 0) {
        h = "http://" + h;
    }
    while (!h.empty() && h.back() == '/') {
        h.pop_back();
    }
    return h;
}

bool server_up() {
    httplib::Client cli(server_host());
    cli.set_connection_timeout(2, 0);
    cli.set_read_timeout(2, 0);
    const auto res = cli.Get("/api/version");
    return res && res->status > 0;
}

bool call_json(const std::string & method, const std::string & path, const json * body, int timeout_s, json & out,
              std::string & err) {
    httplib::Client cli(server_host());
    cli.set_connection_timeout(timeout_s, 0);
    cli.set_read_timeout(timeout_s, 0);
    cli.set_write_timeout(timeout_s, 0);

    httplib::Result res;
    if (method == "GET") {
        res = cli.Get(path);
    } else if (method == "POST") {
        const std::string payload = body ? body->dump() : std::string("{}");
        res                       = cli.Post(path, payload, "application/json");
    } else {
        err = "unsupported method " + method;
        return false;
    }
    if (!res) {
        err = httplib::to_string(res.error());
        return false;
    }
    // callJSON never inspects the status code; only a body that fails to
    // parse as JSON is an error, and an empty body parses as {}.
    const std::string text = trim(res->body);
    if (text.empty()) {
        out = json::object();
        return true;
    }
    json parsed = json::parse(text, nullptr, false);
    if (parsed.is_discarded()) {
        err = "not JSON: " + text;
        return false;
    }
    out = std::move(parsed);
    return true;
}

// -------------------------------------------------------------- local.json

json read_local_json(const std::string & root) {
    const fs::path p = fs::path(root) / "local.json";
    std::ifstream  in(p, std::ios::binary);
    if (!in) {
        return json::object();
    }
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (text.size() >= 3 && static_cast<unsigned char>(text[0]) == 0xEF && static_cast<unsigned char>(text[1]) == 0xBB &&
        static_cast<unsigned char>(text[2]) == 0xBF) {
        text.erase(0, 3);
    }
    if (trim(text).empty()) {
        return json::object();
    }
    json j = json::parse(text, nullptr, false);
    return (j.is_discarded() || !j.is_object()) ? json::object() : j;
}

bool write_local_json(const std::string & root, const json & j, std::string & err) {
    std::ofstream out(fs::path(root) / "local.json", std::ios::binary | std::ios::trunc);
    if (!out) {
        err = "could not write " + (fs::path(root) / "local.json").string();
        return false;
    }
    out << j.dump(2) << '\n';
    return out.good();
}

// ------------------------------------------------------- a timed subprocess

std::pair<std::string, bool> run_with_timeout(const std::string & exe, const std::vector<std::string> & args,
                                              int timeout_ms) {
    const std::string resolved = file_exists(exe) ? exe : which(exe);
    if (resolved.empty()) {
        return {"", false};
    }

    std::vector<std::string> storage;
    storage.push_back(resolved);
    storage.insert(storage.end(), args.begin(), args.end());
    std::vector<const char *> argv;
    for (const auto & s : storage) {
        argv.push_back(s.c_str());
    }
    argv.push_back(nullptr);

    subprocess_s proc{};
    const int    options = subprocess_option_combined_stdout_stderr | subprocess_option_enable_async |
                        subprocess_option_enable_async_no_wait | subprocess_option_no_window |
                        subprocess_option_inherit_environment;
    if (subprocess_create(argv.data(), options, &proc) != 0) {
        return {"", false};
    }
    // RAII: whatever happens below, the handles this call opened are closed
    // before it returns.
    struct Guard {
        subprocess_s * p;
        ~Guard() { subprocess_destroy(p); }
    } guard{&proc};

    std::string  out;
    char         buf[4096];
    const auto   deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    bool         timed_out = false;
    for (;;) {
        const unsigned n = subprocess_read_stdout(&proc, buf, sizeof(buf));
        if (n > 0) {
            out.append(buf, n);
            continue;
        }
        if (!subprocess_alive(&proc)) {
            break;
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            timed_out = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }

    if (timed_out) {
        subprocess_terminate(&proc);
        subprocess_join(&proc, nullptr);
        return {"", false};
    }
    int code = 0;
    subprocess_join(&proc, &code);
    return {trim(out), code == 0};
}

// ------------------------------------------------------- GitHub's releases

std::string repo_slug() {
    const std::string s = get_env("LLMASH_REPO");
    return s.empty() ? "omgitsbase/llmash" : s;
}

std::string release_version(const std::string & tag) {
    std::string v = tag;
    if (!v.empty() && v.front() == 'v') {
        v.erase(0, 1);
    }
    if (!v.empty() && v.front() == '.') {
        v.erase(0, 1);
    }
    const size_t dash = v.find('-');
    if (dash != std::string::npos) {
        v = v.substr(0, dash);
    }
    return v;
}

bool latest_release(const std::string & slug, Release & out, std::string & err) {
    auto session = wrap(WinHttpOpen(L"llmash", WINHTTP_ACCESS_TYPE_DEFAULT_PROXY, WINHTTP_NO_PROXY_NAME,
                                    WINHTTP_NO_PROXY_BYPASS, 0));
    if (!session.ok()) {
        err = "WinHttpOpen failed";
        return false;
    }
    WinHttpSetTimeouts(session, 5000, 5000, 15000, 15000);

    auto connect = wrap(WinHttpConnect(session, L"api.github.com", INTERNET_DEFAULT_HTTPS_PORT, 0));
    if (!connect.ok()) {
        err = "could not reach api.github.com";
        return false;
    }

    const std::wstring path = widen("/repos/" + slug + "/releases/latest");
    auto request = wrap(WinHttpOpenRequest(connect, L"GET", path.c_str(), nullptr, WINHTTP_NO_REFERER,
                                           WINHTTP_DEFAULT_ACCEPT_TYPES, WINHTTP_FLAG_SECURE));
    if (!request.ok()) {
        err = "could not build the request";
        return false;
    }

    const std::wstring headers = L"Accept: application/vnd.github+json\r\nUser-Agent: llmash\r\n";
    if (!WinHttpSendRequest(request, headers.c_str(), static_cast<DWORD>(-1), WINHTTP_NO_REQUEST_DATA, 0, 0, 0) ||
        !WinHttpReceiveResponse(request, nullptr)) {
        err = "GitHub did not answer";
        return false;
    }

    DWORD status = 0, status_size = sizeof(status);
    WinHttpQueryHeaders(request, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER, WINHTTP_HEADER_NAME_BY_INDEX,
                       &status, &status_size, WINHTTP_NO_HEADER_INDEX);

    std::string body;
    for (;;) {
        DWORD avail = 0;
        if (!WinHttpQueryDataAvailable(request, &avail) || avail == 0) {
            break;
        }
        std::vector<char> buf(avail);
        DWORD             got = 0;
        if (!WinHttpReadData(request, buf.data(), avail, &got)) {
            break;
        }
        body.append(buf.data(), got);
        if (body.size() > (1u << 20)) {
            break;
        }
    }

    switch (status) {
        case 200:
            break;
        case 404:
            err = slug + " has no releases yet";
            return false;
        case 403:
            err = "GitHub is rate limiting this address; try again later";
            return false;
        default:
            err = "GitHub answered " + std::to_string(status);
            return false;
    }

    const json j = json::parse(body, nullptr, false);
    if (j.is_discarded()) {
        err = "GitHub's reply was not JSON";
        return false;
    }
    out.tag   = j.value("tag_name", std::string());
    out.draft = j.value("draft", false);
    return true;
}

std::string prog_name() {
    const std::string env_prog = get_env("LLMASH_PROG");
    if (!env_prog.empty()) {
        return env_prog;
    }
    wchar_t buf[MAX_PATH];
    const DWORD n = GetModuleFileNameW(nullptr, buf, MAX_PATH);
    if (n == 0 || n == MAX_PATH) {
        return "llmash";
    }
    std::string stem = fs::path(std::wstring(buf, n)).stem().string();
    std::transform(stem.begin(), stem.end(), stem.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return stem.empty() ? "llmash" : stem;
}

} // namespace clidoc
} // namespace llmash
