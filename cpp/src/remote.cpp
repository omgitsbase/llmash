#include "cli_win.h"
#include "remote.h"

#include "log.h"

#include <httplib.h>
#include <nlohmann/json.hpp>
#include <subprocess.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <regex>
#include <sstream>
#include <thread>

#ifdef _WIN32
#include <windows.h>
#endif

namespace fs = std::filesystem;
using json   = nlohmann::json;

namespace llmash {

namespace {

double now_seconds() {
    return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
}

void sleep_ms(int ms) {
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}

std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

std::string trim(const std::string & s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) {
        return "";
    }
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

std::string trim_right_slash(std::string s) {
    while (!s.empty() && s.back() == '/') {
        s.pop_back();
    }
    return s;
}

double env_float(const char * name, double fallback) {
    const std::string v = env_str(name);
    if (v.empty()) {
        return fallback;
    }
    try {
        return std::stod(v);
    } catch (const std::exception &) {
        return fallback;
    }
}

std::string j_str(const json & j, const char * key) {
    const auto it = j.find(key);
    if (it == j.end() || !it->is_string()) {
        return "";
    }
    return it->get<std::string>();
}

Route parse_route(const json & j) {
    Route r;
    r.match     = j_str(j, "match");
    r.url       = j_str(j, "url");
    r.container = j_str(j, "container");
    r.exe       = j_str(j, "exe");
    r.path      = j_str(j, "path");
    const auto it = j.find("args");
    if (it != j.end() && it->is_array()) {
        for (const auto & a : *it) {
            r.args.push_back(a.is_string() ? a.get<std::string>() : a.dump());
        }
    }
    return r;
}

const std::map<std::string, double> & unit_table() {
    static const std::map<std::string, double> t = {
        {"B", 1.0}, {"KiB", 1024.0}, {"MiB", 1048576.0}, {"GiB", 1073741824.0}, {"TiB", 1099511627776.0},
        {"KB", 1000.0}, {"MB", 1000000.0}, {"GB", 1000000000.0}, {"TB", 1000000000000.0},
    };
    return t;
}

double unit_of(const std::string & u) {
    const auto & t  = unit_table();
    const auto   it = t.find(u);
    return it != t.end() ? it->second : 1.0;
}

const std::regex & load_line_re() {
    static const std::regex re(
        R"(load\s+(\w+)\s+([0-9.]+)%\s+([0-9.]+)\s*([KMGTP]?i?B)\s*/\s*([0-9.]+)\s*([KMGTP]?i?B)\s+([0-9.]+)\s*s)");
    return re;
}

const std::regex & done_line_re() {
    static const std::regex re(R"(model loaded in\s+([0-9.]+)\s*s)");
    return re;
}

const std::regex & port_re() {
    static const std::regex re(R"(:(\d+))");
    return re;
}

bool file_exists(const std::string & p) {
    std::error_code ec;
    return fs::exists(p, ec) && !fs::is_directory(p, ec);
}


#ifdef _WIN32
std::string wide_to_utf8(const wchar_t * s, int len) {
    if (len <= 0) {
        return "";
    }
    const int n = WideCharToMultiByte(CP_UTF8, 0, s, len, nullptr, 0, nullptr, nullptr);
    std::string out(n, '\0');
    WideCharToMultiByte(CP_UTF8, 0, s, len, out.data(), n, nullptr, nullptr);
    return out;
}

// The current process's environment block, one "NAME=value" entry per line.
std::vector<std::string> current_environment() {
    struct Block {
        LPWCH p;
        ~Block() {
            if (p) {
                FreeEnvironmentStringsW(p);
            }
        }
    } block{GetEnvironmentStringsW()};
    std::vector<std::string> out;
    if (!block.p) {
        return out;
    }
    for (const wchar_t * s = block.p; *s != L'\0';) {
        const size_t len = wcslen(s);
        if (s[0] != L'=') { // Windows carries per-drive cwd pseudo-vars named "=C:"
            out.push_back(wide_to_utf8(s, static_cast<int>(len)));
        }
        s += len + 1;
    }
    return out;
}
#endif

// RAII around a subprocess_s: acquires zero OS handles until spawn()
// succeeds, and always releases whatever it holds on destruction.
class Proc {
public:
    Proc() = default;
    ~Proc() { reset(); }
    Proc(const Proc &)             = delete;
    Proc & operator=(const Proc &) = delete;

    bool spawn(const std::vector<std::string> & argv, const std::vector<std::string> * env,
               const std::string & cwd, bool combined_output) {
        reset();
        std::vector<const char *> cargv;
        cargv.reserve(argv.size() + 1);
        for (const auto & a : argv) {
            cargv.push_back(a.c_str());
        }
        cargv.push_back(nullptr);

        int options = subprocess_option_no_window | subprocess_option_search_user_path;
        if (combined_output) {
            options |= subprocess_option_combined_stdout_stderr;
        }
        std::vector<const char *> cenv;
        const char * const * envp = nullptr;
        if (env != nullptr) {
            cenv.reserve(env->size() + 1);
            for (const auto & e : *env) {
                cenv.push_back(e.c_str());
            }
            cenv.push_back(nullptr);
            envp = cenv.data();
        } else {
            options |= subprocess_option_inherit_environment;
        }
        const int rc = subprocess_create_ex(cargv.data(), options, envp, cwd.empty() ? nullptr : cwd.c_str(), &proc_);
        live_ = rc == 0;
        return live_;
    }

    bool   alive() const { return live_ && subprocess_alive(const_cast<subprocess_s *>(&proc_)) != 0; }
    FILE * out() const { return live_ ? subprocess_stdout(const_cast<subprocess_s *>(&proc_)) : nullptr; }
    FILE * err() const { return live_ ? subprocess_stderr(const_cast<subprocess_s *>(&proc_)) : nullptr; }

    unsigned long pid() const {
#ifdef _WIN32
        return live_ ? GetProcessId(static_cast<HANDLE>(proc_.hProcess)) : 0;
#else
        return 0;
#endif
    }

    int join(int * code) { return live_ ? subprocess_join(&proc_, code) : -1; }
    void terminate() {
        if (live_) {
            subprocess_terminate(&proc_);
        }
    }

    // Closes our handles to the child without killing it: a still-running
    // child intentionally outlives us here, exactly as exec.Cmd does in the
    // Go original when nothing explicitly stops it.
    void reset() {
        if (live_) {
            subprocess_destroy(&proc_);
            live_ = false;
        }
    }

private:
    subprocess_s proc_{};
    bool         live_ = false;
};

// taskkill /PID <pid> /T /F, discarding output: killTree in the Go original.
void kill_tree(unsigned long pid) {
    Proc p;
    if (p.spawn({"taskkill", "/PID", std::to_string(pid), "/T", "/F"}, nullptr, "", true)) {
        char buf[256];
        while (p.out() != nullptr && std::fread(buf, 1, sizeof(buf), p.out()) > 0) {
        }
        int code = 0;
        p.join(&code);
    }
}

// Runs argv to completion and returns its combined stdout+stderr along with
// its exit code (1 if it could not even be started, matching docker()'s
// "else if err != nil { code = 1 }" branch).
int run_capture(const std::vector<std::string> & argv, std::string & out) {
    Proc p;
    if (!p.spawn(argv, nullptr, "", true)) {
        return 1;
    }
    char buf[4096];
    size_t n;
    while (p.out() != nullptr && (n = std::fread(buf, 1, sizeof(buf), p.out())) > 0) {
        out.append(buf, n);
    }
    int code = 0;
    p.join(&code);
    out = trim(out);
    return code;
}

// Stop-Process by executable path, for a backend left running by an earlier
// llmash that this one has no handle on: killByExe in the Go original.
void kill_by_exe(const std::string & exe) {
    std::error_code ec;
    fs::path        abs = fs::absolute(exe, ec);
    if (ec) {
        abs = exe;
    }
    const std::string script = "Get-CimInstance Win32_Process -Filter \"Name='" + abs.filename().string() +
                                "'\" | Where-Object { $_.ExecutablePath -eq '" + ps_quote(abs.string()) +
                                "' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }";
    std::string out;
    run_capture({"powershell", "-NoProfile", "-NonInteractive", "-WindowStyle", "Hidden", "-Command", script}, out);
}

} // namespace

// Owns one exe-route child: its process handle, and the thread draining its
// stderr for load-progress lines (and, once tracked progress ends, just to
// keep the pipe from filling and stalling a backend we intend to keep
// running).
struct RemoteRouter::ExeProc {
    Proc        proc;
    std::thread tail;

    ~ExeProc() {
        // A real exec.Cmd in the Go original is left running when nothing
        // calls stopExeRoute; here that would hang this destructor forever
        // waiting on tail to see EOF, so unlike the Go original we always
        // stop the child on teardown. See the port report for why.
        proc.terminate();
        if (tail.joinable()) {
            tail.join();
        }
    }
};

struct RemoteRouter::LoadState {
    double t0 = 0, bytes = 0, total = 0, esec = 0, at = 0, done_at = 0, expect = 0;
    std::unique_ptr<Proc> tail_proc; // "docker logs -f", container routes only
    std::thread           tail;

    ~LoadState() {
        if (tail_proc) {
            tail_proc->terminate();
        }
        if (tail.joinable()) {
            tail.join();
        }
    }
};

RemoteRouter::RemoteRouter(Config cfg, Registry * reg) : cfg_(std::move(cfg)), reg_(reg) {
    default_container_ = env_str("LLMASH_REMOTE_CONTAINER");
    remote_idle_        = env_float("LLMASH_REMOTE_IDLE", 900);
    remote_boot_wait_   = env_float("LLMASH_REMOTE_BOOT_WAIT", 45);
    remote_load_guess_  = env_float("LLMASH_REMOTE_LOAD_GUESS", 20);
    load_expect();
}

RemoteRouter::~RemoteRouter() = default;

std::string RemoteRouter::state_path() const {
    return (fs::path(cfg_.root) / "remote_load.json").string();
}

void RemoteRouter::load_expect() {
    std::ifstream in(state_path(), std::ios::binary);
    if (!in) {
        return;
    }
    const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    const json        j = json::parse(text, nullptr, false);
    if (j.is_discarded() || !j.is_object()) {
        return;
    }
    for (auto it = j.begin(); it != j.end(); ++it) {
        if (it->is_number()) {
            expect_[it.key()] = it->get<double>();
        }
    }
}

void RemoteRouter::save_expect() const {
    json j = json::object();
    for (const auto & kv : expect_) {
        j[kv.first] = kv.second;
    }
    std::ofstream out(state_path(), std::ios::binary | std::ios::trunc);
    if (out) {
        out << j.dump();
    }
}

void RemoteRouter::load_routes() {
    routes_.clear();
    std::string text;
    bool        found = false;
    for (const char * name : {"routes.json", "vllm_routes.json"}) {
        std::ifstream in(fs::path(cfg_.root) / name, std::ios::binary);
        if (in) {
            text.assign((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
            found = true;
            break;
        }
    }
    if (!found) {
        return;
    }
    json j;
    try {
        j = json::parse(text);
    } catch (const std::exception & e) {
        logf("bad routes.json (%s); using defaults", e.what());
        return;
    }
    if (j.is_array()) {
        for (const auto & e : j) {
            if (e.is_object()) {
                routes_.push_back(parse_route(e));
            }
        }
    } else if (j.is_object()) {
        const auto it = j.find("routes");
        if (it != j.end() && it->is_array()) {
            for (const auto & e : *it) {
                if (e.is_object()) {
                    routes_.push_back(parse_route(e));
                }
            }
        }
    } else {
        logf("bad routes.json (not an array or object); using defaults");
        return;
    }
    std::string names;
    for (const auto & r : routes_) {
        names += (names.empty() ? "" : " ") + r.match;
    }
    logf("fast routes: [%s]", names.c_str());
}

std::vector<std::string> RemoteRouter::route_match_names() const {
    std::vector<std::string> out;
    for (const auto & r : routes_) {
        out.push_back(r.match);
    }
    return out;
}

const Route * RemoteRouter::match(const std::string & name) const {
    const std::string low = lower(name);
    for (const auto & r : routes_) {
        if (!r.match.empty() && low.find(lower(r.match)) != std::string::npos) {
            return &r;
        }
    }
    return nullptr;
}

std::string RemoteRouter::url_of(const Route & r) const {
    return trim_right_slash(r.url);
}

std::string RemoteRouter::key_of(const Route & r) const {
    if (!r.exe.empty()) {
        return "exe " + url_of(r);
    }
    return r.container.empty() ? default_container_ : r.container;
}

void RemoteRouter::drop_up(const std::string & url) {
    std::lock_guard<std::mutex> lk(mu_);
    up_cache_.erase(url);
}

namespace {
constexpr double kRouteUpTTL        = 8.0;
constexpr double kRouteDownTTL      = 120.0;
constexpr long   kRouteProbeTimeoutUs = 350 * 1000; // 350ms: routeProbeTimeout in the Go original
constexpr double kRunningTTL        = 5.0;
} // namespace

bool RemoteRouter::is_up(const Route & r) {
    const std::string url = url_of(r);
    const double       now = now_seconds();
    {
        std::lock_guard<std::mutex> lk(mu_);
        const auto it = up_cache_.find(url);
        if (it != up_cache_.end()) {
            const double ttl = it->second.second ? kRouteUpTTL : kRouteDownTTL;
            if (now - it->second.first < ttl) {
                return it->second.second;
            }
        }
    }
    bool good = false;
    if (!url.empty()) {
        httplib::Client cli(url);
        cli.set_connection_timeout(0, kRouteProbeTimeoutUs);
        cli.set_read_timeout(0, kRouteProbeTimeoutUs);
        auto res = cli.Get("/models");
        good     = res && res->status == 200;
    }
    std::lock_guard<std::mutex> lk(mu_);
    up_cache_[url] = {now, good};
    return good;
}

std::mutex & RemoteRouter::lock_for(const std::string & key) {
    std::lock_guard<std::mutex> lk(mu_);
    auto & l = locks_[key];
    if (!l) {
        l = std::make_unique<std::mutex>();
    }
    return *l;
}

double RemoteRouter::idle_for(const std::string & key) const {
    std::lock_guard<std::mutex> lk(mu_);
    const auto it = keep_.find(key);
    return it != keep_.end() ? it->second : remote_idle_;
}

void RemoteRouter::set_keep_alive(const std::string & key, double idle_seconds) {
    std::lock_guard<std::mutex> lk(mu_);
    keep_[key]      = idle_seconds;
    last_used_[key] = now_seconds();
}

double RemoteRouter::expiry(const std::string & key) const {
    const double idle = idle_for(key);
    double       last;
    {
        std::lock_guard<std::mutex> lk(mu_);
        const auto it = last_used_.find(key);
        last          = it != last_used_.end() ? it->second : 0;
    }
    if (last == 0) {
        last = now_seconds();
    }
    if (std::isinf(idle) || idle <= 0) {
        return std::numeric_limits<double>::infinity();
    }
    return last + idle;
}

bool RemoteRouter::docker_running(const std::string & container) {
    const double now = now_seconds();
    {
        std::lock_guard<std::mutex> lk(mu_);
        const auto it = running_cache_.find(container);
        if (it != running_cache_.end() && now - it->second.first < kRunningTTL) {
            return it->second.second;
        }
    }
    std::string out;
    const int   rc   = run_capture({"docker", "inspect", "-f", "{{.State.Running}}", container}, out);
    const bool  good = rc == 0 && out == "true";
    std::lock_guard<std::mutex> lk(mu_);
    running_cache_[container] = {now, good};
    return good;
}

bool RemoteRouter::port_published(const std::string & container, const Route & r) {
    const std::string url   = url_of(r);
    const auto         slash = url.find("//");
    const std::string  after = slash != std::string::npos ? url.substr(slash + 2) : url;
    std::smatch        m;
    if (!std::regex_search(after, m, port_re())) {
        return true;
    }
    const std::string want = ":" + m[1].str();
    for (int i = 0; i < 6; i++) {
        std::string out;
        const int   rc = run_capture({"docker", "port", container}, out);
        if (rc == 0 && out.find(want) != std::string::npos) {
            return true;
        }
        sleep_ms(500);
    }
    return false;
}

bool RemoteRouter::ensure_up(const Route & r) {
    const std::string key = key_of(r);
    if (key.empty()) {
        drop_up(url_of(r));
        return is_up(r);
    }
    std::lock_guard<std::mutex> guard(lock_for(key));
    {
        std::lock_guard<std::mutex> lk(mu_);
        last_used_[key] = now_seconds();
    }
    drop_up(url_of(r));
    if (is_up(r)) {
        return true;
    }
    if (!r.exe.empty()) {
        return start_exe(r, key);
    }

    const std::string container = r.container.empty() ? default_container_ : r.container;
    {
        std::lock_guard<std::mutex> lk(mu_);
        running_cache_.erase(container);
    }
    std::string out;
    int         rc = run_capture({"docker", "start", container}, out);
    if (rc != 0) {
        if (out.size() > 200) {
            out.resize(200);
        }
        logf("%s start failed: %s", container.c_str(), out.c_str());
        return false;
    }
    if (!port_published(container, r)) {
        logf("%s started without its port; restarting it", container.c_str());
        run_capture({"docker", "stop", container}, out);
        sleep_ms(2000);
        rc = run_capture({"docker", "start", container}, out);
        if (rc != 0 || !port_published(container, r)) {
            logf("%s will not publish its port; falling back", container.c_str());
            run_capture({"docker", "stop", container}, out);
            return false;
        }
    }
    logf("%s starting", container.c_str());
    load_begin_docker(container);
    const double t0 = now_seconds();
    while (now_seconds() - t0 < remote_boot_wait_) {
        drop_up(url_of(r));
        if (is_up(r)) {
            logf("%s ready in %.1fs", container.c_str(), now_seconds() - t0);
            load_end(container, true);
            return true;
        }
        sleep_ms(1000);
    }
    load_end(container, false);
    logf("%s did not become ready; falling back to llama.cpp", container.c_str());
    return false;
}

bool RemoteRouter::start_exe(const Route & r, const std::string & key) {
    if (!file_exists(r.exe)) {
        logf("%s: %s is not there", key.c_str(), r.exe.c_str());
        return false;
    }
    std::vector<std::string> argv{r.exe};
    for (const auto & a : r.args) {
        argv.push_back(a);
    }

    std::vector<std::string>        env_storage;
    const std::vector<std::string> * envp = nullptr;
#ifdef _WIN32
    if (!r.path.empty()) {
        env_storage = current_environment();
        env_storage.push_back("PATH=" + r.path + ";" + env_str("PATH"));
        envp = &env_storage;
    }
#endif
    auto ep = std::make_unique<ExeProc>();
    if (!ep->proc.spawn(argv, envp, fs::path(r.exe).parent_path().string(), false)) {
        logf("%s start failed", key.c_str());
        return false;
    }
    ExeProc * raw = ep.get();
    {
        std::lock_guard<std::mutex> lk(mu_);
        exe_procs_[key] = std::move(ep);
    }
    logf("%s starting", key.c_str());
    load_begin_exe(key, *raw);

    const double t0 = now_seconds();
    while (now_seconds() - t0 < remote_boot_wait_) {
        drop_up(url_of(r));
        if (is_up(r)) {
            logf("%s ready in %.1fs", key.c_str(), now_seconds() - t0);
            load_end(key, true);
            return true;
        }
        if (!raw->proc.alive()) {
            logf("%s exited before it served", key.c_str());
            load_end(key, false);
            return false;
        }
        sleep_ms(500);
    }
    load_end(key, false);
    logf("%s did not become ready; falling back to llama.cpp", key.c_str());
    stop_exe(r, key);
    return false;
}

void RemoteRouter::stop_exe(const Route & r, const std::string & key) {
    std::unique_ptr<ExeProc> proc;
    {
        std::lock_guard<std::mutex> lk(mu_);
        const auto it = exe_procs_.find(key);
        if (it != exe_procs_.end()) {
            proc = std::move(it->second);
            exe_procs_.erase(it);
        }
    }
    if (proc && proc->proc.alive()) {
        proc.reset(); // ~ExeProc terminates the child and joins its tail thread
        drop_up(url_of(r));
        return;
    }
    if (!r.exe.empty() && is_up(r)) {
        kill_by_exe(r.exe);
    }
    drop_up(url_of(r));
}

bool RemoteRouter::exe_running(const Route & r, const std::string & key) {
    {
        std::lock_guard<std::mutex> lk(mu_);
        const auto it = exe_procs_.find(key);
        if (it != exe_procs_.end() && it->second->proc.alive()) {
            return true;
        }
    }
    return is_up(r);
}

bool RemoteRouter::stop_for(const std::string & model_name) {
    const Route * r = match(model_name);
    if (r == nullptr) {
        return false;
    }
    const std::string key = key_of(*r);
    if (key.empty()) {
        return false;
    }
    if (!r->exe.empty()) {
        if (!exe_running(*r, key)) {
            return false;
        }
        logf("stopping %s on request", key.c_str());
        stop_exe(*r, key);
    } else {
        const std::string container = r->container.empty() ? default_container_ : r->container;
        if (!docker_running(container)) {
            return false;
        }
        logf("stopping %s on request", container.c_str());
        std::string out;
        run_capture({"docker", "stop", container}, out);
        std::lock_guard<std::mutex> lk(mu_);
        running_cache_.erase(container);
    }
    {
        std::lock_guard<std::mutex> lk(mu_);
        last_used_.erase(key);
        keep_.erase(key);
    }
    drop_up(url_of(*r));
    return true;
}

void RemoteRouter::reap_orphans() {
    for (const auto & r : routes_) {
        if (r.exe.empty()) {
            continue;
        }
        if (is_up(r)) {
            logf("stopping an orphaned %s from an earlier run", fs::path(r.exe).filename().string().c_str());
            kill_by_exe(r.exe);
            drop_up(url_of(r));
        }
    }
}

void RemoteRouter::reap_idle() {
    std::map<std::string, double> snap;
    {
        std::lock_guard<std::mutex> lk(mu_);
        snap = last_used_;
    }
    for (const auto & kv : snap) {
        const std::string & key  = kv.first;
        const double         last = kv.second;
        const double         idle = idle_for(key);
        if (last == 0 || idle <= 0 || std::isinf(idle) || now_seconds() - last < idle) {
            continue;
        }
        const Route * route = nullptr;
        for (const auto & r : routes_) {
            if (key_of(r) == key) {
                route = &r;
                break;
            }
        }
        if (route != nullptr && !route->exe.empty()) {
            if (exe_running(*route, key)) {
                logf("%s idle past %.0fs; stopping", key.c_str(), idle);
                stop_exe(*route, key);
                drop_up(url_of(*route));
            }
        } else if (docker_running(key)) {
            logf("%s idle past %.0fs; stopping", key.c_str(), idle);
            std::string out;
            run_capture({"docker", "stop", key}, out);
            std::lock_guard<std::mutex> lk(mu_);
            running_cache_.erase(key);
            for (const auto & r : routes_) {
                if ((r.container.empty() ? default_container_ : r.container) == key) {
                    up_cache_.erase(url_of(r));
                }
            }
        }
        std::lock_guard<std::mutex> lk(mu_);
        last_used_[key] = 0;
        keep_.erase(key);
    }
}

void RemoteRouter::reaper_loop(const std::atomic<bool> & stop) {
    while (!stop.load()) {
        for (int i = 0; i < 300 && !stop.load(); i++) { // 30s in 100ms slices, so stop is responsive
            sleep_ms(100);
        }
        if (!stop.load()) {
            reap_idle();
        }
    }
}

std::optional<RouteLoad> RemoteRouter::load_progress(const std::string & key) const {
    std::lock_guard<std::mutex> lk(mu_);
    const auto it = loads_.find(key);
    if (it == loads_.end() || it->second->done_at != 0) {
        return std::nullopt;
    }
    const LoadState & st  = *it->second;
    const double       now = now_seconds();
    double             pct;
    if (st.bytes > 0 && st.total > 0) {
        const double rate = st.bytes / std::max(st.esec, 1e-3);
        const double est  = st.bytes + rate * std::max(now - st.at, 0.0);
        pct               = 100.0 * std::min(est, st.total * 0.995) / st.total;
    } else {
        double exp = st.expect != 0 ? st.expect : remote_load_guess_;
        pct        = std::min(95.0, 100.0 * (now - st.t0) / std::max(exp, 1.0));
    }
    pct = std::floor(std::max(0.0, std::min(99.9, pct)) * 10.0) / 10.0;
    RouteLoad out;
    out.pct     = pct;
    out.total   = static_cast<uint64_t>(st.total);
    out.elapsed = std::floor((now - st.t0) * 10.0) / 10.0;
    return out;
}

std::vector<std::string> RemoteRouter::loading_keys() const {
    std::lock_guard<std::mutex> lk(mu_);
    std::vector<std::string> out;
    for (const auto & kv : loads_) {
        out.push_back(kv.first);
    }
    return out;
}

std::vector<std::string> RemoteRouter::models_on(const std::string & key) const {
    {
        std::lock_guard<std::mutex> lk(mu_);
        const auto it = names_cache_.find(key);
        if (it != names_cache_.end()) {
            return it->second;
        }
    }
    std::vector<std::string> out;
    if (reg_ != nullptr) {
        for (const auto & m : reg_->all()) {
            const Route * r = match(m.name);
            if (r != nullptr && key_of(*r) == key) {
                out.push_back(m.name);
            }
        }
    }
    std::lock_guard<std::mutex> lk(mu_);
    names_cache_[key] = out;
    return out;
}

namespace {

// Applies one tailed line to a load state's fields under RemoteRouter::mu_
// (called with the lock held). Returns false once loading has finished.
bool apply_load_line(double & bytes, double & total, double & esec, double & at, double & done_at,
                      const std::string & line, std::map<std::string, double> & expect, const std::string & key) {
    std::smatch m;
    if (std::regex_search(line, m, load_line_re())) {
        bytes = std::stod(m[3].str()) * unit_of(m[4].str());
        total = std::stod(m[5].str()) * unit_of(m[6].str());
        esec  = std::stod(m[7].str());
        at    = now_seconds();
        return true;
    }
    if (std::regex_search(line, m, done_line_re())) {
        expect[key] = std::stod(m[1].str());
        done_at     = now_seconds();
        return false;
    }
    return true;
}

} // namespace

void RemoteRouter::load_begin_docker(const std::string & container) {
    auto st = std::make_unique<LoadState>();
    st->t0        = now_seconds();
    st->tail_proc = std::make_unique<Proc>();

    // Move out (never destroy-while-locked: ~LoadState joins its tail thread,
    // and that thread also needs mu_, so destroying one under mu_ deadlocks).
    std::unique_ptr<LoadState> old;
    {
        std::lock_guard<std::mutex> lk(mu_);
        const auto it = expect_.find(container);
        st->expect    = it != expect_.end() ? it->second : 0;
        const auto lit = loads_.find(container);
        if (lit != loads_.end()) {
            old = std::move(lit->second);
            loads_.erase(lit);
        }
    }
    old.reset(); // joins the superseded tail now, with mu_ free

    Proc * proc = st->tail_proc.get();
    if (!proc->spawn({"docker", "logs", "-f", "--tail", "0", container}, nullptr, "", true)) {
        std::lock_guard<std::mutex> lk(mu_);
        loads_[container] = std::move(st);
        return;
    }
    LoadState * raw = st.get();
    {
        std::lock_guard<std::mutex> lk(mu_);
        loads_[container] = std::move(st);
    }
    raw->tail = std::thread([this, raw, proc, container]() {
        FILE * pipe = proc->out();
        if (pipe == nullptr) {
            return;
        }
        char        buf[4096];
        std::string pending;
        size_t      n;
        while ((n = std::fread(buf, 1, sizeof(buf), pipe)) > 0) {
            pending.append(buf, n);
            size_t pos;
            while ((pos = pending.find('\n')) != std::string::npos) {
                const std::string line = pending.substr(0, pos);
                pending.erase(0, pos + 1);
                std::lock_guard<std::mutex> lk(mu_);
                const auto it = loads_.find(container);
                if (it == loads_.end() || it->second.get() != raw) {
                    return;
                }
                if (!apply_load_line(raw->bytes, raw->total, raw->esec, raw->at, raw->done_at, line, expect_, container)) {
                    save_expect();
                }
            }
        }
    });
}

void RemoteRouter::load_begin_exe(const std::string & key, ExeProc & proc) {
    auto st = std::make_unique<LoadState>();
    st->t0 = now_seconds();
    {
        std::lock_guard<std::mutex> lk(mu_);
        const auto it = expect_.find(key);
        st->expect    = it != expect_.end() ? it->second : 0;
    }
    LoadState * raw = st.get();
    {
        std::lock_guard<std::mutex> lk(mu_);
        loads_[key] = std::move(st);
    }
    FILE * pipe = proc.proc.out();
    proc.tail   = std::thread([this, raw, pipe, key]() {
        char        buf[4096];
        std::string pending;
        size_t      n;
        bool        tracking = true;
        while (pipe != nullptr && (n = std::fread(buf, 1, sizeof(buf), pipe)) > 0) {
            pending.append(buf, n);
            size_t pos;
            while ((pos = pending.find('\n')) != std::string::npos) {
                const std::string line = pending.substr(0, pos);
                pending.erase(0, pos + 1);
                if (!tracking) {
                    continue; // still here only to keep the pipe from filling up
                }
                std::lock_guard<std::mutex> lk(mu_);
                const auto it = loads_.find(key);
                if (it == loads_.end() || it->second.get() != raw) {
                    tracking = false;
                    continue;
                }
                if (!apply_load_line(raw->bytes, raw->total, raw->esec, raw->at, raw->done_at, line, expect_, key)) {
                    save_expect();
                }
            }
        }
    });
}

void RemoteRouter::load_end(const std::string & key, bool ok) {
    std::unique_ptr<LoadState> st;
    {
        std::lock_guard<std::mutex> lk(mu_);
        const auto it = loads_.find(key);
        if (it != loads_.end()) {
            st = std::move(it->second);
            loads_.erase(it);
        }
    }
    if (ok && st && st->done_at == 0) {
        std::lock_guard<std::mutex> lk(mu_);
        if (expect_.find(key) == expect_.end()) {
            expect_[key] = std::floor((now_seconds() - st->t0) * 100.0) / 100.0;
            save_expect();
        }
    }
    // st's destructor (outside the lock) terminates a docker-logs tail and
    // joins it; an exe tail keeps draining and is joined by ~ExeProc instead.
}

} // namespace llmash
