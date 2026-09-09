#include "manager.h"

#include "log.h"

#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0A00
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <psapi.h>

#include <subprocess.h>
#include <httplib.h>

#ifdef _MSC_VER
#pragma comment(lib, "ws2_32.lib")
#pragma comment(lib, "psapi.lib")
#endif

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <mutex>
#include <set>
#include <sstream>
#include <thread>

namespace fs = std::filesystem;

namespace llmash {

namespace {

// how long a load may make no progress at all before it is called stuck
constexpr double kLoadStallTimeout = 600.0;
constexpr double kCtxTrainNative   = 262144.0;

double now_f() {
    return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
}

std::string trim(std::string s) {
    const auto notspace = [](unsigned char c) { return !std::isspace(c); };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), notspace));
    s.erase(std::find_if(s.rbegin(), s.rend(), notspace).base(), s.end());
    return s;
}

std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

bool file_exists(const std::string & p) {
    std::error_code ec;
    return fs::is_regular_file(p, ec);
}


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

// Ensures WSAStartup has run before any socket call in this translation unit.
// httplib.h does not call it itself on Windows.
struct WinsockInit {
    WinsockInit() {
        WSADATA d;
        WSAStartup(MAKEWORD(2, 2), &d);
    }
    ~WinsockInit() { WSACleanup(); }
} g_winsock_init;

// Releases a subprocess_s (pipes, handles) exactly once, however the
// function that owns it returns.
struct ScopedSubprocess {
    subprocess_s proc{};
    bool         valid = false;
    ~ScopedSubprocess() {
        if (valid) {
            subprocess_destroy(&proc);
        }
    }
};

// Runs a short-lived helper to completion (or timeout) and returns its
// combined stdout+stderr.
bool run_capture(const std::vector<std::string> & argv, int timeout_ms, std::string & out) {
    std::vector<const char *> cargv;
    cargv.reserve(argv.size() + 1);
    for (const auto & s : argv) {
        cargv.push_back(s.c_str());
    }
    cargv.push_back(nullptr);

    ScopedSubprocess sp;
    const int        options = subprocess_option_no_window | subprocess_option_inherit_environment |
                        subprocess_option_combined_stdout_stderr | subprocess_option_enable_async |
                        subprocess_option_enable_async_no_wait;
    if (subprocess_create_ex(cargv.data(), options, nullptr, nullptr, &sp.proc) != 0) {
        return false;
    }
    sp.valid = true;

    const auto start = std::chrono::steady_clock::now();
    char       buf[4096];
    for (;;) {
        const unsigned n = subprocess_read_stdout(&sp.proc, buf, sizeof(buf));
        if (n > 0) {
            out.append(buf, n);
            continue;
        }
        if (!subprocess_alive(&sp.proc)) {
            break;
        }
        if (std::chrono::steady_clock::now() - start > std::chrono::milliseconds(timeout_ms)) {
            subprocess_terminate(&sp.proc);
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    int code = 0;
    subprocess_join(&sp.proc, &code);
    return true;
}

std::mutex g_vram_mu;
double     g_vram_at    = -1e9;
double     g_vram_free  = 0;
bool       g_vram_known = false;

// Free VRAM as the driver reports it, cached for two seconds; unknown reads
// as no GPU rather than as an 80 GB card, since assuming the budget on a
// machine with none of it sends every context calculation and offload
// decision the wrong way.
std::pair<double, bool> free_vram_gb() {
    std::lock_guard<std::mutex> lock(g_vram_mu);
    if (now_f() - g_vram_at < 2.0) {
        return {g_vram_free, g_vram_known};
    }
    double free = 0;
    bool   known = false;
    if (const double v = env_float("LLMASH_VRAM_GB", 0); v > 0) {
        free = v, known = true;
    } else {
        std::string out;
        if (run_capture({"nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"}, 8000, out)) {
            std::istringstream iss(out);
            std::string        line;
            if (std::getline(iss, line)) {
                line = trim(line);
                try {
                    free  = std::stod(line) / 1024.0;
                    known = true;
                } catch (const std::exception &) {
                }
            }
        }
    }
    g_vram_at = now_f(), g_vram_free = free, g_vram_known = known;
    return {free, known};
}

double free_ram_gb() {
    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof(ms);
    if (!GlobalMemoryStatusEx(&ms)) {
        return 999.0;
    }
    return static_cast<double>(ms.ullAvailPhys) / static_cast<double>(1ull << 30);
}

std::mutex               g_dev_mu;
double                   g_dev_at = -1e9;
std::vector<std::string> g_dev_cache;

// The devices llama.cpp itself reports. nvidia-smi answers for one vendor
// only, and an AMD/Intel/integrated card still has somewhere to offload to.
std::vector<std::string> offload_devices(const std::string & llama_bin) {
    std::lock_guard<std::mutex> lock(g_dev_mu);
    if (now_f() - g_dev_at < 60.0) {
        return g_dev_cache;
    }
    g_dev_cache.clear();
    g_dev_at = now_f();
    if (llama_bin.empty()) {
        return g_dev_cache;
    }
    std::string out;
    if (!run_capture({llama_bin, "--list-devices"}, 20000, out)) {
        return g_dev_cache;
    }
    std::istringstream iss(out);
    std::string        line;
    while (std::getline(iss, line)) {
        line               = trim(line);
        const auto colon   = line.find(':');
        if (colon == std::string::npos) {
            continue;
        }
        std::string name = trim(line.substr(0, colon));
        if (name.empty() || name.find(' ') != std::string::npos) {
            continue;
        }
        const std::string up = lower(name);
        if (up == "cpu" || up.rfind("cpu", 0) == 0) {
            continue;
        }
        g_dev_cache.push_back(name);
    }
    return g_dev_cache;
}

// can_offload() is a free function per manager.h with no Config to read, so
// it locates llama-server the same minimal way config.cpp's own resolver
// does (LLAMA_BIN, then the install's runtime folder) for its own use; a
// caller that already has a Config (Instance::args) passes its real path in.
std::string guess_llama_bin() {
    std::string v = env_str("LLAMA_BIN");
    if (!v.empty()) {
        return v;
    }
    const std::string exe = exe_dir();
    for (const auto & rel : {"runtime\\llama-server.exe", "llama.cpp\\llama-server.exe"}) {
        std::string cand = (fs::path(exe) / rel).string();
        if (file_exists(cand)) {
            return cand;
        }
    }
    return "";
}

bool can_offload_with(const std::string & llama_bin) {
    if (free_vram_gb().second) {
        return true;
    }
    return !offload_devices(llama_bin).empty();
}

std::mutex           g_pin_mu;
bool                 g_pin_loaded = false;
std::set<std::string> g_pinned;

// Models never evicted for VRAM, from LLMASH_PIN. local.json's own "pin"
// list is not reachable here: Config carries no arbitrary local.json keys
// (see the report for this job).
bool is_pinned(const std::string & name) {
    std::lock_guard<std::mutex> lock(g_pin_mu);
    if (!g_pin_loaded) {
        g_pin_loaded = true;
        std::stringstream ss(env_str("LLMASH_PIN"));
        std::string       tok;
        while (std::getline(ss, tok, ',')) {
            tok = trim(tok);
            if (!tok.empty()) {
                g_pinned.insert(tok);
            }
        }
    }
    return g_pinned.count(name) != 0;
}

std::string sanitize_name(const std::string & s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        out.push_back((std::isalnum(static_cast<unsigned char>(c)) || c == '.' || c == '_' || c == '-') ? c : '_');
    }
    return out;
}

bool own_runtime(const Config & cfg) {
    if (cfg.llama_bin.empty()) {
        return false;
    }
    std::error_code ec;
    return fs::is_regular_file(fs::path(cfg.llama_bin).parent_path() / "RUNTIME.txt", ec);
}

std::string runtime_dir(const Config & cfg) {
    if (cfg.llama_bin.empty()) {
        return "the runtime folder";
    }
    return fs::path(cfg.llama_bin).parent_path().string();
}

std::string explain_load_failure(const std::string & raw, const Config & cfg) {
    const std::string low = lower(raw);
    const auto        has = [&](const char * s) { return low.find(s) != std::string::npos; };

    if (has("unknown model architecture") && !own_runtime(cfg)) {
        return "This llama.cpp build does not know this model's architecture, which usually means the runtime is "
               "older than the model. Run `llmash update -Runtime cuda` (or vulkan, or cpu) to replace the runtime "
               "in " +
               runtime_dir(cfg) + ".";
    }
    if (has("wrong number of tensors") || has("check_tensor_dims") || has("unknown model architecture")) {
        return "llama.cpp cannot load this Ollama-packaged build: it does not carry the tensors llama.cpp expects "
               "for this architecture, which happens when a model is packaged for Ollama's own fork. Run `llmash "
               "pull` for this model again to take the Hugging Face build instead.";
    }
    if (has("unable to allocate") || has("out of memory") || has("cudamalloc")) {
        return "Not enough VRAM to load this model at the requested context size. Lower the context window, or "
               "unload whatever else is resident.";
    }
    if (has("failed to fit") || has("common_fit_params")) {
        return "llama.cpp couldn't fit this model in the available memory. Lower the context window and try again.";
    }
    if (has("no such file") || has("failed to open")) {
        return "The model file is missing from disk.";
    }
    std::string        trimmed = trim(raw);
    std::vector<std::string> lines;
    std::istringstream        iss(trimmed);
    std::string               l;
    while (std::getline(iss, l)) {
        lines.push_back(l);
    }
    for (auto it = lines.rbegin(); it != lines.rend(); ++it) {
        std::string line = trim(*it);
        if (!line.empty()) {
            if (line.size() > 300) {
                line = line.substr(0, 300);
            }
            return line;
        }
    }
    return "llama-server failed to start.";
}

// Owns the child's pipe for its whole life and writes what comes out of it
// to logfile, so tail_log has something to read; touches nothing on
// Instance, since this thread outlives any single call into it.
void drain_to_file(subprocess_s proc, std::string logfile) {
    ScopedSubprocess sp;
    sp.proc  = proc;
    sp.valid = true;
    std::ofstream out(logfile, std::ios::binary | std::ios::trunc);
    char          buf[8192];
    for (;;) {
        const unsigned n = subprocess_read_stdout(&sp.proc, buf, sizeof(buf));
        if (n > 0) {
            if (out) {
                out.write(buf, static_cast<std::streamsize>(n));
                out.flush();
            }
            continue;
        }
        if (!subprocess_alive(&sp.proc)) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
    }
    int code = 0;
    subprocess_join(&sp.proc, &code);
}

} // namespace

// ------------------------------------------------------------- free port

int free_port() {
    const SOCKET s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (s == INVALID_SOCKET) {
        return 0;
    }
    sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port        = 0;
    int port             = 0;
    if (bind(s, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) == 0) {
        sockaddr_in got{};
        int         len = sizeof(got);
        if (getsockname(s, reinterpret_cast<sockaddr *>(&got), &len) == 0) {
            port = ntohs(got.sin_port);
        }
    }
    closesocket(s);
    return port;
}

bool can_offload() { return can_offload_with(guess_llama_bin()); }

// The cores worth giving inference threads, and a mask with one bit per
// core.
static std::pair<int, uint64_t> perf_cores_and_mask() {
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
    if (len == 0) {
        return {0, 0};
    }
    std::vector<char> buf(len);
    if (!GetLogicalProcessorInformationEx(
            RelationProcessorCore, reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *>(buf.data()), &len)) {
        return {0, 0};
    }

    struct CoreInfo {
        int      cls;
        uint64_t first;
    };
    std::vector<CoreInfo> cores;
    size_t                off = 0;
    while (off + sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX) <= buf.size()) {
        auto * info = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *>(buf.data() + off);
        if (info->Size == 0 || off + info->Size > buf.size()) {
            break;
        }
        if (info->Relationship == RelationProcessorCore) {
            const auto & pr = info->Processor;
            // a machine with more than one processor group is left alone:
            // an affinity mask only addresses the group it belongs to
            if (pr.GroupCount == 1 && pr.GroupMask[0].Group == 0) {
                const KAFFINITY mask = pr.GroupMask[0].Mask;
                if (mask != 0) {
                    cores.push_back({static_cast<int>(pr.EfficiencyClass), static_cast<uint64_t>(mask & (~mask + 1))});
                }
            }
        }
        off += info->Size;
    }
    if (cores.empty()) {
        return {0, 0};
    }
    int best = cores[0].cls;
    for (const auto & c : cores) {
        best = std::max(best, c.cls);
    }
    int      n    = 0;
    uint64_t mask = 0;
    for (const auto & c : cores) {
        if (c.cls == best) {
            n++;
            mask |= c.first;
        }
    }
    return {n, mask};
}

std::pair<int, uint64_t> cpu_threads_and_mask() {
    if (const int v = env_int("LLMASH_THREADS", 0); v > 0) {
        return {v, 0};
    }
    const auto     nm = perf_cores_and_mask();
    const unsigned hw = std::thread::hardware_concurrency();
    if (nm.first <= 0 || (hw != 0 && static_cast<unsigned>(nm.first) > hw)) {
        return {0, 0};
    }
    return nm;
}

// --------------------------------------------------------------- Instance

Instance::Instance(Model m, int ctx_in, bool vision_in, Config * cfg)
    : model(std::move(m)), ctx(ctx_in), vision(vision_in), cfg_(cfg) {
    port         = free_port();
    load_mode    = cfg_->load_mode;
    last_used    = now_f();
    keep_alive   = std::numeric_limits<double>::infinity();
    expires_at   = std::numeric_limits<double>::infinity();
}

std::string Instance::url() const { return "http://127.0.0.1:" + std::to_string(port); }

void Instance::touch() {
    std::lock_guard<std::mutex> lock(mu_);
    last_used = now_f();
    expires_at = std::isinf(keep_alive) ? std::numeric_limits<double>::infinity() : last_used + keep_alive;
}

void Instance::set_keep_alive(double ka) {
    {
        std::lock_guard<std::mutex> lock(mu_);
        keep_alive = ka;
    }
    touch();
}

bool Instance::ready() const {
    std::lock_guard<std::mutex> lock(mu_);
    return ready_;
}

bool Instance::on_gpu() const {
    std::lock_guard<std::mutex> lock(mu_);
    return on_gpu_;
}

double Instance::vram_gb() const { return static_cast<double>(model.size) / static_cast<double>(1ull << 30) * 1.05; }

bool Instance::alive() const {
    std::lock_guard<std::mutex> lock(mu_);
    // never started (or a placeholder still being spawned): not dead
    if (pid_ == 0) {
        return true;
    }
    HANDLE h = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, pid_);
    if (!h) {
        return false;
    }
    DWORD code = 0;
    const BOOL ok = GetExitCodeProcess(h, &code);
    CloseHandle(h);
    return ok && code == STILL_ACTIVE;
}

std::string Instance::tail_log(size_t n) const {
    if (logfile.empty()) {
        return "";
    }
    std::ifstream in(logfile, std::ios::binary | std::ios::ate);
    if (!in) {
        return "";
    }
    const std::streamoff size = in.tellg();
    const std::streamoff start = (static_cast<uint64_t>(size) > n) ? size - static_cast<std::streamoff>(n) : 0;
    in.seekg(start);
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

// Progress 0..1: bytes read (dio) or working set (mmap) against the file
// size, whichever is furthest along.
double Instance::progress() const {
    if (ready()) {
        return 1.0;
    }
    if (!alive()) {
        return 0.0;
    }
    unsigned long pid;
    {
        std::lock_guard<std::mutex> lock(mu_);
        pid = pid_;
    }
    if (pid == 0) {
        return 0.0;
    }
    HANDLE h = OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, FALSE, pid);
    if (!h) {
        return 0.0;
    }
    PROCESS_MEMORY_COUNTERS pmc{};
    pmc.cb          = sizeof(pmc);
    const bool haveMem = K32GetProcessMemoryInfo(h, &pmc, sizeof(pmc)) != 0;
    IO_COUNTERS io{};
    const bool  haveIo = GetProcessIoCounters(h, &io) != 0;
    CloseHandle(h);
    if (!haveMem && !haveIo) {
        return 0.0;
    }
    double total = static_cast<double>(model.size);
    if (total < 1) {
        total = 1;
    }
    double best = haveMem ? static_cast<double>(pmc.WorkingSetSize) : 0.0;
    if (haveIo) {
        best = std::max(best, static_cast<double>(io.ReadTransferCount));
    }
    return std::clamp(best / total, 0.0, 0.99);
}

bool tune_enabled(const std::string & name) {
    static std::set<std::string> off;
    static bool                  loaded = false;
    if (!loaded) {
        loaded = true;
        std::stringstream ss(env_str("LLMASH_TUNE_OFF"));
        std::string       tok;
        while (std::getline(ss, tok, ',')) {
            off.insert(lower(trim(tok)));
        }
    }
    return off.count(name) == 0;
}

// Flags the caller set by hand win over the tuned ones.
std::vector<std::string> drop_overridden(const std::vector<std::string> & tuned,
                                         const std::vector<std::string> & extra) {
    std::set<std::string> set;
    for (const std::string & x : extra) {
        if (!x.empty() && x[0] == '-') {
            set.insert(x);
        }
    }
    if (set.empty()) {
        return tuned;
    }
    std::vector<std::string> out;
    for (size_t i = 0; i < tuned.size(); i++) {
        if (tuned[i].empty() || tuned[i][0] != '-') {
            out.push_back(tuned[i]);
            continue;
        }
        size_t n = 1;
        while (i + n < tuned.size() && !tuned[i + n].empty() && tuned[i + n][0] != '-') {
            n++;
        }
        if (set.count(tuned[i]) == 0) {
            out.insert(out.end(), tuned.begin() + i, tuned.begin() + i + n);
        }
        i += n - 1;
    }
    return out;
}

std::vector<std::string> Instance::args() const {
    std::vector<std::string> a{cfg_->llama_bin, "-m", model.path, "--host", "127.0.0.1", "--port", std::to_string(port)};

    if (can_offload_with(cfg_->llama_bin)) {
        a.insert(a.end(), {"-ngl", "999"});
    } else if (const auto tm = cpu_threads_and_mask(); tm.first > 0) {
        // nothing to offload to, so the thread count is the whole game
        a.insert(a.end(), {"-t", std::to_string(tm.first), "-tb", std::to_string(tm.first)});
        if (tm.second != 0) {
            std::ostringstream hex;
            hex << std::hex << tm.second;
            a.insert(a.end(), {"-C", hex.str(), "--cpu-strict", "1"});
        }
    }

    const int parallel = cfg_->parallel > 0 ? cfg_->parallel : 1;
    a.insert(a.end(), {"-c", std::to_string(ctx * parallel), "--jinja", "--no-webui", "-fa", "on", "--cache-type-k",
                       cfg_->kv_type, "--cache-type-v", cfg_->kv_type, "--parallel", std::to_string(parallel)});

    if (!plain_args_) {
        // only a build of ours is known to take these
        a.insert(a.end(), {"--load-mode", load_mode});
        if (free_vram_gb().second) {
            a.push_back("-bs");
        }
    }

    const int native = model.ctx_train > 0 ? model.ctx_train : (cfg_->ctx > 0 ? cfg_->ctx : 8192);
    if (ctx > native) {
        std::ostringstream scale;
        scale.setf(std::ios::fixed);
        scale.precision(4);
        scale << static_cast<double>(ctx) / static_cast<double>(native);
        a.insert(a.end(), {"--rope-scaling", "yarn", "--rope-scale", scale.str(), "--yarn-orig-ctx",
                           std::to_string(native)});
    }

    const std::vector<std::string> extra  = launch_extra_for(*cfg_, model.name);
    const Tuning                   tuning = auto_tune();
    if (!tuning.flags.empty()) {
        const std::vector<std::string> tuned = drop_overridden(tuning.flags, extra);
        a.insert(a.end(), tuned.begin(), tuned.end());
    }
    a.insert(a.end(), extra.begin(), extra.end());

    // The projector is loaded up front only when this turn needs it, or when
    // nothing has asked for it to be held back.
    const bool on_demand = env_str("LLMASH_MMPROJ_ON_DEMAND", "1") == "1";
    if (!model.projector.empty() && file_exists(model.projector) &&
        (vision || !(mmproj_blocked(*cfg_, model.name) || on_demand))) {
        a.insert(a.end(), {"--mmproj", model.projector});
    }

    // no GPU: speculation costs more than it saves.
    if (!free_vram_gb().second) {
        return a;
    }

    // A model's own head is trained with its weights and wins over a
    // downloaded one.
    const int mtp_draft = env_int("LLMASH_MTP_DRAFT", 3);
    if (model.has_mtp) {
        a.insert(a.end(), {"--spec-type", "draft-mtp", "--spec-draft-n-max", std::to_string(mtp_draft)});
    } else if (!model.mtp_path.empty() && file_exists(model.mtp_path)) {
        a.insert(a.end(), {"--spec-type", "draft-mtp", "--model-draft", model.mtp_path, "-ngld", "999",
                           "--spec-draft-n-max", std::to_string(mtp_draft)});
    } else {
        const std::string fallback = env_str("LLMASH_SPEC_FALLBACK", "ngram-mod");
        if (!fallback.empty() && fallback != "none") {
            a.insert(a.end(), {"--spec-type", fallback});
            if (fallback.rfind("ngram", 0) != 0) {
                a.insert(a.end(), {"--spec-draft-n-max", std::to_string(mtp_draft)});
            }
        }
    }
    return a;
}

std::string Instance::start() {
    log_line("loading " + model.name + " ctx=" + std::to_string(ctx) + " port=" + std::to_string(port) +
             " mode=" + load_mode);
    const std::string log_dir = (fs::path(cfg_->root) / "logs").string();
    std::error_code   ec;
    fs::create_directories(log_dir, ec);
    logfile = (fs::path(log_dir) / (sanitize_name(model.name) + ".log")).string();

    const std::vector<std::string> argv = args();
    std::vector<const char *>      cargv;
    cargv.reserve(argv.size() + 1);
    for (const auto & s : argv) {
        cargv.push_back(s.c_str());
    }
    cargv.push_back(nullptr);

    subprocess_s proc{};
    const int    options = subprocess_option_no_window | subprocess_option_inherit_environment |
                        subprocess_option_combined_stdout_stderr | subprocess_option_enable_async;
    if (subprocess_create_ex(cargv.data(), options, nullptr, nullptr, &proc) != 0) {
        err = "failed to start llama-server";
        mark_loaded();
        return err;
    }

    {
        std::lock_guard<std::mutex> lock(mu_);
        pid_ = GetProcessId(reinterpret_cast<HANDLE>(proc.hProcess));
    }
    std::thread(drain_to_file, proc, logfile).detach();

    const double t0 = now_f();
    httplib::Client client("127.0.0.1", port);
    client.set_connection_timeout(2, 0);
    client.set_read_timeout(2, 0);

    double lastPct = -1.0, lastMove = now_f();
    for (;;) {
        if (const double pct = progress(); pct > lastPct + 0.001) {
            lastPct  = pct;
            lastMove = now_f();
        }
        if (now_f() - lastMove > kLoadStallTimeout) {
            break;
        }
        if (!alive()) {
            std::string out = tail_log(24000);
            std::string low = out;
            std::transform(low.begin(), low.end(), low.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            const bool argReject = low.find("unknown argument") != std::string::npos ||
                                   low.find("invalid argument") != std::string::npos ||
                                   low.find("unrecognized argument") != std::string::npos;
            if (!plain_args_ && argReject) {
                log_line(model.name + ": this llama.cpp build rejected an argument, retrying without the optional ones");
                plain_args_ = true;
                return start();
            }
            const bool dioIssue = low.find("direct") != std::string::npos || low.find("dio") != std::string::npos ||
                                  low.find("unsupported") != std::string::npos;
            if (load_mode != "mmap" && dioIssue) {
                log_line(model.name + ": DirectIO unavailable here, retrying with mmap");
                load_mode = "mmap";
                return start();
            }
            err = explain_load_failure(out, *cfg_);
            log_line("FAILED to load " + model.name + ": " + err);
            if (out.size() > 1200) {
                out = out.substr(out.size() - 1200);
            }
            log_line(out);
            mark_loaded();
            return err;
        }
        if (auto res = client.Get("/health"); res && res->status == 200) {
            std::string low = tail_log(24000);
            std::transform(low.begin(), low.end(), low.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            const bool onGpu = low.find("no usable gpu") == std::string::npos &&
                               low.find("failed to initialize cuda") == std::string::npos;
            {
                std::lock_guard<std::mutex> lock(mu_);
                ready_  = true;
                on_gpu_ = onGpu;
            }
            if (!onGpu) {
                log_line(model.name + ": no usable GPU, the weights are in system RAM");
            }
            log_line("ready " + model.name + " in " + std::to_string(now_f() - t0) + "s");
            mark_loaded();
            return "";
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
    }
    stop();
    mark_loaded();
    err = "timed out waiting for llama-server";
    return err;
}

void Instance::mark_loaded() {
    // No wait primitive is exposed for this signal (see the report): the one
    // caller in the Go original that blocked on it can never observe a not-
    // yet-finished load here, since Manager::get serializes every load
    // behind one global lock before calling start().
}

void Instance::stop() {
    unsigned long pid;
    {
        std::lock_guard<std::mutex> lock(mu_);
        pid = pid_;
    }
    if (pid != 0) {
        if (HANDLE h = OpenProcess(PROCESS_TERMINATE | SYNCHRONIZE, FALSE, pid)) {
            TerminateProcess(h, 1);
            WaitForSingleObject(h, 10000);
            CloseHandle(h);
        }
    }
    std::lock_guard<std::mutex> lock(mu_);
    ready_ = false;
}

// --------------------------------------------------------------- Manager

namespace {

Instance * find_by_name(std::vector<std::unique_ptr<Instance>> & live, const std::string & name) {
    for (auto & p : live) {
        if (p->model.name == name) {
            return p.get();
        }
    }
    return nullptr;
}

void erase_ptr(std::vector<std::unique_ptr<Instance>> & live, Instance * target) {
    live.erase(std::remove_if(live.begin(), live.end(), [&](const std::unique_ptr<Instance> & p) { return p.get() == target; }),
               live.end());
}

// Serializes every model load, matching manager.go's package-level `load`
// mutex ("held across a load, like the Python lock"); manager.h leaves no
// room for it as a Manager field, so it lives here instead.
std::mutex g_load_mu;

} // namespace

Manager::Manager(Config cfg, Registry * reg) : cfg_(std::move(cfg)), reg_(reg) {}

std::vector<Instance *> Manager::loaded() {
    std::lock_guard<std::mutex> lock(mu_);
    std::vector<Instance *>     out;
    for (auto & p : live_) {
        if (p->ready()) {
            out.push_back(p.get());
        }
    }
    return out;
}

void Manager::drop_dead() {
    std::vector<Instance *> dead;
    {
        std::lock_guard<std::mutex> lock(mu_);
        for (auto & p : live_) {
            if (!p->alive()) {
                dead.push_back(p.get());
            }
        }
    }
    for (Instance * in : dead) {
        log_line(in->model.name + " died underneath us, dropping it");
        in->stop();
    }
    if (!dead.empty()) {
        std::lock_guard<std::mutex> lock(mu_);
        for (Instance * in : dead) {
            erase_ptr(live_, in);
        }
    }
}

void Manager::evict_for(double need_gb, const std::string & keep) {
    std::vector<Instance *> loaded_list;
    {
        std::lock_guard<std::mutex> lock(mu_);
        for (auto & p : live_) {
            if (p->ready()) {
                loaded_list.push_back(p.get());
            }
        }
    }
    double used = 0;
    for (Instance * in : loaded_list) {
        if (in->model.name != keep) {
            used += in->vram_gb();
        }
    }
    const double vram_budget = env_float("LLMASH_VRAM_GB", 80);
    const double ram_floor   = env_float("LLMASH_RAM_FLOOR", 12);
    const double busy_grace  = env_float("LLMASH_BUSY_GRACE", 120);
    const bool   tight       = free_ram_gb() < ram_floor;
    if (used + need_gb <= vram_budget && !tight) {
        return;
    }
    if (tight) {
        log_line("system RAM down to " + std::to_string(free_ram_gb()) + " GB, evicting to make room");
    }
    std::sort(loaded_list.begin(), loaded_list.end(), [](Instance * a, Instance * b) { return a->last_used < b->last_used; });
    for (Instance * in : loaded_list) {
        if (in->model.name == keep || is_pinned(in->model.name)) {
            continue;
        }
        if (now_f() - in->last_used < busy_grace) {
            log_line("not evicting " + in->model.name + ": used " + std::to_string(now_f() - in->last_used) + "s ago");
            continue;
        }
        log_line("evicting " + in->model.name + " to free " + std::to_string(in->vram_gb()) + " GB");
        in->stop();
        {
            std::lock_guard<std::mutex> lock(mu_);
            erase_ptr(live_, in);
        }
        used -= in->vram_gb();
        if (used + need_gb <= vram_budget && free_ram_gb() >= ram_floor) {
            return;
        }
    }
}

int Manager::fit_ctx(const Model & m, int ctx) {
    const double weights = static_cast<double>(m.size) / static_cast<double>(1ull << 30);
    const int    parallel = cfg_.parallel > 0 ? cfg_.parallel : 1;
    const double want    = weights * (1.0 + static_cast<double>(ctx) * parallel / kCtxTrainNative);
    double       freeV;
    {
        std::lock_guard<std::mutex> lock(mu_);
        freeV = free_vram_gb().first;
        if (Instance * cur = find_by_name(live_, m.name)) {
            freeV += cur->vram_gb();
        }
    }
    const double vram_headroom = env_float("LLMASH_VRAM_HEADROOM", 6);
    const double room          = std::max(4.0, freeV - vram_headroom);
    if (want <= room) {
        return ctx;
    }
    const double step    = 32768.0;
    const double allowed = std::max(0.0, room / std::max(weights, 0.1) - 1.0) * kCtxTrainNative;
    const int    fitted  = static_cast<int>(std::max(8192.0, std::floor(allowed / step) * step));
    if (fitted < ctx) {
        log_line("ctx " + std::to_string(ctx) + " would need ~" + std::to_string(want) + " GB on the card with " +
                 std::to_string(freeV) + " GB free; using " + std::to_string(fitted) + " instead");
        return fitted;
    }
    return ctx;
}

Instance * Manager::get(const std::string & name, int ctx, double keep_alive, bool vision, std::string & err) {
    drop_dead();
    const Model * m = reg_->find(name);
    if (!m) {
        err = "model not found";
        return nullptr;
    }
    if (!file_exists(m->path)) {
        err = "missing from disk";
        return nullptr;
    }
    if (ctx <= 0) {
        ctx = cfg_.ctx > 0 ? cfg_.ctx : 8192;
    }

    {
        std::lock_guard<std::mutex> lock(mu_);
        if (Instance * cur = find_by_name(live_, m->name); cur && cur->ready() && cur->ctx >= ctx) {
            cur->set_keep_alive(keep_alive);
            return cur;
        }
    }

    // manager.go: the trained context is the ceiling unless local.json
    // forces one (ctx_override) or lifts it (ctx_max).
    const int    native  = m->ctx_train > 0 ? m->ctx_train : (cfg_.ctx > 0 ? cfg_.ctx : 8192);
    const int    forced  = ctx_target(cfg_, m->name);
    const int    ceiling = ctx_ceiling(cfg_, m->name, native);
    const double weights = static_cast<double>(m->size) / static_cast<double>(1ull << 30);
    double       need    = weights * 1.05;
    if (forced > 0) {
        ctx  = forced;
        need = weights * 1.05 + weights * 0.4 * (static_cast<double>(ctx) / kCtxTrainNative);
    } else if (ceiling > native) {
        if (ctx > ceiling) {
            ctx = ceiling;
        }
        ctx  = fit_ctx(*m, ctx);
        need = weights * 1.05 + weights * 0.4 * (static_cast<double>(ctx) / kCtxTrainNative);
    } else {
        if (ctx > native) {
            ctx = native;
        }
        ctx = fit_ctx(*m, ctx);
    }

    std::lock_guard<std::mutex> load_lock(g_load_mu);
    Instance *                  inst  = nullptr;
    bool                        fresh = false;
    {
        std::lock_guard<std::mutex> lock(mu_);
        inst = find_by_name(live_, m->name);
        if (inst != nullptr && inst->ctx < ctx) {
            log_line("reloading " + m->name + " for a larger context (" + std::to_string(inst->ctx) + " -> " +
                     std::to_string(ctx) + ")");
            inst->stop();
            erase_ptr(live_, inst);
            inst = nullptr;
        }
        fresh = (inst == nullptr);
    }
    if (fresh) {
        evict_for(need, m->name);
        auto       owned = std::make_unique<Instance>(*m, ctx, vision, &cfg_);
        Instance * raw   = owned.get();
        {
            std::lock_guard<std::mutex> lock(mu_);
            live_.push_back(std::move(owned));
        }
        inst = raw;
    }
    if (fresh) {
        if (const std::string start_err = inst->start(); !start_err.empty()) {
            inst->stop();
            {
                std::lock_guard<std::mutex> lock(mu_);
                erase_ptr(live_, inst);
            }
            err = start_err;
            return nullptr;
        }
    }
    if (!inst->ready()) {
        err = inst->err.empty() ? "model failed to load" : inst->err;
        return nullptr;
    }
    inst->set_keep_alive(keep_alive);
    return inst;
}

bool Manager::unload(const std::string & name) {
    std::string key = name;
    if (const Model * m = reg_->find(name)) {
        key = m->name;
    }
    Instance * inst = nullptr;
    {
        std::lock_guard<std::mutex> lock(mu_);
        inst = find_by_name(live_, key);
        if (inst != nullptr) {
            erase_ptr(live_, inst);
        }
    }
    if (inst == nullptr) {
        return false;
    }
    log_line("unloading " + key);
    inst->stop();
    return true;
}

Instance * Manager::find(const std::string & name) {
    std::lock_guard<std::mutex> lock(mu_);
    if (Instance * exact = find_by_name(live_, name)) {
        return exact;
    }
    if (name.empty()) {
        return nullptr;
    }
    for (auto & p : live_) {
        if (p->model.name.find(name) != std::string::npos) {
            return p.get();
        }
    }
    return nullptr;
}

void Manager::shutdown() {
    std::vector<std::unique_ptr<Instance>> all;
    {
        std::lock_guard<std::mutex> lock(mu_);
        all = std::move(live_);
        live_.clear();
    }
    for (auto & p : all) {
        p->stop();
    }
}

void Manager::reap_idle() {
    const double             now = now_f();
    std::vector<std::string> gone;
    {
        std::lock_guard<std::mutex> lock(mu_);
        for (auto & p : live_) {
            if (is_pinned(p->model.name) || !p->ready() || now <= p->expires_at || now - p->last_used < 5) {
                continue;
            }
            gone.push_back(p->model.name);
        }
    }
    for (const std::string & name : gone) {
        log_line(name + " idle past keep_alive");
        unload(name);
    }
}

Tuning auto_tune() {
    Tuning     t;
    const auto add = [&](const std::string & note, std::initializer_list<std::string> flags) {
        t.flags.insert(t.flags.end(), flags);
        t.why += (t.why.empty() ? "" : ", ") + note;
    };

    // A conversation that comes back should not pay for its prompt twice.
    if (const int n = env_int("LLMASH_CACHE_REUSE", 256); n > 0 && tune_enabled("cache-reuse")) {
        add("cache-reuse " + std::to_string(n), {"--cache-reuse", std::to_string(n)});
    }

    // Prompt caches for slots that are not resident live in host RAM.
    if (tune_enabled("cache-ram")) {
        int mib = env_int("LLMASH_CACHE_RAM_MB", 0);
        if (mib == 0) {
            mib = static_cast<int>(free_ram_gb() * 1024 / 4);
            mib = std::max(8192, std::min(32768, mib));
        }
        if (mib > 0) {
            add("cache-ram " + std::to_string(mib) + " MiB", {"-cram", std::to_string(mib)});
        }
    }

    // Prompt processing runs in physical batches; the stock 512 leaves a big
    // card idle.
    if (tune_enabled("batch")) {
        int        ub = env_int("LLMASH_UBATCH", 0);
        int        b  = env_int("LLMASH_BATCH", 0);
        const auto fv = free_vram_gb();
        if (ub == 0 && fv.second && fv.first > 24) {
            ub = 2048;
            b  = 4096;
        }
        if (ub > 0) {
            if (b < ub) {
                b = ub * 2;
            }
            add("batch " + std::to_string(b) + "/" + std::to_string(ub),
                {"-b", std::to_string(b), "-ub", std::to_string(ub)});
        }
    }

    if (const int p = env_int("LLMASH_PRIO", 1); p > 0 && tune_enabled("prio")) {
        add("prio " + std::to_string(p), {"--prio", std::to_string(p)});
    }
    return t;
}

} // namespace llmash
