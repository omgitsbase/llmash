#include "manager.h"

#include "gguf.h"

#include "draft.h"
#include "log.h"
#include "platform.h"

#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0A00
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <psapi.h>
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sched.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/sysinfo.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

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

#ifdef _WIN32
// httplib.h does not call WSAStartup itself.
struct WinsockInit {
    WinsockInit() {
        WSADATA d;
        WSAStartup(MAKEWORD(2, 2), &d);
    }
    ~WinsockInit() { WSACleanup(); }
} g_winsock_init;
#endif

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
double     g_vram_total = 0;
bool       g_vram_known = false;

void read_vram_locked() {
    if (now_f() - g_vram_at < 2.0) {
        return;
    }
    double free = 0, total = 0;
    bool   known = false;
    if (const double v = env_float("LLMASH_VRAM_GB", 0); v > 0) {
        free = total = v, known = true;
    } else {
        std::string out;
        if (run_capture({"nvidia-smi", "--query-gpu=memory.free,memory.total", "--format=csv,noheader,nounits"}, 8000,
                        out)) {
            std::istringstream iss(out);
            std::string        line;
            if (std::getline(iss, line)) {
                const size_t comma = line.find(',');
                try {
                    free  = std::stod(trim(line.substr(0, comma))) / 1024.0;
                    known = true;
                    if (comma != std::string::npos) {
                        total = std::stod(trim(line.substr(comma + 1))) / 1024.0;
                    }
                } catch (const std::exception &) {
                }
            }
        }
    }
    g_vram_at = now_f(), g_vram_free = free, g_vram_total = total, g_vram_known = known;
}

std::pair<double, bool> free_vram_gb() {
    std::lock_guard<std::mutex> lock(g_vram_mu);
    read_vram_locked();
    return {g_vram_free, g_vram_known};
}

// What the card holds in total, which is what a prompt batch has to fit
// beside rather than inside whatever is already resident.
double total_vram_gb() {
    std::lock_guard<std::mutex> lock(g_vram_mu);
    read_vram_locked();
    return g_vram_total;
}

// What one process may hold on the card. Under WDDM every video allocation is
// backed by commit charge, RAM plus pagefile, so a process is refused once it
// has taken what is left of that, however much of the card is free: measured
// at 79 GiB with 80.5 GiB of commit free, 59 with 60. A llama-server is one
// process, so a model and its cache have to fit in this, not in the card. A
// larger pagefile raises it. Read live, since other programs move it.
#ifdef _WIN32
double gpu_process_budget_gb(double forced) {
    if (forced > 0) {
        return forced;
    }
    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof(ms);
    if (!GlobalMemoryStatusEx(&ms)) {
        return 0.0;
    }
    // a bare process reaches avail minus 1.5; llama-server also holds a few GB
    // of host memory out of the same commit
    const double avail = static_cast<double>(ms.ullAvailPageFile) / static_cast<double>(1ull << 30);
    return std::max(0.0, avail - 5.0);
}
#else
double gpu_process_budget_gb(double forced) { return forced; }
#endif

double free_ram_gb() {
#ifdef _WIN32
    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof(ms);
    if (!GlobalMemoryStatusEx(&ms)) {
        return 999.0;
    }
    return static_cast<double>(ms.ullAvailPhys) / static_cast<double>(1ull << 30);
#else
    // MemAvailable counts reclaimable cache; sysinfo's freeram does not
    std::ifstream mi("/proc/meminfo");
    std::string   line;
    while (std::getline(mi, line)) {
        if (line.rfind("MemAvailable:", 0) == 0) {
            try {
                return std::stod(line.substr(13)) / (1024.0 * 1024.0);
            } catch (const std::exception &) {
                break;
            }
        }
    }
    struct sysinfo si{};
    if (sysinfo(&si) != 0) {
        return 999.0;
    }
    return static_cast<double>(si.freeram) * static_cast<double>(si.mem_unit) / static_cast<double>(1ull << 30);
#endif
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

std::string guess_llama_bin() {
    std::string v = env_str("LLAMA_BIN");
    if (!v.empty()) {
        return v;
    }
    const std::string exe = exe_dir();
    for (const auto & rel : {"runtime", "llama.cpp"}) {
        std::string cand = (fs::path(exe) / rel / llama_server_exe()).string();
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

// Models never evicted for VRAM.
bool is_pinned(const Config & cfg, const std::string & name) {
    return std::find(cfg.pin.begin(), cfg.pin.end(), name) != cfg.pin.end();
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

    if (has("unknown model architecture")) {
        std::string arch;
        if (const size_t q = low.find("architecture: '"); q != std::string::npos) {
            const size_t a = q + 15, b = low.find('\'', a);
            arch = b != std::string::npos ? low.substr(a, b - a) : "";
        }
        return "The runtime's llama.cpp does not know this model's architecture" + (arch.empty() ? "" : " (" + arch + ")") +
               ": the model is newer than the runtime in " + runtime_dir(cfg) + ". " +
               (own_runtime(cfg) ? "It needs a newer llmash runtime; nothing about the file is wrong."
                                 : "Run `llmash update -Runtime cuda` (or vulkan, or cpu) to replace it.") +
               " A newer llama-server can serve this model on its own: name its folder under runtime in local.json, keyed by the model's name.";
    }
    if (has("no kernel image") || has("unsupported toolchain")) {
        return "The runtime in " + runtime_dir(cfg) + " has no CUDA code this card can run. Run `llmash update`; "
               "if it still fails, update the NVIDIA driver.";
    }
    if (has("wrong number of tensors") || has("check_tensor_dims")) {
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
#ifdef _WIN32
    const SOCKET s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (s == INVALID_SOCKET) {
        return 0;
    }
#else
    const int s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (s < 0) {
        return 0;
    }
#endif
    sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port        = 0;
    int port             = 0;
    if (bind(s, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) == 0) {
        sockaddr_in got{};
#ifdef _WIN32
        int len = sizeof(got);
#else
        socklen_t len = sizeof(got);
#endif
        if (getsockname(s, reinterpret_cast<sockaddr *>(&got), &len) == 0) {
            port = ntohs(got.sin_port);
        }
    }
#ifdef _WIN32
    closesocket(s);
#else
    close(s);
#endif
    return port;
}

bool can_offload() { return can_offload_with(guess_llama_bin()); }

bool has_thinking_block(const std::string & chat_template) {
    const std::string t = lower(chat_template);
    return t.find("<think>") != std::string::npos || t.find("enable_thinking") != std::string::npos ||
           t.find("reasoning_content") != std::string::npos;
}

// The cores worth giving inference threads, and a mask with one bit per
// core.
#ifdef _WIN32
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
#else
// One thread per physical core, efficient cores dropped by cpu_capacity the
// way the Windows side drops them by EfficiencyClass.
static std::pair<int, uint64_t> perf_cores_and_mask() {
    const fs::path base("/sys/devices/system/cpu");
    std::error_code ec;
    if (!fs::is_directory(base, ec)) {
        return {0, 0};
    }

    struct Cpu {
        int      id       = 0;
        long     capacity = -1;
        std::string siblings;
    };
    std::vector<Cpu> cpus;
    for (int i = 0; i < 64; i++) {
        const fs::path dir = base / ("cpu" + std::to_string(i));
        if (!fs::is_directory(dir, ec)) {
            continue;
        }
        Cpu c;
        c.id = i;
        std::ifstream cap(dir / "cpu_capacity");
        if (cap) {
            cap >> c.capacity;
        }
        std::ifstream sib(dir / "topology" / "thread_siblings_list");
        if (sib) {
            std::getline(sib, c.siblings);
        }
        cpus.push_back(std::move(c));
    }
    if (cpus.empty()) {
        return {0, 0};
    }

    long best = -1;
    for (const Cpu & c : cpus) {
        best = std::max(best, c.capacity);
    }

    std::set<std::string> seen;
    int      n    = 0;
    uint64_t mask = 0;
    for (const Cpu & c : cpus) {
        if (best > 0 && c.capacity != best) {
            continue;
        }
        const std::string key = c.siblings.empty() ? std::to_string(c.id) : c.siblings;
        if (!seen.insert(key).second) {
            continue;
        }
        n++;
        mask |= (1ull << c.id);
    }
    return {n, mask};
}
#endif

std::pair<int, uint64_t> cpu_threads_and_mask() {
    if (const int v = env_int("LLMASH_THREADS", 0); v > 0) {
        return {v, 0};
    }
    const auto     nm = perf_cores_and_mask();
    const unsigned hw = std::thread::hardware_concurrency();
    if (nm.first <= 0 || (hw != 0 && static_cast<unsigned>(nm.first) > hw)) {
        return {0, 0};
    }
    if (nm.first < 4) {
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

double Instance::vram_gb() const {
    const double GB    = static_cast<double>(1ull << 30);
    const double cache = kv_on_gpu ? model.kv_bytes_tok * kv_type_scale(kv_type) * ctx : 0.0;
    return static_cast<double>(model.size) / GB * 1.05 + (cache + model.state_bytes) / GB;
}

bool Instance::alive() const {
    std::lock_guard<std::mutex> lock(mu_);
    // never started (or a placeholder still being spawned): not dead
    if (pid_ == 0) {
        return true;
    }
#ifdef _WIN32
    HANDLE h = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, pid_);
    if (!h) {
        return false;
    }
    DWORD code = 0;
    const BOOL ok = GetExitCodeProcess(h, &code);
    CloseHandle(h);
    return ok && code == STILL_ACTIVE;
#else
    return kill(static_cast<pid_t>(pid_), 0) == 0;
#endif
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
#ifdef _WIN32
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
#else
    double resident = 0, read_bytes = 0;
    {
        std::ifstream st("/proc/" + std::to_string(pid) + "/statm");
        double        total = 0, res = 0;
        if (st && (st >> total >> res)) {
            resident = res * static_cast<double>(sysconf(_SC_PAGESIZE));
        }
    }
    {
        std::ifstream io("/proc/" + std::to_string(pid) + "/io");
        std::string   line;
        while (std::getline(io, line)) {
            if (line.rfind("read_bytes:", 0) == 0) {
                try {
                    read_bytes = std::stod(line.substr(11));
                } catch (const std::exception &) {
                }
                break;
            }
        }
    }
    const bool haveMem = resident > 0;
    const bool haveIo  = read_bytes > 0;
    if (!haveMem && !haveIo) {
        return 0.0;
    }
#endif
    double total = static_cast<double>(model.size);
    if (total < 1) {
        total = 1;
    }
#ifdef _WIN32
    double best = haveMem ? static_cast<double>(pmc.WorkingSetSize) : 0.0;
    if (haveIo) {
        best = std::max(best, static_cast<double>(io.ReadTransferCount));
    }
#else
    double best = resident;
    if (haveIo) {
        best = std::max(best, read_bytes);
    }
#endif
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

std::vector<std::string> Instance::args() {
    // Another llama-server named for this model gets only the flags every build takes.
    const std::string named   = runtime_for(*cfg_, model.name);
    const bool        foreign = !named.empty();
    if (foreign) {
        plain_args_ = true;
    }
    std::vector<std::string> a{foreign ? named : cfg_->llama_bin, "-m", model.path, "--host", "127.0.0.1", "--port", std::to_string(port)};

    // -ngl is left unset: llama.cpp defaults it to auto and fits what it can on
    // the card, keeping the rest in RAM. Naming a number takes that away.
    if (!can_offload_with(cfg_->llama_bin)) {
        if (const auto tm = cpu_threads_and_mask(); tm.first > 0) {
            // nothing to offload to, so the thread count is the whole game
            a.insert(a.end(), {"-t", std::to_string(tm.first), "-tb", std::to_string(tm.first)});
            if (tm.second != 0) {
                std::ostringstream hex;
                hex << std::hex << tm.second;
                a.insert(a.end(), {"-C", hex.str(), "--cpu-strict", "1"});
            }
        }
    }

    const int parallel = cfg_->parallel > 0 ? cfg_->parallel : 1;
    const std::string kv = kv_type.empty() ? cfg_->kv_type : kv_type;
    a.insert(a.end(), {"-c", std::to_string(ctx * parallel), "--jinja", "--no-webui", "-fa", "on", "--cache-type-k",
                       kv, "--cache-type-v", kv, "--parallel", std::to_string(parallel)});
    if (!kv_on_gpu) {
        a.push_back("--no-kv-offload");  // the cache stays in system RAM, the weights on the card
    }
    if (env_int("LLMASH_WORKER_VERBOSE", 0) != 0) {
        a.insert(a.end(), {"-lv", "4"});  // everything llama-server does, into logs/<model>.log
    }

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
    const Tuning                   tuning = auto_tune(ctx);
    if (!tuning.flags.empty()) {
        const std::vector<std::string> tuned = drop_overridden(tuning.flags, extra);
        a.insert(a.end(), tuned.begin(), tuned.end());
        tune_note = tuning.why;
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
        spec_note = "none (no GPU)";
        return a;
    }
    if (foreign) {
        spec_note = "none (other runtime)";
        return a;
    }

    // A model's own head is trained with its weights and wins over a
    // downloaded one; a downloaded one wins over self-speculation.
    const int         mtp_draft  = env_int("LLMASH_MTP_DRAFT", 3);
    const int         dspark_max = env_int("LLMASH_DSPARK_DRAFT", 6);
    const int         dspark_min = env_int("LLMASH_DSPARK_DRAFT_MIN", 6);
    const std::string mtp        = !model.mtp_path.empty() ? model.mtp_path : sidecar_path(model.path, ".mtp.gguf");
    const std::string eagle3     = sidecar_path(model.path, ".eagle3.gguf");
    const std::string dspark     = dspark_path(model.path);
    const std::string draft      = sidecar_path(model.path, ".draft.gguf");
    const std::string fallback   = env_str("LLMASH_SPEC_FALLBACK", "ngram-mod");
    // LLMASH_SPEC_STACK=1 offers the lookup drafter beside the model's own
    // head. Off: the fork gives lookup the first refusal, and on prompts seen
    // once it wins the round with worse guesses than the head would have made
    // (Qwen3.6-A3B -4.8% on prose). The gain only appears when a prompt is
    // repeated, which is the drafter replaying its own last answer.
    const bool stack = fallback.rfind("ngram", 0) == 0 && env_int("LLMASH_SPEC_STACK", 0) != 0;
    const auto with_lookup = [&](const char * primary) {
        return stack ? fallback + "," + primary : std::string(primary);
    };
    if (model.has_mtp) {
        a.insert(a.end(), {"--spec-type", with_lookup("draft-mtp"), "--spec-draft-n-max", std::to_string(mtp_draft)});
    } else if (!mtp.empty() && file_exists(mtp)) {
        a.insert(a.end(), {"--spec-type", with_lookup("draft-mtp"), "--model-draft", mtp, "-ngld", "999",
                           "--spec-draft-n-max", std::to_string(mtp_draft)});
        spec_note = "mtp";
    } else if (!eagle3.empty() && file_exists(eagle3)) {
        a.insert(a.end(), {"--spec-type", with_lookup("draft-eagle3"), "--model-draft", eagle3, "-ngld", "999",
                           "--spec-draft-n-max", std::to_string(mtp_draft)});
        spec_note = "eagle3";
    } else if (!dspark.empty() && file_exists(dspark)) {
        a.insert(a.end(), {"--spec-type", with_lookup("draft-dspark"), "--model-draft", dspark, "-ngld", "999",
                           "--spec-draft-n-max", std::to_string(dspark_max), "--spec-draft-n-min",
                           std::to_string(dspark_min)});
        spec_note = "dspark";
    } else if (!draft.empty() && file_exists(draft)) {
        a.insert(a.end(), {"--spec-type", with_lookup("draft-simple"), "--model-draft", draft, "-ngld", "999",
                           "--spec-draft-n-max", std::to_string(mtp_draft)});
        spec_note = "draft";
    } else {
        if (!fallback.empty() && fallback != "none") {
            a.insert(a.end(), {"--spec-type", fallback});
            if (fallback.rfind("ngram", 0) != 0) {
                a.insert(a.end(), {"--spec-draft-n-max", std::to_string(mtp_draft)});
            }
            spec_note = fallback;
        }
    }
    if (stack && !spec_note.empty()) {
        spec_note += "+lookup";
    }
    return a;
}

// The drafter named in the log the way the flag reads.
std::string spec_why(const std::string & kind) {
    if (kind.empty()) {
        return "";
    }
    if (kind.rfind("ngram", 0) == 0) {
        return "self-speculation " + kind;
    }
    return "speculation " + kind;
}

std::string Instance::start() {
    log_line("loading " + model.name + " ctx=" + std::to_string(ctx) + " port=" + std::to_string(port) +
             " mode=" + load_mode);
    const std::string log_dir = (fs::path(cfg_->root) / "logs").string();
    std::error_code   ec;
    fs::create_directories(log_dir, ec);
    logfile = (fs::path(log_dir) / (sanitize_name(model.name) + ".log")).string();

    const std::vector<std::string> argv = args();
    if (!tune_note.empty() || !spec_note.empty()) {
        std::string why = tune_note;
        if (const std::string s = spec_why(spec_note); !s.empty()) {
            why += (why.empty() ? "" : ", ") + s;
        }
        log_line(model.name + " tuned: " + why);
    }
    {
        std::string line;
        for (size_t i = 1; i < argv.size(); i++) {
            line += (i > 1 ? " " : "") + argv[i];
        }
        log_line(model.name + " args: " + line);
    }
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
#ifdef _WIN32
        pid_ = GetProcessId(reinterpret_cast<HANDLE>(proc.hProcess));
#else
        pid_ = static_cast<unsigned long>(proc.child);
#endif
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
    stalled = true;
    std::string last;
    std::istringstream lines(tail_log(4000));
    for (std::string l; std::getline(lines, l);) {
        if (const std::string t = trim(l); !t.empty()) {
            last = t;
        }
    }
    err = "llama-server stopped making progress for " + std::to_string(static_cast<int>(kLoadStallTimeout / 60)) +
          " minutes while loading" + (last.empty() ? "" : " (its log ends at \"" + last + "\")") + "; the log is " +
          logfile;
    return err;
}

void Instance::mark_loaded() {
}

void Instance::stop() {
    unsigned long pid;
    {
        std::lock_guard<std::mutex> lock(mu_);
        pid = pid_;
    }
    if (pid != 0) {
#ifdef _WIN32
        if (HANDLE h = OpenProcess(PROCESS_TERMINATE | SYNCHRONIZE, FALSE, pid)) {
            TerminateProcess(h, 1);
            WaitForSingleObject(h, 10000);
            CloseHandle(h);
        }
#else
        ::kill(static_cast<pid_t>(pid), SIGTERM);
        for (int waited = 0; waited < 10000; waited += 50) {
            if (::kill(static_cast<pid_t>(pid), 0) != 0) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        ::kill(static_cast<pid_t>(pid), SIGKILL);
#endif
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

std::vector<Instance *> Manager::live() {
    std::lock_guard<std::mutex> lock(mu_);
    std::vector<Instance *>     out;
    for (auto & p : live_) {
        out.push_back(p.get());
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
    const double total = total_vram_gb();
    double vram_budget = env_float("LLMASH_VRAM_GB", total > 0 ? total - env_float("LLMASH_VRAM_HEADROOM", 6) : 80);
    // The OS ceiling binds well before the card does, so room measured against
    // the card alone says a load will fit when it cannot.
    if (const double budget = gpu_process_budget_gb(cfg_.gpu_budget_gb); budget > 0) {
        vram_budget = std::min(vram_budget, budget);
    }
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
    // Two passes. The first leaves alone a model somebody used moments ago,
    // which is what the grace period is for. The second takes it anyway: a
    // model held back for its recent user is no good to them if the request
    // that needed the room fails instead, and nobody is served at all.
    const auto sweep = [&](bool respect_grace) {
        for (Instance *& in : loaded_list) {
            if (in == nullptr) {
                continue;
            }
            if (in->model.name == keep || is_pinned(cfg_, in->model.name)) {
                in = nullptr;
                continue;
            }
            if (respect_grace && now_f() - in->last_used < busy_grace) {
                log_line("leaving " + in->model.name + " for now: used " + std::to_string(now_f() - in->last_used) +
                         "s ago");
                continue;
            }
            const double freed = in->vram_gb();
            log_line("evicting " + in->model.name + " to free " + std::to_string(freed) + " GB");
            in->stop();
            {
                std::lock_guard<std::mutex> lock(mu_);
                erase_ptr(live_, in);   // the pointer dies here, so the slot is cleared
            }
            in = nullptr;
            used -= freed;
            if (used + need_gb <= vram_budget && free_ram_gb() >= ram_floor) {
                return true;
            }
        }
        return false;
    };
    if (sweep(true)) {
        return;
    }
    sweep(false);
}

Manager::Fit Manager::fit_report(const Model & m) {
    const double GB = static_cast<double>(1ull << 30);
    Fit          f;
    const uint64_t in_ram = std::min(m.size, m.input_bytes);
    f.weights_gb = static_cast<double>(m.size - in_ram) / GB;
    f.ram_gb     = static_cast<double>(in_ram) / GB;
    f.per_tok_gb = m.kv_bytes_tok / GB;
    f.state_gb   = m.state_bytes / GB;
    f.native     = m.ctx_train;
    f.free_gb    = free_vram_gb().first;
    {
        std::lock_guard<std::mutex> lock(mu_);
        if (Instance * cur = find_by_name(live_, m.name)) {
            f.free_gb += cur->vram_gb();
        }
    }
    f.total_gb   = total_vram_gb();
    f.compute_gb = f.total_gb > 24 ? 1.5 : 0.8;  // a micro-batch of logits and the attention scratch
    return f;
}

double Manager::fit_room(const Fit & f) const {
    double room = f.free_gb - env_float("LLMASH_VRAM_HEADROOM", 6);
    if (const double budget = gpu_process_budget_gb(cfg_.gpu_budget_gb); budget > 0) {
        room = std::min(room, budget - 1.0);  // one process holds the model and its cache
    }
    return std::max(2.0, room);
}

// The largest context at or under `ctx` whose cache, at `kv_scale` of f16, fits.
int Manager::fit_at(const Fit & f, int ctx, double kv_scale) const {
    const int    parallel = cfg_.parallel > 0 ? cfg_.parallel : 1;
    const double fixed    = f.weights_gb * 1.05 + f.state_gb * parallel + f.compute_gb;
    const double per_tok  = f.per_tok_gb * kv_scale * parallel;
    const double room     = fit_room(f);
    if (per_tok <= 0 || fixed + scratch_gb(ctx) + per_tok * ctx <= room) {
        return ctx;
    }
    // the scratch shrinks as the context does, so step down until it fits
    const double step = 4096.0;
    int          fits = static_cast<int>(std::max(step, std::floor((room - fixed) / per_tok / step) * step));
    while (fits > step && fixed + scratch_gb(fits) + per_tok * fits > room) {
        fits -= static_cast<int>(step);
    }
    return std::min(fits, ctx);
}

nlohmann::json Manager::fit_json(const Model & m, int ctx) {
    const Fit    f     = fit_report(m);
    const int    asked = ctx > 0 ? ctx : (f.native > 0 ? f.native : 8192);
    const double f16   = kv_type_scale(cfg_.kv_type);
    const double q8    = kv_type_scale("q8_0");
    const int    parallel = cfg_.parallel > 0 ? cfg_.parallel : 1;
    return nlohmann::json{{"model", m.name},
                          {"native", f.native},
                          {"asked", asked},
                          {"kv_type", cfg_.kv_type},
                          {"known", f.per_tok_gb > 0},
                          {"free_gb", f.free_gb},
                          {"total_gb", f.total_gb},
                          {"room_gb", fit_room(f)},
                          {"budget_gb", gpu_process_budget_gb(cfg_.gpu_budget_gb)},
                          {"weights_gb", f.weights_gb},
                          {"ram_gb", f.ram_gb},
                          {"state_gb", f.state_gb},
                          {"compute_gb", f.compute_gb},
                          {"scratch_gb", scratch_gb(asked)},
                          {"cache_gb", f.per_tok_gb * f16 * asked * parallel},
                          {"cache_gb_q8", f.per_tok_gb * q8 * asked * parallel},
                          {"fitted", fit_at(f, asked, f16)},
                          {"fitted_q8", fit_at(f, asked, q8)}};
}

int Manager::v1_ctx(const Model & m) {
    if (const int forced = ctx_target(cfg_, m.name); forced > 0) {
        return forced;
    }
    if (const auto it = cfg_.fit.find(m.name); it != cfg_.fit.end() && it->second.ctx > 0) {
        return it->second.ctx;
    }
    if (!env_str("LLMASH_V1_CTX").empty()) {
        return env_int("LLMASH_V1_CTX", 32768);
    }
    const int native = m.ctx_train > 0 ? m.ctx_train : (cfg_.ctx > 0 ? cfg_.ctx : 8192);
    int       ctx    = ctx_ceiling(cfg_, m.name, native);
    if (cfg_.ctx_cap > 0 && ctx > cfg_.ctx_cap) {
        ctx = cfg_.ctx_cap;
    }
    return fit_ctx(m, ctx, v1_prefs(m));
}

LoadPrefs Manager::v1_prefs(const Model & m) const {
    LoadPrefs p;
    if (const auto it = cfg_.fit.find(m.name); it != cfg_.fit.end()) {
        p.kv_type   = it->second.kv_type;
        p.kv_on_gpu = it->second.kv_on_gpu;
    }
    return p;
}

void Manager::set_config(const Config & c) {
    std::lock_guard<std::mutex> lock(mu_);
    cfg_ = c;
}

int Manager::fit_ctx(const Model & m, int ctx, const LoadPrefs & prefs) {
    if (ctx <= 0 || !prefs.kv_on_gpu) {
        return ctx;  // a cache in system RAM has nothing to fit on the card
    }
    const Fit f = fit_report(m);
    if (f.per_tok_gb <= 0) {
        return ctx;  // an architecture the header did not describe: llama.cpp fits what it can
    }
    const double scale  = kv_type_scale(prefs.kv_type.empty() ? cfg_.kv_type : prefs.kv_type);
    const int    fitted = fit_at(f, ctx, scale);
    if (fitted < ctx) {
        char buf[240];
        const double budget = gpu_process_budget_gb(cfg_.gpu_budget_gb);
        std::snprintf(buf, sizeof(buf), "ctx %d needs %.1f GB (%.1f weights, %.1f cache), %.1f free%s: using %d", ctx,
                      f.weights_gb * 1.05 + f.state_gb + f.compute_gb + f.per_tok_gb * scale * ctx, f.weights_gb,
                      f.per_tok_gb * scale * ctx, f.free_gb,
                      budget > 0 && budget - 1.0 < f.free_gb - 6 ? (", " + std::to_string(static_cast<int>(budget)) + " GB the most Windows will back for one process").c_str() : "",
                      fitted);
        log_line(std::string(m.name) + ": " + buf);
    }
    return fitted;
}

Instance * Manager::get(const std::string & name, int ctx, double keep_alive, bool vision, std::string & err,
                        const LoadPrefs & prefs) {
    drop_dead();
    const std::optional<Model> m = reg_->find(name);
    if (!m) {
        err = "model not found";
        return nullptr;
    }
    if (!file_exists(m->path)) {
        err = "missing from disk";
        return nullptr;
    }
    if (m->incomplete) {
        err = name + " is a download that did not finish; pull it again, or rm it to free the space";
        return nullptr;
    }
    // A cache type recorded for the model by `ctx --kv` holds on every path
    // unless the request names its own. The OpenAI path did this already; the
    // Ollama path loaded at the default and quietly ignored the setting.
    LoadPrefs p = prefs;
    if (p.kv_type.empty()) {
        if (const auto it = cfg_.fit.find(m->name); it != cfg_.fit.end()) {
            p.kv_type   = it->second.kv_type;
            p.kv_on_gpu = it->second.kv_on_gpu;
        }
    }
    const std::string kv = p.kv_type.empty() ? cfg_.kv_type : lower(p.kv_type);
    if (ctx <= 0) {
        ctx = cfg_.ctx > 0 ? cfg_.ctx : 8192;
    }
    if (cfg_.ctx_cap > 0 && ctx > cfg_.ctx_cap) {
        log_line(name + ": ctx " + std::to_string(ctx) + " asked, " + std::to_string(cfg_.ctx_cap) + " is the server's limit");
        ctx = cfg_.ctx_cap;
    }

    {
        std::lock_guard<std::mutex> lock(mu_);
        Instance *                  cur = find_by_name(live_, m->name);
        if (cur != nullptr && vision && !cur->vision && !m->projector.empty() && file_exists(m->projector)) {
            log_line(m->name + ": media turn, reloading with projector");
            cur->stop();
            erase_ptr(live_, cur);
            cur = nullptr;
        }
        if (cur != nullptr && cur->ready() && cur->ctx >= ctx && cur->kv_type == kv && cur->kv_on_gpu == p.kv_on_gpu) {
            cur->set_keep_alive(keep_alive);
            return cur;
        }
    }

    // The trained context is the ceiling unless local.json forces one
    // (ctx_override) or lifts it (ctx_max), or the request itself asks past it:
    // then the model runs under YaRN, up to four times its length.
    const int    native   = m->ctx_train > 0 ? m->ctx_train : (cfg_.ctx > 0 ? cfg_.ctx : 8192);
    const int    forced   = ctx_target(cfg_, m->name);
    const int    yarn_max = m->ctx_train > 0 ? std::max(1, env_int("LLMASH_YARN_MAX", 4)) * native : native;
    const int    ceiling  = std::max(ctx_ceiling(cfg_, m->name, native), std::min(ctx, yarn_max));
    if (forced > 0) {
        ctx = forced;
    } else {
        if (ctx > ceiling) {
            ctx = ceiling;
        }
        ctx = fit_ctx(*m, ctx, prefs);
    }
    if (ctx > native) {
        char buf[96];
        std::snprintf(buf, sizeof(buf), "ctx %d is past the trained %d: running under YaRN x%.2f", ctx, native,
                      static_cast<double>(ctx) / native);
        log_line(m->name + ": " + buf);
    }
    const double weights = static_cast<double>(m->size) / static_cast<double>(1ull << 30);
    const double cache   = p.kv_on_gpu ? m->kv_bytes_tok * kv_type_scale(kv) * ctx : 0.0;
    const double need    = weights * 1.05 + (cache + m->state_bytes) / static_cast<double>(1ull << 30);

    std::lock_guard<std::mutex> load_lock(g_load_mu);
    Instance *                  inst  = nullptr;
    bool                        fresh = false;
    {
        std::lock_guard<std::mutex> lock(mu_);
        inst = find_by_name(live_, m->name);
        if (inst != nullptr && (inst->ctx < ctx || inst->kv_type != kv || inst->kv_on_gpu != p.kv_on_gpu)) {
            log_line("reloading " + m->name + " for a larger context or another cache (" + std::to_string(inst->ctx) +
                     " " + inst->kv_type + (inst->kv_on_gpu ? "" : " in RAM") + " -> " + std::to_string(ctx) + " " + kv +
                     (p.kv_on_gpu ? "" : " in RAM") + ")");
            inst->stop();
            erase_ptr(live_, inst);
            inst = nullptr;
        }
        fresh = (inst == nullptr);
    }
    if (fresh) {
        {
            std::lock_guard<std::mutex> lock(mu_);
            if (const auto it = stalls_.find(m->name); it != stalls_.end() && it->second >= 2) {
                err = m->name + " stalled twice while loading, so llmash will not start it again until the server "
                                "restarts; the log is in " + (fs::path(cfg_.root) / "logs").string();
                return nullptr;
            }
        }
        evict_for(need, m->name);
        auto       owned = std::make_unique<Instance>(*m, ctx, vision, &cfg_);
        owned->kv_type   = kv;
        owned->kv_on_gpu = p.kv_on_gpu;
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
            const bool stalled = inst->stalled;
            int        stalls  = 0;
            {
                std::lock_guard<std::mutex> lock(mu_);
                erase_ptr(live_, inst);
                if (stalled) {
                    stalls = ++stalls_[m->name];
                }
            }
            err = start_err;
            if (stalls >= 2) {
                err += ". It has stalled twice; llmash will not start it again until the server restarts";
            }
            return nullptr;
        }
        std::lock_guard<std::mutex> lock(mu_);
        stalls_.erase(m->name);
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
    if (const std::optional<Model> m = reg_->find(name)) {
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
            if (is_pinned(cfg_, p->model.name) || !p->ready() || now <= p->expires_at || now - p->last_used < 5) {
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

// The micro-batch a context can afford: the attention scratch grows with
// both, and at a million tokens 2048 wants 8 GB of it.
int ubatch_for(int ctx) {
    if (const int ub = env_int("LLMASH_UBATCH", 0); ub > 0) {
        return ub;
    }
    return ctx > 524288 ? 512 : ctx > 262144 ? 1024 : 2048;
}

double scratch_gb(int ctx) {
    return 0.5 + static_cast<double>(ubatch_for(ctx)) * ctx * 2.0 / static_cast<double>(1ull << 30);
}

Tuning auto_tune(int ctx) {
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

    if (tune_enabled("batch")) {
        int        ub = env_int("LLMASH_UBATCH", 0);
        int        b  = env_int("LLMASH_BATCH", 0);
        const auto fv = free_vram_gb();
        if (ub == 0 && fv.second && std::max(total_vram_gb(), fv.first) > 24) {
            ub = ubatch_for(ctx);
            b  = ub * 2;
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
