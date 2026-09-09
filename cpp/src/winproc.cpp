#include "winproc.h"

#ifdef _WIN32
#include <windows.h>
#include <tlhelp32.h>
#else
#include <csignal>
#include <unistd.h>
#endif

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

namespace llmash {
namespace {

std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

#ifdef _WIN32
std::string to_utf8(const std::wstring & w) {
    if (w.empty()) {
        return "";
    }
    const int n = WideCharToMultiByte(CP_UTF8, 0, w.c_str(), static_cast<int>(w.size()), nullptr, 0, nullptr, nullptr);
    std::string s(static_cast<size_t>(n), '\0');
    WideCharToMultiByte(CP_UTF8, 0, w.c_str(), static_cast<int>(w.size()), s.data(), n, nullptr, nullptr);
    return s;
}
#endif

std::string image_path(unsigned long pid) {
#ifdef _WIN32
    HANDLE h = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, pid);
    if (h == nullptr) {
        return "";
    }
    wchar_t buf[MAX_PATH];
    DWORD   n  = MAX_PATH;
    const bool ok = QueryFullProcessImageNameW(h, 0, buf, &n) != FALSE;
    CloseHandle(h);
    return ok ? to_utf8(std::wstring(buf, n)) : std::string();
#else
    std::error_code ec;
    const fs::path  target = fs::read_symlink("/proc/" + std::to_string(pid) + "/exe", ec);
    return ec ? std::string() : target.string();
#endif
}

fs::path pid_path(const std::string & root) { return fs::path(root) / "cache" / "server.pid"; }

} // namespace

std::vector<RunningProcess> running_processes() {
    std::vector<RunningProcess> out;
#ifdef _WIN32
    HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    if (snap == INVALID_HANDLE_VALUE) {
        return out;
    }
    PROCESSENTRY32W e{};
    e.dwSize = sizeof(e);
    if (Process32FirstW(snap, &e)) {
        do {
            RunningProcess p;
            p.pid  = e.th32ProcessID;
            p.ppid = e.th32ParentProcessID;
            p.name = lower(to_utf8(e.szExeFile));
            out.push_back(std::move(p));
        } while (Process32NextW(snap, &e));
    }
    CloseHandle(snap);
#else
    std::error_code ec;
    for (auto it = fs::directory_iterator("/proc", ec); !ec && it != fs::directory_iterator(); ++it) {
        const std::string name = it->path().filename().string();
        if (name.empty() || !std::all_of(name.begin(), name.end(),
                                         [](unsigned char c) { return std::isdigit(c) != 0; })) {
            continue;
        }
        RunningProcess p;
        p.pid = std::stoul(name);

        // comm can hold spaces and brackets, so it is read to the last ')'
        std::ifstream st(it->path() / "stat");
        std::string   line;
        if (st && std::getline(st, line)) {
            const size_t open  = line.find('(');
            const size_t close = line.rfind(')');
            if (open != std::string::npos && close != std::string::npos && close > open) {
                p.name = lower(line.substr(open + 1, close - open - 1));
                std::istringstream rest(line.substr(close + 1));
                std::string        state;
                unsigned long      ppid = 0;
                if (rest >> state >> ppid) {
                    p.ppid = ppid;
                }
            }
        }
        out.push_back(std::move(p));
    }
#endif
    return out;
}

std::vector<RunningProcess> processes_under(const std::string & dir) {
    std::vector<RunningProcess> out;
    std::error_code             ec;
    const fs::path              want = fs::weakly_canonical(fs::path(dir), ec);
    if (ec || dir.empty()) {
        return out;
    }
    const std::string want_s = lower(want.generic_string());
    for (RunningProcess & p : running_processes()) {
        if (p.pid <= 4) {
            continue;
        }
        p.path = image_path(p.pid);
        if (p.path.empty()) {
            continue;
        }
        if (lower(fs::path(p.path).generic_string()).rfind(want_s, 0) == 0) {
            out.push_back(p);
        }
    }
    return out;
}

int kill_tree(unsigned long pid, const std::string & child_name) {
    int killed = 0;
    if (pid != 0 && !child_name.empty()) {
        for (const RunningProcess & p : running_processes()) {
            if (p.ppid == pid && p.name == child_name && kill_pid(p.pid)) {
                killed++;
            }
        }
    }
    if (kill_pid(pid)) {
        killed++;
    }
    return killed;
}

bool kill_pid(unsigned long pid) {
#ifdef _WIN32
    HANDLE h = OpenProcess(PROCESS_TERMINATE, FALSE, pid);
    if (h == nullptr) {
        return false;
    }
    const bool ok = TerminateProcess(h, 1) != FALSE;
    CloseHandle(h);
    return ok;
#else
    return ::kill(static_cast<pid_t>(pid), SIGKILL) == 0;
#endif
}

bool pid_alive(unsigned long pid, const std::string & expect_name) {
    if (pid == 0) {
        return false;
    }
#ifdef _WIN32
    HANDLE h = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, pid);
    if (h == nullptr) {
        return false;
    }
    DWORD      code = 0;
    const bool live = GetExitCodeProcess(h, &code) && code == STILL_ACTIVE;
    CloseHandle(h);
    if (!live) {
        return false;
    }
#else
    if (::kill(static_cast<pid_t>(pid), 0) != 0) {
        return false;
    }
#endif
    if (expect_name.empty()) {
        return true;
    }
    const std::string path = image_path(pid);
    if (path.empty()) {
        return false;
    }
    const std::string got = lower(fs::path(path).filename().string());
    const std::string want = lower(expect_name);
#ifdef _WIN32
    return got == want;
#else
    const size_t dot = want.rfind(".exe");
    return got == want || (dot != std::string::npos && got == want.substr(0, dot));
#endif
}

unsigned long read_pid_file(const std::string & root) {
    std::ifstream in(pid_path(root), std::ios::binary);
    unsigned long pid = 0;
    if (in) {
        in >> pid;
    }
    return pid;
}

void write_pid_file(const std::string & root, unsigned long pid) {
    std::error_code ec;
    fs::create_directories(pid_path(root).parent_path(), ec);
    std::ofstream out(pid_path(root), std::ios::binary | std::ios::trunc);
    if (out) {
        out << pid;
    }
}

void remove_pid_file(const std::string & root) {
    std::error_code ec;
    fs::remove(pid_path(root), ec);
}

} // namespace llmash
