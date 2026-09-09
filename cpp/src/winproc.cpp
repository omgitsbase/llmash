#include "winproc.h"

#include <windows.h>

#include <tlhelp32.h>

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

std::string to_utf8(const std::wstring & w) {
    if (w.empty()) {
        return "";
    }
    const int n = WideCharToMultiByte(CP_UTF8, 0, w.c_str(), static_cast<int>(w.size()), nullptr, 0, nullptr, nullptr);
    std::string s(static_cast<size_t>(n), '\0');
    WideCharToMultiByte(CP_UTF8, 0, w.c_str(), static_cast<int>(w.size()), s.data(), n, nullptr, nullptr);
    return s;
}

std::string image_path(unsigned long pid) {
    HANDLE h = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, pid);
    if (h == nullptr) {
        return "";
    }
    wchar_t buf[MAX_PATH];
    DWORD   n  = MAX_PATH;
    const bool ok = QueryFullProcessImageNameW(h, 0, buf, &n) != FALSE;
    CloseHandle(h);
    return ok ? to_utf8(std::wstring(buf, n)) : std::string();
}

fs::path pid_path(const std::string & root) { return fs::path(root) / "cache" / "server.pid"; }

} // namespace

std::vector<RunningProcess> running_processes() {
    std::vector<RunningProcess> out;
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
    return out;
}

std::vector<RunningProcess> processes_under(const std::string & dir) {
    std::vector<RunningProcess> out;
    std::error_code             ec;
    const fs::path              want = fs::weakly_canonical(fs::path(dir), ec);
    if (ec || dir.empty()) {
        return out;
    }
    const std::string want_s = lower(want.string());
    for (RunningProcess & p : running_processes()) {
        if (p.pid <= 4) {
            continue;
        }
        p.path = image_path(p.pid);
        if (p.path.empty()) {
            continue;
        }
        const std::string got = lower(p.path);
        if (got.rfind(want_s, 0) == 0) {
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
    HANDLE h = OpenProcess(PROCESS_TERMINATE, FALSE, pid);
    if (h == nullptr) {
        return false;
    }
    const bool ok = TerminateProcess(h, 1) != FALSE;
    CloseHandle(h);
    return ok;
}

bool pid_alive(unsigned long pid, const std::string & expect_name) {
    if (pid == 0) {
        return false;
    }
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
    if (expect_name.empty()) {
        return true;
    }
    // an id can be reused, so the image has to agree
    const std::string path = image_path(pid);
    return !path.empty() && lower(fs::path(path).filename().string()) == lower(expect_name);
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
