#include "update.h"

#include "cli_util.h"
#include "cli_run.h"
#include "version.h"

#include <windows.h>

#include <cstdio>
#include <filesystem>
#include <string>

namespace fs = std::filesystem;

namespace llmash {
namespace {

// A machine-wide lock so two updates cannot race each other.
class UpdateLock {
public:
    UpdateLock() {
        h_ = CreateMutexW(nullptr, TRUE, L"Local\\llmash-update");
        if (h_ == nullptr) {
            held_ = true; // no mutex available: do not block the update
            return;
        }
        if (GetLastError() == ERROR_ALREADY_EXISTS) {
            held_ = WaitForSingleObject(h_, 2000) == WAIT_OBJECT_0;
        } else {
            held_ = true;
        }
    }
    ~UpdateLock() {
        if (h_ != nullptr) {
            if (held_) {
                ReleaseMutex(h_);
            }
            CloseHandle(h_);
        }
    }
    bool held() const { return held_; }

private:
    HANDLE h_    = nullptr;
    bool   held_ = false;
};

std::wstring widen(const std::string & s) {
    if (s.empty()) {
        return L"";
    }
    const int    n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, nullptr, 0);
    std::wstring w(static_cast<size_t>(n), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, w.data(), n);
    w.resize(static_cast<size_t>(n - 1));
    return w;
}

// Runs the installer that ships beside the program, on our console, and
// waits for it. The installer fetches the release itself.
int run_installer(const std::string & script, const std::string & root) {
    std::wstring cmd = L"powershell -NoProfile -ExecutionPolicy RemoteSigned -File \"" + widen(script) +
                       L"\" -Dir \"" + widen(root) + L"\" -Yes";
    STARTUPINFOW        si{};
    si.cb = sizeof(si);
    PROCESS_INFORMATION pi{};
    if (!CreateProcessW(nullptr, cmd.data(), nullptr, nullptr, TRUE, 0, nullptr, nullptr, &si, &pi)) {
        return -1;
    }
    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD code = 1;
    GetExitCodeProcess(pi.hProcess, &code);
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    return static_cast<int>(code);
}

int die(const std::string & message) {
    std::fprintf(stderr, "%s\n", message.c_str());
    return 1;
}

} // namespace

int cmd_update(const std::vector<std::string> & args, const Config & cfg) {
    bool force = false;
    for (const std::string & a : args) {
        if (a == "--force") {
            force = true;
        }
    }
    const std::string prog = clidoc::prog_name();

    UpdateLock lock;
    if (!lock.held()) {
        return die("another " + prog + " update is already running on this machine");
    }

    std::error_code ec;
    if (fs::is_regular_file(fs::path(cfg.root) / "build.py", ec)) {
        std::printf("this is a source checkout at %s, not an install.\nUpdate it with:  python build.py --here\n",
                    cfg.root.c_str());
        return 1;
    }

    clidoc::Release rel;
    std::string     err;
    if (!clidoc::latest_release(clidoc::repo_slug(), rel, err)) {
        return die("could not reach GitHub: " + err);
    }
    const std::string here  = version_string(cfg);
    const std::string there = clidoc::release_version(rel.tag);
    std::printf("installed %s, %s has %s\n", here.c_str(), clidoc::repo_slug().c_str(), there.c_str());
    if (here == there && !force) {
        std::printf("already up to date\n");
        return 0;
    }

    const std::string script = (fs::path(cfg.root) / "install.ps1").string();
    if (!fs::is_regular_file(script, ec)) {
        std::printf("the installer is not beside the program; run this instead:\n\n"
                    "  irm https://raw.githubusercontent.com/%s/main/install.ps1 | iex\n",
                    clidoc::repo_slug().c_str());
        return 1;
    }

    std::printf("updating %s to %s\n\n", cfg.root.c_str(), there.c_str());
    std::fflush(stdout);
    const int code = run_installer(script, cfg.root);
    if (code != 0) {
        return die("the installer stopped: exit " + std::to_string(code));
    }
    return 0;
}

} // namespace llmash
