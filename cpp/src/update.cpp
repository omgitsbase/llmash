#include "update.h"

#include "cli_util.h"
#include "cli_run.h"
#include "platform.h"
#include "pull.h"
#include "version.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/file.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

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
#ifdef _WIN32
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
#else
        fd_ = ::open("/tmp/llmash-update.lock", O_CREAT | O_RDWR, 0644);
        if (fd_ < 0) {
            held_ = true;
            return;
        }
        held_ = ::flock(fd_, LOCK_EX | LOCK_NB) == 0;
#endif
    }
    ~UpdateLock() {
#ifdef _WIN32
        if (h_ != nullptr) {
            if (held_) {
                ReleaseMutex(h_);
            }
            CloseHandle(h_);
        }
#else
        if (fd_ >= 0) {
            if (held_) {
                ::flock(fd_, LOCK_UN);
            }
            ::close(fd_);
        }
#endif
    }
    bool held() const { return held_; }

private:
#ifdef _WIN32
    HANDLE h_ = nullptr;
#else
    int fd_ = -1;
#endif
    bool held_ = false;
};

#ifdef _WIN32
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
#endif

const char * installer_name() {
#ifdef _WIN32
    return "install.ps1";
#else
    return "install.sh";
#endif
}

// Runs the installer that ships beside the program, on our console, and
// waits for it. The installer fetches the release itself.
int run_installer(const std::string & script, const std::string & root) {
#ifdef _WIN32
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
#else
    const pid_t child = ::fork();
    if (child < 0) {
        return -1;
    }
    if (child == 0) {
        std::string s = script, d = root;
        char * argv[] = {const_cast<char *>("sh"), s.data(), const_cast<char *>("--dir"), d.data(),
                         const_cast<char *>("--yes"), nullptr};
        ::execvp("sh", argv);
        ::_exit(127);
    }
    int status = 0;
    if (::waitpid(child, &status, 0) < 0) {
        return -1;
    }
    return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
#endif
}

int die(const std::string & message) {
    std::fprintf(stderr, "%s\n", message.c_str());
    return 1;
}

// The installer that came with this install, wherever the layout put it.
//
// `cfg.root` is the folder holding the running executable, and the installer
// puts a second copy of the program in `<root>/bin` so one shim is on PATH.
// Run from PATH, root is therefore `...\llmash\bin` while install.ps1 sits in
// its parent, so the old single check never found it and `update` told every
// user to go and paste the one-liner instead.
std::string find_installer(const Config & cfg) {
    std::error_code ec;
    const fs::path  root = fs::path(cfg.root);
    for (const fs::path & dir : {root, root.parent_path()}) {
        if (dir.empty()) {
            continue;
        }
        const fs::path p = dir / installer_name();
        if (fs::is_regular_file(p, ec)) {
            return p.string();
        }
    }
    return "";
}

// Failing that, fetch it from the release being installed. An install that
// cannot update itself is worse than one that downloads 30 KB to do it.
std::string fetch_installer(const clidoc::Release & rel) {
    std::string url;
    for (const auto & a : rel.assets) {
        if (a.first == installer_name()) {
            url = a.second;
            break;
        }
    }
    if (url.empty()) {
        return "";
    }
    std::error_code ec;
    const fs::path  dest = fs::temp_directory_path(ec) / (std::string("llmash-") + installer_name());
    std::string     err;
    if (!fetch_blob(url, dest.string(), 0, [](int64_t) {}, err)) {
        return "";
    }
    return fs::is_regular_file(dest, ec) ? dest.string() : "";
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

    std::string script = find_installer(cfg);
    if (script.empty()) {
        std::printf("fetching the installer for %s\n", there.c_str());
        std::fflush(stdout);
        script = fetch_installer(rel);
    }
    if (script.empty()) {
#ifdef _WIN32
        std::printf("could not get the installer; run this instead:\n\n"
                    "  irm https://raw.githubusercontent.com/%s/main/install.ps1 | iex\n",
                    clidoc::repo_slug().c_str());
#else
        std::printf("could not get the installer; run this instead:\n\n"
                    "  curl -fsSL https://raw.githubusercontent.com/%s/main/install.sh | sh\n",
                    clidoc::repo_slug().c_str());
#endif
        return 1;
    }

    // Install where this install lives, which is the folder holding the
    // installer, not `cfg.root` — from PATH that is the `bin` copy, and
    // passing it would lay a second install down inside the first.
    const std::string target = fs::path(script).parent_path().string();
    const std::string where  = fs::is_regular_file(fs::path(target) / "llmash.exe", ec)
                                   ? target
                                   : fs::path(cfg.root).parent_path().string();
    std::printf("updating %s to %s\n\n", where.c_str(), there.c_str());
    std::fflush(stdout);
    const int code = run_installer(script, where);
    if (code != 0) {
        return die("the installer stopped: exit " + std::to_string(code));
    }
    return 0;
}

} // namespace llmash
