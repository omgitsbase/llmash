#include "runtime_host.h"

#include "platform.h"

#ifdef _WIN32
#include <windows.h>
#include <shellapi.h>
#endif

#include <cstdio>
#include <filesystem>

namespace fs = std::filesystem;

namespace llmash {
namespace {

#ifdef _WIN32
// int llama_server(int, char **)
constexpr const char * kServerEntry = "?llama_server@@YAHHPEAPEAD@Z";
constexpr const wchar_t * kServerLibrary = L"llama-server-impl.dll";

std::string narrow(const std::wstring & s) {
    if (s.empty()) {
        return "";
    }
    const int n = WideCharToMultiByte(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), nullptr, 0, nullptr, nullptr);
    std::string out(static_cast<size_t>(n), '\0');
    WideCharToMultiByte(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), out.data(), n, nullptr, nullptr);
    return out;
}

std::wstring quote(const std::wstring & arg) {
    if (!arg.empty() && arg.find_first_of(L" \t\n\v\"") == std::wstring::npos) {
        return arg;
    }
    std::wstring out = L"\"";
    for (auto it = arg.begin();; ++it) {
        size_t backslashes = 0;
        while (it != arg.end() && *it == L'\\') {
            ++it;
            ++backslashes;
        }
        if (it == arg.end()) {
            out.append(backslashes * 2, L'\\');
            break;
        }
        out.append(*it == L'"' ? backslashes * 2 + 1 : backslashes, L'\\');
        out.push_back(*it);
    }
    out.push_back(L'"');
    return out;
}

std::wstring marker_name(unsigned long pid) { return L"Local\\llmash-runtime-" + std::to_wstring(pid); }

std::wstring own_program() {
    std::wstring buf(MAX_PATH, L'\0');
    for (;;) {
        const DWORD n = GetModuleFileNameW(nullptr, buf.data(), static_cast<DWORD>(buf.size()));
        if (n == 0) {
            return L"";
        }
        if (n < buf.size()) {
            buf.resize(n);
            return buf;
        }
        buf.resize(buf.size() * 2);
    }
}

// the job closes with this process, so terminating the host stops the server too
int run_tied(const std::vector<std::wstring> & wargs) {
    std::wstring cmdline;
    for (const auto & a : wargs) {
        cmdline += (cmdline.empty() ? L"" : L" ") + quote(a);
    }
    HANDLE job = CreateJobObjectW(nullptr, nullptr);
    if (job != nullptr) {
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION lim{};
        lim.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        SetInformationJobObject(job, JobObjectExtendedLimitInformation, &lim, sizeof(lim));
    }
    STARTUPINFOW si{};
    si.cb = sizeof(si);
    si.dwFlags    = STARTF_USESTDHANDLES;
    si.hStdInput  = GetStdHandle(STD_INPUT_HANDLE);
    si.hStdOutput = GetStdHandle(STD_OUTPUT_HANDLE);
    si.hStdError  = GetStdHandle(STD_ERROR_HANDLE);
    PROCESS_INFORMATION pi{};
    if (!CreateProcessW(wargs[0].c_str(), cmdline.data(), nullptr, nullptr, TRUE, CREATE_SUSPENDED | CREATE_NO_WINDOW,
                        nullptr, nullptr, &si, &pi)) {
        std::fprintf(stderr, "llmash: could not start %s (error %lu)\n", narrow(wargs[0]).c_str(), GetLastError());
        return 1;
    }
    if (job != nullptr) {
        AssignProcessToJobObject(job, pi.hProcess);
    }
    ResumeThread(pi.hThread);
    CloseHandle(pi.hThread);
    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD code = 1;
    GetExitCodeProcess(pi.hProcess, &code);
    CloseHandle(pi.hProcess);
    return static_cast<int>(code);
}
#endif

} // namespace

std::vector<std::string> hosted_command(const std::vector<std::string> & argv) {
#ifdef _WIN32
    const std::wstring self = own_program();
    if (argv.empty() || self.empty()) {
        return argv;
    }
    std::vector<std::string> out{narrow(self), kRuntimeHostArg};
    out.insert(out.end(), argv.begin(), argv.end());
    return out;
#else
    return argv;
#endif
}

int runtime_host_main() {
#ifdef _WIN32
    int      wargc = 0;
    LPWSTR * wargv = CommandLineToArgvW(GetCommandLineW(), &wargc);
    if (wargv == nullptr || wargc < 3) {
        std::fputs("usage: llmashw runtime <llama-server.exe> [arguments...]\n", stderr);
        return 2;
    }
    std::vector<std::wstring> wargs(wargv + 2, wargv + wargc);
    LocalFree(wargv);

    HANDLE marker = CreateEventW(nullptr, TRUE, FALSE, marker_name(GetCurrentProcessId()).c_str());
    (void) marker;  // held until the process ends

    const fs::path dir = fs::path(wargs[0]).parent_path();
    SetDllDirectoryW(dir.c_str());
    SetCurrentDirectoryW(dir.c_str());  // where a build that loads its backends at run time looks
    HMODULE lib   = LoadLibraryExW((dir / kServerLibrary).c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    using entry_t = int (*)(int, char **);
    const auto entry = lib != nullptr ? reinterpret_cast<entry_t>(GetProcAddress(lib, kServerEntry)) : nullptr;
    if (entry == nullptr) {
        return run_tied(wargs);
    }

    // llama.cpp only rebuilds UTF-8 argv itself when argc matches the real command line
    std::vector<std::string> args;
    args.reserve(wargs.size());
    for (const auto & w : wargs) {
        args.push_back(narrow(w));
    }
    std::vector<char *> ptrs;
    ptrs.reserve(args.size() + 1);
    for (auto & a : args) {
        ptrs.push_back(a.data());
    }
    ptrs.push_back(nullptr);
    return entry(static_cast<int>(args.size()), ptrs.data());
#else
    std::fputs("llmash runtime is only used on Windows\n", stderr);
    return 2;
#endif
}

bool is_runtime_host(unsigned long pid) {
#ifdef _WIN32
    HANDLE h = OpenEventW(SYNCHRONIZE, FALSE, marker_name(pid).c_str());
    if (h == nullptr) {
        return false;
    }
    CloseHandle(h);
    return true;
#else
    (void) pid;
    return false;
#endif
}

std::vector<std::string> runtime_process_names() {
#ifdef _WIN32
    return {llama_server_exe(), llmash_daemon_exe(), llmash_exe()};
#else
    return {llama_server_exe()};
#endif
}

} // namespace llmash
