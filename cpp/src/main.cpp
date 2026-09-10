#include "cli_commands.h"
#include "cli_console.h"
#include "cli_http.h"
#include "cli_run.h"
#include "cmd_doctor.h"
#include "cmd_models.h"
#include "config.h"
#include "help.h"
#include "platform.h"
#include "launch.h"
#include "registry.h"
#include "serve.h"
#include "tray.h"
#include "update.h"
#include "version.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace llmash;

namespace {

// argv[0] is whatever name the user typed, and the help text is re-addressed
// to it: installed as both `llmash` and `ollama`.
std::string prog_name(const char * argv0) {
    std::string s = argv0 ? argv0 : "llmash";
    const size_t slash = s.find_last_of("\\/");
    if (slash != std::string::npos) {
        s = s.substr(slash + 1);
    }
    const size_t dot = s.rfind('.');
    if (dot != std::string::npos) {
        s = s.substr(0, dot);
    }
    for (char & c : s) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return s.empty() ? "llmash" : s;
}

// llmashw.exe is the same program without a console.
bool windowed_exe() {
#ifndef _WIN32
    return false;
#else
    wchar_t buf[MAX_PATH];
    const DWORD n = GetModuleFileNameW(nullptr, buf, MAX_PATH);
    if (n == 0 || n == MAX_PATH) {
        return false;
    }
    std::wstring s(buf, n);
    const size_t slash = s.find_last_of(L"\\/");
    if (slash != std::wstring::npos) {
        s = s.substr(slash + 1);
    }
    const size_t dot = s.rfind(L'.');
    if (dot != std::wstring::npos) {
        s = s.substr(0, dot);
    }
    return !s.empty() && (s.back() == L'w' || s.back() == L'W');
#endif
}

#ifdef _WIN32
UINT g_saved_cp = 0;
#endif

void console_setup() {
#ifdef _WIN32
    for (DWORD h : {STD_OUTPUT_HANDLE, STD_ERROR_HANDLE}) {
        DWORD mode = 0;
        if (GetConsoleMode(GetStdHandle(h), &mode)) {
            SetConsoleMode(GetStdHandle(h), mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
        }
    }
    if (is_console_stdout()) {
        g_saved_cp = GetConsoleOutputCP();
        SetConsoleOutputCP(CP_UTF8);
    }
#endif
}

void console_restore() {
#ifdef _WIN32
    if (g_saved_cp != 0) {
        SetConsoleOutputCP(g_saved_cp);
    }
#endif
}

int print_help(const std::string & prog, const std::string & topic) {
    if (topic.empty()) {
        std::fputs(prog_text(help_text(), prog).c_str(), stdout);
        return 0;
    }
    if (const std::string * h = command_help(topic)) {
        std::fputs(prog_text(*h, prog).c_str(), stdout);
        return 0;
    }
    std::fprintf(stderr, "Unknown help topic [`%s`]\n", topic.c_str());
    std::fputs(help_text().c_str(), stderr);
    return 1;
}

std::string alias_of(const std::string & cmd) {
    if (cmd == "ls") {
        return "list";
    }
    if (cmd == "start") {
        return "serve";
    }
    if (cmd == "set") {
        return "models";
    }
    return cmd;
}

int dispatch(const std::string & prog, const std::string & cmd, const std::vector<std::string> & args,
             bool verbose) {
    Config cfg = load_config();

    if (cmd == "tray") {
        if (windowed_exe()) {
            return tray_main(cfg);
        }
        std::string err;
        if (!cmd_tray(cfg, err)) {
            std::fprintf(stderr, "%s\n", err.c_str());
            return 1;
        }
        return 0;
    }
    if (cmd == "run") {
        RunArgs r = parse_run_args(args);
        r.verbose = r.verbose || verbose;
        return cmd_run(r);
    }
    if (cmd == "doctor") {
        cmd_doctor();
        return 0;
    }
    if (cmd == "models") {
        cmd_models(args);
        return 0;
    }
    if (cmd == "update") {
        return cmd_update(args, cfg);
    }
    if (cmd == "launch") {
        cmd_launch(cfg, args);
    }

    Registry reg(cfg);
    if (cmd == "pulldraft") {
        return cmd_pulldraft(args, reg);
    }
    if (cmd == "link")      { return cmd_link(args, cfg); }
    if (cmd == "unlink")    { return cmd_unlink(); }
    if (cmd == "uninstall") { return cmd_uninstall(args, cfg); }
    if (cmd == "push")      { return cmd_push(); }
    if (cmd == "signin")    { return cmd_signin(); }
    if (cmd == "signout")   { return cmd_signout(); }

    ApiClient api(cfg);
    if (cmd == "list")                     { return cmd_list(args, api, cfg); }
    if (cmd == "ps")                       { return cmd_ps(args, api, cfg); }
    if (cmd == "show")                     { return cmd_show(args, api); }
    if (cmd == "rm")                       { return cmd_rm(args, api); }
    if (cmd == "stop")                     { return cmd_stop(args, api); }
    if (cmd == "pull" || cmd == "install") { return cmd_pull(args, api); }
    if (cmd == "create")                   { return cmd_create(args, api); }
    if (cmd == "cp")                       { return cmd_cp(args, api); }

    std::fprintf(stderr, "Error: unknown command \"%s\" for \"%s\"\n", cmd.c_str(), prog.c_str());
    return 1;
}

} // namespace

int main(int argc, char ** argv) {
    std::vector<std::string> argv_all;
    for (int i = 1; i < argc; i++) {
        argv_all.emplace_back(argv[i]);
    }

    // The server is this same program, and needs none of the console setup.
    if (!argv_all.empty() && (argv_all[0] == "serve" || argv_all[0] == "start")) {
        return cmd_serve(std::vector<std::string>(argv_all.begin() + 1, argv_all.end()));
    }

    console_setup();
    const std::string prog = prog_name(argc > 0 ? argv[0] : nullptr);
    if (std::getenv("LLMASH_PROG") == nullptr) {
        set_env("LLMASH_PROG", prog);
    }

    // cobra accepts the global flags before the subcommand; peel them off.
    bool   verbose = false;
    size_t at      = 0;
    while (at < argv_all.size() && (argv_all[at] == "--verbose" || argv_all[at] == "--nowordwrap")) {
        verbose = verbose || argv_all[at] == "--verbose";
        at++;
    }
    std::vector<std::string> rest(argv_all.begin() + static_cast<long long>(at), argv_all.end());

    int code = 0;
    try {
        if (rest.empty()) {
            launch_menu(load_config());
        }
        const std::string first = rest[0];
        if (first == "-v" || first == "--version") {
            std::printf("%s version is %s\n", prog.c_str(), version_string(load_config()).c_str());
            console_restore();
            return 0;
        }
        if (first == "-h" || first == "--help") {
            code = print_help(prog, "");
            console_restore();
            return code;
        }
        const std::string cmd = alias_of(first);
        std::vector<std::string> args(rest.begin() + 1, rest.end());
        if (cmd == "help") {
            code = print_help(prog, args.empty() ? "" : alias_of(args[0]));
            console_restore();
            return code;
        }
        if (command_help(cmd) == nullptr) {
            std::fprintf(stderr, "Error: unknown command \"%s\" for \"%s\"\n", first.c_str(), prog.c_str());
            console_restore();
            return 1;
        }
        // -h/--help before any `--` separator prints the cobra-style help.
        for (const std::string & t : args) {
            if (t == "--") {
                break;
            }
            if (t == "-h" || t == "--help") {
                std::fputs(prog_text(*command_help(cmd), prog).c_str(), stdout);
                console_restore();
                return 0;
            }
        }
        code = dispatch(prog, cmd, args, verbose);
    } catch (const CliUsageError & e) {
        std::fprintf(stderr, "%s\n", e.what());
        code = 1;
    } catch (const CliExit & e) {
        code = e.code;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "%s\n", e.what());
        code = 1;
    }
    console_restore();
    return code;
}
