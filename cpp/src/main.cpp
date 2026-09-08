#include "cli_run.h"
#include "cmd_doctor.h"
#include "cmd_models.h"
#include "config.h"
#include "help.h"
#include "registry.h"
#include "tray.h"

#include <cstdio>
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
    return s.empty() ? "llmash" : s;
}

int print_help(const std::string & prog, const std::vector<std::string> & rest) {
    if (!rest.empty()) {
        if (const std::string * topic = command_help(rest[0])) {
            std::fputs(prog_text(*topic, prog).c_str(), stdout);
            return 0;
        }
    }
    std::fputs(prog_text(help_text(), prog).c_str(), stdout);
    return 0;
}

} // namespace

int main(int argc, char ** argv) {
    const std::string prog = prog_name(argc > 0 ? argv[0] : nullptr);

    std::vector<std::string> args;
    for (int i = 2; i < argc; i++) {
        args.emplace_back(argv[i]);
    }
    const std::string cmd = argc > 1 ? argv[1] : "";

    if (cmd.empty() || cmd == "help" || cmd == "-h" || cmd == "--help") {
        return print_help(prog, args);
    }
    if (cmd == "-v" || cmd == "--version") {
        std::printf("%s version is 0.4.0\n", prog.c_str());
        return 0;
    }

    Config cfg = load_config();

    try {
        if (cmd == "run") {
            return cmd_run(parse_run_args(args));
        }
        if (cmd == "doctor") {
            cmd_doctor();
            return 0;
        }
        if (cmd == "models") {
            cmd_models(args);
            return 0;
        }
        if (cmd == "tray") {
            std::string err;
            if (!cmd_tray(cfg, err)) {
                std::fprintf(stderr, "%s\n", err.c_str());
                return 1;
            }
            return 0;
        }

        for (const char * c : {"list", "ls", "ps", "show", "rm", "stop", "pull",
                               "install", "create", "cp", "push", "signin", "signout"}) {
            if (cmd == c) {
                std::fprintf(stderr,
                             "%s: not ported to the C++ build yet (cmds.go). Use the Go build for this one.\n",
                             cmd.c_str());
                return 1;
            }
        }
    } catch (const CliUsageError & e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    } catch (const CliExit & e) {
        return e.code;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }

    std::fprintf(stderr, "Error: unknown command \"%s\" for \"%s\"\n", cmd.c_str(), prog.c_str());
    return 1;
}
