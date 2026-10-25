#pragma once

#include "config.h"

#include <nlohmann/json.hpp>

#include <map>
#include <string>
#include <vector>

namespace llmash {

struct Integration {
    std::string title;
    std::string exe;
};

// The executable `exe` in one of `dirs`, PATHEXT applied on Windows; "" when none holds it.
std::string which_in(const std::string & exe, const std::vector<std::string> & dirs);

// The folders installers put a per-user tool in without touching PATH, Claude Code's own installer among them.
std::vector<std::string> usual_tool_dirs();

// A tool to launch wherever it was installed: this process's PATH, then PATH as saved now (a tool installed after the
// terminal opened), then the usual install folders.
std::string find_tool(const std::string & exe);

// The integrations `llmash launch <name>` knows, keyed the way the command
// line names them, and the older/alternate names that resolve to one of them.
const std::map<std::string, Integration> & integrations();
const std::map<std::string, std::string> & integration_aliases();

// "" if name is neither an integration key nor a known alias for one.
std::string resolve_integration(const std::string & name);

// integrations.json beside the program: the model saved per integration by
// `llmash launch <name> --model <m>`.
std::string  profile_file(const Config & cfg);
nlohmann::json load_profiles(const Config & cfg);
void         save_profiles(const Config & cfg, const nlohmann::json & profiles);
std::string  profile_model(const nlohmann::json & profiles, const std::string & key);

// Parsed form of `llmash launch [name] [--model M] [--config] [--restore]
// [-y|--yes] [-- extra...]`.
struct LaunchArgs {
    std::string name;
    std::string model;
    bool        config  = false;
    bool        restore = false;
    std::vector<std::string> extra;
};

// Sets `error` and returns a default-constructed LaunchArgs on a bad flag or
// a flag combination that needs a name none was given; never throws.
LaunchArgs parse_launch_args(const std::vector<std::string> & args, std::string & error);

// OLLAMA_HOST, defaulted and normalized exactly like the Go CLI's `host`.
std::string ollama_host();

// The env vars an OpenAI-compatible CLI needs to talk to this server; each
// entry is "NAME=value", ready to hand to a child process.
std::vector<std::string> generic_launch_env(const std::string & host, const std::string & model);

// env with Claude Code pointed at this server: no Anthropic key, every model slot set to model, ctx its window.
std::vector<std::string> claude_launch_env(std::vector<std::string> env, const std::string & host, const std::string & model,
                                           int ctx);

[[noreturn]] void launch_claude(const Config & cfg, const std::string & model, const std::vector<std::string> & extra);
[[noreturn]] void launch_generic(const std::string & key, const std::string & model,
                                  const std::vector<std::string> & extra);

// Bare `llmash launch`: an interactive picker on a real console, the help
// text otherwise. Does not return.
[[noreturn]] void launch_menu(const Config & cfg);

// The full `llmash launch [args...]` command. Does not return.
[[noreturn]] void cmd_launch(const Config & cfg, const std::vector<std::string> & args);

} // namespace llmash
