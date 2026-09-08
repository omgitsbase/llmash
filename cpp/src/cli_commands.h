#pragma once

// Port of cmd/llmash/cmds.go: the CLI-side implementation of list, ps, show,
// pull, rm, cp, stop, pulldraft, create, push, link, unlink, uninstall,
// signin and signout. Every command talks to the local server named by
// OLLAMA_HOST/LLMASH_PORT over the same /api and /cli routes cmd/llmash's
// Go server (and this port's own, elsewhere in cpp/) both serve; none of
// them touch llama.cpp or a model file directly.
//
// What's deliberately NOT a pixel-for-pixel port, and why:
//   - console.go/progress.go's animated redraw (ANSI cursor moves, ticking
//     spinners/bars) is UI presentation, not cmds.go logic, and isn't part
//     of this file in the Go tree either. pull's progress is reported here
//     as one printed line per status/bar change instead of an in-place
//     redraw -- same information, simpler renderer. The key-handling state
//     machines those files also own (read_pick, ask_number_feed) ARE ported
//     faithfully, in cli_console.h, since cmds.go calls them directly.
//   - draft_install.go's cmdPullDraft depends on draft.go/draft_verify.go
//     (Hugging Face drafter search + GGUF spec matching), a separate
//     subsystem this job was not given. cmd_pulldraft below ports the
//     surrounding flow byte-for-byte (already-has-a-drafter checks, confirm
//     prompt, messages) but the candidate search itself is a stub that
//     always reports none found -- see find_drafter_candidates.

#include "cli_http.h"
#include "cli_run.h"
#include "config.h"
#include "registry.h"

#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <vector>

namespace llmash {

// Thrown by die()/cli_exit() to unwind to a command's own entry point,
// standing in for Go's os.Exit(code): unlike os.Exit this runs destructors
// on the way out, which is what we want for RAII resources (never bare
// exit()/abort() from inside this module).

[[noreturn]] void die(const std::string & message);

// ------------------------------------------------------------ arg parsing

struct ParsedArgs {
    std::vector<std::string>          pos;
    std::vector<std::string>          bool_flags_set;
    std::vector<std::pair<std::string, std::string>> vals;

    bool        has_flag(const std::string & name) const;
    bool        has_val(const std::string & name) const;
    std::string val(const std::string & name) const;
};

// Flags in any position, `--x=v` or `--x v`; anything else starting with a
// dash is `Error: unknown flag: '<token>'` (main.go's parseSimple).
ParsedArgs parse_simple(const std::vector<std::string> & args, const std::vector<std::string> & bool_flags,
                        const std::vector<std::string> & val_flags);

// -------------------------------------------------------- pure logic units

struct QuantInfo {
    std::string name;
    int64_t     size  = 0;
    int         files = 0;
};

struct Tiers {
    int tiny = 0, medium = 0, large = 0;
};

double bits_of(const std::string & quant_name);
Tiers  tiers_of(const std::vector<QuantInfo> & quants);
// The quantisation tag embedded in a GGUF file name (pull.go's quantTag),
// e.g. "Qwen3-8B-Q4_K_M-00001-of-00002.gguf" -> "Q4_K_M".
std::string quant_tag(const std::string & file_name);

bool is_hf_ref(const std::string & model_ref);

// Go's url.QueryEscape: space becomes '+', not %20.
std::string url_query_escape(const std::string & s);

// crypto/rand + base64.RawURLEncoding, ported for link_key's bearer token.
std::string base64_url_encode(const unsigned char * data, size_t len);

// filepath.Rel(dir, path) having no ".." prefix: is `path` at or under `dir`.
bool path_under(const std::string & path, const std::string & dir);

// net.DialTimeout("tcp", 127.0.0.1:port, 600ms): is something listening.
bool port_open(int port);

// filterRows: the rows of tags()/ps()'s "models" array whose name has
// `prefix` (case-folded when fold is true, the way `list` folds and `ps`
// does not).
std::vector<nlohmann::json> filter_rows(const nlohmann::json & doc, const std::string & prefix, bool fold);

std::string render_list(const std::vector<nlohmann::json> & rows);
std::string render_ps(const std::vector<nlohmann::json> & rows);

// A long value list cut to what fits a narrow column, Go-slice-literal
// styled ("[a b ...+3 more]"), for `show -v`'s Metadata table.
std::string elide(const std::vector<std::string> & values, int target);

void show_info(const nlohmann::json & resp, bool verbose, std::string & out);

// ---------------------------------------------------------------- commands

int cmd_list(const std::vector<std::string> & args, ApiClient & api, const Config & cfg);
int cmd_ps(const std::vector<std::string> & args, ApiClient & api, const Config & cfg);
int cmd_show(const std::vector<std::string> & args, ApiClient & api);
int cmd_rm(const std::vector<std::string> & args, ApiClient & api);
int cmd_stop(const std::vector<std::string> & args, ApiClient & api);
int cmd_pull(const std::vector<std::string> & args, ApiClient & api);
int cmd_create(const std::vector<std::string> & args, ApiClient & api);
int cmd_cp(const std::vector<std::string> & args, ApiClient & api);
int cmd_push();
int cmd_signin();
int cmd_signout();
int cmd_link(const std::vector<std::string> & args, const Config & cfg);
int cmd_unlink();
int cmd_uninstall(const std::vector<std::string> & args, const Config & cfg);
int cmd_pulldraft(const std::vector<std::string> & args, Registry & reg);

// The name main() was invoked as ("llmash", "ollama", a shim's own name),
// substituted into every message above that carries `prog`. Defaults to
// "llmash"; main.cpp owns detecting the real one (LLMASH_PROG / argv[0]'s
// basename), same as Go's main.go init() does today.
void               set_prog(const std::string & prog);
const std::string & prog();

// One entry point for main.cpp to route a parsed command line through:
// dispatches to the matching cmd_* above (constructing its own ApiClient
// from cfg), or returns false for a name none of these commands own.
bool dispatch(const std::string & cmd, const std::vector<std::string> & args, Config & cfg, Registry & reg,
              int & exit_code);

} // namespace llmash
