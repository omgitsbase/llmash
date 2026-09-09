#pragma once

// `llmash run`: the interactive chat client.

#include <nlohmann/json.hpp>

#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace llmash {

// ------------------------------------------------------- command-line flags

// Same fields as Go's runOpts (main.go's parseRun).
struct RunArgs {
    std::string model;
    std::string prompt;
    std::string format;
    std::string keepalive;
    std::string think;
    bool        think_set    = false;
    bool        verbose      = false;
    bool        nowordwrap   = false;
    bool        hidethinking = false;
    int         ctx          = 0;
    int         dimensions   = 0;
    std::optional<bool>   truncate;
    std::optional<double> temperature;
};

// Raised for anything cobra would have rejected with a die() and exit(1): an
// unknown flag, a flag missing its argument, an invalid --think level, no
// model argument.
struct CliUsageError : std::runtime_error {
    explicit CliUsageError(const std::string & msg) : std::runtime_error(msg) {}
};

// Mirrors main.go's parseRun. `args` excludes the "run" word itself, matching
// cobra's rest := argv[1:].
RunArgs parse_run_args(const std::vector<std::string> & args);

// ----------------------------------------------------------- pure helpers

// The format flag on the wire: nothing, "json", or a parsed JSON schema
// object (falling back to the raw string if it doesn't parse).
nlohmann::json format_value(const std::string & f);

// The value types ollama's api.FormatParams gives a `/set parameter`.
// Throws CliUsageError on a value that doesn't fit the parameter's type.
nlohmann::json format_param(const std::string & key, const std::vector<std::string> & vals);

// Terminal cell width: two columns for the wide CJK/emoji blocks, zero for
// combining marks (a pragmatic subset of Unicode's East Asian Width table).
int disp_width(const std::string & utf8);

struct DisplayState {
    int         line_length = 0;
    std::string word_buffer;
};

// Streams `content` to stdout, wrapping at the terminal width when word_wrap
// is set.
void display_response(const std::string & content, bool word_wrap, DisplayState & state);

struct FileExtraction {
    std::string              text;
    std::vector<std::string> images_b64;
    bool                     ok = true; // false if a referenced file could not be read
};

// A path with the shell's own backslash-escaping undone (`\ ` -> ` `, etc).
std::string normalize_file_path(const std::string & fp);

// Splits a prompt into its text and the base64 of any image or audio files
// named in it (a bare path ending in .jpg/.jpeg/.png/.webp/.wav).
FileExtraction extract_file_data(const std::string & input);

std::string encode_base64(const std::string & bytes);

// ------------------------------------------------------------ run options

using Message = nlohmann::json; // {"role":.., "content":.., ...}, as in Go's `message map[string]any`

// Mirrors Go's runOptions. think/keep_alive/options carry `any`-typed wire
// values, so they stay as json rather than a fixed C++ type.
struct RunOptions {
    std::string           model, parent_model;
    std::string           system;
    std::string           prompt;
    std::vector<Message>  messages;
    std::vector<Message>  loaded_messages;
    nlohmann::json         options = nlohmann::json::object();
    std::string           format;
    nlohmann::json         think;      // null, bool, or a level string
    bool                   hide_thinking = false;
    nlohmann::json         keep_alive; // null, string, or number
    bool                   word_wrap     = true;
    bool                   multi_modal   = false;
    bool                   verbose       = false;
    bool                   show_connect  = false;
    std::vector<std::string> images;

    RunOptions copy() const;

    // {"model":.., "options":.., "format":.., "think":.., "keep_alive":..}
    // plus whatever the caller adds, the same union order the Go body() uses.
    nlohmann::json body(const nlohmann::json & extra) const;
};

bool has_cap(const nlohmann::json & model_info, const std::string & want);

// Thinking defaults on for a model that supports it, unless the flag said
// otherwise.
void infer_thinking(const nlohmann::json & model_info, RunOptions & o, bool explicit_set);

// ------------------------------------------------------------------ entry

// The whole `llmash run <model> [prompt]` command: resolves the model,
// streams one reply or drops into the interactive REPL. Returns the process
// exit code (0 normally; die() paths raise CliExit instead of calling
// std::exit so this stays callable from a test).
int cmd_run(const RunArgs & args);

// die()'s C++ shape: printed to stderr, then unwound instead of exiting so
// callers (main, or a test) choose when the process actually ends.
struct CliExit : std::runtime_error {
    int code;
    explicit CliExit(int c, const std::string & msg = "") : std::runtime_error(msg), code(c) {}
};

} // namespace llmash
