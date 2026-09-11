#pragma once

#include <map>
#include <string>
#include <vector>

namespace llmash {

// local.json beside the program, with the environment winning over it.
struct Config {
    std::string root;        // the install directory
    std::string models_root; // OLLAMA_MODELS, or models_root in local.json
    std::string gguf_dir;    // LLMASH_GGUF, or gguf_dir; empty means <root>/gguf
    std::vector<std::string> extra_roots;
    std::string llama_bin;   // llama-server.exe
    int         port         = 11434;
    int         public_port  = 11435;
    int         ctx          = 8192;
    int         parallel     = 1;
    std::string kv_type      = "f16";
    std::string load_mode    = "dio";
    std::string keep_alive   = "15m";
    std::map<std::string, int> ctx_override; // local.json ctx_override, keys lowercased
    std::map<std::string, int> ctx_max;      // local.json ctx_max
    std::map<std::string, std::vector<std::string>> launch_extra; // extra llama-server flags, by name fragment
    std::vector<std::string> no_mmproj;      // names whose projector is not loaded up front
    std::vector<std::string> pin;            // LLMASH_PIN, or local.json pin: never evicted
};

// Never throws: an unreadable file leaves the defaults.
Config load_config();

// A forced context for a model whose name contains a ctx_override key, else 0.
int ctx_target(const Config & cfg, const std::string & name);
// ctx_max lifts the trained context when its key matches and it is larger.
int ctx_ceiling(const Config & cfg, const std::string & name, int native);

// The launch_extra entries whose key appears in the model's name.
std::vector<std::string> launch_extra_for(const Config & cfg, const std::string & name);
bool mmproj_blocked(const Config & cfg, const std::string & name);

std::string exe_dir();

std::string env_str(const char * name, const std::string & fallback = "");
int         env_int(const char * name, int fallback);

// Case-insensitive, which is what Windows means by the same folder.
bool same_dir(const std::string & a, const std::string & b);

} // namespace llmash
