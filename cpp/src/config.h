#pragma once

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
};

// Never throws: an unreadable file leaves the defaults.
Config load_config();

std::string exe_dir();

std::string env_str(const char * name, const std::string & fallback = "");
int         env_int(const char * name, int fallback);

// Case-insensitive, which is what Windows means by the same folder.
bool same_dir(const std::string & a, const std::string & b);

} // namespace llmash
