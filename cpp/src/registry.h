#pragma once

#include "config.h"
#include "gguf.h"

#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace llmash {

struct Model {
    std::string name;
    std::string path;
    std::string quant;
    std::string arch;
    uint64_t    size = 0;
    uint64_t    input_bytes = 0;      // of that, what llama.cpp keeps in RAM as the input layer
    bool        has_mtp = false;
    bool        in_library = false;   // read-only: never deleted from
    bool        incomplete = false;   // a `.part` file: listed and removable, never loaded
    std::string mtp_path;             // a sidecar drafter, when there is one
    // An Ollama-store model is a manifest plus shared blobs, so `path` alone
    // cannot delete it. Empty for a loose GGUF.
    std::string manifest;
    std::string store_root;

    std::string digest;      // sha256 of the manifest blob, or derived
    std::string family;      // general.basename, size suffix stripped
    std::string param_size;  // "8B", "35B-A3B"
    std::string tmpl;        // tokenizer.chat_template
    std::string projector;   // an mmproj sidecar, when there is one
    std::string system;      // the store's system layer
    std::string params_text; // the store's params layer, "key value" per line
    int         num_ctx = 0;  // the Modelfile's
    double      modified = 0;
    int         ctx_train = 0;
    int         experts = 0;
    int         experts_used = 0;
    double      kv_bytes_tok = 0;   // f16 cache per token of context, from the header
    double      state_bytes  = 0;   // per-sequence state that does not grow with the context
    std::vector<std::string> caps;
};

// The folders read, and whether a path came from one of them.
class Registry {
public:
    explicit Registry(Config cfg);

    std::vector<Model>   all();
    // A name that matches no model exactly still finds the one model it can
    // only mean; `loose` also lets a prefix or a fragment of the name do that.
    std::optional<Model> find(const std::string & name, bool loose = true);

    std::vector<std::string> library_dirs() const;
    bool                     in_library(const std::string & path) const;

    bool is_ollama_store(const std::string & dir) const;

    void invalidate();

private:
    Config             cfg_;
    mutable std::mutex mu_;
    std::vector<Model> cache_;
    bool               loaded_ = false;

    void scan();
    void scan_library(const std::string & dir, std::vector<Model> & out, const std::vector<std::string> & skip) const;
    void scan_ollama_store(const std::string & root, std::vector<Model> & out) const;
};

// Subfolders included, hidden folders and links skipped.
std::vector<std::string> walk_gguf(const std::string & dir, const std::vector<std::string> & skip = {});

// The `<name>.gguf.part` files a stopped pull left behind.
std::vector<std::string> walk_partials(const std::string & dir, const std::vector<std::string> & skip = {});

// loose_name with `.partial` appended to the tag.
std::string partial_name(const std::string & path);

// The display name a loose file gets when nothing else names it.
std::string loose_name(const std::string & path);

} // namespace llmash
