#pragma once

#include "config.h"
#include "gguf.h"

#include <cstdint>
#include <string>
#include <vector>

namespace llmash {

struct Model {
    std::string name;
    std::string path;
    std::string quant;
    std::string arch;
    uint64_t    size = 0;
    bool        has_mtp = false;
    bool        in_library = false;   // read-only: never deleted from
    std::string mtp_path;             // a sidecar drafter, when there is one

    std::string digest;      // sha256 of the manifest blob, or derived
    std::string family;      // general.basename, size suffix stripped
    std::string param_size;  // "8B", "35B-A3B"
    std::string tmpl;        // tokenizer.chat_template
    std::string projector;   // an mmproj sidecar, when there is one
    std::string system;      // the store's system layer
    std::string params_text; // the store's params layer, "key value" per line
    double      modified = 0;
    int         ctx_train = 0;
    int         experts = 0;
    int         experts_used = 0;
    std::vector<std::string> caps;
};

// The folders read, and whether a path came from one of them.
class Registry {
public:
    explicit Registry(Config cfg);

    std::vector<Model> all();
    const Model *      find(const std::string & name);

    std::vector<std::string> library_dirs() const;
    bool                     in_library(const std::string & path) const;

    bool is_ollama_store(const std::string & dir) const;

    void invalidate();

private:
    Config             cfg_;
    std::vector<Model> cache_;
    bool               loaded_ = false;

    void scan();
    void scan_library(const std::string & dir, std::vector<Model> & out) const;
    void scan_ollama_store(const std::string & root, std::vector<Model> & out) const;
};

// Subfolders included, hidden folders and links skipped.
std::vector<std::string> walk_gguf(const std::string & dir);

// The display name a loose file gets when nothing else names it.
std::string loose_name(const std::string & path);

} // namespace llmash
