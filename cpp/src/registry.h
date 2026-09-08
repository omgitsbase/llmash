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
