#include "registry.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <regex>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace llmash {

namespace {

std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

bool ends_with(const std::string & s, const std::string & suffix) {
    return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// A later shard of a split model is loaded through the first one.
const std::regex & shard_re() {
    static const std::regex re(R"(-\d{5}-of-\d{5}$)");
    return re;
}
const std::regex & first_shard_re() {
    static const std::regex re(R"(-00001-of-\d{5}$)");
    return re;
}

std::string stem_of(const std::string & path) {
    return fs::path(path).stem().string();
}

// Drafters and projectors sit beside a model and are not models themselves.
bool is_sidecar(const std::string & stem) {
    const std::string s = lower(stem);
    for (const char * tag : {".mtp", ".eagle3", ".dspark", ".draft", ".mmproj", "-mmproj",
                             "eagle3", "dspark", "draftmodel", "speculator"}) {
        if (s.find(tag) != std::string::npos) {
            return true;
        }
    }
    return s.rfind("mmproj", 0) == 0;
}

std::string pretty_params(uint64_t n) {
    char buf[32];
    if (n >= 1000000000000ull)   { std::snprintf(buf, sizeof(buf), "%.1fT", n / 1e12); }
    else if (n >= 1000000000ull) { std::snprintf(buf, sizeof(buf), "%.1fB", n / 1e9);  }
    else if (n >= 1000000ull)    { std::snprintf(buf, sizeof(buf), "%.2fM", n / 1e6);  }
    else                         { std::snprintf(buf, sizeof(buf), "%llu", (unsigned long long) n); }
    return buf;
}

// Go's sizeInName, [-_ ][A-Za-z]?\d+(\.\d+)?[bBmM] at a word boundary: the
// name is cut where the first such size starts.
std::string family_of(const GGUFInfo & g) {
    const std::string & base = g.basename;
    static const std::regex size_re(R"([-_ ][A-Za-z]?\d+(\.\d+)?[bBmM])");

    auto it  = std::sregex_iterator(base.begin(), base.end(), size_re);
    const auto end = std::sregex_iterator();
    for (; it != end; ++it) {
        const size_t after = static_cast<size_t>(it->position(0) + it->length(0));
        if (after < base.size() && (std::isalnum(static_cast<unsigned char>(base[after])) != 0)) {
            continue; // "35Bx" is not a size
        }
        std::string cut = base.substr(0, static_cast<size_t>(it->position(0)));
        while (!cut.empty() && (cut.back() == '-' || cut.back() == '_' || cut.back() == ' ')) cut.pop_back();
        return cut.empty() ? g.arch : cut;
    }
    return base.empty() ? g.arch : base;
}

std::vector<std::string> caps_for(const GGUFInfo & g, const std::string & projector) {
    if (g.has_pooling) {
        return {"embedding"};
    }
    std::vector<std::string> caps{"completion"};
    if (!projector.empty()) {
        caps.push_back("vision");
    }
    const std::string tl = lower(g.chat_template);
    const bool oss = g.arch == "gpt-oss";
    if (oss || tl.find("tool") != std::string::npos || tl.find("function") != std::string::npos) {
        caps.push_back("tools");
    }
    if (oss || tl.find("think") != std::string::npos || tl.find("reason") != std::string::npos) {
        caps.push_back("thinking");
    }
    return caps;
}

// an mmproj sidecar beside the weights
std::string find_projector_for(const std::string & path) {
    const fs::path dir = fs::path(path).parent_path();
    const std::string stem = fs::path(path).stem().string();
    std::error_code ec;
    for (const auto & cand : { dir / (stem + ".mmproj.gguf"), dir / ("mmproj-" + stem + ".gguf") }) {
        if (fs::exists(cand, ec)) return cand.string();
    }
    for (auto it = fs::directory_iterator(dir, ec); !ec && it != fs::directory_iterator(); ++it) {
        const std::string n = lower(it->path().filename().string());
        if (n.rfind("mmproj", 0) == 0 && n.size() > 5 && n.substr(n.size() - 5) == ".gguf") {
            return it->path().string();
        }
    }
    return "";
}

} // namespace

std::string loose_name(const std::string & path) {
    std::string stem = stem_of(path);
    std::smatch m;
    if (std::regex_search(stem, m, first_shard_re())) {
        stem = stem.substr(0, m.position(0));
    }
    return lower(stem);
}

std::vector<std::string> walk_gguf(const std::string & dir) {
    std::vector<std::string> out;
    std::error_code ec;
    if (!fs::is_directory(dir, ec)) {
        return out;
    }
    auto it = fs::recursive_directory_iterator(
        dir, fs::directory_options::skip_permission_denied, ec);
    if (ec) {
        return out;
    }
    for (auto end = fs::recursive_directory_iterator(); it != end; it.increment(ec)) {
        if (ec) {
            ec.clear();
            continue;
        }
        const fs::path p = it->path();
        const std::string base = p.filename().string();
        if (it->is_directory(ec)) {
            if (!base.empty() && base[0] == '.') {
                it.disable_recursion_pending();
            }
            continue;
        }
        if (it->is_symlink(ec)) {
            continue;
        }
        if (lower(p.extension().string()) == ".gguf") {
            out.push_back(p.string());
        }
    }
    std::sort(out.begin(), out.end());
    return out;
}

Registry::Registry(Config cfg) : cfg_(std::move(cfg)) {}

void Registry::invalidate() {
    loaded_ = false;
    cache_.clear();
}

std::vector<std::string> Registry::library_dirs() const {
    std::vector<std::string> dirs;
    const auto add = [&](const std::string & d) {
        if (d.empty()) {
            return;
        }
        for (const auto & e : dirs) {
            if (same_dir(e, d)) {
                return;
            }
        }
        dirs.push_back(d);
    };
    if (!cfg_.gguf_dir.empty()) {
        add(cfg_.gguf_dir);
    }
    if (!cfg_.models_root.empty() && !is_ollama_store(cfg_.models_root)) {
        add(cfg_.models_root);
    }
    if (!cfg_.models_root.empty()) {
        add((fs::path(cfg_.models_root) / "gguf").string());
    }
    for (const auto & e : cfg_.extra_roots) {
        add((fs::path(e) / "gguf").string());
        if (!is_ollama_store(e)) {
            add(e);
        }
    }
    return dirs;
}

bool Registry::is_ollama_store(const std::string & dir) const {
    std::error_code ec;
    return fs::is_directory(fs::path(dir) / "manifests", ec);
}

bool Registry::in_library(const std::string & path) const {
    std::error_code ec;
    const fs::path abs = fs::weakly_canonical(fs::path(path), ec);
    const std::string a = lower((ec ? fs::path(path) : abs).string());
    for (const auto & d : library_dirs()) {
        ec.clear();
        const fs::path da = fs::weakly_canonical(fs::path(d), ec);
        std::string b = lower((ec ? fs::path(d) : da).string());
        if (!b.empty() && b.back() != '\\' && b.back() != '/') {
            b += '\\';
        }
        if (a.rfind(b, 0) == 0) {
            return true;
        }
    }
    return false;
}

void Registry::scan_library(const std::string & dir, std::vector<Model> & out) const {
    for (const auto & path : walk_gguf(dir)) {
        const std::string stem = stem_of(path);
        if (is_sidecar(stem)) {
            continue;
        }
        std::smatch m;
        if (std::regex_search(stem, m, shard_re()) && !std::regex_search(stem, first_shard_re())) {
            continue;
        }
        const GGUFInfo g = read_gguf(path);
        if (!g.ok) {
            continue;
        }
        Model mo;
        mo.name       = loose_name(path);
        mo.path       = path;
        mo.quant      = g.quant;
        mo.arch       = g.arch;
        mo.has_mtp    = g.has_mtp;
        mo.in_library = true;
        mo.family       = family_of(g);
        mo.param_size   = g.n_params > 0 ? pretty_params(g.n_params) : g.size_label;
        mo.tmpl         = g.chat_template;
        mo.ctx_train    = static_cast<int>(g.ctx_train);
        mo.experts      = g.experts;
        mo.experts_used = g.experts_used;
        mo.projector    = find_projector_for(path);
        mo.caps         = caps_for(g, mo.projector);

        std::error_code ec;
        mo.size = static_cast<uint64_t>(fs::file_size(path, ec));

        const fs::path side = fs::path(path).parent_path() / (stem + ".mtp.gguf");
        if (fs::exists(side, ec)) {
            mo.mtp_path = side.string();
        }
        out.push_back(std::move(mo));
    }
}

void Registry::scan_ollama_store(const std::string & root, std::vector<Model> & out) const {
    const fs::path manifests = fs::path(root) / "manifests";
    std::error_code ec;
    if (!fs::is_directory(manifests, ec)) {
        return;
    }
    for (auto it = fs::recursive_directory_iterator(manifests, fs::directory_options::skip_permission_denied, ec);
         it != fs::recursive_directory_iterator(); it.increment(ec)) {
        if (ec) {
            ec.clear();
            continue;
        }
        if (!it->is_regular_file(ec)) {
            continue;
        }
        std::ifstream in(it->path(), std::ios::binary);
        if (!in) {
            continue;
        }
        json j = json::parse(in, nullptr, false);
        if (j.is_discarded()) {
            continue;
        }
        std::string blob;
        for (const auto & layer : j.value("layers", json::array())) {
            if (layer.value("mediaType", std::string()) == "application/vnd.ollama.image.model") {
                blob = layer.value("digest", std::string());
                break;
            }
        }
        if (blob.empty()) {
            continue;
        }
        std::replace(blob.begin(), blob.end(), ':', '-');
        const fs::path path = fs::path(root) / "blobs" / blob;
        if (!fs::exists(path, ec)) {
            continue;
        }

        // manifests/<registry>/<namespace>/<name>/<tag>
        const fs::path rel = fs::relative(it->path(), manifests, ec);
        std::string name = rel.parent_path().filename().string();
        const std::string tag = rel.filename().string();
        if (name.empty()) {
            continue;
        }
        name += ":" + tag;

        const GGUFInfo g = read_gguf(path.string());
        Model mo;
        mo.name    = lower(name);
        mo.path    = path.string();
        mo.quant   = g.quant;
        mo.arch    = g.arch;
        mo.has_mtp = g.has_mtp;
        mo.size    = static_cast<uint64_t>(fs::file_size(path, ec));
        mo.digest       = blob;
        mo.family       = family_of(g);
        mo.param_size   = g.n_params > 0 ? pretty_params(g.n_params) : g.size_label;
        mo.tmpl         = g.chat_template;
        mo.ctx_train    = static_cast<int>(g.ctx_train);
        mo.experts      = g.experts;
        mo.experts_used = g.experts_used;
        mo.projector    = find_projector_for(path.string());
        mo.caps         = caps_for(g, mo.projector);
        out.push_back(std::move(mo));
    }
}

void Registry::scan() {
    std::vector<Model> found;
    if (!cfg_.models_root.empty() && is_ollama_store(cfg_.models_root)) {
        scan_ollama_store(cfg_.models_root, found);
    }
    for (const auto & e : cfg_.extra_roots) {
        if (is_ollama_store(e)) {
            scan_ollama_store(e, found);
        }
    }
    for (const auto & d : library_dirs()) {
        scan_library(d, found);
    }

    std::vector<Model> uniq;
    for (auto & m : found) {
        const auto seen = std::find_if(uniq.begin(), uniq.end(),
                                       [&](const Model & o) { return o.name == m.name; });
        if (seen == uniq.end()) {
            uniq.push_back(std::move(m));
        }
    }
    cache_  = std::move(uniq);
    loaded_ = true;
}

std::vector<Model> Registry::all() {
    if (!loaded_) {
        scan();
    }
    return cache_;
}

const Model * Registry::find(const std::string & name) {
    if (!loaded_) {
        scan();
    }
    const std::string want = lower(name);
    for (const auto & m : cache_) {
        if (m.name == want) {
            return &m;
        }
    }
    for (const auto & m : cache_) {
        if (m.name.rfind(want + ":", 0) == 0 || want + ":latest" == m.name) {
            return &m;
        }
    }
    return nullptr;
}

} // namespace llmash
