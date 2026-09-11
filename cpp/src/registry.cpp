#include "registry.h"

#include "caps.h"
#include "sha256.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <chrono>
#include <cmath>
#include <iterator>
#include <map>
#include <regex>
#include <set>

#ifndef _WIN32
#include <sys/stat.h>
#endif

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
    static const std::regex re(R"(-\d+-of-\d+$)");
    return re;
}
const std::regex & first_shard_re() {
    static const std::regex re(R"(-0*1-of-\d+$)");
    return re;
}
const std::regex & quant_re() {
    static const std::regex re(R"(-(?:i?q\d+(?:_[a-z0-9]+)*|f16|bf16|f32|mxfp4)$)", std::regex::icase);
    return re;
}

std::map<std::string, std::string> read_aliases(const std::string & dir) {
    std::map<std::string, std::string> out;
    std::ifstream in(fs::path(dir) / "aliases.json", std::ios::binary);
    if (!in) {
        return out;
    }
    const json j = json::parse(in, nullptr, false);
    if (j.is_object()) {
        for (const auto & [k, v] : j.items()) {
            if (v.is_string()) {
                out[k] = v.get<std::string>();
            }
        }
    }
    return out;
}

// Unix time is what the API reports, and the two platforms count from
// different epochs.
double mtime_unix(const fs::path & p) {
#ifdef _WIN32
    std::error_code ec;
    const auto      t = fs::last_write_time(p, ec);
    if (ec) {
        return 0;
    }
    // Seconds, not nanoseconds: MSVC's file clock counts from 1601, and
    // 400-odd years of nanoseconds overflows the count.
    const auto secs = std::chrono::duration_cast<std::chrono::seconds>(t.time_since_epoch()).count();
    return static_cast<double>(secs) - 11644473600.0;
#else
    struct stat st{};
    if (::stat(p.string().c_str(), &st) != 0) {
        return 0;
    }
    return static_cast<double>(st.st_mtime);
#endif
}

std::string read_text_file(const fs::path & p) {
    std::ifstream in(p, std::ios::binary);
    if (!in) {
        return "";
    }
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

// Go's %v of a JSON value.
std::string go_value(const json & v) {
    if (v.is_string()) {
        return v.get<std::string>();
    }
    if (v.is_number_integer()) {
        return std::to_string(v.get<long long>());
    }
    if (v.is_number_float()) {
        const double d = v.get<double>();
        if (d == std::floor(d) && std::fabs(d) < 1e15) {
            return std::to_string(static_cast<long long>(d));
        }
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%g", d);
        return buf;
    }
    if (v.is_boolean()) {
        return v.get<bool>() ? "true" : "false";
    }
    if (v.is_array()) {
        std::string out = "[";
        for (size_t i = 0; i < v.size(); i++) {
            out += (i ? " " : "") + go_value(v[i]);
        }
        return out + "]";
    }
    return v.dump();
}

std::string params_lines(const std::string & text) {
    const json j = json::parse(text, nullptr, false);
    if (!j.is_object()) {
        return "";
    }
    std::string out;
    for (const auto & [k, v] : j.items()) {
        out += (out.empty() ? "" : "\n") + k + " " + go_value(v);
    }
    return out;
}

std::string stem_of(const std::string & path) {
    return fs::path(path).stem().string();
}

// Drafters and projectors sit beside a model and are not models themselves.
bool is_sidecar(const std::string & stem) {
    const std::string s = lower(stem);
    if (s.find("mmproj") != std::string::npos) {
        return true;
    }
    for (const char * tag : {"eagle3", "dspark", "draftmodel", "speculator"}) {
        if (s.find(tag) != std::string::npos) {
            return true;
        }
    }
    for (const char * tag : {".mtp", ".draft"}) {
        if (ends_with(s, tag)) {
            return true;
        }
    }
    return false;
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

// registry.go's looseName without the alias: shard and quantisation
// suffixes off, lower case, underscores to dashes, and a :gguf tag.
std::string loose_name(const std::string & path) {
    std::string stem = std::regex_replace(stem_of(path), shard_re(), "");
    stem             = std::regex_replace(stem, quant_re(), "");
    stem             = lower(stem);
    std::replace(stem.begin(), stem.end(), '_', '-');
    return stem + ":gguf";
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
    const std::string a = lower((ec ? fs::path(path) : abs).generic_string());
    for (const auto & d : library_dirs()) {
        ec.clear();
        const fs::path da = fs::weakly_canonical(fs::path(d), ec);
        std::string b = lower((ec ? fs::path(d) : da).generic_string());
        if (!b.empty() && b.back() != '/') {
            b += '/';
        }
        if (a.rfind(b, 0) == 0) {
            return true;
        }
    }
    return false;
}

void Registry::scan_library(const std::string & dir, std::vector<Model> & out) const {
    const std::map<std::string, std::string> aliases = read_aliases(dir);
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
        const auto alias = aliases.find(fs::path(path).filename().string());
        mo.name       = alias != aliases.end() && !alias->second.empty() ? alias->second : loose_name(path);
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
        mo.caps         = caps_for(g, mo.projector, dir);

        std::error_code ec;
        mo.size     = static_cast<uint64_t>(fs::file_size(path, ec));
        mo.modified = mtime_unix(path);
        mo.digest   = "sha256:" + sha256_hex(fs::path(path).filename().string() + ":" +
                                             std::to_string(mo.size) + ":" +
                                             std::to_string(static_cast<long long>(mo.modified)));

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
        std::string blob, raw_digest, tmpl_blob, system_blob, params_blob, config_blob;
        for (const auto & layer : j.value("layers", json::array())) {
            const std::string mt = layer.value("mediaType", std::string());
            std::string       d  = layer.value("digest", std::string());
            const std::string raw = d;
            std::replace(d.begin(), d.end(), ':', '-');
            if (mt == "application/vnd.ollama.image.model") {
                blob       = d;
                raw_digest = raw;
            } else if (mt == "application/vnd.ollama.image.template") {
                tmpl_blob = d;
            } else if (mt == "application/vnd.ollama.image.system") {
                system_blob = d;
            } else if (mt == "application/vnd.ollama.image.params") {
                params_blob = d;
            }
        }
        if (blob.empty()) {
            continue;
        }
        if (j.contains("config") && j["config"].is_object()) {
            config_blob = j["config"].value("digest", std::string());
            std::replace(config_blob.begin(), config_blob.end(), ':', '-');
        }
        const fs::path blobs = fs::path(root) / "blobs";
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
        // Ollama's config blob names the family, size and quantisation.
        std::string family, param_size, quant;
        if (!config_blob.empty()) {
            const json c = json::parse(read_text_file(blobs / config_blob), nullptr, false);
            if (c.is_object()) {
                family = c.value("model_family", std::string());
                if (family.empty() && c.contains("model_families") && c["model_families"].is_array() &&
                    !c["model_families"].empty() && c["model_families"][0].is_string()) {
                    family = c["model_families"][0].get<std::string>();
                }
                param_size = c.value("model_type", std::string());
                quant      = c.value("file_type", std::string());
            }
        }
        Model mo;
        mo.name    = lower(name);
        mo.path    = path.string();
        mo.quant   = quant.empty() ? g.quant : quant;
        mo.arch    = g.arch;
        mo.has_mtp = g.has_mtp;
        mo.size    = static_cast<uint64_t>(fs::file_size(path, ec));
        mo.manifest     = it->path().string();
        mo.store_root   = root;
        mo.digest       = raw_digest;
        mo.family       = family.empty() ? g.arch : family;
        mo.param_size   = !param_size.empty() ? param_size : (g.n_params > 0 ? pretty_params(g.n_params) : "");
        mo.tmpl         = tmpl_blob.empty() ? g.chat_template : read_text_file(blobs / tmpl_blob);
        if (mo.tmpl.empty()) {
            mo.tmpl = g.chat_template;
        }
        mo.system       = system_blob.empty() ? "" : read_text_file(blobs / system_blob);
        mo.params_text  = params_blob.empty() ? "" : params_lines(read_text_file(blobs / params_blob));
        mo.modified = mtime_unix(it->path());
        mo.ctx_train    = static_cast<int>(g.ctx_train);
        mo.experts      = g.experts;
        mo.experts_used = g.experts_used;
        mo.projector    = find_projector_for(path.string());
        mo.caps         = caps_for(g, mo.projector, (fs::path(root) / "gguf").string());
        out.push_back(std::move(mo));
    }
}

void Registry::scan() {
    std::vector<Model> found;
    for (const auto & d : library_dirs()) {
        scan_library(d, found);
    }
    // Two files that strip to the same name: the first keeps it, the next is
    // told apart by its quantisation, or by a counter when it has none.
    std::set<std::string> taken;
    for (auto & m : found) {
        if (taken.count(m.name) != 0) {
            const std::string stem = std::regex_replace(stem_of(m.path), shard_re(), "");
            std::string       base = m.name;
            if (ends_with(base, ":gguf")) {
                base = base.substr(0, base.size() - 5);
            }
            std::string name = m.name;
            std::smatch q;
            if (std::regex_search(stem, q, quant_re()) && ends_with(m.name, ":gguf")) {
                std::string tag = lower(q.str(0).substr(1));
                std::replace(tag.begin(), tag.end(), '_', '-');
                name = base + ":" + tag;
            }
            for (int n = 2; taken.count(name) != 0; n++) {
                name = base + ":gguf-" + std::to_string(n);
            }
            m.name = name;
        }
        taken.insert(m.name);
    }
    if (!cfg_.models_root.empty() && is_ollama_store(cfg_.models_root)) {
        scan_ollama_store(cfg_.models_root, found);
    }
    for (const auto & e : cfg_.extra_roots) {
        if (is_ollama_store(e)) {
            scan_ollama_store(e, found);
        }
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
    // registry.go's get: a bare name is name:latest, and failing that the
    // loose file name:gguf; a tagged name is looked up as given.
    const std::string want  = lower(name);
    const auto        exact = [this](const std::string & n) -> const Model * {
        for (const auto & m : cache_) {
            if (m.name == n) {
                return &m;
            }
        }
        return nullptr;
    };
    if (want.find(':') == std::string::npos) {
        if (const Model * m = exact(want + ":latest")) {
            return m;
        }
        return exact(want + ":gguf");
    }
    return exact(want);
}

} // namespace llmash
