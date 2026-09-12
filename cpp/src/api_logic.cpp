#include "api_logic.h"

#include <subprocess.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <limits>
#include <random>
#include <regex>
#include <set>
#include <stdexcept>
#include <system_error>

#ifdef _WIN32
// RtlGenRandom, the boring way to get bytes from advapi32 without pulling in
// bcrypt: declared by hand so windows.h stays out of this file.
extern "C" unsigned char __stdcall SystemFunction036(void * buffer, unsigned long length);
#endif

namespace fs = std::filesystem;
using json   = nlohmann::json;

namespace llmash {

namespace {

std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

std::string upper(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
    return s;
}

std::string trim(const std::string & s) {
    const auto is_space = [](unsigned char c) { return std::isspace(c) != 0; };
    size_t     b        = 0;
    size_t     e        = s.size();
    while (b < e && is_space(static_cast<unsigned char>(s[b]))) {
        b++;
    }
    while (e > b && is_space(static_cast<unsigned char>(s[e - 1]))) {
        e--;
    }
    return s.substr(b, e - b);
}

bool file_exists(const std::string & p) {
    if (p.empty()) {
        return false;
    }
    std::error_code ec;
    return fs::is_regular_file(fs::path(p), ec);
}

std::string base_name(const std::string & p) { return fs::path(p).filename().string(); }
std::string stem_of(const std::string & p) { return fs::path(p).stem().string(); }

bool is_gguf(const std::string & p) { return lower(fs::path(p).extension().string()) == ".gguf"; }

const std::regex & shard_re() {
    static const std::regex re(R"(-\d+-of-\d+$)");
    return re;
}

std::string strip_shard(const std::string & stem) {
    return std::regex_replace(stem, shard_re(), "");
}

// registry.go's quantSuffix, applied until it stops matching, then "-UD".
std::string pair_stem(std::string stem) {
    static const std::regex quant(R"(-(?:i?q\d+(?:_[a-z0-9]+)*|f16|bf16|f32|mxfp4)$)",
                                  std::regex::icase);
    static const std::regex ud(R"(-ud$)", std::regex::icase);
    std::string             prev;
    while (prev != stem) {
        prev = stem;
        stem = std::regex_replace(stem, quant, "");
    }
    return lower(std::regex_replace(stem, ud, ""));
}

// crypto/rand + base64.RawURLEncoding: no padding, '-' and '_'.
std::string b64url(const unsigned char * data, size_t len) {
    static const char * alphabet =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    std::string out;
    for (size_t i = 0; i < len; i += 3) {
        const unsigned n    = static_cast<unsigned>(data[i]) << 16 |
                           (i + 1 < len ? static_cast<unsigned>(data[i + 1]) << 8 : 0) |
                           (i + 2 < len ? static_cast<unsigned>(data[i + 2]) : 0);
        const size_t   have = len - i;
        out += alphabet[(n >> 18) & 63];
        out += alphabet[(n >> 12) & 63];
        if (have > 1) {
            out += alphabet[(n >> 6) & 63];
        }
        if (have > 2) {
            out += alphabet[n & 63];
        }
    }
    return out;
}

void random_bytes(unsigned char * out, size_t n) {
#ifdef _WIN32
    if (SystemFunction036(out, static_cast<unsigned long>(n)) != 0) {
        return;
    }
#endif
    std::random_device rd;
    for (size_t i = 0; i < n; i++) {
        out[i] = static_cast<unsigned char>(rd() & 0xFF);
    }
}

json read_json_file(const fs::path & p) {
    std::ifstream in(p, std::ios::binary);
    if (!in) {
        return json::object();
    }
    const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    json              j = json::parse(text, nullptr, /*allow_exceptions=*/false);
    if (j.is_discarded() || !j.is_object()) {
        return json::object();
    }
    return j;
}

std::vector<std::string> caps_or_completion(const Model & m) {
    if (m.caps.empty()) {
        return {"completion"};
    }
    return m.caps;
}

json error_obj(const std::string & msg) { return json{{"error", msg}}; }

double parse_ka_string(const std::string & raw) {
    const std::string s = trim(raw);
    if (s == "-1" || s == "-1s") {
        return std::numeric_limits<double>::infinity();
    }
    const auto parse = [](const std::string & x, double & out) {
        try {
            size_t used = 0;
            out         = std::stod(x, &used);
            return used == x.size() && !x.empty();
        } catch (const std::exception &) {
            return false;
        }
    };
    const auto ends_with = [&](const char * suf) {
        const size_t n = std::strlen(suf);
        return s.size() >= n && s.compare(s.size() - n, n, suf) == 0;
    };
    double v = 0;
    if (ends_with("ms")) {
        if (parse(s.substr(0, s.size() - 2), v)) {
            return v / 1000;
        }
    } else if (ends_with("s")) {
        if (parse(s.substr(0, s.size() - 1), v)) {
            return v;
        }
    } else if (ends_with("m")) {
        if (parse(s.substr(0, s.size() - 1), v)) {
            return v * 60;
        }
    } else if (ends_with("h")) {
        if (parse(s.substr(0, s.size() - 1), v)) {
            return v * 3600;
        }
    } else if (parse(s, v)) {
        if (v < 0) {
            return std::numeric_limits<double>::infinity();
        }
        return v;
    }
    return 300;
}

} // namespace

// ------------------------------------------------------------ formatting

double now_unix() {
    const auto d = std::chrono::system_clock::now().time_since_epoch();
    return std::chrono::duration_cast<std::chrono::duration<double>>(d).count();
}

std::string iso(double unix_seconds) {
    long long us = static_cast<long long>(std::llround(unix_seconds * 1e6));
    long long secs = us / 1000000;
    long long frac = us % 1000000;
    if (frac < 0) {
        frac += 1000000;
        secs -= 1;
    }
    const std::time_t t = static_cast<std::time_t>(secs);
    std::tm           tm{};
#ifdef _WIN32
    if (gmtime_s(&tm, &t) != 0) {
        return "";
    }
#else
    if (gmtime_r(&t, &tm) == nullptr) {
        return "";
    }
#endif
    char stamp[64];
    if (std::strftime(stamp, sizeof(stamp), "%Y-%m-%dT%H:%M:%S", &tm) == 0) {
        return "";
    }
    std::string out = stamp;
    if (frac != 0) {
        char digits[16];
        std::snprintf(digits, sizeof(digits), "%06lld", frac);
        std::string f = digits;
        while (!f.empty() && f.back() == '0') {
            f.pop_back();
        }
        out += "." + f;
    }
    return out + "+00:00";
}

std::string human_bytes(double n) {
    static const char * units[] = {"B", "KB", "MB", "GB", "TB"};
    char                buf[64];
    for (const char * u : units) {
        if (n < 1024) {
            const bool whole = std::strcmp(u, "B") == 0 || std::strcmp(u, "KB") == 0;
            std::snprintf(buf, sizeof(buf), whole ? "%.0f %s" : "%.1f %s", n, u);
            return buf;
        }
        n /= 1024;
    }
    std::snprintf(buf, sizeof(buf), "%.1f PB", n);
    return buf;
}

// ------------------------------------------------------------ model JSON

bool loadable(const Model & m) {
    const std::string fam = lower(m.family);
    return !(fam == "gptoss" || fam == "glm4moelite");
}

int advertised_ctx(const Config & cfg) { return cfg.ctx > 0 ? cfg.ctx : 8192; }

// config.go's advertisedCtx: the trained context, forced by ctx_override or
// lifted by ctx_max.
int advertised_ctx(const Model & m, const Config & cfg) {
    const int native = m.ctx_train > 0 ? m.ctx_train : advertised_ctx(cfg);
    const int forced = ctx_target(cfg, m.name);
    return forced > 0 ? forced : ctx_ceiling(cfg, m.name, native);
}

std::string digest_of(const Model & m) {
    const std::string seed =
        m.name + "|" + std::to_string(m.size) + "|" + m.quant + "|" + m.arch;
    std::string out;
    uint64_t    h = 1469598103934665603ULL;
    for (int block = 0; block < 4; block++) {
        h ^= static_cast<uint64_t>(block) * 0x9E3779B97F4A7C15ULL;
        for (const unsigned char c : seed) {
            h ^= c;
            h *= 1099511628211ULL;
        }
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(h));
        out += buf;
    }
    return out;
}

json tag_entry_json(const Model & m, const Config & cfg) {
    json families = json::array();
    if (!m.family.empty()) {
        families.push_back(m.family);
    }
    std::string digest = m.digest.empty() ? digest_of(m) : m.digest;
    if (digest.rfind("sha256:", 0) == 0) {
        digest = digest.substr(7);
    }
    return json{
        {"name", m.name},
        {"model", m.name},
        {"modified_at", iso(m.modified)},
        {"size", m.size},
        {"digest", digest},
        {"details",
         json{{"parent_model", ""},
              {"format", "gguf"},
              {"family", m.family},
              {"families", families},
              {"parameter_size", m.param_size},
              {"quantization_level", m.quant},
              {"context_length", advertised_ctx(m, cfg)},
              {"expert_count", m.experts},
              {"expert_used_count", m.experts_used}}},
        {"capabilities", caps_or_completion(m)},
        {"loadable", loadable(m)},
    };
}

json tags_json(const std::vector<Model> & models, const Config & cfg) {
    json rows = json::array();
    for (const auto & m : models) {
        if (file_exists(m.path) && loadable(m)) {
            rows.push_back(tag_entry_json(m, cfg));
        }
    }
    return json{{"models", rows}};
}

json v1_entry_json(const Model & m, const Config & cfg) {
    const int ctx = advertised_ctx(m, cfg);
    return json{
        {"id", m.name},
        {"object", "model"},
        {"created", static_cast<int64_t>(m.modified)},
        {"owned_by", "llmash"},
        {"context_length", ctx},
        {"max_model_len", ctx},
        {"max_context_length", ctx},
        {"context_window", ctx},
    };
}

json v1_models_json(const std::vector<Model> & models, const Config & cfg) {
    json data = json::array();
    for (const auto & m : models) {
        if (file_exists(m.path) && loadable(m)) {
            data.push_back(v1_entry_json(m, cfg));
        }
    }
    return json{{"object", "list"}, {"data", data}};
}

json show_json(const Model & m, const Config & cfg) {
    const std::string arch = m.family.empty() ? "llama" : m.family;
    const int         ctx  = advertised_ctx(m, cfg);
    return json{
        {"license", ""},
        {"modelfile", "FROM " + m.path},
        {"parameters", m.params_text},
        {"template", m.tmpl},
        {"system", m.system},
        {"details", tag_entry_json(m, cfg)["details"]},
        {"model_info",
         json{{"general.architecture", arch},
              {arch + ".context_length", ctx},
              {"context_length", ctx},
              {"general.parameter_count", m.param_size}}},
        {"capabilities", caps_or_completion(m)},
    };
}

json openai_error_json(const std::string & message, const std::string & type,
                       const std::string & code) {
    return json{{"error", json{{"message", message}, {"type", type}, {"code", code}}}};
}

json ps_entry_json(const InstanceView & v, const Config & cfg) {
    json e = tag_entry_json(v.model, cfg);

    const int64_t size = static_cast<int64_t>(v.vram_gb * (1LL << 30));
    e["size"]          = size;
    // only what actually went to the card counts as resident there
    e["size_vram"] = v.on_gpu ? size : static_cast<int64_t>(0);
    if (std::isinf(v.expires_at) || v.expires_at > 1e15) {
        e["expires_at"] = "9999-12-31T23:59:59Z";
    } else {
        e["expires_at"] = iso(v.expires_at);
    }
    e["context_length"] = v.ctx;
    return e;
}

json ps_json(const std::vector<InstanceView> & live, const Config & cfg) {
    json out = json::array();
    for (const auto & v : live) {
        if (!v.ready) {
            continue;
        }
        out.push_back(ps_entry_json(v, cfg));
    }
    return json{{"models", out}};
}

// ---------------------------------------------------------------- auth

bool check_api_key(const std::string & authorization_header, const std::string & x_api_key_header,
                    const std::string & expected_key) {
    if (expected_key.empty()) {
        return false;
    }
    std::string key = x_api_key_header;
    if (authorization_header.size() > 7 && lower(authorization_header.substr(0, 7)) == "bearer ") {
        key = trim(authorization_header.substr(7));
    }
    if (key.size() != expected_key.size()) {
        return false;
    }
    unsigned char diff = 0;
    for (size_t i = 0; i < key.size(); i++) {
        diff |= static_cast<unsigned char>(key[i]) ^ static_cast<unsigned char>(expected_key[i]);
    }
    return diff == 0;
}

std::string link_key(const Config & cfg) {
    const std::string from_env = env_str("LLMASH_LINK_KEY");
    if (!from_env.empty()) {
        return from_env;
    }
    const fs::path file = fs::path(cfg.root) / "link.json";
    json           d    = read_json_file(file);
    const auto     it   = d.find("key");
    if (it != d.end() && it->is_string() && !it->get<std::string>().empty()) {
        return it->get<std::string>();
    }

    unsigned char raw[24];
    random_bytes(raw, sizeof(raw));
    const std::string key = "sk-llmash-" + b64url(raw, sizeof(raw));
    d["key"]              = key;
    d["public_port"]      = cfg.public_port;
    std::ofstream out(file, std::ios::binary | std::ios::trunc);
    if (out) {
        out << d.dump(2);
    }
    return key;
}

// -------------------------------------------------------- keep_alive

double parse_keep_alive(const json & v, const std::string & default_keep) {
    if (v.is_null() || v.is_discarded()) {
        return default_keep.empty() ? 300.0 : parse_ka_string(default_keep);
    }
    if (v.is_number()) {
        const double t = v.get<double>();
        return t < 0 ? std::numeric_limits<double>::infinity() : t;
    }
    if (v.is_string()) {
        return parse_ka_string(v.get<std::string>());
    }
    return parse_ka_string(v.dump());
}

double keep_alive_out(double ka) {
    if (std::isinf(ka) && ka > 0) {
        return -1;
    }
    return ka;
}

// ----------------------------------------------------- create / copy

std::string loose_dir(const Config & cfg) {
    const std::string from_env = env_str("LLMASH_GGUF");
    if (!from_env.empty()) {
        return from_env;
    }
    if (!cfg.gguf_dir.empty()) {
        return cfg.gguf_dir;
    }
    const std::string base = cfg.models_root.empty() ? cfg.root : cfg.models_root;
    return (fs::path(base) / "gguf").string();
}

std::string safe_model_name(const std::string & name) {
    const size_t      colon = name.find(':');
    const std::string head  = colon == std::string::npos ? name : name.substr(0, colon);
    std::string       out;
    out.reserve(head.size());
    for (const char c : head) {
        const bool keep = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
                          (c >= '0' && c <= '9') || c == '.' || c == '_' || c == '-';
        out += keep ? c : '-';
    }
    return out;
}

std::string find_projector(const std::string & gguf_path) {
    const std::string stem = strip_shard(stem_of(gguf_path));
    const fs::path    dir  = fs::path(gguf_path).parent_path();

    for (const std::string & name : {stem + ".mmproj.gguf", stem + "-mmproj.gguf"}) {
        const fs::path cand = dir / name;
        if (file_exists(cand.string())) {
            return cand.string();
        }
    }

    std::vector<std::string> cands;
    std::error_code          ec;
    for (auto it = fs::directory_iterator(dir, ec); !ec && it != fs::directory_iterator(); ++it) {
        const std::string base = it->path().filename().string();
        const std::string low  = lower(base);
        if (low.find("mmproj") != std::string::npos && low.size() > 5 &&
            low.compare(low.size() - 5, 5, ".gguf") == 0) {
            cands.push_back(it->path().string());
        }
    }
    std::sort(cands.begin(), cands.end());

    static const std::regex tail(R"(^[.\-_]*mmproj[.\-_a-z0-9]*\.gguf$)", std::regex::icase);
    const std::string       low_stem = lower(stem);
    for (const auto & c : cands) {
        const std::string low = lower(base_name(c));
        if (low.rfind(low_stem, 0) == 0 && std::regex_match(low.substr(low_stem.size()), tail)) {
            return c;
        }
    }

    static const std::regex mm_tail(R"([.-]mmproj$)", std::regex::icase);
    static const std::regex mm_head(R"(^mmproj[.\-_])", std::regex::icase);
    const std::string       want = pair_stem(stem);
    for (const auto & c : cands) {
        std::string base = std::regex_replace(stem_of(c), mm_tail, "");
        base             = std::regex_replace(base, mm_head, "");
        if (pair_stem(base) == want) {
            return c;
        }
    }
    return "";
}

void copy_file_preserving_mtime(const std::string & src, const std::string & dst) {
    std::error_code ec;
    fs::copy_file(fs::path(src), fs::path(dst), fs::copy_options::overwrite_existing, ec);
    if (ec) {
        throw std::runtime_error(ec.message());
    }
    const auto when = fs::last_write_time(fs::path(src), ec);
    if (!ec) {
        std::error_code ignored;
        fs::last_write_time(fs::path(dst), when, ignored);
    }
}

void quantize_gguf(const Config & cfg, const std::string & src, const std::string & dst,
                    const std::string & level) {
    const fs::path qexe = fs::path(cfg.llama_bin).parent_path() / "llama-quantize.exe";
    if (!file_exists(qexe.string())) {
        throw std::runtime_error("llama-quantize.exe not found next to llama-server");
    }
    Subprocess proc;
    if (!proc.start({qexe.string(), "--allow-requantize", src, dst, upper(level)})) {
        throw std::runtime_error("llama-quantize failed (could not start)");
    }
    const int code = proc.join();
    if (code != 0) {
        throw std::runtime_error("llama-quantize failed (exit status " + std::to_string(code) + ")");
    }
}

void run_create(const Config & cfg, const std::string & name, const std::string & from,
                const std::string & quantize, const std::string & draft_quantize,
                const Emit & emit) {
    const std::string src = from;
    if (!is_gguf(src) || !file_exists(src)) {
        emit(error_obj("'" + from + "' isn't a local .gguf, so there's nothing to import."));
        return;
    }
    const std::string dest_dir = loose_dir(cfg);
    std::error_code   ec;
    fs::create_directories(fs::path(dest_dir), ec);

    const std::string safe = safe_model_name(name);
    const std::string dest = (fs::path(dest_dir) / (safe + ".gguf")).string();
    if (file_exists(dest)) {
        emit(error_obj("'" + base_name(dest) + "' already exists in " + dest_dir +
                       ". Remove it first (rm) or pick another name."));
        return;
    }
    const double size = static_cast<double>(fs::file_size(fs::path(src), ec));

    try {
        if (!quantize.empty()) {
            emit(json{{"status", "quantizing " + base_name(src) + " -> " + base_name(dest) + " (" +
                                     upper(quantize) + ") ..."}});
            quantize_gguf(cfg, src, dest, quantize);
        } else {
            emit(json{{"status", "importing " + base_name(src) + " -> " + dest + " (" +
                                     human_bytes(ec ? 0 : size) + ") ..."}});
            copy_file_preserving_mtime(src, dest);
        }
        if (!draft_quantize.empty()) {
            const std::string d = (fs::path(dest_dir) / (safe + ".draft.gguf")).string();
            emit(json{{"status", "quantizing " + base_name(src) + " -> " + base_name(d) + " (" +
                                     upper(draft_quantize) + ") ..."}});
            quantize_gguf(cfg, src, d, draft_quantize);
        }
    } catch (const std::exception & e) {
        emit(error_obj(e.what()));
        return;
    }

    const std::string mm = find_projector(src);
    if (!mm.empty()) {
        try {
            copy_file_preserving_mtime(mm, (fs::path(dest_dir) / (safe + ".mmproj.gguf")).string());
        } catch (const std::exception &) {
            // Go ignores this copy's error too; the model itself is already in.
        }
        emit(json{{"status", "also imported its projector (" + base_name(mm) + ")"}});
    }
    emit(json{{"status", "created '" + name + "'  (llmash serves it as " + loose_name(dest) + ")"}});
    emit(json{{"status", "success"}});
}

void run_copy(const Config & cfg, const Model & source, const std::string & destination,
              const Emit & emit) {
    if (!file_exists(source.path) || !is_gguf(source.path)) {
        emit(error_obj("can't copy '" + source.name +
                       "': its backing file isn't a GGUF llmash can duplicate."));
        return;
    }
    const std::string dest_dir = loose_dir(cfg);
    std::error_code   ec;
    fs::create_directories(fs::path(dest_dir), ec);

    const std::string dest =
        (fs::path(dest_dir) / (safe_model_name(destination) + ".gguf")).string();
    if (file_exists(dest)) {
        emit(error_obj("'" + base_name(dest) + "' already exists in " + dest_dir +
                       ". Remove it first (rm) or pick another name."));
        return;
    }
    const double size = static_cast<double>(fs::file_size(fs::path(source.path), ec));
    emit(json{{"status", "copying " + base_name(source.path) + " -> " + base_name(dest) + " (" +
                             human_bytes(ec ? 0 : size) + ") ..."}});
    try {
        copy_file_preserving_mtime(source.path, dest);
    } catch (const std::exception & e) {
        emit(error_obj(e.what()));
        return;
    }
    emit(json{{"status", "copied '" + source.name + "' to '" + destination + "'"}});
    emit(json{{"status", "success"}});
}

// ------------------------------------------------------------- delete

json read_manifest(const fs::path & p) {
    std::ifstream in(p, std::ios::binary);
    if (!in) {
        return json::object();
    }
    const json j = json::parse(in, nullptr, false);
    return j.is_object() ? j : json::object();
}

// Every blob a manifest names, config layer included.
std::vector<std::string> digests_of(const json & j) {
    std::vector<std::string> out;
    json layers = j.value("layers", json::array());
    if (j.contains("config") && j["config"].is_object()) {
        layers.push_back(j["config"]);
    }
    for (const auto & layer : layers) {
        std::string d = layer.value("digest", std::string());
        if (!d.empty()) {
            std::replace(d.begin(), d.end(), ':', '-');
            out.push_back(d);
        }
    }
    return out;
}

// Every blob digest named by any manifest under a store, so a blob shared
// with a model that is staying is never collected.
std::set<std::string> digests_in_use(const fs::path & manifests,
                                     const std::string & except) {
    std::set<std::string> used;
    std::error_code       ec;
    for (auto it = fs::recursive_directory_iterator(
             manifests, fs::directory_options::skip_permission_denied, ec);
         it != fs::recursive_directory_iterator(); it.increment(ec)) {
        if (ec) {
            ec.clear();
            continue;
        }
        if (!it->is_regular_file(ec) || it->path().string() == except) {
            continue;
        }
        for (const auto & d : digests_of(read_manifest(it->path()))) {
            used.insert(d);
        }
    }
    return used;
}

// Remove a model the Ollama way: drop its manifest, then the blobs no
// remaining manifest still refers to.
DeleteOutcome delete_from_store(const Model & m) {
    DeleteOutcome out;
    std::error_code ec;

    const std::vector<std::string> mine = digests_of(read_manifest(fs::path(m.manifest)));

    if (!fs::remove(fs::path(m.manifest), ec) || ec) {
        out.status = 500;
        out.body   = error_obj("could not remove " + m.name + "'s manifest: " +
                             (ec ? ec.message() : std::string("it is still there")));
        return out;
    }
    // Ollama leaves the empty <name> folder behind; tidy it so `list` on the
    // store does not show a directory with nothing in it.
    fs::path dir = fs::path(m.manifest).parent_path();
    for (int i = 0; i < 3 && fs::is_empty(dir, ec) && !ec; i++) {
        if (!fs::remove(dir, ec) || ec) {
            break;
        }
        dir = dir.parent_path();
    }

    std::vector<std::string> removed{m.name};
    const auto used = digests_in_use(fs::path(m.store_root) / "manifests", m.manifest);
    uint64_t freed = 0;
    for (const auto & d : mine) {
        if (used.count(d)) {
            continue;
        }
        const fs::path blob = fs::path(m.store_root) / "blobs" / d;
        const auto     sz   = fs::file_size(blob, ec);
        if (!ec && fs::remove(blob, ec) && !ec) {
            freed += sz;
            removed.push_back(d);
        }
    }
    out.status = 200;
    out.body   = json{{"status", "success"}, {"removed", removed},
                      {"freed", freed}};
    return out;
}

DeleteOutcome run_delete(const Model & m, const Config & cfg) {
    DeleteOutcome out;
    // in_library covers every folder read in place, and one of them is the
    // folder pulls are written to. Refusing there means rm can never undo a
    // pull, so only the folders llmash did not write are protected.
    if (m.in_library && !same_dir(fs::path(m.path).parent_path().string(), loose_dir(cfg))) {
        out.status = 409;
        out.body   = error_obj(m.name + " is read from " + m.path +
                             ", a folder llmash only reads; delete the file yourself");
        return out;
    }
    std::vector<std::string> removed;
    // An Ollama-store model is a manifest naming blobs that other manifests
    // may also name, so deleting `path` is both wrong and not enough: the
    // blob carries no .gguf extension, the shard scan below matched nothing,
    // and `rm` answered "nothing to delete" for a model plainly in the list.
    if (!m.manifest.empty()) {
        return delete_from_store(m);
    }
    if (file_exists(m.path)) {
        const fs::path    dir  = fs::path(m.path).parent_path();
        const std::string stem = strip_shard(stem_of(m.path));

        std::vector<std::string> parts;
        std::error_code          ec;
        for (auto it = fs::directory_iterator(dir, ec); !ec && it != fs::directory_iterator();
             ++it) {
            const std::string base = it->path().filename().string();
            if (base.rfind(stem, 0) == 0 && lower(it->path().extension().string()) == ".gguf") {
                parts.push_back(it->path().string());
            }
        }
        std::sort(parts.begin(), parts.end());
        for (const auto & part : parts) {
            std::error_code rm;
            if (!fs::remove(fs::path(part), rm) || rm) {
                out.status = 500;
                out.body   = error_obj("could not delete " + base_name(part) + ": " +
                                     (rm ? rm.message() : std::string("file still present")));
                return out;
            }
            removed.push_back(base_name(part));
        }
    }
    if (removed.empty()) {
        out.status = 409;
        out.body   = error_obj("found " + m.name + " but nothing to delete. Its file is at " +
                             m.path + ", outside the model directory");
        return out;
    }
    out.status = 200;
    out.body   = json{{"status", "success"}, {"removed", removed}};
    return out;
}

// -------------------------------------------------------------- cli cache

std::string CliTextCache::get(const std::string & kind, double ttl_seconds,
                              const std::function<std::string()> & build) {
    {
        std::lock_guard<std::mutex> lock(mu_);
        const auto                  it = cache_.find(kind);
        if (it != cache_.end() && now_unix() - it->second.at < ttl_seconds) {
            return it->second.text;
        }
    }
    std::string text = build ? build() : std::string();
    {
        std::lock_guard<std::mutex> lock(mu_);
        cache_[kind] = Entry{now_unix(), text};
    }
    return text;
}

void CliTextCache::invalidate() {
    std::lock_guard<std::mutex> lock(mu_);
    cache_.clear();
}

// -------------------------------------------------------------- process

Subprocess::Subprocess() : proc_(std::make_unique<subprocess_s>()) {}

Subprocess::~Subprocess() {
    if (started_) {
        subprocess_join(proc_.get(), nullptr);
        subprocess_destroy(proc_.get());
    }
}

bool Subprocess::start(const std::vector<std::string> & args) {
    if (started_ || args.empty()) {
        return false;
    }
    std::vector<const char *> argv;
    argv.reserve(args.size() + 1);
    for (const auto & a : args) {
        argv.push_back(a.c_str());
    }
    argv.push_back(nullptr);
    const int options = subprocess_option_no_window | subprocess_option_inherit_environment;
    started_          = subprocess_create(argv.data(), options, proc_.get()) == 0;
    return started_;
}

int Subprocess::join() {
    if (!started_) {
        return -1;
    }
    int code = -1;
    if (subprocess_join(proc_.get(), &code) != 0) {
        return -1;
    }
    return code;
}

} // namespace llmash
