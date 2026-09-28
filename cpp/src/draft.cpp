#include "draft.h"
#include "caps.h"
#include "gguf_io.h"

#include "gguf.h"
#include "pull.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>

namespace fs = std::filesystem;
using json   = nlohmann::json;

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

bool starts_with(const std::string & s, const std::string & prefix) { return s.rfind(prefix, 0) == 0; }

bool contains(const std::string & hay, const std::string & needle) {
    return hay.find(needle) != std::string::npos;
}

void replace_all(std::string & s, const std::string & from, const std::string & to) {
    if (from.empty()) {
        return;
    }
    for (size_t i = s.find(from); i != std::string::npos; i = s.find(from, i + to.size())) {
        s.replace(i, from.size(), to);
    }
}

std::string stem_of(const std::string & p) { return fs::path(p).stem().string(); }

const std::regex & shard_suffix_re() {
    static const std::regex re(R"(-\d+-of-\d+$)");
    return re;
}

const std::regex & quant_suffix_re() {
    static const std::regex re(R"(-(?:i?q\d+(?:_[a-z0-9]+)*|f16|bf16|f32|mxfp4|nvfp4|rco-\d+)$)", std::regex::icase);
    return re;
}

std::string strip_shard(const std::string & stem) { return std::regex_replace(stem, shard_suffix_re(), ""); }

// registry.go's pairStem: the quantisation is not part of a model's identity,
// so two files that differ only by it pair with the same sidecar.
std::string pair_stem(std::string stem) {
    std::string prev;
    while (prev != stem) {
        prev = stem;
        stem = std::regex_replace(stem, quant_suffix_re(), "");
    }
    static const std::regex ud(R"(-ud$)", std::regex::icase);
    return lower(std::regex_replace(stem, ud, ""));
}

std::string trim_set(const std::string & s, const char * set) {
    size_t b = 0, e = s.size();
    while (b < e && std::strchr(set, s[b]) != nullptr) b++;
    while (e > b && std::strchr(set, s[e - 1]) != nullptr) e--;
    return s.substr(b, e - b);
}

// ------------------------------------------------- the target's own header

// The keys hfRepoOf and modelStem read.
struct RepoMeta {
    std::string bm_repo_url, repo_url, bm_org, bm_name, size_label, name, basename;
};

RepoMeta read_repo_meta(const std::string & path) {
    RepoMeta      m;
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return m;
    }
    char magic[4] = {};
    if (!in.read(magic, 4) || std::memcmp(magic, "GGUF", 4) != 0) {
        return m;
    }
    GGUFReader g(in);
    g.u32(); // version
    g.u64(); // tensor count
    const uint64_t n_kv = g.u64();
    if (g.bad()) {
        return m;
    }
    for (uint64_t i = 0; i < n_kv && !g.bad(); i++) {
        const std::string k = g.str();
        const uint32_t    t = g.u32();
        if (g.bad()) {
            break;
        }
        if (t != GGUF_STRING) {
            g.skip_value(t);
            continue;
        }
        const std::string v = g.str();
        if (g.bad()) {
            break;
        }
        if (k == "general.base_model.0.repo_url") {
            m.bm_repo_url = v;
        } else if (k == "general.repo_url") {
            m.repo_url = v;
        } else if (k == "general.base_model.0.organization") {
            m.bm_org = v;
        } else if (k == "general.base_model.0.name") {
            m.bm_name = v;
        } else if (k == "general.size_label") {
            m.size_label = v;
        } else if (k == "general.name") {
            m.name = v;
        } else if (k == "general.basename") {
            m.basename = v;
        }
    }
    return m;
}

std::string hf_repo_of(const RepoMeta & m) {
    static const std::regex re(R"(huggingface\.co/([^/\s]+/[^/\s]+))");
    for (const std::string & s : {m.bm_repo_url, m.repo_url}) {
        std::smatch hit;
        if (std::regex_search(s, hit, re)) {
            std::string r = hit[1].str();
            while (!r.empty() && r.back() == '/') {
                r.pop_back();
            }
            return r;
        }
    }
    if (!m.bm_org.empty() && !m.bm_name.empty()) {
        std::string n = m.bm_name;
        std::replace(n.begin(), n.end(), ' ', '-');
        return m.bm_org + "/" + n;
    }
    return "";
}

// The skip has to be exact: scan_gguf has already read the array's element
// type and length, which is why GGUFReader::skip_value cannot be used here.
void skip_array_body(GGUFReader & g, uint32_t et, uint64_t n) {
    switch (et) {
        case GGUF_U8:
        case GGUF_I8:
        case GGUF_BOOL: g.skip(static_cast<int64_t>(n)); return;
        case GGUF_U16:
        case GGUF_I16: g.skip(static_cast<int64_t>(n) * 2); return;
        case GGUF_U32:
        case GGUF_I32:
        case GGUF_F32: g.skip(static_cast<int64_t>(n) * 4); return;
        case GGUF_U64:
        case GGUF_I64:
        case GGUF_F64: g.skip(static_cast<int64_t>(n) * 8); return;
        case GGUF_STRING:
            for (uint64_t i = 0; i < n && !g.bad(); i++) {
                const uint64_t ln = g.u64();
                if (g.bad()) {
                    return;
                }
                g.skip(static_cast<int64_t>(ln));
            }
            return;
        default:
            for (uint64_t i = 0; i < n && !g.bad(); i++) {
                g.skip_value(et);
            }
            return;
    }
}

// registry.go's findMtp/findEagle3/findDraft and findDspark, which the C++
// Model carries no fields for.
std::string sidecar_named(const std::string & gguf, const char * suffix) {
    const std::string stem = strip_shard(stem_of(gguf));
    const fs::path    c    = fs::path(gguf).parent_path() / (stem + suffix);
    std::error_code   ec;
    return fs::is_regular_file(c, ec) ? c.string() : std::string();
}

std::string find_dspark(const std::string & gguf) {
    if (const std::string direct = sidecar_named(gguf, ".dspark.gguf"); !direct.empty()) {
        return direct;
    }
    const std::string want = pair_stem(strip_shard(stem_of(gguf)));
    const fs::path    dir  = fs::path(gguf).parent_path();
    std::error_code   ec;
    std::vector<std::string> cands;
    for (auto it = fs::directory_iterator(dir, ec); !ec && it != fs::directory_iterator(); ++it) {
        const std::string name = it->path().filename().string();
        if (ends_with(lower(name), ".dspark.gguf")) {
            cands.push_back(it->path().string());
        }
    }
    std::sort(cands.begin(), cands.end());
    for (const auto & c : cands) {
        std::string s = stem_of(c);
        if (ends_with(s, ".dspark")) {
            s = s.substr(0, s.size() - 7);
        }
        if (pair_stem(s) == want) {
            return c;
        }
    }
    return "";
}

} // namespace

// ================================================================ helpers

std::string sidecar_path(const std::string & gguf, const char * suffix) { return sidecar_named(gguf, suffix); }

std::string dspark_path(const std::string & gguf) { return find_dspark(gguf); }

// An MTP drafter is trained against the model, not one quantisation of it: <stem>.mtp.gguf first, else one beside
// another build of the same model (Gemma's assistant drafter serves its Q4, Q8 and RCO builds alike).
std::string mtp_path(const std::string & gguf) {
    if (const std::string direct = sidecar_named(gguf, ".mtp.gguf"); !direct.empty()) {
        return direct;
    }
    // an Ollama store's blobs are content-addressed; nothing there is named for a model
    if (fs::path(gguf).parent_path().filename() == "blobs") {
        return "";
    }
    const std::string        want = pair_stem(strip_shard(stem_of(gguf)));
    std::error_code          ec;
    std::vector<std::string> cands;
    for (auto it = fs::directory_iterator(fs::path(gguf).parent_path(), ec); !ec && it != fs::directory_iterator(); ++it) {
        if (ends_with(lower(it->path().filename().string()), ".mtp.gguf")) {
            cands.push_back(it->path().string());
        }
    }
    std::sort(cands.begin(), cands.end());
    for (const auto & c : cands) {
        std::string st = stem_of(c);
        if (ends_with(lower(st), ".mtp")) {
            st = st.substr(0, st.size() - 4);
        }
        if (pair_stem(strip_shard(st)) == want) {
            return c;
        }
    }
    return "";
}

std::string normalise(const std::string & s) {
    std::string out;
    for (const char c : lower(s)) {
        if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9')) {
            out += c;
        }
    }
    return out;
}

// A DFlash drafter is trained against a base model and drafts for any quantisation of it, so
// besides <stem>.dflash.gguf it is matched by the base model the target's header names.
std::string dflash_path(const std::string & gguf) {
    if (const std::string direct = sidecar_named(gguf, ".dflash.gguf"); !direct.empty()) {
        return direct;
    }
    const std::string want = normalise(read_repo_meta(gguf).bm_name);
    if (want.empty()) {
        return "";
    }
    static const std::regex tag(R"([-_.]?dflash\d*)", std::regex::icase);
    std::error_code          ec;
    std::vector<std::string> cands;
    for (auto it = fs::directory_iterator(fs::path(gguf).parent_path(), ec); !ec && it != fs::directory_iterator(); ++it) {
        const std::string name = lower(it->path().filename().string());
        if (name.find("dflash") != std::string::npos && ends_with(name, ".gguf")) {
            cands.push_back(it->path().string());
        }
    }
    std::sort(cands.begin(), cands.end());
    for (const auto & c : cands) {
        if (normalise(std::regex_replace(pair_stem(strip_shard(stem_of(c))), tag, "")) == want) {
            return c;
        }
    }
    return "";
}

// A size already in the name (27B, 0.6B, E4B) is the one repos are named by; a header's size label can count the
// parameters another way (gemma-4-E4B's says 7.5B).
static bool names_size(const std::string & name) {
    static const std::regex re(R"((^|[-_.])[eE]?[0-9]+(\.[0-9]+)?[bBmM]([-_.]|$))");
    return std::regex_search(name, re);
}

// The packaging words and the quantisation taken off a name, spaces to dashes.
static std::string bare_name(std::string name) {
    std::replace(name.begin(), name.end(), ' ', '-');
    for (const char * junk : {"-GGUF", "-gguf", "-it-GGUF", "-UD", "-Instruct", "-MTP", "-mtp", "-NVFP4", "-nvfp4"}) {
        if (ends_with(name, junk)) {
            name = name.substr(0, name.size() - std::strlen(junk));
        }
    }
    name = std::regex_replace(name, quant_suffix_re(), "");
    return trim_set(name, "-_. ");
}

static bool is_base_of(const std::string & base, const std::string & name) {
    const std::string b = normalise(base), n = normalise(name);
    return b == n + "base" || b == n;
}

ModelIdent identify(const Model & m) {
    const RepoMeta meta = read_repo_meta(m.path);
    ModelIdent     id;

    // the model's own name: what the converter wrote (Qwen3.5 + 4B), else the registry name with its tag, else
    // the file's name; the base it names is not it, even when that base is the same model before tuning
    std::string       name = m.name;
    const size_t      colon = name.find(':');
    const std::string tail  = colon == std::string::npos ? std::string() : name.substr(colon + 1);
    name                    = colon == std::string::npos ? name : name.substr(0, colon);
    if (!meta.basename.empty() && !meta.size_label.empty()) {
        // a converter that wrote the whole name as the basename (Qwen3.5-4B) does not get its size twice
        name = names_size(meta.basename) || meta.size_label.find('x') != std::string::npos ? meta.basename
                                                                                             : meta.basename + "-" + meta.size_label;
    } else if (!meta.name.empty() && names_size(meta.name)) {
        name = meta.name;
    } else if (colon != std::string::npos && tail != "latest" && tail != "gguf") {
        name += "-" + tail;  // a registry tag carries the size (e2b, 26b-a4b)
    } else if (!meta.size_label.empty() && !names_size(name) && meta.size_label.find('x') == std::string::npos) {
        name += "-" + meta.size_label;
    }
    id.name = bare_name(name);

    // A header that cannot name the model (no basename, a name without its size: Agents-A1's says "A1") is named by
    // where the file came from: the repository the pull was asked for, or the model the hub's card says the
    // repository quantized. A card naming a fine-tune's base counts as the header naming it.
    const bool  poor      = meta.basename.empty() && !names_size(meta.name);
    std::string base_repo = hf_repo_of(meta);
    if (poor || base_repo.empty()) {
        const SourceRecord src = read_source(fs::path(m.path).parent_path().string(), fs::path(m.path).filename().string());
        if (!src.repo.empty()) {
            if (poor && !src.from.empty() && src.from != src.repo) {
                id.name = bare_name(src.from.substr(src.from.find('/') + 1));
            } else {
                std::string       relation;
                const std::string card = hf_base_model(src.repo, &relation);
                if (!card.empty() && poor && (relation == "quantized" || relation.empty())) {
                    id.name = bare_name(card.substr(card.find('/') + 1));
                } else if (!card.empty() && base_repo.empty()) {
                    base_repo = "https://huggingface.co/" + card;
                }
            }
        }
    }
    // Meta-Llama-3.1-8B is published everywhere as Llama-3.1-8B
    if (starts_with(id.name, "Meta-")) {
        id.name = id.name.substr(5);
    }

    // the base the header names: the same model before instruction tuning is not another model; anything else is
    if (!base_repo.empty()) {
        std::string base = bare_name(base_repo.substr(base_repo.find('/') + 1));
        if (ends_with(lower(base), "-base")) {
            base = base.substr(0, base.size() - 5);
        }
        if (!is_base_of(base, id.name) && !base.empty()) {
            id.tuned_from = base;
        }
    }
    return id;
}

std::string model_stem(const Model & m) { return identify(m).name; }

bool hub_info_reason(const std::string & repo, HubModel & out, std::string & err) {
    const HttpResult r =
        http_request("https://huggingface.co/api/models/" + repo, "GET", "",
                     {"User-Agent: llmash", "Accept: application/json"});
    if (!r.error.empty()) {
        err = r.error;
        return false;
    }
    if (r.status == 429) {
        err = "Hugging Face is rate limiting this address; try again in a minute";
        return false;
    }
    if (r.status != 200) {
        err = "hub answered " + std::to_string(r.status);
        return false;
    }
    const json doc = json::parse(r.body, nullptr, false);
    if (doc.is_discarded() || !doc.is_object()) {
        err = "hub answered with something that is not JSON";
        return false;
    }
    out = HubModel{};
    if (doc.contains("id") && doc["id"].is_string()) {
        out.id = doc["id"].get<std::string>();
    }
    if (doc.contains("downloads") && doc["downloads"].is_number()) {
        out.downloads = doc["downloads"].get<int>();
    }
    if (doc.contains("tags") && doc["tags"].is_array()) {
        for (const auto & t : doc["tags"]) {
            if (t.is_string()) {
                out.tags.push_back(t.get<std::string>());
            }
        }
    }
    if (doc.contains("cardData") && doc["cardData"].is_object() && doc["cardData"].contains("base_model")) {
        const json & bm = doc["cardData"]["base_model"];
        if (bm.is_string()) {
            out.base_models.push_back(bm.get<std::string>());
        } else if (bm.is_array()) {
            for (const auto & b : bm) {
                out.base_models.push_back(b.is_string() ? b.get<std::string>() : b.dump());
            }
        }
    }
    return true;
}

std::string foreign_base(const std::vector<std::string> & bases, const std::string & want) {
    static const char * markers[] = {"abliterat", "uncensored", "caption",  "distill",  "merge",
                                     "roleplay",  "magic",      "agentic",  "heretic",  "aggressive"};
    for (const auto & b : bases) {
        std::string  name = b;
        const size_t i    = name.find_last_of('/');
        if (i != std::string::npos) {
            name = name.substr(i + 1);
        }
        // the model itself, whatever its name says it was made from (a distill is its own base)
        if (!want.empty() && normalise(name) == want) {
            continue;
        }
        const std::string low = lower(b);
        for (const char * m : markers) {
            if (contains(low, m)) {
                return b;
            }
        }
        std::string rest = normalise(name);
        if (!want.empty()) {
            const size_t p = rest.find(want);
            if (p != std::string::npos) {
                rest.erase(p, want.size());
            }
        }
        for (const char * innocuous :
             {"instruct", "it",     "chat",   "base",   "gguf",   "hf",     "llamacpp", "llama",
              "nvfp",     "fp",     "unsloth", "quant",
              "speculator", "eagle3", "eagle", "dspark", "dflash", "draft",  "model", "assistant",
              "f16",      "bf16",   "fp16",   "q4km",   "q4",     "q8",     "iq4xs",    "test", "preview",
              "v1",       "v2",     "v3",     "0",      "1",      "2",      "3",        "4",    "5",
              "6",        "7",      "8",      "9"}) {
            replace_all(rest, innocuous, "");
        }
        if (rest.size() > 3) {
            return b;
        }
    }
    return "";
}

std::string other_runtime(const std::string & repo) {
    for (const char * r : {"mlx", "exl3", "exl2", "bpw", "ov-int", "openvino", "awq", "gptq", "-int4", "-int8",
                           "rocm", "trt", "tensorrt"}) {
        if (contains(lower(repo), r)) {
            std::string s = r;
            if (starts_with(s, "-")) {
                s = s.substr(1);
            }
            return s;
        }
    }
    return "";
}

bool prefer_quant(const std::string & a, const std::string & b) {
    static const char * order[] = {"q4_k_m", "q4_k", "iq4_xs", "q5_k_m", "q8_0", "f16", "bf16"};
    const auto          rank    = [](const std::string & s) {
        const std::string low = lower(s);
        int               i   = 0;
        for (const char * q : order) {
            if (contains(low, q)) {
                return i;
            }
            i++;
        }
        return i;
    };
    return rank(a) < rank(b);
}

// ============================================================= candidates

bool assistant_named(const std::string & repo) {
    std::string  base = lower(repo);
    const size_t i    = base.find_last_of('/');
    if (i != std::string::npos) {
        base = base.substr(i + 1);
    }
    return contains(base, "-assistant") || contains(base, "_assistant") || contains(base, ".assistant");
}

bool consider_repo(const HubModel & hit, const std::string & want, const std::string & stem, const Say & say,
                   DraftCand & out) {
    const auto tell = [&](const std::string & line) {
        if (say) {
            say(line);
        }
    };

    const DraftKind * kind = kind_of(hit.id);
    if (kind == nullptr) {
        for (const auto & t : hit.tags) {
            kind = kind_of(t);
            if (kind != nullptr) {
                break;
            }
        }
    }
    // Gemma 4 publishes its MTP drafter as <model>-assistant, whatever the repo's tags say; the word also names chat
    // fine-tunes, so the file's architecture decides (fits_target)
    if (assistant_named(hit.id)) {
        kind = &draft_kinds()[0];
    }
    if (kind == nullptr) {
        return false;
    }

    HubModel    info;
    std::string err;
    if (!hub_info_reason(hit.id, info, err)) {
        tell("    " + hit.id + ": " + err);
        return false;
    }
    const std::vector<std::string> & bases = info.base_models;

    bool matched = false;
    for (const auto & b : bases) {
        if (contains(normalise(b), want)) {
            matched = true;
        }
    }
    if (!matched && !contains(normalise(hit.id), want)) {
        tell("    " + hit.id + ": does not name " + stem + " as a base");
        return false;
    }
    if (const std::string derived = foreign_base(bases, want); !derived.empty()) {
        tell("    " + hit.id + ": built for " + derived + ", a different model");
        return false;
    }
    if (const std::string runtime = other_runtime(hit.id); !runtime.empty()) {
        tell("    " + hit.id + ": a " + runtime + " build, which llama.cpp does not load");
        return false;
    }
    if (bases.empty()) {
        if (const std::string derived = foreign_base({hit.id}, want); !derived.empty()) {
            tell("    " + hit.id + ": the name says it is a project of its own, not a drafter for " + stem);
            return false;
        }
    }

    HubFile best;
    for (const auto & f : hub_files(hit.id)) {
        if (f.type != "file" || !ends_with(lower(f.path), ".gguf")) {
            continue;
        }
        if (best.path.empty() || prefer_quant(f.path, best.path)) {
            best = f;
        }
    }
    if (best.path.empty()) {
        tell("    " + hit.id + ": " + kind->name + ", but only safetensors (needs converting)");
        return false;
    }
    if (best.size > DRAFT_SIZE_CEILING) {
        tell("    " + hit.id + ": " + human_bytes(best.size) + " is too big for a drafter; that is a whole model");
        return false;
    }

    out       = DraftCand{};
    out.repo  = hit.id;
    out.file  = best.path;
    out.kind  = kind;
    out.size  = best.size;
    out.bases = bases;
    out.score = kind->rank;
    if (info.downloads > 100) {
        out.score += 5;
    }
    if (best.size > 0 && best.size < (1ll << 30)) {
        out.score += 3;
    }
    out.note = kind->name + ", " + human_bytes(best.size);
    return true;
}

// A full build of the model that carries its MTP head (an -MTP GGUF): the head alone is fetched from it, as a
// sidecar. The candidate's size is the file's; the fetch is a fraction of it.
static bool consider_embedded(const HubModel & hit, const std::string & name, const Say & say, DraftCand & out) {
    const std::string low = lower(hit.id);
    if (low.find("mtp") == std::string::npos || low.find("gguf") == std::string::npos) {
        return false;
    }
    if (!gguf_of_model(hit, name)) {
        say("    " + hit.id + ": not a build of " + name);
        return false;
    }
    if (const std::string runtime = other_runtime(hit.id); !runtime.empty()) {
        return false;
    }
    HubFile best;
    for (const auto & f : hub_files(hit.id)) {
        const std::string fl = lower(f.path);
        if (f.type != "file" || !ends_with(fl, ".gguf") || contains(fl, "mmproj") || kind_of(f.path) != nullptr) {
            continue;
        }
        if (best.path.empty() || prefer_quant(f.path, best.path)) {
            best = f;
        }
    }
    if (best.path.empty()) {
        return false;
    }
    out          = DraftCand{};
    out.repo     = hit.id;
    out.file     = best.path;
    out.kind     = &draft_kinds()[0];
    out.size     = best.size;
    out.embedded = true;
    out.score    = out.kind->rank + (hit.downloads > 100 ? 5 : 0);
    out.note     = "MTP head, extracted from a " + human_bytes(best.size) + " build";
    return true;
}

static std::vector<DraftCand> search_for(const std::string & name, const Say & say,
                                         const std::chrono::steady_clock::time_point & until) {
    const std::string        want = normalise(name);
    std::vector<std::string> seen;
    std::vector<DraftCand>   cands;
    for (const std::string & q : {name + " MTP GGUF", name + " assistant", name + " dflash", name + " eagle3",
                                  name + " dspark", name + " speculator", name + " draft GGUF"}) {
        if (std::chrono::steady_clock::now() > until) {
            break;
        }
        for (const auto & hit : hub_search(q, 25)) {
            if (std::find(seen.begin(), seen.end(), hit.id) != seen.end()) {
                continue;
            }
            seen.push_back(hit.id);
            DraftCand c;
            if (consider_repo(hit, want, name, say, c) || consider_embedded(hit, name, say, c)) {
                cands.push_back(std::move(c));
            }
        }
    }
    std::stable_sort(cands.begin(), cands.end(),
                     [](const DraftCand & a, const DraftCand & b) { return a.score > b.score; });
    return cands;
}

// Drafters for the model these weights are; when none is published under its own name and the header names the
// model it was tuned from, that model's, said so: they share the vocabulary and width, and draft somewhat less well.
std::vector<DraftCand> find_drafters(const Model & m, bool verbose, const Say & say_in, double budget_s) {
    const ModelIdent id = identify(m);
    if (id.name.empty()) {
        return {};
    }
    // a budget (the offer after a pull) ends the search early; `pulldraft` runs it whole
    const auto until = std::chrono::steady_clock::now() +
                       std::chrono::milliseconds(static_cast<long long>((budget_s > 0 ? budget_s : 3600) * 1000));
    const Say say = [&](const std::string & line) {
        if (verbose && say_in) {
            say_in(line);
        }
    };
    say("  looking for a drafter trained on " + id.name);
    std::vector<DraftCand> cands = search_for(id.name, say, until);
    if (cands.empty() && !id.tuned_from.empty()) {
        say("  none under its own name; looking under " + id.tuned_from + ", which it was tuned from");
        cands = search_for(id.tuned_from, say, until);
        for (DraftCand & c : cands) {
            c.via = id.tuned_from;
            c.note += ", trained on " + id.tuned_from + " which this model was tuned from";
        }
    }
    return cands;
}

// =========================================================== verification

bool GGUFSpec::hidden() const { return arch == "eagle3" || arch == "dflash" || arch == "dspark"; }

std::string scan_gguf(std::istream & in, bool want_tensors, GGUFSpec & s) {
    s = GGUFSpec{};

    char magic[4] = {};
    if (!in.read(magic, 4) || std::memcmp(magic, "GGUF", 4) != 0) {
        return "not a GGUF file";
    }
    GGUFReader     g(in);
    const uint32_t ver = g.u32();
    if (g.bad()) {
        return "not a GGUF file";
    }
    if (ver < 2 || ver > 3) {
        return "GGUF version " + std::to_string(ver) + ", which this llama.cpp does not read";
    }
    const uint64_t n_tensors = g.u64();
    const uint64_t n_kv      = g.u64();
    if (g.bad()) {
        return "not a GGUF file";
    }
    s.tensors = static_cast<int64_t>(n_tensors);
    if (s.tensors == 0) {
        return "no tensors in the file";
    }

    // Everything below may run out of bytes: a truncated header is not an
    // error, it is a partial answer.
    s.partial = true;
    for (uint64_t i = 0; i < n_kv; i++) {
        const std::string k = g.str();
        if (g.bad()) {
            return "";
        }
        const uint32_t t = g.u32();
        if (g.bad()) {
            return "";
        }
        if (t != GGUF_ARRAY) {
            if (t == GGUF_STRING) {
                const std::string v = g.str();
                if (g.bad()) {
                    return "";
                }
                if (k == "general.architecture") {
                    s.arch = v;
                }
                continue;
            }
            int64_t    v       = 0;
            const bool numeric = g.read_int(t, v);
            if (g.bad()) {
                return "";
            }
            if (!numeric) {
                continue;
            }
            if (ends_with(k, ".embedding_length")) {
                s.embed = v;
            } else if (ends_with(k, ".embedding_length_out")) {
                s.embed_out = v;
            } else if (ends_with(k, ".block_count")) {
                s.blocks = v;
            } else if (ends_with(k, ".vocab_size") && s.vocab == 0) {
                s.vocab = v;
            }
            continue;
        }
        // An array's length comes before its contents, which is all the
        // vocabulary count is.
        const uint32_t et = g.u32();
        const uint64_t n  = g.u64();
        if (g.bad()) {
            return "";
        }
        if (k == "tokenizer.ggml.tokens") {
            s.vocab = static_cast<int64_t>(n);
        }
        if (ends_with(k, ".target_layers") && n <= 64 && et >= GGUF_U32 && et <= GGUF_I32) {
            for (uint64_t j = 0; j < n; j++) {
                int64_t id = 0;
                g.read_int(et, id);
                if (g.bad()) {
                    return "";
                }
                s.layers.push_back(id);
            }
            continue;
        }
        skip_array_body(g, et, n);
        if (g.bad()) {
            return "";
        }
    }
    if (!want_tensors) {
        s.partial = false;
        return "";
    }

    for (uint64_t i = 0; i < n_tensors; i++) {
        const std::string name = g.str();
        if (g.bad()) {
            return "";
        }
        const uint32_t nd = g.u32();
        if (g.bad() || nd > 4) {
            return "";
        }
        std::vector<int64_t> dims;
        for (uint32_t d = 0; d < nd; d++) {
            const uint64_t v = g.u64();
            if (g.bad()) {
                return "";
            }
            dims.push_back(static_cast<int64_t>(v));
        }
        g.u32(); // ggml type
        g.u64(); // offset
        if (g.bad()) {
            return "";
        }
        if ((name == "fc.weight" || ends_with(name, ".fc.weight")) && !dims.empty()) {
            s.enc = dims[0];
        }
        if (name.rfind("blk.", 0) == 0 && name.find(".nextn.") != std::string::npos) {
            s.nextn_layer = std::max<int64_t>(s.nextn_layer, std::atoll(name.c_str() + 4));
        }
    }
    s.partial = false;
    return "";
}

std::string spec_of_file(const std::string & path, GGUFSpec & out) {
    out = GGUFSpec{};
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return "could not open " + path;
    }
    return scan_gguf(in, true, out);
}

std::string spec_of_url(const std::string & url, GGUFSpec & out, bool & unreadable) {
    unreadable = false;
    out        = GGUFSpec{};
    std::string last;
    for (const int64_t window : {1ll << 20, 24ll << 20}) {
        char range[64];
        std::snprintf(range, sizeof(range), "bytes=0-%lld", static_cast<long long>(window - 1));
        const HttpResult r = http_request(url, "GET", range, {"User-Agent: llmash"});
        if (!r.error.empty()) {
            unreadable = true;
            return r.error;
        }
        if (r.status != 200 && r.status != 206) {
            unreadable = true;
            switch (r.status) {
                case 401:
                case 403: return "the hub wants an account for this file";
                case 429: return "the hub is rate limiting this address";
                default: return "the hub answered " + std::to_string(r.status);
            }
        }
        const std::string body =
            r.body.size() > static_cast<size_t>(window) ? r.body.substr(0, static_cast<size_t>(window)) : r.body;
        std::istringstream in(body, std::ios::binary);
        GGUFSpec           s;
        const std::string  err = scan_gguf(in, true, s);
        if (!err.empty()) {
            out = s;
            return err;
        }
        if (!s.partial) {
            out = s;
            return "";
        }
        last = "its header is longer than " + human_bytes(window);
    }
    out.partial = true;
    unreadable  = true;
    return last;
}

std::string pairs(const GGUFSpec & target, const GGUFSpec & draft) {
    if (draft.arch.empty()) {
        return "no architecture in its header";
    }
    if (target.vocab > 0 && draft.vocab > 0) {
        const int64_t diff = target.vocab - draft.vocab;
        if (diff > 128 || diff < -128) {
            return "its vocabulary has " + std::to_string(draft.vocab) + " tokens against this model's " +
                   std::to_string(target.vocab);
        }
    }
    // an assistant drafter works in the target's own hidden width
    if (draft.embed_out > 0 && target.embed > 0 && draft.embed_out != target.embed) {
        return "it drafts for a model " + std::to_string(draft.embed_out) + " wide, and this one is " +
               std::to_string(target.embed);
    }
    if (draft.nextn_layer >= 0 && target.blocks > 0 && draft.nextn_layer != target.blocks) {
        return "its MTP head sits after layer " + std::to_string(draft.nextn_layer) + ", and this model has " +
               std::to_string(target.blocks);
    }
    if (!draft.hidden()) {
        return "";
    }
    if (draft.layers.empty()) {
        return "it is a " + draft.arch + " drafter but names no layers to read";
    }
    if (draft.arch == "eagle3" && draft.layers.size() != 3) {
        return "EAGLE-3 reads exactly 3 layers, this one names " + std::to_string(draft.layers.size());
    }
    if (target.blocks > 0) {
        for (const int64_t id : draft.layers) {
            if (id < 0 || id >= target.blocks) {
                return "it reads layer " + std::to_string(id) + ", and this model has " +
                       std::to_string(target.blocks);
            }
        }
    }
    if (draft.enc > 0 && target.embed > 0) {
        const int64_t want = static_cast<int64_t>(draft.layers.size()) * target.embed;
        if (draft.enc != want) {
            return "its encoder takes " + std::to_string(draft.enc) + " values where this model gives " +
                   std::to_string(want);
        }
    }
    return "";
}

std::string fits_target(const Model & m, DraftCand & c) {
    GGUFSpec target;
    if (!spec_of_file(m.path, target).empty()) {
        return ""; // a target we cannot read is not the candidate's fault
    }
    GGUFSpec    draft;
    bool        unreadable = false;
    const std::string err  = spec_of_url(hf_download_url(c.repo, c.file), draft, unreadable);
    if (unreadable) {
        return "";
    }
    if (!err.empty()) {
        return err;
    }
    if (assistant_named(c.repo) && !contains(lower(draft.arch), "assistant")) {
        return "it is a whole " + draft.arch + " model, not an assistant drafter";
    }
    if (c.embedded && draft.nextn_layer < 0) {
        return "it carries no MTP head after all";
    }
    // a head published on its own (blk.N.nextn.* and little else) is an MTP drafter however its repository is named;
    // as a plain draft model the runtime could not load it
    if (draft.nextn_layer >= 0 && draft.tensors < 64 && (c.kind == nullptr || c.kind->name != "mtp")) {
        c.kind     = &draft_kinds()[0];
        c.embedded = false;
        c.note     = "MTP head, " + human_bytes(c.size);
        c.score    = std::max(c.score, c.kind->rank);
    }
    return pairs(target, draft);
}

// The head out of a full build: the file's own header (the runtime reads a sidecar as the model it belongs to)
// with only the head's block, the embedding and the output kept, each fetched by its byte range.
static bool extract_mtp_head(const std::string & url, const std::string & tmp, const ProgressFn & progress,
                             std::string & err) {
    std::string    head;
    ggufio::Layout l;
    for (const int64_t window : {32ll << 20, 128ll << 20}) {
        if (!hub_head(url, window, head, err)) {
            return false;
        }
        l = ggufio::layout_from(head);
        if (l.error.empty()) {
            break;
        }
    }
    if (!l.error.empty()) {
        err = "could not read the build's header: " + l.error;
        return false;
    }
    int64_t layer = -1;
    for (const auto & t : l.tensors) {
        if (t.name.rfind("blk.", 0) == 0 && t.name.find(".nextn.") != std::string::npos) {
            layer = std::max<int64_t>(layer, std::atoll(t.name.c_str() + 4));
        }
    }
    if (layer < 0) {
        err = "the build carries no MTP head";
        return false;
    }
    ggufio::Layout side = l;
    side.tensors.clear();
    side.data_start.clear();
    const std::string prefix = "blk." + std::to_string(layer) + ".";
    int64_t           total  = 0;
    for (const auto & t : l.tensors) {
        if (t.name.rfind(prefix, 0) == 0 || t.name == "token_embd.weight" || t.name == "output.weight" ||
            t.name == "output_norm.weight") {
            side.tensors.push_back(t);
            total += t.bytes;
        }
    }
    if (ggufio::write_header(tmp, side, err) < 0) {
        return false;
    }
    std::ofstream out(tmp, std::ios::binary | std::ios::app);
    int64_t       done = 0;
    for (const auto & t : side.tensors) {
        const int64_t from = l.file_offset(t);
        for (int64_t at = 0; at < t.bytes;) {
            const int64_t     piece = std::min<int64_t>(t.bytes - at, 64ll << 20);
            std::vector<char> buf(static_cast<size_t>(piece));
            if (!hub_span(url, from + at, piece, buf.data(), err)) {
                return false;
            }
            out.write(buf.data(), static_cast<std::streamsize>(piece));
            at += piece;
            done += piece;
            if (progress) {
                progress(done);
            }
        }
        const int64_t pad = (t.bytes + l.align - 1) / l.align * l.align - t.bytes;
        for (int64_t k = 0; k < pad; k++) {
            out.put('\0');
        }
    }
    if (!out) {
        err = "could not write " + tmp;
        return false;
    }
    return true;
}

// ============================================================= installing

bool writable(const std::string & dir) {
    const auto      ns = std::chrono::steady_clock::now().time_since_epoch().count();
    char            name[64];
    std::snprintf(name, sizeof(name), ".llmash-%llx", static_cast<unsigned long long>(ns));
    const fs::path  probe = fs::path(dir) / name;
    std::error_code ec;
    {
        std::ofstream f(probe, std::ios::binary);
        if (!f) {
            return false;
        }
    }
    fs::remove(probe, ec);
    return true;
}

std::string draft_path(const Model & m, const DraftKind & kind, const Config & cfg) {
    const std::string stem = strip_shard(stem_of(m.path));
    std::string       dir  = fs::path(m.path).parent_path().string();
    std::error_code   ec;
    if (!fs::is_directory(dir, ec) || !writable(dir)) {
        dir = loose_dir(cfg);
    }
    return (fs::path(dir) / (stem + "." + kind.name + ".gguf")).string();
}

std::string installed_drafter(const Model & m) {
    std::error_code ec;
    if (!m.mtp_path.empty() && fs::is_regular_file(m.mtp_path, ec)) {
        return "an MTP head";
    }
    if (!mtp_path(m.path).empty()) {
        return "an MTP head";
    }
    if (!find_dspark(m.path).empty()) {
        return "a DSpark drafter";
    }
    if (!sidecar_named(m.path, ".draft.gguf").empty()) {
        return "a draft model";
    }
    if (!sidecar_named(m.path, ".eagle3.gguf").empty()) {
        return "an EAGLE-3 drafter";
    }
    return "";
}

std::string has_own_drafter(const Model & m) {
    if (const std::string d = installed_drafter(m); !d.empty()) {
        return d;
    }
    if (!m.classifier.empty() || is_embedding(read_gguf(m.path))) {
        return "nothing to draft: it generates no text";
    }
    if (m.has_mtp || read_gguf(m.path).has_mtp) {
        return "an MTP head of its own";
    }
    return "";
}

std::string spec_fallback() { return env_str("LLMASH_SPEC_FALLBACK", "ngram-mod"); }

std::string verify_draft(const Model & m, const std::string & path) {
    GGUFSpec          draft;
    const std::string derr = spec_of_file(path, draft);
    if (!derr.empty()) {
        return derr;
    }
    GGUFSpec target;
    if (!spec_of_file(m.path, target).empty()) {
        return "";
    }
    return pairs(target, draft);
}

std::string install_draft(const Model & m, const DraftCand & c, const Config & cfg, const ProgressFn & progress,
                          std::string & err) {
    if (c.kind == nullptr) {
        err = "no drafter kind on the candidate";
        return "";
    }
    const std::string dest = draft_path(m, *c.kind, cfg);
    std::error_code   ec;
    if (fs::is_regular_file(dest, ec)) {
        return dest;
    }
    const std::string tmp = dest + ".part";
    fs::remove(tmp, ec);

    const bool ok = c.embedded ? extract_mtp_head(hf_download_url(c.repo, c.file), tmp, progress, err)
                               : fetch_blocks(hf_download_url(c.repo, c.file), tmp, c.size, progress, err);
    if (!ok) {
        fs::remove(tmp, ec);
        fs::remove(tmp + ".idx", ec);
        return "";
    }
    if (const std::string why = verify_draft(m, tmp); !why.empty()) {
        fs::remove(tmp, ec);
        fs::remove(tmp + ".idx", ec);
        err = why;
        return "";
    }
    fs::rename(tmp, dest, ec);
    if (ec) {
        err = ec.message();
        fs::remove(tmp, ec);
        return "";
    }
    fs::remove(tmp + ".idx", ec);
    return dest;
}

bool drafter_from(const std::string & ref_in, DraftCand & out, std::string & err) {
    std::string ref = ref_in;
    for (const char * p : {"https://huggingface.co/", "http://huggingface.co/", "huggingface.co/", "hf.co/", "hf:"}) {
        if (starts_with(lower(ref), p)) {
            ref = ref.substr(std::strlen(p));
            break;
        }
    }
    std::string repo = ref, want;
    if (const size_t c = ref.find_first_of(":@"); c != std::string::npos) {
        repo = ref.substr(0, c);
        want = lower(ref.substr(c + 1));
    }
    if (std::count(repo.begin(), repo.end(), '/') != 1) {
        err = repo + " is not a Hugging Face repository (org/name)";
        return false;
    }
    HubModel info;
    if (!hub_info_reason(repo, info, err)) {
        return false;
    }
    HubFile best;
    for (const auto & f : hub_files(repo)) {
        if (f.type != "file" || !ends_with(lower(f.path), ".gguf")) {
            continue;
        }
        const std::string base = lower(fs::path(f.path).filename().string());
        if (!want.empty()) {
            // a file named in full wins over one that only contains the name (a quantisation, say)
            if (base == want || (contains(base, want) && (best.path.empty() || prefer_quant(f.path, best.path)))) {
                best = f;
                if (base == want) {
                    break;
                }
            }
            continue;
        }
        if (best.path.empty() || prefer_quant(f.path, best.path)) {
            best = f;
        }
    }
    if (best.path.empty()) {
        err = want.empty() ? repo + " holds no GGUF file" : "no GGUF file in " + repo + " matches " + want;
        return false;
    }
    // typed by its name, else by the architecture its header names; anything else drafts as a plain model
    const DraftKind * kind = assistant_named(repo) ? &draft_kinds()[0] : kind_of(repo);
    if (kind == nullptr) {
        kind = kind_of(best.path);
    }
    if (kind == nullptr) {
        GGUFSpec s;
        bool     unreadable = false;
        spec_of_url(hf_download_url(repo, best.path), s, unreadable);
        const std::string arch = lower(s.arch);
        for (const auto & k : draft_kinds()) {
            if (arch == k.name) {
                kind = &k;
            }
        }
        if (kind == nullptr) {
            kind = contains(arch, "assistant") ? &draft_kinds()[0] : &draft_kinds().back();
        }
    }
    out       = DraftCand{};
    out.repo  = repo;
    out.file  = best.path;
    out.kind  = kind;
    out.size  = best.size;
    out.bases = info.base_models;
    out.note  = kind->name + ", " + human_bytes(best.size);
    return true;
}

} // namespace llmash
