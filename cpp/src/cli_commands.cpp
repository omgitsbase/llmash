// See cli_commands.h for what is deliberately
// not a pixel-for-pixel port and why.

#define _CRT_RAND_S

#include "cli_commands.h"
#include "platform.h"

#include "winproc.h"

#include "cli_console.h"
#include "cli_format.h"
#include "cli_process.h"
#include "cli_win.h"
#include "gguf.h"

#include <subprocess.h>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#endif

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <regex>
#include <set>
#include <sstream>

namespace fs = std::filesystem;
using json   = nlohmann::json;

namespace llmash {

namespace {

// ------------------------------------------------------------- primitives

std::string g_prog = "llmash";

// U+2191 U+2193 as UTF-8 bytes: a narrow literal is otherwise transcoded to
// the compiler's execution code page, which has no arrows.
const std::string kArrows = "\xe2\x86\x91\xe2\x86\x93";

std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

std::string upper(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
    return s;
}

bool equal_fold(const std::string & a, const std::string & b) { return lower(a) == lower(b); }

bool has_prefix(const std::string & s, const std::string & p) { return s.rfind(p, 0) == 0; }

std::string sprintf_str(const char * fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    va_list copy;
    va_copy(copy, ap);
    const int n = std::vsnprintf(nullptr, 0, fmt, copy);
    va_end(copy);
    std::string out;
    if (n > 0) {
        out.resize(static_cast<size_t>(n));
        std::vsnprintf(&out[0], static_cast<size_t>(n) + 1, fmt, ap);
    }
    va_end(ap);
    return out;
}

// strconv.FormatFloat(f, 'f', -1, 64): the fewest decimals that still parse
// back to the same double, never in exponent form.
std::string format_f(double v) {
    char buf[512];
    for (int p = 0; p <= 17; p++) {
        std::snprintf(buf, sizeof(buf), "%.*f", p, v);
        if (std::strtod(buf, nullptr) == v) {
            return buf;
        }
    }
    return buf;
}

// Go's %g: shortest representation that round-trips.
std::string format_g(double v) {
    char buf[64];
    for (int p = 1; p <= 17; p++) {
        std::snprintf(buf, sizeof(buf), "%.*g", p, v);
        if (std::strtod(buf, nullptr) == v) {
            return buf;
        }
    }
    return buf;
}

// fmt.Sprint of one decoded JSON value.
std::string any_string(const json & v) {
    if (v.is_string()) {
        return v.get<std::string>();
    }
    if (v.is_boolean()) {
        return v.get<bool>() ? "true" : "false";
    }
    if (v.is_number()) {
        return format_g(v.get<double>());
    }
    if (v.is_null()) {
        return "<nil>";
    }
    return v.dump();
}

// http.go's str/num/sub/list, over nlohmann's object type.
std::string j_str(const json & m, const std::string & k) {
    if (!m.is_object()) {
        return "";
    }
    const auto it = m.find(k);
    if (it == m.end() || it->is_null()) {
        return "";
    }
    return any_string(*it);
}

double j_num(const json & m, const std::string & k) {
    if (!m.is_object()) {
        return 0;
    }
    const auto it = m.find(k);
    if (it == m.end() || !it->is_number()) {
        return 0;
    }
    return it->get<double>();
}

json j_sub(const json & m, const std::string & k) {
    if (m.is_object()) {
        const auto it = m.find(k);
        if (it != m.end() && it->is_object()) {
            return *it;
        }
    }
    return json::object();
}

json j_list(const json & m, const std::string & k) {
    if (m.is_object()) {
        const auto it = m.find(k);
        if (it != m.end() && it->is_array()) {
            return *it;
        }
    }
    return json::array();
}

std::string first_of(const std::vector<std::string> & vals) {
    for (const auto & v : vals) {
        if (!v.empty()) {
            return v;
        }
    }
    return "";
}

bool file_exists(const std::string & p) {
    std::error_code ec;
    return !p.empty() && fs::is_regular_file(p, ec);
}

std::string read_file(const std::string & p, bool & ok) {
    std::ifstream in(fs::path(p), std::ios::binary);
    if (!in) {
        ok = false;
        return "";
    }
    ok = true;
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

double now_seconds() {
    return static_cast<double>(
        std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count());
}

// exec.Command(...).CombinedOutput(), which is what cmds.go reads back from
// tailscale when a funnel call fails.
std::pair<std::string, bool> combined_output(const std::vector<std::string> & argv) {
    std::vector<const char *> cargv;
    cargv.reserve(argv.size() + 1);
    for (const auto & a : argv) {
        cargv.push_back(a.c_str());
    }
    cargv.push_back(nullptr);

    subprocess_s proc{};
    const int    options = subprocess_option_combined_stdout_stderr | subprocess_option_no_window |
                        subprocess_option_inherit_environment;
    if (subprocess_create(cargv.data(), options, &proc) != 0) {
        return {"", false};
    }
    struct Closer {
        subprocess_s * p;
        ~Closer() { subprocess_destroy(p); }
    } closer{&proc};

    std::string out;
    if (FILE * f = subprocess_stdout(&proc)) {
        char   buf[4096];
        size_t n;
        while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) {
            out.append(buf, n);
        }
    }
    int code = -1;
    subprocess_join(&proc, &code);
    return {out, code == 0};
}

// ------------------------------------------------------------ the server

void need_server(ApiClient & api) {
    if (api.up()) {
        return;
    }
    std::fprintf(stderr, "llmash isn't running at %s.\nStart it with:  %s serve\n", api.host().c_str(),
                 prog().c_str());
    throw CliExit(1);
}

// run.go's showModel: the model's /api/show document and the status it came
// back with.
std::pair<json, int> show_model(ApiClient & api, const std::string & name) {
    const json body = json{{"model", name}};
    ApiResult  r;
    const json d = api.call_json("POST", "/api/show", &body, 60, r);
    if (!r.ok) {
        die("Error: " + r.error);
    }
    return {d, r.status};
}

// net/http's Response.Status ("404 Not Found"), which cmdStop falls back to
// when the server sends no "error" field of its own.
std::string status_line(int code) {
    static const std::map<int, const char *> texts{
        {200, "OK"},          {201, "Created"},      {202, "Accepted"},         {204, "No Content"},
        {400, "Bad Request"}, {401, "Unauthorized"}, {403, "Forbidden"},        {404, "Not Found"},
        {405, "Method Not Allowed"},                 {409, "Conflict"},         {413, "Request Entity Too Large"},
        {429, "Too Many Requests"},                  {500, "Internal Server Error"},
        {501, "Not Implemented"},                    {502, "Bad Gateway"},      {503, "Service Unavailable"},
        {504, "Gateway Timeout"}};
    const auto it = texts.find(code);
    return std::to_string(code) + (it == texts.end() ? "" : std::string(" ") + it->second);
}

std::string arg0(const std::vector<std::string> & args) { return args.empty() ? std::string() : args[0]; }

// ----------------------------------------------------- list / ps plumbing

// The server keeps the finished tables in cache\ beside a heartbeat; while
// the heartbeat is fresh the file is the answer and no socket is opened.
bool cached_table(const Config & cfg, const std::string & name, std::string & out) {
    std::error_code ec;
    const fs::path  alive = fs::path(cfg.root) / "cache" / "alive";
    const auto      mtime = fs::last_write_time(alive, ec);
    if (ec) {
        return false;
    }
    const auto age = std::chrono::duration_cast<std::chrono::seconds>(fs::file_time_type::clock::now() - mtime);
    if (age > std::chrono::seconds(15)) {
        return false;
    }
    bool              ok   = false;
    const std::string text = read_file((fs::path(cfg.root) / "cache" / name).string(), ok);
    if (!ok || text.empty()) {
        return false;
    }
    out = text;
    return true;
}

// The pre-rendered table, or false if this server doesn't serve one (an
// actual ollama, say), in which case the caller renders it itself.
bool server_table(ApiClient & api, const std::string & path) {
    const ApiResult r = api.call("GET", path, nullptr, 60);
    if (!r.ok) {
        need_server(api);
        die("error: " + r.error);
    }
    if (r.status != 200 || !has_prefix(r.body, "NAME")) {
        return false;
    }
    std::fputs(r.body.c_str(), stdout);
    return true;
}

std::string short12(const std::string & digest) {
    std::string d = digest;
    if (has_prefix(d, "sha256:")) {
        d = d.substr(7);
    }
    if (d.size() > 12) {
        d = d.substr(0, 12);
    }
    return d;
}

// ------------------------------------------------------------ show blocks

std::vector<std::vector<std::string>> head_lines(const std::string & s, int n) {
    std::vector<std::vector<std::string>> rows;
    int                                   count = 0;
    for (const auto & ln : split_lines(s)) {
        const std::string text = trim(ln);
        if (text.empty()) {
            continue;
        }
        count++;
        if (n < 0 || count <= n) {
            rows.push_back({"", text});
        }
    }
    if (n >= 0 && count > n) {
        rows.push_back({"", "..."});
    }
    return rows;
}

// ------------------------------------------------------------ pull events

// progress.go's animated bars are not part of this port (see the header):
// one line per status change, and one per percent of a layer.
class PullProgress {
public:
    void status(const std::string & s) {
        if (s == last_status_) {
            return;
        }
        last_status_ = s;
        std::fprintf(stderr, "%s\n", s.c_str());
    }

    void layer(const std::string & digest, int64_t completed, int64_t total) {
        std::string name = trim(digest);
        if (has_prefix(digest, "sha256:")) {
            name = trim(digest.substr(7));
            if (name.size() > 12) {
                name = name.substr(0, 12);
            }
        }
        const int pct = total > 0 ? static_cast<int>(completed * 100 / total) : 0;
        int &     seen = pct_[digest];
        if (pct == seen && completed < total) {
            return;
        }
        seen = pct;
        std::fprintf(stderr, "pulling %s: %3d%% %s/%s\n", name.c_str(), pct, human_bytes(completed).c_str(),
                     human_bytes(total).c_str());
    }

private:
    std::string             last_status_;
    std::map<std::string, int> pct_;
};

void pull_stream(ApiClient & api, const json & body) {
    PullProgress p;
    std::string  failed;
    std::string  err;
    const bool   ok = api.stream("/api/pull", body, [&](const json & ev) {
        const std::string e = j_str(ev, "error");
        if (!e.empty()) {
            failed = e;
            return false;
        }
        const std::string digest = j_str(ev, "digest");
        if (!digest.empty()) {
            const int64_t completed = static_cast<int64_t>(j_num(ev, "completed"));
            if (completed == 0) {
                return true; // the server's size announcement, before any bytes
            }
            p.layer(digest, completed, static_cast<int64_t>(j_num(ev, "total")));
            return true;
        }
        const std::string st = j_str(ev, "status");
        if (!st.empty()) {
            p.status(st);
        }
        return true;
    }, err);
    if (!failed.empty()) {
        std::fprintf(stderr, "Error: %s\n", failed.c_str());
        throw CliExit(1);
    }
    if (!ok) {
        die("Error: " + err);
    }
}

// The server owns the model store, so it does the copying and quantizing;
// this side reads the Modelfile and prints what the server reports.
void stream_statuses(ApiClient & api, const std::string & path, const json & body) {
    std::string err;
    bool        bailed = false;
    const bool  ok     = api.stream(path, body, [&](const json & ev) {
        const std::string e = j_str(ev, "error");
        if (!e.empty()) {
            std::fprintf(stderr, "%s\n", e.c_str());
            bailed = true;
            return false;
        }
        const std::string s = j_str(ev, "status");
        if (!s.empty() && s != "success") {
            std::printf("%s\n", s.c_str());
        }
        return true;
    }, err);
    if (bailed) {
        throw CliExit(1);
    }
    if (!ok) {
        die("error: " + err);
    }
}

// ------------------------------------------------------------ build picker

std::string pad_to(const std::string & s, int width, bool left_align) {
    const int n = width - static_cast<int>(s.size());
    if (n <= 0) {
        return s;
    }
    return left_align ? s + std::string(static_cast<size_t>(n), ' ') : std::string(static_cast<size_t>(n), ' ') + s;
}

std::string build_row(const std::string & label, const QuantInfo & q) {
    return pad_to(label, 7, true) + " " + pad_to(q.name, 12, true) + " " + pad_to(human_bytes(q.size), 8, false);
}

// chooseBuild lists a repository's builds and, when it ships them, its MTP
// heads, and asks for one of each.
std::pair<std::string, std::string> choose_build(ApiClient & api, const std::string & model, std::string quant) {
    ApiResult  r;
    const json d = api.call_json("GET", "/api/quants?repo=" + url_query_escape(model), nullptr, 60, r);
    if (!r.ok || r.status != 200) {
        return {quant, ""};
    }
    const std::string      repo = j_str(d, "repo");
    std::vector<QuantInfo> quants;
    for (const auto & q : j_list(d, "quants")) {
        QuantInfo qi;
        qi.name  = j_str(q, "name");
        qi.size  = static_cast<int64_t>(j_num(q, "size"));
        qi.files = static_cast<int>(j_num(q, "files"));
        quants.push_back(qi);
    }

    std::string chosen = quant;
    if (!quants.empty()) {
        const Tiers t = tiers_of(quants);
        const std::vector<std::string> basic{build_row("tiny", quants[t.tiny]), build_row("medium", quants[t.medium]),
                                             build_row("large", quants[t.large])};
        std::vector<std::string> full;
        for (const auto & q : quants) {
            full.push_back(build_row("", q));
        }
        int start = 2;
        if (!quant.empty()) {
            for (size_t i = 0; i < quants.size(); i++) {
                if (equal_fold(quants[i].name, quant)) {
                    start = static_cast<int>(i);
                }
            }
        }
        bool advanced = !quant.empty() && start != t.tiny && start != t.medium && start != t.large;
        int  cursor   = start;
        if (!advanced) {
            cursor = start == t.tiny ? 0 : start == t.medium ? 1 : start == t.large ? 2 : 0;
            if (quant.empty()) {
                cursor = 2;
            }
        }
        const int tier[3] = {t.tiny, t.medium, t.large};
        for (;;) {
            const int n = advanced ? pick_menu(repo + ", every build:", full, cursor,
                                               kArrows + " move   enter choose   a back", "aA")
                                   : pick_menu(repo + ", which build?", basic, cursor,
                                               kArrows + " move   enter choose   a all builds", "aA");
            if (n == -1) {
                throw CliExit(1);
            }
            if (n < -1) {
                advanced = !advanced;
                cursor   = advanced ? tier[std::min(std::max(cursor, 0), 2)] : 2;
                continue;
            }
            chosen = advanced ? quants[static_cast<size_t>(n)].name : quants[static_cast<size_t>(tier[n])].name;
            break;
        }
    }

    const json heads = j_list(d, "mtp");
    if (heads.empty()) {
        return {chosen, ""};
    }
    std::vector<std::string> names;
    std::vector<int64_t>     sizes;
    for (const auto & h : heads) {
        names.push_back(j_str(h, "name"));
        sizes.push_back(static_cast<int64_t>(j_num(h, "size")));
    }
    int def = 0;
    for (const auto & want : {chosen, std::string("Q8_0"), std::string("BF16")}) {
        int found = -1;
        for (size_t i = 0; i < names.size(); i++) {
            if (equal_fold(quant_tag(names[i]), want)) {
                found = static_cast<int>(i);
            }
        }
        if (found >= 0) {
            def = found;
            break;
        }
    }
    std::printf("Use the suggested MTP head, %s (%s)? [Y/n/a] ", names[static_cast<size_t>(def)].c_str(),
                human_bytes(sizes[static_cast<size_t>(def)]).c_str());
    for (;;) {
        const int k = read_pick([]() { return raw_getch(); });
        if (k == 'y' || k == 'Y' || k == kPickEnter) {
            std::printf("yes\n");
            return {chosen, names[static_cast<size_t>(def)]};
        }
        if (k == 'n' || k == 'N' || k == kPickEsc) {
            std::printf("no\n");
            return {chosen, ""};
        }
        if (k == 'a' || k == 'A') {
            std::printf("choose\n");
            std::vector<std::string> rows{"none"};
            for (size_t i = 0; i < names.size(); i++) {
                rows.push_back(pad_to(names[i], 40, true) + " " + pad_to(human_bytes(sizes[i]), 8, false));
            }
            const int n = pick_menu("MTP heads:", rows, def + 1, kArrows + " move   enter choose", "");
            if (n <= 0) {
                return {chosen, ""};
            }
            return {chosen, names[static_cast<size_t>(n - 1)]};
        }
    }
}

// ------------------------------------------------------------- drafters

// draft.go/draft_verify.go (the Hugging Face drafter search and its GGUF
// spec matching) are a separate subsystem this module was not given, so the
// search reports nothing and the surrounding flow -- which is cmds.go's --
// runs unchanged.
struct DraftCandidate {
    std::string repo;
    std::string note;
};

std::vector<DraftCandidate> find_drafter_candidates(const std::string & gguf_path, Registry & reg) {
    (void) gguf_path;
    (void) reg;
    return {};
}

std::string stem_of(const std::string & p) { return fs::path(p).stem().string(); }

// registry.go's shardSuffix, so a "-00001-of-00003" shard finds the sidecar
// named after the whole model.
std::string unsharded_stem(const std::string & p) {
    static const std::regex shard(R"(-\d+-of-\d+$)");
    return std::regex_replace(stem_of(p), shard, "");
}

// registry.go's findMtp/findDspark/findDraft/findEagle3, in the order
// draft_install.go's installedDrafter asks about them.
std::string installed_drafter(const std::string & gguf) {
    const fs::path    dir  = fs::path(gguf).parent_path();
    const std::string stem = unsharded_stem(gguf);
    struct Sidecar {
        const char * suffix;
        const char * label;
    };
    for (const Sidecar & s : {Sidecar{".mtp.gguf", "an MTP head"}, Sidecar{".dspark.gguf", "a DSpark drafter"},
                              Sidecar{".draft.gguf", "a draft model"}, Sidecar{".eagle3.gguf", "an EAGLE-3 drafter"}}) {
        if (file_exists((dir / (stem + s.suffix)).string())) {
            return s.label;
        }
    }
    return "";
}

bool has_mtp_head(const std::string & gguf) { return read_gguf(gguf).has_mtp; }

std::string has_own_drafter(const std::string & gguf) {
    const std::string d = installed_drafter(gguf);
    if (!d.empty()) {
        return d;
    }
    return has_mtp_head(gguf) ? "an MTP head of its own" : "";
}

// draft_install.go's modelFor: the local .gguf behind a model name, or ""
// when the name is not backed by a file on this machine.
std::string model_gguf(ApiClient & api, const std::string & name) {
    auto [info, code] = show_model(api, name);
    if (code != 200) {
        die("Error: " + first_of({j_str(info, "error"), "model not found"}));
    }
    std::string gguf = j_str(info, "modelfile");
    if (has_prefix(gguf, "FROM ")) {
        gguf = gguf.substr(5);
    }
    return file_exists(gguf) ? gguf : "";
}

std::string spec_fallback() { return env_str("LLMASH_SPEC_FALLBACK", "ngram-mod"); }

// Runs at the end of a pull, and says nothing when there is nothing to offer.
void offer_draft(ApiClient & api, Registry & reg, const std::string & name) {
    if (!is_console_stdin() || !is_console_stdout()) {
        return;
    }
    const std::string gguf = model_gguf(api, name);
    if (gguf.empty() || !has_own_drafter(gguf).empty()) {
        return;
    }
    std::printf("\n%slooking for a draft model...%s ", kDim, kReset);
    if (find_drafter_candidates(gguf, reg).empty()) {
        std::printf("%snone published%s\n", kDim, kReset);
        return;
    }
}

// ---------------------------------------------------------------- link

int funnel_port() { return env_int("LLMASH_FUNNEL_PORT", 8443); }

std::string tailscale_exe() {
    for (const char * p : {"C:\\Program Files\\Tailscale\\tailscale.exe",
                           "C:\\Program Files (x86)\\Tailscale\\tailscale.exe"}) {
        if (file_exists(p)) {
            return p;
        }
    }
    return which_exe("tailscale");
}

std::string link_file(const Config & cfg) { return (fs::path(cfg.root) / "link.json").string(); }

json read_link(const Config & cfg) {
    bool              ok = false;
    const std::string b  = read_file(link_file(cfg), ok);
    if (!ok) {
        return json::object();
    }
    json d = json::parse(b, nullptr, false);
    return d.is_object() ? d : json::object();
}

void write_link(const Config & cfg, const json & d) {
    std::ofstream out(fs::path(link_file(cfg)), std::ios::binary | std::ios::trunc);
    if (out) {
        out << d.dump(2);
    }
}

// Same source of truth as the server: env var, else link.json, else mint one.
std::string link_key(const Config & cfg) {
    const std::string fromenv = env_str("LLMASH_LINK_KEY");
    if (!fromenv.empty()) {
        return fromenv;
    }
    json              d = read_link(cfg);
    const std::string k = j_str(d, "key");
    if (!k.empty()) {
        return k;
    }
    unsigned char raw[24];
    for (size_t i = 0; i < sizeof(raw); i += 4) {
        unsigned int v = 0;
#ifdef _WIN32
        if (rand_s(&v) != 0) {
            v = static_cast<unsigned int>(std::rand());
        }
#else
        v = static_cast<unsigned int>(std::rand());
#endif
        std::memcpy(raw + i, &v, 4);
    }
    const std::string key = "sk-llmash-" + base64_url_encode(raw, sizeof(raw));
    d["key"]              = key;
    d["public_port"]      = cfg.public_port;
    write_link(cfg, d);
    return key;
}

std::string tailnet_name(const std::string & ts) {
    const ProcessResult r = run_hidden({ts, "status", "--json"}, nullptr, true, true);
    if (!r.started || r.exit_code != 0) {
        return "";
    }
    const json st = json::parse(r.out, nullptr, false);
    if (!st.is_object()) {
        return "";
    }
    std::string dns = j_str(j_sub(st, "Self"), "DNSName");
    while (!dns.empty() && dns.back() == '.') {
        dns.pop_back();
    }
    return dns;
}

void report_copy(bool ok, const std::string & what) {
    if (ok) {
        std::printf("  %s%s%s\n", kGreen, what.c_str(), kReset);
    } else {
        std::printf("  (copy failed)\n");
    }
}

// ------------------------------------------------------------- uninstall

// Commands the installer puts on PATH. `llamash` was this project's old
// name and is only here so an uninstall clears it.
const char * const kShimNames[] = {"llmash", "ollama", "llamash"};

int stop_server(const Config & cfg) {
    // The server records its own id; llama-server children go with it, and
    // anything of ours still running out of the install root follows.
    int killed = 0;
    const unsigned long pid = read_pid_file(cfg.root);
    if (pid_alive(pid)) {
        killed += kill_tree(pid, llama_server_exe());
    }
    remove_pid_file(cfg.root);
    for (const RunningProcess & p : processes_under(cfg.root)) {
        if ((p.name == llmash_daemon_exe() || p.name == llmash_exe() || p.name == llama_server_exe()) &&
            p.pid != current_pid() && kill_pid(p.pid)) {
            killed++;
        }
    }
    return killed;
}


// ----------------------------------------------------------- flag shapes

struct ShowOpts {
    std::string model;
    bool        modelfile = false, tmpl = false, parameters = false, system = false, lic = false, verbose = false;
};

ShowOpts parse_show(const std::vector<std::string> & args) {
    const ParsedArgs o = parse_simple(
        args, {"--modelfile", "--template", "--parameters", "--system", "--license", "-v", "--verbose"}, {});
    if (o.pos.empty()) {
        die("Error: requires at least 1 arg(s), only received 0");
    }
    ShowOpts s;
    s.model      = o.pos[0];
    s.modelfile  = o.has_flag("--modelfile");
    s.tmpl       = o.has_flag("--template");
    s.parameters = o.has_flag("--parameters");
    s.system     = o.has_flag("--system");
    s.lic        = o.has_flag("--license");
    s.verbose    = o.has_flag("-v") || o.has_flag("--verbose");
    return s;
}

} // namespace

// ============================================================== public API

void                set_prog(const std::string & p) { g_prog = p.empty() ? "llmash" : p; }
const std::string & prog() { return g_prog; }

[[noreturn]] void die(const std::string & message) {
    std::fprintf(stderr, "%s\n", message.c_str());
    throw CliExit(1, message);
}

// ------------------------------------------------------------ arg parsing

bool ParsedArgs::has_flag(const std::string & name) const {
    return std::find(bool_flags_set.begin(), bool_flags_set.end(), name) != bool_flags_set.end();
}

bool ParsedArgs::has_val(const std::string & name) const {
    for (const auto & kv : vals) {
        if (kv.first == name) {
            return true;
        }
    }
    return false;
}

std::string ParsedArgs::val(const std::string & name) const {
    for (const auto & kv : vals) {
        if (kv.first == name) {
            return kv.second;
        }
    }
    return "";
}

ParsedArgs parse_simple(const std::vector<std::string> & args, const std::vector<std::string> & bool_flags,
                        const std::vector<std::string> & val_flags) {
    const std::set<std::string> bools(bool_flags.begin(), bool_flags.end());
    const std::set<std::string> vals(val_flags.begin(), val_flags.end());

    ParsedArgs o;
    for (size_t i = 0; i < args.size(); i++) {
        const std::string & t = args[i];
        if (!has_prefix(t, "-") || t == "-") {
            o.pos.push_back(t);
            continue;
        }
        std::string name    = t;
        std::string value;
        bool        has_value = false;
        const size_t eq       = t.find('=');
        if (eq != std::string::npos && eq > 0) {
            name      = t.substr(0, eq);
            value     = t.substr(eq + 1);
            has_value = true;
        }
        if (bools.count(name) != 0) {
            if (!o.has_flag(name)) {
                o.bool_flags_set.push_back(name);
            }
            continue;
        }
        if (vals.count(name) != 0) {
            if (!has_value) {
                if (i + 1 >= args.size()) {
                    die("Error: flag needs an argument: " + name);
                }
                value = args[++i];
            }
            bool replaced = false;
            for (auto & kv : o.vals) {
                if (kv.first == name) {
                    kv.second = value;
                    replaced  = true;
                    break;
                }
            }
            if (!replaced) {
                o.vals.emplace_back(name, value);
            }
            continue;
        }
        die("Error: unknown flag: '" + t + "'");
    }
    return o;
}

// ------------------------------------------------------- pure logic units

double bits_of(const std::string & quant_name) {
    const std::string up = upper(quant_name);
    if (up.find("F32") != std::string::npos) {
        return 32;
    }
    if (up.find("F16") != std::string::npos) {
        return 16;
    }
    static const std::regex re(R"((?:IQ|Q|TQ)([1-8]))");
    std::smatch             m;
    if (!std::regex_search(up, m, re)) {
        return 0;
    }
    double b = static_cast<double>(m[1].str()[0] - '0');
    if (up.find("_K") != std::string::npos || up.find("XL") != std::string::npos) {
        b += 0.5;
    }
    return b;
}

Tiers tiers_of(const std::vector<QuantInfo> & quants) {
    const auto find = [&](const std::vector<std::string> & names) {
        for (const auto & want : names) {
            for (size_t i = 0; i < quants.size(); i++) {
                if (equal_fold(quants[i].name, want)) {
                    return static_cast<int>(i);
                }
            }
        }
        return -1;
    };
    const auto by_bits = [&](double min, double max, bool smallest) {
        int best = -1;
        for (size_t i = 0; i < quants.size(); i++) {
            const double b = bits_of(quants[i].name);
            if (b < min || b > max) {
                continue;
            }
            if (best < 0 || (smallest && quants[i].size < quants[static_cast<size_t>(best)].size) ||
                (!smallest && quants[i].size > quants[static_cast<size_t>(best)].size)) {
                best = static_cast<int>(i);
            }
        }
        return best;
    };

    Tiers     t;
    const int n = static_cast<int>(quants.size());
    t.large     = find({"Q8_0", "Q6_K", "UD-Q8_K_XL", "UD-Q6_K_XL", "Q5_K_M"});
    if (t.large < 0) {
        t.large = by_bits(5, 8.5, false);
        if (t.large < 0) {
            t.large = n - 1;
        }
    }
    t.medium = find({"Q4_K_M", "UD-Q4_K_XL", "Q4_K_S", "IQ4_XS", "IQ4_NL", "Q4_0", "Q5_K_S"});
    if (t.medium < 0) {
        t.medium = by_bits(4, 5.5, false);
        if (t.medium < 0) {
            t.medium = n / 2;
        }
    }
    t.tiny = find({"Q3_K_M", "UD-Q3_K_XL", "IQ3_M", "IQ3_XS", "Q3_K_S", "IQ3_XXS", "UD-IQ3_XXS"});
    if (t.tiny < 0) {
        t.tiny = by_bits(2.5, 3.9, true);
        if (t.tiny < 0) {
            t.tiny = 0;
        }
    }
    return t;
}


bool is_hf_ref(const std::string & model_ref) {
    for (const char * p : {"hf:", "hf.co/", "huggingface.co/"}) {
        if (has_prefix(model_ref, p)) {
            return true;
        }
    }
    return false;
}

std::string url_query_escape(const std::string & s) {
    static const char * hex = "0123456789ABCDEF";
    std::string         out;
    out.reserve(s.size());
    for (const unsigned char c : s) {
        if (std::isalnum(c) != 0 || c == '-' || c == '_' || c == '.' || c == '~') {
            out.push_back(static_cast<char>(c));
        } else if (c == ' ') {
            out.push_back('+');
        } else {
            out.push_back('%');
            out.push_back(hex[c >> 4]);
            out.push_back(hex[c & 0x0F]);
        }
    }
    return out;
}

std::string base64_url_encode(const unsigned char * data, size_t len) {
    static const char * alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    std::string         out;
    out.reserve((len * 4 + 2) / 3);
    size_t i = 0;
    for (; i + 3 <= len; i += 3) {
        const uint32_t v = (static_cast<uint32_t>(data[i]) << 16) | (static_cast<uint32_t>(data[i + 1]) << 8) |
                           static_cast<uint32_t>(data[i + 2]);
        out.push_back(alphabet[(v >> 18) & 0x3F]);
        out.push_back(alphabet[(v >> 12) & 0x3F]);
        out.push_back(alphabet[(v >> 6) & 0x3F]);
        out.push_back(alphabet[v & 0x3F]);
    }
    const size_t rest = len - i;
    if (rest == 1) {
        const uint32_t v = static_cast<uint32_t>(data[i]) << 16;
        out.push_back(alphabet[(v >> 18) & 0x3F]);
        out.push_back(alphabet[(v >> 12) & 0x3F]);
    } else if (rest == 2) {
        const uint32_t v = (static_cast<uint32_t>(data[i]) << 16) | (static_cast<uint32_t>(data[i + 1]) << 8);
        out.push_back(alphabet[(v >> 18) & 0x3F]);
        out.push_back(alphabet[(v >> 12) & 0x3F]);
        out.push_back(alphabet[(v >> 6) & 0x3F]);
    }
    return out;
}

bool path_under(const std::string & path, const std::string & dir) {
    if (path.empty() || dir.empty()) {
        return false;
    }
    const fs::path rel = fs::path(path).lexically_normal().lexically_relative(fs::path(dir).lexically_normal());
    if (rel.empty()) {
        return false;
    }
    return rel.native().rfind(fs::path("..").native(), 0) != 0;
}

bool port_open(int port) {
#ifdef _WIN32
    struct WsaGuard {
        bool ok = false;
        WsaGuard() {
            WSADATA d;
            ok = WSAStartup(MAKEWORD(2, 2), &d) == 0;
        }
        ~WsaGuard() {
            if (ok) {
                WSACleanup();
            }
        }
    } wsa;
    if (!wsa.ok) {
        return false;
    }

    const SOCKET s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (s == INVALID_SOCKET) {
        return false;
    }
    struct SockGuard {
        SOCKET s;
        ~SockGuard() { closesocket(s); }
    } guard{s};

    u_long nonblocking = 1;
    ioctlsocket(s, FIONBIO, &nonblocking);

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port   = htons(static_cast<u_short>(port));
    InetPtonA(AF_INET, "127.0.0.1", &addr.sin_addr);

    if (connect(s, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) == 0) {
        return true;
    }
    if (WSAGetLastError() != WSAEWOULDBLOCK) {
        return false;
    }
    fd_set  wr;
    FD_ZERO(&wr);
    FD_SET(s, &wr);
    timeval tv{0, 600 * 1000};
    if (select(0, nullptr, &wr, nullptr, &tv) <= 0) {
        return false;
    }
    int       err = 0;
    int       len = sizeof(err);
    if (getsockopt(s, SOL_SOCKET, SO_ERROR, reinterpret_cast<char *>(&err), &len) != 0) {
        return false;
    }
    return err == 0;
#else
    (void) port;
    return false;
#endif
}

std::vector<json> filter_rows(const json & doc, const std::string & prefix, bool fold) {
    std::vector<json> out;
    for (const auto & v : j_list(doc, "models")) {
        if (!v.is_object()) {
            continue;
        }
        std::string name = j_str(v, "name");
        std::string want = prefix;
        if (fold) {
            name = lower(name);
            want = lower(want);
        }
        if (has_prefix(name, want)) {
            out.push_back(v);
        }
    }
    return out;
}



std::string elide(const std::vector<std::string> & values, int target) {
    int total = 1, show = 0;
    for (size_t i = 0; i < values.size(); i++) {
        int w = disp_width(values[i]);
        if (i > 0) {
            w += 2;
        }
        if (total + w > target && i > 0) {
            break;
        }
        total += w;
        show++;
    }
    if (static_cast<size_t>(show) < values.size()) {
        const std::vector<std::string> head(values.begin(), values.begin() + show);
        return "[" + join(head, " ") +
               sprintf_str(" ...+%d more]", static_cast<int>(values.size()) - show);
    }
    return "[" + join(values, " ") + "]";
}

// The blocks `ollama show` prints, in its order and its table layout.
void show_info(const json & resp, bool verbose, std::string & out) {
    const auto render = [&](const std::string & header, const std::vector<std::vector<std::string>> & rows) {
        out += "  " + header + "\n";
        Table t;
        if (header == "Template" || header == "System" || header == "License") {
            t.wrap_at = 100;
        }
        for (const auto & r : rows) {
            t.add(r);
        }
        out += t.str();
        out += "\n";
    };

    const json info = j_sub(resp, "model_info");
    const json det  = j_sub(resp, "details");

    const auto num_str = [&](const std::string & key, std::string & value) {
        const auto it = info.find(key);
        if (it == info.end() || !it->is_number()) {
            return false;
        }
        value = format_f(it->get<double>());
        return true;
    };

    std::vector<std::vector<std::string>> rows;
    const std::string                     arch = j_str(info, "general.architecture");
    if (!info.empty()) {
        if (!arch.empty()) {
            rows.push_back({"", "architecture", arch});
        }
        std::string param = j_str(det, "parameter_size");
        if (param.empty()) {
            const auto it = info.find("general.parameter_count");
            if (it != info.end() && it->is_number()) {
                param = human_number(static_cast<uint64_t>(it->get<double>()));
            }
        }
        if (!param.empty()) {
            rows.push_back({"", "parameters", param});
        }
        std::string s;
        if (num_str(arch + ".context_length", s)) {
            rows.push_back({"", "context length", s});
        }
        if (num_str(arch + ".embedding_length", s)) {
            rows.push_back({"", "embedding length", s});
        }
    } else {
        rows.push_back({"", "architecture", j_str(det, "family")});
        rows.push_back({"", "parameters", j_str(det, "parameter_size")});
    }
    rows.push_back({"", "quantization", j_str(det, "quantization_level")});
    render("Model", rows);

    const json caps = j_list(resp, "capabilities");
    if (!caps.empty()) {
        rows.clear();
        for (const auto & c : caps) {
            rows.push_back({"", any_string(c)});
        }
        render("Capabilities", rows);
    }

    const std::string params = j_str(resp, "parameters");
    if (!params.empty()) {
        rows.clear();
        for (const auto & ln : split_lines(params)) {
            if (trim(ln).empty()) {
                continue;
            }
            std::vector<std::string> row{""};
            for (const auto & f : split_fields(ln)) {
                row.push_back(f);
            }
            rows.push_back(row);
        }
        render("Parameters", rows);
    }

    if (verbose && !info.empty()) {
        std::vector<std::string> keys;
        for (auto it = info.begin(); it != info.end(); ++it) {
            keys.push_back(it.key());
        }
        std::sort(keys.begin(), keys.end());
        rows.clear();
        for (const auto & k : keys) {
            const json & v = info.at(k);
            std::string  text;
            if (v.is_boolean()) {
                text = v.get<bool>() ? "true" : "false";
            } else if (v.is_string()) {
                text = v.get<std::string>();
            } else if (v.is_number()) {
                text = format_g(v.get<double>());
            } else if (v.is_array()) {
                std::vector<std::string> items;
                items.reserve(v.size());
                for (const auto & e : v) {
                    items.push_back(any_string(e));
                }
                text = elide(items, 10);
            } else if (v.is_null()) {
                text = "<nil>";
            } else {
                text = "map[string]interface {}";
            }
            rows.push_back({"", k, text});
        }
        render("Metadata", rows);
    }

    const std::string system = j_str(resp, "system");
    if (!system.empty()) {
        render("System", head_lines(system, 2));
    }
    const std::string lic = j_str(resp, "license");
    if (!lic.empty()) {
        render("License", head_lines(lic, 2));
    }
}

// ---------------------------------------------------------------- commands

// A name prefix filters the table, the way ollama's does; without one the
// server's pre-rendered copy is the answer.
int cmd_list(const std::vector<std::string> & args, ApiClient & api, const Config & cfg) {
    try {
        const std::vector<std::string> pos = parse_simple(args, {}, {}).pos;
        if (pos.empty()) {
            std::string cached;
            if (cached_table(cfg, "list.txt", cached)) {
                std::fputs(cached.c_str(), stdout);
                return 0;
            }
            if (server_table(api, "/cli/list")) {
                return 0;
            }
        }
        need_server(api);
        ApiResult  r;
        const json d = api.call_json("GET", "/api/tags", nullptr, 60, r);
        if (!r.ok) {
            die("error: " + r.error);
        }
        std::fputs(render_list(filter_rows(d, arg0(pos), true)).c_str(), stdout);
        return 0;
    } catch (const CliExit & e) {
        return e.code;
    }
}

int cmd_ps(const std::vector<std::string> & args, ApiClient & api, const Config & cfg) {
    try {
        const std::vector<std::string> pos = parse_simple(args, {}, {}).pos;
        if (pos.empty()) {
            std::string cached;
            if (cached_table(cfg, "ps.txt", cached)) {
                std::fputs(cached.c_str(), stdout);
                return 0;
            }
            if (server_table(api, "/cli/ps")) {
                return 0;
            }
        }
        need_server(api);
        ApiResult  r;
        const json d = api.call_json("GET", "/api/ps", nullptr, 60, r);
        if (!r.ok) {
            die("error: " + r.error);
        }
        std::fputs(render_ps(filter_rows(d, arg0(pos), false)).c_str(), stdout);
        return 0;
    } catch (const CliExit & e) {
        return e.code;
    }
}

int cmd_show(const std::vector<std::string> & args, ApiClient & api) {
    try {
        const ShowOpts o = parse_show(args);
        need_server(api);
        int set = 0;
        for (const bool b : {o.lic, o.modelfile, o.parameters, o.system, o.tmpl}) {
            if (b) {
                set++;
            }
        }
        if (set > 1) {
            die("Error: only one of '--license', '--modelfile', '--parameters', '--system', or '--template' can be "
                "specified");
        }
        auto [d, code] = show_model(api, o.model);
        if (code != 200) {
            die("Error: " + first_of({j_str(d, "error"), "model not found"}));
        }
        if (o.lic) {
            std::printf("%s\n", j_str(d, "license").c_str());
        } else if (o.modelfile) {
            std::printf("%s\n", j_str(d, "modelfile").c_str());
        } else if (o.parameters) {
            std::printf("%s\n", j_str(d, "parameters").c_str());
        } else if (o.system) {
            std::fputs(j_str(d, "system").c_str(), stdout);
        } else if (o.tmpl) {
            std::fputs(j_str(d, "template").c_str(), stdout);
        } else {
            std::string out;
            show_info(d, o.verbose, out);
            std::fputs(out.c_str(), stdout);
        }
        return 0;
    } catch (const CliExit & e) {
        return e.code;
    }
}

int cmd_rm(const std::vector<std::string> & args, ApiClient & api) {
    try {
        const ParsedArgs o = parse_simple(args, {}, {});
        if (o.pos.empty()) {
            die("Error: requires at least 1 arg(s), only received 0");
        }
        need_server(api);
        for (const auto & name : o.pos) {
            const json      body = json{{"model", name}};
            const ApiResult r    = api.call("DELETE", "/api/delete", &body, 60);
            if (!r.ok) {
                die("Error: " + r.error);
            }
            if (r.status >= 400) {
                std::string msg = trim(r.body);
                const json  d   = json::parse(r.body, nullptr, false);
                if (d.is_object() && !j_str(d, "error").empty()) {
                    msg = j_str(d, "error");
                }
                die("Error: " + msg);
            }
            std::printf("deleted '%s'\n", name.c_str());
        }
        return 0;
    } catch (const CliExit & e) {
        return e.code;
    }
}

// Unloading is a generate request with keep_alive 0, and like ollama it says
// nothing when it worked.
int cmd_stop(const std::vector<std::string> & args, ApiClient & api) {
    try {
        const ParsedArgs o = parse_simple(args, {}, {});
        if (o.pos.empty()) {
            die("Error: requires at least 1 arg(s), only received 0");
        }
        need_server(api);
        for (const auto & name : o.pos) {
            if (show_model(api, name).second == 404) {
                die("Error: couldn't find model \"" + name + "\" to stop");
            }
            const json body = json{{"model", name}, {"keep_alive", 0}, {"prompt", ""}};
            ApiResult  r;
            const json d = api.call_json("POST", "/api/generate", &body, 60, r);
            if (!r.ok) {
                die("Error: " + r.error);
            }
            if (r.status >= 400) {
                die("Error: " + first_of({j_str(d, "error"), status_line(r.status)}));
            }
        }
        return 0;
    } catch (const CliExit & e) {
        return e.code;
    }
}

int cmd_pull(const std::vector<std::string> & args, ApiClient & api) {
    try {
        const ParsedArgs o = parse_simple(args, {"--insecure", "--draft", "--no-draft"}, {"--quant", "-q"});
        if (o.pos.empty()) {
            die("Error: requires at least 1 arg(s), only received 0");
        }
        const std::string model = o.pos[0];
        std::string       quant = first_of({o.val("--quant"), o.val("-q")});
        const bool        offer = !o.has_flag("--no-draft");

        need_server(api);
        const bool  interactive = is_console_stdin() && is_console_stdout();
        std::string repo, as;
        if (is_hf_ref(model)) {
            repo = model;
        } else {
            ApiResult  r;
            const json d = api.call_json("GET", "/api/resolve?model=" + url_query_escape(model), nullptr, 180, r);
            if (r.ok && r.status == 200 && j_str(d, "source") == "hf") {
                std::printf("%s: the registry build %s, which llama.cpp does not load.\n", model.c_str(),
                            j_str(d, "reason").c_str());
                std::printf("Taking %s from Hugging Face instead.\n", j_str(d, "repo").c_str());
                repo = "hf:" + j_str(d, "repo");
                as   = model;
                if (quant.empty()) {
                    quant = j_str(d, "quant");
                }
            }
        }
        std::string mtp;
        if (!repo.empty() && interactive && repo.find('@') == std::string::npos) {
            std::tie(quant, mtp) = choose_build(api, repo, quant);
        }
        json body = json{{"model", first_of({repo, model})}};
        if (!quant.empty()) {
            body["quant"] = quant;
        }
        if (!as.empty()) {
            body["as"] = as;
        }
        if (!mtp.empty()) {
            body["mtp"] = mtp;
        }
        pull_stream(api, body);
        if (offer && mtp.empty()) {
            Config   cfg = load_config();
            Registry reg(cfg);
            offer_draft(api, reg, first_of({as, model}));
        }
        return 0;
    } catch (const CliExit & e) {
        return e.code;
    }
}

int cmd_create(const std::vector<std::string> & args, ApiClient & api) {
    try {
        const ParsedArgs o =
            parse_simple(args, {"--experimental"}, {"-f", "--file", "-q", "--quantize", "--draft-quantize"});
        if (o.pos.empty()) {
            die("Error: requires at least 1 arg(s), only received 0");
        }
        const std::string name           = o.pos[0];
        const std::string file           = first_of({o.val("-f"), o.val("--file"), "Modelfile"});
        const std::string quantize       = first_of({o.val("-q"), o.val("--quantize")});
        const std::string draft_quantize = o.val("--draft-quantize");

        bool              ok   = false;
        const std::string text = read_file(file, ok);
        if (!ok) {
            die("Modelfile not found: " + file);
        }
        std::string from;
        for (const auto & line : split_lines(text)) {
            const std::string s = trim(line);
            if (has_prefix(upper(s), "FROM ")) {
                from = trim(s.substr(5));
                while (!from.empty() && (from.front() == '"' || from.front() == '\'')) {
                    from.erase(from.begin());
                }
                while (!from.empty() && (from.back() == '"' || from.back() == '\'')) {
                    from.pop_back();
                }
                break;
            }
        }
        if (from.empty()) {
            die(file + " has no FROM line, so there is nothing to import.");
        }
        need_server(api);
        if (!equal_fold(fs::path(from).extension().string(), ".gguf") || !file_exists(from)) {
            std::string loose = "the model store's gguf folder";
            ApiResult   r;
            const json  d = api.call_json("GET", "/api/paths", nullptr, 5, r);
            if (r.ok && !j_str(d, "loose_dir").empty()) {
                loose = j_str(d, "loose_dir");
            }
            std::fprintf(stderr,
                         "'%s' isn't a local .gguf, so there's nothing to import.\n"
                         "llmash builds models from GGUF files, not Modelfile layers:\n"
                         "  - local file : put its .gguf in %s (it appears in `%s list` automatically),\n"
                         "                 or point a Modelfile's FROM at the .gguf and re-run create\n"
                         "  - from the hub: %s pull hf:owner/repo\n",
                         from.c_str(), loose.c_str(), prog().c_str(), prog().c_str());
            throw CliExit(1);
        }
        std::error_code   ec;
        const std::string abs = fs::absolute(from, ec).lexically_normal().string();
        stream_statuses(api, "/api/create",
                        json{{"model", name}, {"from", abs}, {"quantize", quantize},
                             {"draft_quantize", draft_quantize}});
        return 0;
    } catch (const CliExit & e) {
        return e.code;
    }
}

int cmd_cp(const std::vector<std::string> & args, ApiClient & api) {
    try {
        const ParsedArgs o = parse_simple(args, {}, {});
        if (o.pos.size() < 2) {
            die("Error: requires at least 2 arg(s), only received " + std::to_string(o.pos.size()));
        }
        need_server(api);
        stream_statuses(api, "/api/copy", json{{"source", o.pos[0]}, {"destination", o.pos[1]}});
        return 0;
    } catch (const CliExit & e) {
        return e.code;
    }
}

int cmd_push() {
    std::fprintf(stderr,
                 "llmash has no model registry to push to. It serves local GGUFs only, so there's nothing to "
                 "upload.\nTo share this llmash instead, expose it over Tailscale with `%s link`.\n",
                 prog().c_str());
    return 1;
}

int cmd_signin() {
    std::fprintf(stderr,
                 "llmash is fully local: there is no ollama.com account to sign in to.\n"
                 "To reach this server remotely, use `%s link` instead.\n",
                 prog().c_str());
    return 1;
}

int cmd_signout() {
    std::printf("not signed in to ollama.com (llmash is fully local)\n");
    return 0;
}

int cmd_link(const std::vector<std::string> & args, const Config & cfg) {
    try {
        const ParsedArgs o   = parse_simple(args, {"--off"}, {});
        const bool       off = o.has_flag("--off");

        const std::string ts = tailscale_exe();
        if (ts.empty()) {
            die("Tailscale isn't installed (or not on PATH).");
        }
        if (!port_open(cfg.public_port)) {
            die("llmash's public port " + std::to_string(cfg.public_port) +
                " isn't up, so the tunnel would point at nothing.\nRestart llmash (it opens that port on boot), then "
                "run `" +
                prog() + " link` again.");
        }
        const std::string dns = tailnet_name(ts);
        if (dns.empty()) {
            die("Couldn't read your Tailscale name. Is `tailscale` logged in and up?");
        }
        const std::string key = link_key(cfg);
        if (!off) {
            const auto [out, ok] = combined_output({ts, "funnel", "--bg",
                                                    "--https=" + std::to_string(funnel_port()),
                                                    "127.0.0.1:" + std::to_string(cfg.public_port)});
            if (!ok) {
                die("Funnel setup failed:\n" + trim(out));
            }
        }
        const std::string url = "https://" + dns + ":" + std::to_string(funnel_port());
        const std::string bar = std::string(kDim) + std::string(66, '-') + kReset;
        std::printf("\n%s\n", bar.c_str());
        std::printf("  %sllmash is public%s  %s- an Ollama-compatible API, keyed%s\n", kBold, kReset, kDim, kReset);
        std::printf("%s\n", bar.c_str());
        std::printf("  %sURL%s  %s%s%s%s\n", kDim, kReset, kBold, kCyan, url.c_str(), kReset);
        std::printf("  %skey%s  %s%s%s\n", kDim, kReset, kGreen, key.c_str(), kReset);
        std::printf("%s\n", bar.c_str());
        std::printf("  %scurl:%s\n", kDim, kReset);
        const std::string shortkey = key.size() > 14 ? key.substr(0, 14) : key;
        std::printf("  %scurl %s/api/chat -H \"Authorization: Bearer %s...\" \\%s\n", kDim, url.c_str(),
                    shortkey.c_str(), kReset);
        std::printf("  %s     -d '{\"model\":\"qwen3.6:27b\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}'%s\n",
                    kDim, kReset);
        std::printf("%s\n", bar.c_str());
        if (off) {
            std::printf("  %s--off: the tunnel was not (re)started. To take the public API down, run `%s unlink`.%s\n",
                        kDim, prog().c_str(), kReset);
            std::printf("%s\n", bar.c_str());
        }
        if (!is_console_stdin() || !is_console_stdout()) {
            return 0;
        }
        std::printf("  %sc%s copy link   %sk%s copy key   %sq%s done %s(tunnel stays up in the background)%s\n", kBold,
                    kReset, kBold, kReset, kBold, kReset, kDim, kReset);
        for (;;) {
            const int ch = raw_getch();
            if (ch == 'c' || ch == 'C') {
                report_copy(clip_copy(url), "link copied");
            } else if (ch == 'k' || ch == 'K') {
                report_copy(clip_copy(key), "key copied");
            } else if (ch == 'q' || ch == 'Q' || ch == '\r' || ch == '\n' || ch == 0x1b || ch == 0x03) {
                return 0;
            }
        }
    } catch (const CliExit & e) {
        return e.code;
    }
}

int cmd_unlink() {
    try {
        const std::string ts = tailscale_exe();
        if (ts.empty()) {
            die("Tailscale isn't installed (or not on PATH).");
        }
        // Only our port, so any other funnel on the machine is left alone.
        const auto [out, ok] = combined_output({ts, "funnel", "--https=" + std::to_string(funnel_port()), "off"});
        if (!ok) {
            die(trim(out));
        }
        std::printf("public API on :%d is off\n", funnel_port());
        return 0;
    } catch (const CliExit & e) {
        return e.code;
    }
}

int cmd_uninstall(const std::vector<std::string> & args, const Config & cfg) {
    try {
        const ParsedArgs o    = parse_simple(args, {"--keep"}, {});
        const bool       keep = o.has_flag("--keep");

        const std::string bin_dir     = (fs::path(cfg.root) / "bin").string();
        const std::string startup_dir = (fs::path(env_str("APPDATA")) / "Microsoft" / "Windows" / "Start Menu" /
                                         "Programs" / "Startup")
                                            .string();
        const std::string startup_lnk = (fs::path(startup_dir) / "llmash.lnk").string();

        // A directory the installer did not create (a source checkout
        // registered by hand) is never deleted; only the registration goes.
        bool        dev = true;
        bool        ok  = false;
        std::string b   = read_file((fs::path(cfg.root) / "install.json").string(), ok);
        if (ok) {
            if (b.size() >= 3 && static_cast<unsigned char>(b[0]) == 0xEF && static_cast<unsigned char>(b[1]) == 0xBB &&
                static_cast<unsigned char>(b[2]) == 0xBF) {
                b.erase(0, 3);
            }
            const json info = json::parse(b, nullptr, false);
            if (info.is_object() && !info.empty()) {
                const auto it = info.find("dev");
                dev           = it != info.end() && it->is_boolean() && it->get<bool>();
            }
        }
        const bool keep_dir = keep || dev;
        std::printf("Removing %s from %s\n", prog().c_str(), cfg.root.c_str());

        if (dev) {
            std::printf("  left the server running (not an installer directory)\n");
        } else if (const int n = stop_server(cfg); n > 0) {
            std::printf("  stopped %d process%s\n", n, n != 1 ? "es" : "");
        } else {
            std::printf("  server was not running\n");
        }

        std::vector<std::string> shims;
        for (const char * name : kShimNames) {
            bool found = false;
            for (const char * ext : {".exe", ".cmd"}) {
                const std::string p = (fs::path(bin_dir) / (std::string(name) + ext)).string();
                if (file_exists(p)) {
                    shims.push_back(p);
                    found = true;
                }
            }
            if (found) {
                std::printf("  removing command: %s\n", name);
            }
        }
        // Another install may own the startup entry; leave that one alone.
        if (const std::string target = shortcut_target(startup_lnk); !target.empty()) {
            if (path_under(target, cfg.root)) {
                std::error_code ec;
                fs::remove(startup_lnk, ec);
                std::printf("  removed the startup entry\n");
            } else {
                std::printf("  left the startup entry alone (it starts %s)\n", target.c_str());
            }
        }
        const std::string disabled = (fs::path(startup_dir) / "Ollama.lnk.disabled").string();
        if (file_exists(disabled)) {
            std::error_code ec;
            fs::rename(disabled, fs::path(startup_dir) / "Ollama.lnk", ec);
            std::printf("  restored Ollama's startup entry\n");
        }
        if (reg_delete_hkcu_key("Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\llmash")) {
            std::printf("  removed from Settings > Apps\n");
        }
        if (remove_from_user_path(bin_dir)) {
            std::printf("  removed from PATH\n");
        }

        // The program that started us lives in the directory being deleted,
        // so the last step goes to a hidden PowerShell that waits for this
        // process to exit and retries the delete for a few seconds.
        const auto retry = [](const std::string & target, bool recurse) {
            const std::string flag = recurse ? " -Recurse" : "";
            return "for ($i = 0; $i -lt 30; $i++) { Remove-Item -LiteralPath '" + ps_quote(target) + "'" + flag +
                   " -Force -EA SilentlyContinue; if (-not (Test-Path -LiteralPath '" + ps_quote(target) +
                   "')) { break }; Start-Sleep -Milliseconds 300 }";
        };
        std::vector<std::string> steps;
        if (keep_dir) {
            for (const auto & s : shims) {
                steps.push_back(retry(s, false));
            }
            if (steps.empty()) {
                steps.push_back("$null");
            }
            std::printf("  left %s in place%s\n", cfg.root.c_str(),
                        keep ? " (--keep)" : " (not an installer directory)");
        } else {
            steps.push_back(retry(cfg.root, true));
            std::printf("  %s will be removed in a moment\n", cfg.root.c_str());
        }
        const std::string script = "Wait-Process -Id " + std::to_string(current_pid()) +
                                   " -EA SilentlyContinue; Start-Sleep -Milliseconds 500; " + join(steps, "; ");
        hidden_powershell(script, false);
        std::printf("done. Your models were left where they are.\n");
        return 0;
    } catch (const CliExit & e) {
        return e.code;
    }
}

int cmd_pulldraft(const std::vector<std::string> & args, Registry & reg) {
    try {
        const ParsedArgs o = parse_simple(args, {"--yes", "-y", "--force"}, {});
        if (o.pos.empty()) {
            die("Error: requires at least 1 arg(s), only received 0");
        }
        const std::string name  = o.pos[0];
        const bool        force = o.has_flag("--force");

        Config    cfg = load_config();
        ApiClient api(cfg);
        need_server(api);

        const std::string gguf = model_gguf(api, name);
        if (gguf.empty()) {
            die("Error: " + name + " is not a local GGUF, so there is nothing to pair a drafter with");
        }
        if (has_mtp_head(gguf)) {
            std::printf("%s has an MTP head of its own, trained with these exact weights.\n", name.c_str());
            std::printf("There is nothing to look for.\n");
            return 0;
        }
        if (const std::string installed = installed_drafter(gguf); !installed.empty() && !force) {
            std::printf("%s already has %s installed.\n", name.c_str(), installed.c_str());
            std::printf("`%s pulldraft %s --force` fetches it again.\n", prog().c_str(), name.c_str());
            return 0;
        }

        std::printf("looking for a draft model for %s\n", name.c_str());
        if (find_drafter_candidates(gguf, reg).empty()) {
            std::printf("\nnothing published for this model. A drafter has to be trained against\n");
            std::printf("these exact weights, and either none exists or the ones that do ship\n");
            std::printf("only safetensors. %s keeps its self-speculation (%s).\n", name.c_str(),
                        spec_fallback().c_str());
            return 0;
        }
        return 0;
    } catch (const CliExit & e) {
        return e.code;
    }
}

bool dispatch(const std::string & cmd, const std::vector<std::string> & args, Config & cfg, Registry & reg,
              int & exit_code) {
    const std::string name = cmd == "ls" ? "list" : cmd;

    if (name == "push") {
        exit_code = cmd_push();
        return true;
    }
    if (name == "signin") {
        exit_code = cmd_signin();
        return true;
    }
    if (name == "signout") {
        exit_code = cmd_signout();
        return true;
    }
    if (name == "link") {
        exit_code = cmd_link(args, cfg);
        return true;
    }
    if (name == "unlink") {
        exit_code = cmd_unlink();
        return true;
    }
    if (name == "uninstall") {
        exit_code = cmd_uninstall(args, cfg);
        return true;
    }
    if (name == "pulldraft") {
        exit_code = cmd_pulldraft(args, reg);
        return true;
    }

    ApiClient api(cfg);
    if (name == "list") {
        exit_code = cmd_list(args, api, cfg);
        return true;
    }
    if (name == "ps") {
        exit_code = cmd_ps(args, api, cfg);
        return true;
    }
    if (name == "show") {
        exit_code = cmd_show(args, api);
        return true;
    }
    if (name == "rm") {
        exit_code = cmd_rm(args, api);
        return true;
    }
    if (name == "stop") {
        exit_code = cmd_stop(args, api);
        return true;
    }
    if (name == "pull" || name == "install") {
        exit_code = cmd_pull(args, api);
        return true;
    }
    if (name == "create") {
        exit_code = cmd_create(args, api);
        return true;
    }
    if (name == "cp") {
        exit_code = cmd_cp(args, api);
        return true;
    }
    return false;
}

} // namespace llmash
