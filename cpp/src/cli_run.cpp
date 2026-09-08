#include "cli_format.h"
#include "cli_run.h"

#include <httplib.h>

#include <windows.h>
#include <io.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <regex>
#include <set>
#include <sstream>
#include <thread>
#include <utility>

// Cross-file pieces run.go leans on that belong to other command modules
// (cmds.go's cmdList/showInfo, pull.go's cmdPull, draft_install.go's
// confirm, readline.go's raw-mode editor, progress.go's multi-state
// renderer) are not ported here — they are a different Go file each, and
// this job is run.go only.

namespace llmash {

using json = nlohmann::json;

namespace {

// --------------------------------------------------------------- wire JSON
// Loose accessors matching http.go's str/num/sub/list: wrong type or a
// missing key is not an error, just an empty result.

std::string j_str(const json & m, const std::string & k) {
    if (!m.is_object()) return "";
    const auto it = m.find(k);
    if (it == m.end() || it->is_null()) return "";
    if (it->is_string()) return it->get<std::string>();
    return it->dump();
}

double j_num(const json & m, const std::string & k) {
    if (!m.is_object()) return 0;
    const auto it = m.find(k);
    if (it == m.end() || !it->is_number()) return 0;
    return it->get<double>();
}

json j_sub(const json & m, const std::string & k) {
    if (!m.is_object()) return json::object();
    const auto it = m.find(k);
    if (it == m.end() || !it->is_object()) return json::object();
    return *it;
}

json j_list(const json & m, const std::string & k) {
    if (!m.is_object()) return json::array();
    const auto it = m.find(k);
    if (it == m.end() || !it->is_array()) return json::array();
    return *it;
}

std::string first_of(std::initializer_list<std::string> vals) {
    for (const auto & v : vals) {
        if (!v.empty()) return v;
    }
    return "";
}

// ------------------------------------------------------------- number text

std::optional<long long> try_parse_int_strict(const std::string & s) {
    if (s.empty()) return std::nullopt;
    size_t i = 0;
    if (s[i] == '+' || s[i] == '-') i++;
    if (i == s.size()) return std::nullopt;
    for (size_t j = i; j < s.size(); j++) {
        if (!std::isdigit(static_cast<unsigned char>(s[j]))) return std::nullopt;
    }
    try {
        return std::stoll(s);
    } catch (const std::exception &) {
        return std::nullopt;
    }
}

// fmt.Sscan's leading-integer read for --ctx/--dimensions: a bad value is
// silently left at its zero default, never an error the user sees.
int scan_leading_int(const std::string & s) {
    size_t i = 0;
    if (i < s.size() && (s[i] == '+' || s[i] == '-')) i++;
    const size_t start = i;
    while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i]))) i++;
    if (i == start) return 0;
    try {
        return std::stoi(s.substr(0, i));
    } catch (const std::exception &) {
        return 0;
    }
}

std::optional<double> try_parse_double(const std::string & s) {
    if (s.empty()) return std::nullopt;
    try {
        size_t pos = 0;
        const double v = std::stod(s, &pos);
        if (pos != s.size()) return std::nullopt;
        return v;
    } catch (const std::exception &) {
        return std::nullopt;
    }
}

// strconv.ParseBool's exact accepted spellings.
std::optional<bool> try_parse_bool(const std::string & s) {
    static const std::set<std::string> truthy = {"1", "t", "T", "TRUE", "true", "True"};
    static const std::set<std::string> falsy  = {"0", "f", "F", "FALSE", "false", "False"};
    if (truthy.count(s)) return true;
    if (falsy.count(s)) return false;
    return std::nullopt;
}

std::string go_slice_repr(const std::vector<std::string> & vals) {
    std::string s = "[";
    for (size_t i = 0; i < vals.size(); i++) {
        if (i) s += " ";
        s += vals[i];
    }
    return s + "]";
}

} // namespace

// ==================================================== cobra's `run` flags

RunArgs parse_run_args(const std::vector<std::string> & args) {
    static const std::set<std::string> think_levels = {"true", "false", "high", "medium", "low", "max"};
    static const std::set<std::string> bool_flags = {"--verbose", "--nowordwrap", "--hidethinking", "--insecure"};
    static const std::set<std::string> val_flags  = {"--format", "--keepalive", "--dimensions", "--width",
                                                        "--height", "--steps", "--seed", "--negative", "--ctx",
                                                        "--temperature"};

    RunArgs r;
    std::vector<std::string> pos;

    for (size_t i = 0; i < args.size(); i++) {
        const std::string & t = args[i];
        if (t.empty() || t[0] != '-' || t == "-") {
            pos.push_back(t);
            continue;
        }

        std::string name = t, val;
        bool        has_val = false;
        const auto  eq = t.find('=');
        if (eq != std::string::npos && eq > 0) {
            name    = t.substr(0, eq);
            val     = t.substr(eq + 1);
            has_val = true;
        }

        if (name == "--think") {
            // cobra's optional-value flag: bare --think is true, only
            // --think=level carries a value.
            r.think_set = true;
            r.think     = "true";
            if (has_val) {
                if (!think_levels.count(val)) {
                    throw CliUsageError("Error: invalid value for --think: \"" + val +
                                         "\" (must be true, false, high, medium, low, or max)");
                }
                r.think = val;
            }
            continue;
        }
        if (name == "--truncate") {
            bool b = true;
            if (has_val) {
                const auto parsed = try_parse_bool(val);
                if (!parsed) throw CliUsageError("Error: invalid argument \"" + val + "\" for \"--truncate\" flag");
                b = *parsed;
            }
            r.truncate = b;
            continue;
        }
        if (bool_flags.count(name)) {
            if (name == "--verbose") r.verbose = true;
            else if (name == "--nowordwrap") r.nowordwrap = true;
            else if (name == "--hidethinking") r.hidethinking = true;
            // --insecure is accepted for parity with pull/create but run
            // itself does nothing with it, same as the Go switch's silent case.
            continue;
        }
        if (val_flags.count(name)) {
            if (!has_val) {
                if (i + 1 >= args.size()) throw CliUsageError("Error: flag needs an argument: " + name);
                val = args[++i];
            }
            if (name == "--format") r.format = val;
            else if (name == "--keepalive") r.keepalive = val;
            else if (name == "--ctx") r.ctx = scan_leading_int(val);
            else if (name == "--dimensions") r.dimensions = scan_leading_int(val);
            else if (name == "--temperature") {
                const auto f = try_parse_double(val);
                if (f) r.temperature = f;
            }
            // width/height/steps/seed/negative are image-gen flags accepted
            // by the shared parser; `run` has nowhere to put them.
            continue;
        }
        throw CliUsageError("Error: unknown flag: '" + t + "'");
    }

    if (pos.empty()) throw CliUsageError("Error: requires at least 1 arg(s), only received 0");
    r.model = pos[0];
    std::string prompt;
    for (size_t i = 1; i < pos.size(); i++) {
        if (i > 1) prompt += " ";
        prompt += pos[i];
    }
    r.prompt = prompt;
    return r;
}

// ======================================================= the wire helpers

json format_value(const std::string & f) {
    if (f.empty()) return nullptr;
    if (f == "json") return "json";
    const json v = json::parse(f, nullptr, false);
    if (!v.is_discarded()) return v;
    return f;
}

json format_param(const std::string & key, const std::vector<std::string> & vals) {
    static const std::set<std::string> int_params = {"seed", "num_predict", "top_k", "num_ctx", "repeat_last_n",
                                                        "num_gpu", "num_keep", "num_batch", "num_thread", "main_gpu",
                                                        "mirostat"};
    static const std::set<std::string> bool_params = {"numa", "low_vram", "use_mmap", "use_mlock"};

    if (key == "stop") {
        json arr = json::array();
        for (const auto & v : vals) arr.push_back(v);
        return arr;
    }
    const std::string first_val = vals.empty() ? "" : vals[0];
    if (int_params.count(key)) {
        const auto n = try_parse_int_strict(first_val);
        if (!n) throw CliUsageError("invalid int value " + go_slice_repr(vals));
        return *n;
    }
    if (bool_params.count(key)) {
        const auto b = try_parse_bool(first_val);
        if (!b) throw CliUsageError("invalid bool value " + go_slice_repr(vals));
        return *b;
    }
    const auto f = try_parse_double(first_val);
    if (!f) throw CliUsageError("invalid float value " + go_slice_repr(vals));
    return *f;
}

// ---------------------------------------------------------------- display

namespace {

std::vector<std::string> utf8_runes(const std::string & s) {
    std::vector<std::string> out;
    size_t                   i = 0;
    while (i < s.size()) {
        const unsigned char c = static_cast<unsigned char>(s[i]);
        size_t              len = 1;
        if ((c & 0x80) == 0x00) len = 1;
        else if ((c & 0xE0) == 0xC0) len = 2;
        else if ((c & 0xF0) == 0xE0) len = 3;
        else if ((c & 0xF8) == 0xF0) len = 4;
        len = std::min(len, s.size() - i);
        out.push_back(s.substr(i, len));
        i += len;
    }
    return out;
}

uint32_t decode_one(const std::string & rune) {
    const auto c0 = static_cast<unsigned char>(rune[0]);
    if (rune.size() == 1) return c0;
    if (rune.size() == 2) return ((c0 & 0x1F) << 6) | (static_cast<unsigned char>(rune[1]) & 0x3F);
    if (rune.size() == 3) {
        return ((c0 & 0x0F) << 12) | ((static_cast<unsigned char>(rune[1]) & 0x3F) << 6) |
               (static_cast<unsigned char>(rune[2]) & 0x3F);
    }
    return ((c0 & 0x07) << 18) | ((static_cast<unsigned char>(rune[1]) & 0x3F) << 12) |
           ((static_cast<unsigned char>(rune[2]) & 0x3F) << 6) | (static_cast<unsigned char>(rune[3]) & 0x3F);
}

// Same ranges as table.go's dispWidth. Combining marks are a hand-picked
// subset of Unicode's Mn/Me blocks rather than a full category table.
int codepoint_width(uint32_t r) {
    if (r == 0) return 0;
    if ((r >= 0x0300 && r <= 0x036F) || (r >= 0x1AB0 && r <= 0x1AFF) || (r >= 0x1DC0 && r <= 0x1DFF) ||
        (r >= 0x20D0 && r <= 0x20FF) || (r >= 0xFE20 && r <= 0xFE2F)) {
        return 0;
    }
    if (r >= 0x1100 &&
        (r <= 0x115F || r == 0x2329 || r == 0x232A || (r >= 0x2E80 && r <= 0xA4CF && r != 0x303F) ||
         (r >= 0xAC00 && r <= 0xD7A3) || (r >= 0xF900 && r <= 0xFAFF) || (r >= 0xFE30 && r <= 0xFE6F) ||
         (r >= 0xFF00 && r <= 0xFF60) || (r >= 0xFFE0 && r <= 0xFFE6) || (r >= 0x1F300 && r <= 0x1F64F) ||
         (r >= 0x1F900 && r <= 0x1F9FF) || (r >= 0x20000 && r <= 0x3FFFD))) {
        return 2;
    }
    return 1;
}

bool is_console(HANDLE h) {
    DWORD mode;
    return h != INVALID_HANDLE_VALUE && GetConsoleMode(h, &mode) != 0;
}

bool is_console(FILE * f) {
    return is_console(reinterpret_cast<HANDLE>(_get_osfhandle(_fileno(f))));
}

int term_width() {
    CONSOLE_SCREEN_BUFFER_INFO info;
    if (!GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &info)) return 100;
    const int w = info.srWindow.Right - info.srWindow.Left + 1;
    return w > 0 ? w : 100;
}

} // namespace


void display_response(const std::string & content, bool word_wrap, DisplayState & state) {
    int w = term_width();
    if (w == 0) w = 80;

    if (word_wrap && w >= 10) {
        for (const auto & ch : utf8_runes(content)) {
            if (state.line_length + 1 > w - 5) {
                if (disp_width(state.word_buffer) > w - 10) {
                    std::cout << state.word_buffer << ch;
                    state.word_buffer.clear();
                    state.line_length = 0;
                    continue;
                }
                const int a = disp_width(state.word_buffer);
                if (a > 0) std::cout << "\x1b[" << a << "D";
                std::cout << "\x1b[K\n" << state.word_buffer << ch;
                state.line_length = disp_width(state.word_buffer) + disp_width(ch);
                continue;
            }
            std::cout << ch;
            state.line_length += disp_width(ch);
            if (disp_width(ch) >= 2) {
                state.word_buffer.clear();
                continue;
            }
            if (ch == " " || ch == "\t") state.word_buffer.clear();
            else if (ch == "\n" || ch == "\r") {
                state.line_length = 0;
                state.word_buffer.clear();
            } else {
                state.word_buffer += ch;
            }
        }
        return;
    }
    std::cout << state.word_buffer << content;
    state.word_buffer.clear();
}

// ------------------------------------------------------------- file paths

std::string normalize_file_path(const std::string & fp) {
    static const std::string escapable = " ()[]{}$&;'\\*?~";
    std::string              out;
    out.reserve(fp.size());
    for (size_t i = 0; i < fp.size(); i++) {
        if (fp[i] == '\\' && i + 1 < fp.size() && escapable.find(fp[i + 1]) != std::string::npos) {
            out += fp[i + 1];
            i++;
        } else {
            out += fp[i];
        }
    }
    return out;
}

std::string encode_base64(const std::string & bytes) {
    static const char tbl[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string        out;
    out.reserve(((bytes.size() + 2) / 3) * 4);
    size_t i = 0;
    for (; i + 2 < bytes.size(); i += 3) {
        const uint32_t n = (static_cast<unsigned char>(bytes[i]) << 16) |
                           (static_cast<unsigned char>(bytes[i + 1]) << 8) | static_cast<unsigned char>(bytes[i + 2]);
        out += tbl[(n >> 18) & 0x3F];
        out += tbl[(n >> 12) & 0x3F];
        out += tbl[(n >> 6) & 0x3F];
        out += tbl[n & 0x3F];
    }
    const size_t rem = bytes.size() - i;
    if (rem == 1) {
        const uint32_t n = static_cast<unsigned char>(bytes[i]) << 16;
        out += tbl[(n >> 18) & 0x3F];
        out += tbl[(n >> 12) & 0x3F];
        out += "==";
    } else if (rem == 2) {
        const uint32_t n = (static_cast<unsigned char>(bytes[i]) << 16) | (static_cast<unsigned char>(bytes[i + 1]) << 8);
        out += tbl[(n >> 18) & 0x3F];
        out += tbl[(n >> 12) & 0x3F];
        out += tbl[(n >> 6) & 0x3F];
        out += "=";
    }
    return out;
}

namespace {
const std::regex & file_re() {
    static const std::regex re(R"((?:[a-zA-Z]:)?(?:\./|/|\\)[\S\\ ]+?\.(jpg|jpeg|png|webp|wav)\b)",
                                std::regex::icase);
    return re;
}
} // namespace

FileExtraction extract_file_data(const std::string & input) {
    FileExtraction out;
    out.text = input;
    auto begin = std::sregex_iterator(input.begin(), input.end(), file_re());
    for (auto it = begin; it != std::sregex_iterator(); ++it) {
        const std::string fp  = it->str();
        const std::string nfp = normalize_file_path(fp);
        std::ifstream      in(nfp, std::ios::binary);
        if (!in) {
            std::printf("Couldn't process image: \"open %s: file not found\"\n", nfp.c_str());
            out.ok = false;
            return out;
        }
        std::string bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        std::printf("Added image '%s'\n", nfp.c_str());
        size_t pos;
        while ((pos = out.text.find(fp)) != std::string::npos) out.text.erase(pos, fp.size());
        out.images_b64.push_back(encode_base64(bytes));
    }
    // strings.TrimSpace
    size_t a = out.text.find_first_not_of(" \t\r\n");
    size_t b = out.text.find_last_not_of(" \t\r\n");
    out.text = (a == std::string::npos) ? "" : out.text.substr(a, b - a + 1);
    return out;
}

// ------------------------------------------------------------ run options

RunOptions RunOptions::copy() const {
    // vector<Message> and json copy-construct deeply already; nothing more
    // to do than what the compiler-generated copy gives us.
    return *this;
}

json RunOptions::body(const json & extra) const {
    json b;
    b["model"] = model;
    if (!options.empty()) b["options"] = options;
    const json fv = format_value(format);
    if (!fv.is_null()) b["format"] = fv;
    if (!think.is_null()) b["think"] = think;
    if (!keep_alive.is_null()) b["keep_alive"] = keep_alive;
    if (extra.is_object()) {
        for (auto it = extra.begin(); it != extra.end(); ++it) b[it.key()] = it.value();
    }
    return b;
}

bool has_cap(const json & model_info, const std::string & want) {
    for (const auto & c : j_list(model_info, "capabilities")) {
        if (c.is_string() && c.get<std::string>() == want) return true;
    }
    return false;
}

void infer_thinking(const json & model_info, RunOptions & o, bool explicit_set) {
    if (explicit_set) return;
    o.think = has_cap(model_info, "thinking") ? json(true) : json(nullptr);
}

// ================================================================ the HTTP client

namespace {

std::string resolve_host() {
    const char * env = std::getenv("OLLAMA_HOST");
    std::string  h   = (env && *env) ? env : "";
    if (h.empty()) h = "http://127.0.0.1:11434";
    if (h.rfind("http", 0) != 0) h = "http://" + h;
    while (!h.empty() && h.back() == '/') h.pop_back();
    return h;
}

std::string prog_name() {
    const char * p = std::getenv("LLMASH_PROG");
    return (p && *p) ? p : "llmash";
}

struct HttpResult {
    bool        ok     = false; // transport succeeded and the body (if any) was valid JSON
    int         status = 0;
    json        body   = json::object();
    std::string error;
};

HttpResult http_call_json(const std::string & method, const std::string & path, const json * body, int timeout_sec) {
    httplib::Client cli(resolve_host());
    cli.set_connection_timeout(timeout_sec, 0);
    cli.set_read_timeout(timeout_sec, 0);
    cli.set_write_timeout(timeout_sec, 0);

    httplib::Result res = (method == "GET") ? cli.Get(path) : cli.Post(path, body ? body->dump() : "", "application/json");

    HttpResult out;
    if (!res) {
        out.error = httplib::to_string(res.error());
        return out;
    }
    out.status = res->status;
    if (!res->body.empty()) {
        const json j = json::parse(res->body, nullptr, false);
        if (j.is_discarded()) {
            out.error = "not JSON: " + res->body;
            return out;
        }
        out.body = j;
    }
    out.ok = true;
    return out;
}

enum class StreamOutcome { Ok, Cancelled, TransportError, BadNdjson };

struct StreamResult {
    StreamOutcome outcome;
    std::string   error;
};

// One ndjson line at a time, as http.go's stream() does.
StreamResult http_stream(const std::string & path, const json & body, const std::atomic<bool> & cancelled,
                          const std::function<bool(const json &)> & on_event) {
    httplib::Client cli(resolve_host());
    cli.set_connection_timeout(5, 0);
    cli.set_read_timeout(3600, 0);
    cli.set_write_timeout(30, 0);

    std::string buf;
    bool        bad_ndjson    = false;
    bool        stopped_early = false;
    bool        was_cancelled = false;

    const httplib::ContentReceiver receiver = [&](const char * data, size_t len) -> bool {
        buf.append(data, len);
        size_t pos;
        while ((pos = buf.find('\n')) != std::string::npos) {
            std::string line = buf.substr(0, pos);
            buf.erase(0, pos + 1);
            while (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.find_first_not_of(" \t") == std::string::npos) continue;
            const json ev = json::parse(line, nullptr, false);
            if (ev.is_discarded()) {
                bad_ndjson    = true;
                stopped_early = true;
                return false;
            }
            if (!on_event(ev)) {
                stopped_early = true;
                return false;
            }
        }
        if (cancelled.load()) {
            was_cancelled = true;
            stopped_early = true;
            return false;
        }
        return true;
    };

    const httplib::Result res = cli.Post(path, httplib::Headers{}, body.dump(), "application/json", receiver);

    if (was_cancelled) return {StreamOutcome::Cancelled, ""};
    if (bad_ndjson) return {StreamOutcome::BadNdjson, "the server sent a response that isn't NDJSON"};
    if (!res && !stopped_early) return {StreamOutcome::TransportError, httplib::to_string(res.error())};
    return {StreamOutcome::Ok, ""};
}

bool server_up() {
    for (const int wait_sec : {4, 8}) {
        if (http_call_json("GET", "/api/version", nullptr, wait_sec).ok) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
    }
    return false;
}

// die()'s C++ shape (see cli_run.h): print, then unwind via CliExit rather
// than calling std::exit, so a test can catch it instead of the process
// dying.
[[noreturn]] void die(const std::string & msg) {
    std::fprintf(stderr, "%s\n", msg.c_str());
    throw CliExit(1, msg);
}

void need_server() {
    if (server_up()) return;
    std::fprintf(stderr, "llmash isn't running at %s.\nStart it with:  %s serve\n", resolve_host().c_str(),
                 prog_name().c_str());
    throw CliExit(1);
}

} // namespace

// ---------------------------------------------------- Ctrl+C, RAII-scoped

namespace {

std::atomic<bool> g_interrupted{false};

BOOL WINAPI console_ctrl_handler(DWORD ctrl_type) {
    if (ctrl_type == CTRL_C_EVENT || ctrl_type == CTRL_BREAK_EVENT) {
        g_interrupted.store(true);
        return TRUE;
    }
    return FALSE;
}

// interruptible()'s C++ shape: SetConsoleCtrlHandler in place of Go's
// signal.Notify, registered and torn down around one request the same way
// interruptible()/its cancel func are.
class InterruptGuard {
public:
    InterruptGuard() {
        g_interrupted.store(false);
        installed_ = ::SetConsoleCtrlHandler(console_ctrl_handler, TRUE) != 0;
    }
    ~InterruptGuard() {
        if (installed_) ::SetConsoleCtrlHandler(console_ctrl_handler, FALSE);
    }
    InterruptGuard(const InterruptGuard &)             = delete;
    InterruptGuard & operator=(const InterruptGuard &) = delete;

    bool interrupted() const { return g_interrupted.load(); }
    bool is_installed() const { return installed_; }

private:
    bool installed_ = false;
};

// newProgress+newSpinner's C++ shape: run.go only ever adds one spinner to
// one progress and stops it the same call, so the two collapse into one RAII
// guard instead of the general multi-state renderer in progress.go.
class Spinner {
public:
    explicit Spinner(std::ostream & out) : out_(out) { thread_ = std::thread([this] { run(); }); }
    ~Spinner() {
        stop_.store(true);
        if (thread_.joinable()) thread_.join();
    }
    Spinner(const Spinner &)             = delete;
    Spinner & operator=(const Spinner &) = delete;

    // Called once the first token/error arrives, matching stopAndClear().
    void stop_and_clear() {
        if (stop_.exchange(true)) return;
        if (thread_.joinable()) thread_.join();
        std::lock_guard<std::mutex> lk(mu_);
        out_ << "\r\x1b[K" << std::flush;
    }

    static std::atomic<int> & live_count() {
        static std::atomic<int> n{0};
        return n;
    }

private:
    void run() {
        live_count().fetch_add(1);
        static const char * frames[] = {"\xe2\xa0\x8b", "\xe2\xa0\x99", "\xe2\xa0\xb9", "\xe2\xa0\xb8", "\xe2\xa0\xbc",
                                          "\xe2\xa0\xb4", "\xe2\xa0\xa6", "\xe2\xa0\xa7", "\xe2\xa0\x87", "\xe2\xa0\x8f"};
        int i = 0;
        while (!stop_.load()) {
            {
                std::lock_guard<std::mutex> lk(mu_);
                out_ << "\r" << frames[i % 10] << " " << std::flush;
            }
            i++;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        live_count().fetch_sub(1);
    }

    std::ostream &    out_;
    std::mutex        mu_;
    std::thread       thread_;
    std::atomic<bool> stop_{false};
};

constexpr const char * kColorGrey    = "\x1b[38;5;245m";
constexpr const char * kColorDefault = "\x1b[0m";

std::string think_open(bool plain) {
    if (plain) return "Thinking...\n";
    return std::string(kColorGrey) + "\x1b[1m" + "Thinking...\n" + kColorDefault + kColorGrey;
}

std::string think_close(bool plain) {
    if (plain) return "...done thinking.\n\n";
    return std::string(kColorGrey) + "\x1b[1m" + "...done thinking.\n\n" + kColorDefault;
}

std::string render_tool_calls(const json & calls, bool plain) {
    std::string out, expl, vals;
    if (!plain) {
        expl = std::string(kColorGrey) + "\x1b[1m";
        vals = kColorDefault;
        out += expl;
    }
    for (size_t i = 0; i < calls.size(); i++) {
        const json fn = j_sub(calls[i], "function");
        const std::string args = fn.contains("arguments") ? fn.at("arguments").dump() : "null";
        if (i > 0) out += "\n";
        out += "  Model called a non-existent function '" + vals + j_str(fn, "name") + expl +
               "()' with arguments: " + vals + args + expl;
    }
    if (!plain) out += kColorDefault;
    return out;
}

// Approximates Go's time.Duration.String(): "1.234s", "150ms", "2h1m3s".
std::string format_go_duration(double ns) {
    if (ns == 0) return "0s";
    const bool neg = ns < 0;
    double     u   = std::fabs(ns);
    std::ostringstream oss;
    if (u < 1e9) {
        double      scaled;
        const char * unit;
        int          prec;
        if (u < 1e3) { scaled = u; unit = "ns"; prec = 0; }
        else if (u < 1e6) { scaled = u / 1e3; unit = "\xc2\xb5s"; prec = 3; }
        else { scaled = u / 1e6; unit = "ms"; prec = 3; }
        oss.setf(std::ios::fixed);
        oss << std::setprecision(prec) << scaled;
        std::string s = oss.str();
        if (prec > 0) {
            while (!s.empty() && s.back() == '0') s.pop_back();
            if (!s.empty() && s.back() == '.') s.pop_back();
        }
        return (neg ? "-" : "") + s + unit;
    }
    long long total_s = static_cast<long long>(u / 1e9);
    double     frac    = (u - total_s * 1e9) / 1e9;
    long long  mins    = total_s / 60;
    long long  secs    = total_s % 60;
    long long  hours   = mins / 60;
    mins %= 60;
    if (hours > 0) oss << hours << "h";
    if (hours > 0 || mins > 0) oss << mins << "m";
    std::ostringstream fs;
    fs.setf(std::ios::fixed);
    fs << std::setprecision(9) << (static_cast<double>(secs) + frac);
    std::string s = fs.str();
    while (!s.empty() && s.back() == '0') s.pop_back();
    if (!s.empty() && s.back() == '.') s.pop_back();
    oss << s << "s";
    return (neg ? "-" : "") + oss.str();
}

void print_summary(const json & m) {
    const auto dur = [&](const char * k) { return j_num(m, k); };
    if (const double d = dur("total_duration"); d > 0) {
        std::fprintf(stderr, "total duration:       %s\n", format_go_duration(d).c_str());
    }
    if (const double d = dur("load_duration"); d > 0) {
        std::fprintf(stderr, "load duration:        %s\n", format_go_duration(d).c_str());
    }
    const double pc = j_num(m, "prompt_eval_count");
    if (pc > 0) std::fprintf(stderr, "prompt eval count:    %d token(s)\n", static_cast<int>(pc));
    if (const double d = dur("prompt_eval_duration"); d > 0) {
        std::fprintf(stderr, "prompt eval duration: %s\n", format_go_duration(d).c_str());
        std::fprintf(stderr, "prompt eval rate:     %.2f tokens/s\n", pc / (d / 1e9));
    }
    const double ec = j_num(m, "eval_count");
    if (ec > 0) std::fprintf(stderr, "eval count:           %d token(s)\n", static_cast<int>(ec));
    if (const double d = dur("eval_duration"); d > 0) {
        std::fprintf(stderr, "eval duration:        %s\n", format_go_duration(d).c_str());
        std::fprintf(stderr, "eval rate:            %.2f tokens/s\n", ec / (d / 1e9));
    }
}

// ----------------------------------------------- stand-ins for other files

// draft_install.go's confirm(): a plain y/N prompt on stdin.
bool confirm(const std::string & question) {
    std::fprintf(stderr, "%s [y/N] ", question.c_str());
    std::string line;
    if (!std::getline(std::cin, line)) return false;
    return line == "y" || line == "Y" || line == "yes" || line == "Yes";
}

// pull.go's cmdPull is a whole other command (download, verify, decompress).
void stand_in_pull(const std::string & name) {
    die("Error: '" + name + "' is not installed and pulling models isn't wired into this build yet; " +
        "install it first with the model's own release, then run it.");
}

// cmds.go's showInfo() lays this out as aligned tables; this keeps the same
// blocks and order without table.go's column layout.
void print_model_info(const json & resp) {
    const json info = j_sub(resp, "model_info");
    const json det   = j_sub(resp, "details");
    std::printf("  Model\n");
    const std::string arch = j_str(info, "general.architecture");
    if (!info.empty()) {
        if (!arch.empty()) std::printf("    architecture        %s\n", arch.c_str());
        std::string param = j_str(det, "parameter_size");
        if (!param.empty()) std::printf("    parameters          %s\n", param.c_str());
        if (info.contains(arch + ".context_length")) {
            std::printf("    context length      %s\n", j_str(info, arch + ".context_length").c_str());
        }
    } else {
        std::printf("    architecture        %s\n", j_str(det, "family").c_str());
        std::printf("    parameters          %s\n", j_str(det, "parameter_size").c_str());
    }
    std::printf("    quantization        %s\n\n", j_str(det, "quantization_level").c_str());
    const json caps = j_list(resp, "capabilities");
    if (!caps.empty()) {
        std::printf("  Capabilities\n");
        for (const auto & c : caps) std::printf("    %s\n", c.is_string() ? c.get<std::string>().c_str() : c.dump().c_str());
        std::printf("\n");
    }
}

// cmds.go's cmdList, minus table.go's aligned columns: enough for the
// REPL's own /list to show something real.
void stand_in_list() {
    const HttpResult r = http_call_json("GET", "/api/tags", nullptr, 60);
    if (!r.ok) die("error: " + r.error);
    for (const auto & m : j_list(r.body, "models")) std::printf("%s\n", j_str(m, "name").c_str());
}

} // namespace

// ============================================================ model calls

namespace {

std::pair<json, int> show_model(const std::string & name) {
    const json body = {{"model", name}};
    const HttpResult r = http_call_json("POST", "/api/show", &body, 60);
    if (!r.ok) die("Error: " + r.error);
    return {r.body, r.status};
}

json show_or_pull(RunOptions & o) {
    auto [d, code] = show_model(o.model);
    if (code == 200) return d;
    if (code != 404) die("Error: " + first_of({j_str(d, "error"), "could not read that model"}));
    if (is_console(stdin) && is_console(stdout) && !confirm(o.model + " is not on this machine. Pull it?")) {
        throw CliExit(1);
    }
    stand_in_pull(o.model);
    std::tie(d, code) = show_model(o.model);
    if (code != 200) die("Error: " + first_of({j_str(d, "error"), "model not found"}));
    return d;
}

std::string http_status_text(int code) {
    switch (code) {
        case 400: return "400 Bad Request";
        case 404: return "404 Not Found";
        case 500: return "500 Internal Server Error";
        case 503: return "503 Service Unavailable";
        default: return std::to_string(code);
    }
}

// A blank generate request loads (or with keep_alive 0, unloads) the model.
void load_or_unload_model(RunOptions & o) {
    Spinner sp(std::cerr);
    const json body = o.body({{"prompt", ""}, {"stream", false}});
    const HttpResult r = http_call_json("POST", "/api/generate", &body, 600);
    sp.stop_and_clear();
    if (!r.ok) throw std::runtime_error(r.error);
    if (r.status >= 400) throw std::runtime_error(first_of({j_str(r.body, "error"), http_status_text(r.status)}));
}

std::string create_model(const json & req) {
    std::atomic<bool> cancelled{false};
    std::string       bad;
    const StreamResult sr = http_stream("/api/create", req, cancelled, [&](const json & ev) {
        if (const std::string e = j_str(ev, "error"); !e.empty()) {
            bad = e;
            return false;
        }
        return true;
    });
    if (!bad.empty()) return bad;
    if (sr.outcome == StreamOutcome::TransportError || sr.outcome == StreamOutcome::BadNdjson) return sr.error;
    return "";
}

void ensure_thinking_support(const std::string & model) {
    auto [info, code] = show_model(model);
    if (code != 200) return;
    if (!has_cap(info, "thinking")) {
        std::fprintf(stderr, "warning: model \"%s\" does not support thinking output\n", model.c_str());
    }
}

Message chat_once(RunOptions o) {
    InterruptGuard guard;
    Spinner        sp(std::cerr);

    DisplayState state;
    std::string  thinking, full, role = "assistant", api_err;
    json         latest;
    bool         opened = false, closed = false, stopped = false;

    json msgs = json::array();
    for (const auto & m : o.messages) msgs.push_back(m);
    const json body = o.body({{"messages", msgs}, {"stream", true}});

    const StreamResult sr = http_stream("/api/chat", body, g_interrupted, [&](const json & ev) {
        if (const std::string e = j_str(ev, "error"); !e.empty()) {
            api_err = e;
            return false;
        }
        const json m = j_sub(ev, "message");
        if (!j_str(m, "content").empty() || !o.hide_thinking) {
            if (!stopped) {
                sp.stop_and_clear();
                stopped = true;
            }
        }
        latest = ev;
        if (const std::string rr = j_str(m, "role"); !rr.empty()) role = rr;
        if (const std::string t = j_str(m, "thinking"); !t.empty() && !o.hide_thinking) {
            if (!opened) {
                std::cout << think_open(false);
                opened = true;
                closed = false;
            }
            thinking += t;
            display_response(t, o.word_wrap, state);
        }
        const std::string content = j_str(m, "content");
        const json        calls   = j_list(m, "tool_calls");
        if (opened && !closed && (!content.empty() || !calls.empty())) {
            if (thinking.empty() || thinking.back() != '\n') std::cout << "\n";
            std::cout << think_close(false);
            opened = false;
            closed = true;
            state  = DisplayState{};
        }
        full += content;
        if (!calls.empty()) std::cout << render_tool_calls(calls, false);
        display_response(content, o.word_wrap, state);
        return true;
    });
    if (!stopped) sp.stop_and_clear();

    if (!api_err.empty()) throw std::runtime_error(api_err);
    if (sr.outcome == StreamOutcome::Cancelled) return nullptr;
    if (sr.outcome == StreamOutcome::TransportError || sr.outcome == StreamOutcome::BadNdjson) {
        throw std::runtime_error(sr.error);
    }
    if (!o.messages.empty()) std::cout << "\n\n";
    if (o.verbose && !latest.is_null()) print_summary(latest);
    return Message{{"role", role}, {"thinking", thinking}, {"content", full}};
}

void generate_once(RunOptions o) {
    InterruptGuard guard;
    Spinner        sp(std::cerr);

    DisplayState state;
    std::string  thinking, api_err;
    json         latest;
    bool         opened = false, closed = false, stopped = false;
    const bool   plain = !is_console(stdout);

    json extra = {{"prompt", o.prompt}, {"stream", true}};
    if (!o.system.empty()) extra["system"] = o.system;
    if (!o.images.empty()) extra["images"] = o.images;

    const StreamResult sr = http_stream("/api/generate", o.body(extra), g_interrupted, [&](const json & ev) {
        if (const std::string e = j_str(ev, "error"); !e.empty()) {
            api_err = e;
            return false;
        }
        latest = ev;
        const std::string content = j_str(ev, "response");
        if (!content.empty() || !o.hide_thinking) {
            if (!stopped) {
                sp.stop_and_clear();
                stopped = true;
            }
        }
        if (const std::string t = j_str(ev, "thinking"); !t.empty() && !o.hide_thinking) {
            if (!opened) {
                std::cout << think_open(plain);
                opened = true;
                closed = false;
            }
            thinking += t;
            display_response(t, o.word_wrap, state);
        }
        const json calls = j_list(ev, "tool_calls");
        if (opened && !closed && (!content.empty() || !calls.empty())) {
            if (thinking.empty() || thinking.back() != '\n') std::cout << "\n";
            std::cout << think_close(plain);
            opened = false;
            closed = true;
            state  = DisplayState{};
        }
        display_response(content, o.word_wrap, state);
        if (!calls.empty()) std::cout << render_tool_calls(calls, plain);
        return true;
    });
    if (!stopped) sp.stop_and_clear();

    if (!api_err.empty()) die("Error: " + api_err);
    if (sr.outcome == StreamOutcome::Cancelled) return;
    if (sr.outcome == StreamOutcome::TransportError || sr.outcome == StreamOutcome::BadNdjson) die("Error: " + sr.error);

    if (!o.prompt.empty()) std::cout << "\n\n";
    if (latest.is_null()) return;
    if (!latest.value("done", false)) return;
    if (o.verbose) print_summary(latest);
}

void embed_once(const RunOptions & o, const std::optional<bool> & truncate, int dimensions) {
    json body = {{"model", o.model}, {"input", o.prompt}};
    if (!o.keep_alive.is_null()) body["keep_alive"] = o.keep_alive;
    if (truncate) body["truncate"] = *truncate;
    if (dimensions > 0) body["dimensions"] = dimensions;

    const HttpResult r = http_call_json("POST", "/api/embed", &body, 600);
    if (!r.ok) die("Error: " + r.error);
    if (r.status >= 400) die("Error: " + first_of({j_str(r.body, "error"), http_status_text(r.status)}));
    const json embs = j_list(r.body, "embeddings");
    if (embs.empty()) die("Error: no embeddings returned");
    std::printf("%s\n", embs[0].dump().c_str());
}

} // namespace

// ================================================================= the REPL

namespace {

enum class ReadOutcome { Ok, Eof };

// Stand-in for readline.go's raw-mode editor (history navigation, Ctrl+G
// external-editor, bracketed paste) — that file is its own module and isn't
// ported here.
class SimpleEditor {
public:
    bool    use_alt          = false;
    bool    pasting          = false;
    bool    history_enabled  = true;

    std::pair<std::string, ReadOutcome> readline(const std::string & prompt) {
        std::fputs(prompt.c_str(), stdout);
        std::fflush(stdout);
        std::string line;
        if (!std::getline(std::cin, line)) return {"", ReadOutcome::Eof};
        if (!line.empty() && line.back() == '\r') line.pop_back();
        return {line, ReadOutcome::Ok};
    }
};

void usage_main(bool multi_modal) {
    std::fprintf(stderr, "Available Commands:\n");
    std::fprintf(stderr, "  /set            Set session variables\n");
    std::fprintf(stderr, "  /show           Show model information\n");
    std::fprintf(stderr, "  /load <model>   Load a session or model\n");
    std::fprintf(stderr, "  /save <model>   Save your current session\n");
    std::fprintf(stderr, "  /clear          Clear session context\n");
    std::fprintf(stderr, "  /bye            Exit\n");
    std::fprintf(stderr, "  /?, /help       Help for a command\n");
    std::fprintf(stderr, "  /? shortcuts    Help for keyboard shortcuts\n\n");
    std::fprintf(stderr, "Use \"\"\" to begin a multi-line message.\n");
    if (multi_modal) {
        std::fprintf(stderr, "Use \\path\\to\\file to include .jpg, .png, .webp images, or .wav audio files.\n");
    }
    std::fprintf(stderr, "\n");
}

void usage_set() {
    std::fprintf(stderr, "Available Commands:\n");
    std::fprintf(stderr, "  /set parameter ...     Set a parameter\n");
    std::fprintf(stderr, "  /set system <string>   Set system message\n");
    std::fprintf(stderr, "  /set history           Enable history\n");
    std::fprintf(stderr, "  /set nohistory         Disable history\n");
    std::fprintf(stderr, "  /set wordwrap          Enable wordwrap\n");
    std::fprintf(stderr, "  /set nowordwrap        Disable wordwrap\n");
    std::fprintf(stderr, "  /set format json       Enable JSON mode\n");
    std::fprintf(stderr, "  /set noformat          Disable formatting\n");
    std::fprintf(stderr, "  /set verbose           Show LLM stats\n");
    std::fprintf(stderr, "  /set quiet             Disable LLM stats\n");
    std::fprintf(stderr, "  /set think             Enable thinking\n");
    std::fprintf(stderr, "  /set nothink           Disable thinking\n\n");
}

void usage_show() {
    std::fprintf(stderr, "Available Commands:\n");
    std::fprintf(stderr, "  /show info         Show details for this model\n");
    std::fprintf(stderr, "  /show license      Show model license\n");
    std::fprintf(stderr, "  /show modelfile    Show Modelfile for this model\n");
    std::fprintf(stderr, "  /show parameters   Show parameters for this model\n");
    std::fprintf(stderr, "  /show system       Show system message\n");
    std::fprintf(stderr, "  /show template     Show prompt template\n\n");
}

void usage_shortcuts() {
    std::fprintf(stderr, "Available keyboard shortcuts:\n");
    std::fprintf(stderr, "  Ctrl + a            Move to the beginning of the line (Home)\n");
    std::fprintf(stderr, "  Ctrl + e            Move to the end of the line (End)\n");
    std::fprintf(stderr, "   Alt + b            Move back (left) one word\n");
    std::fprintf(stderr, "   Alt + f            Move forward (right) one word\n");
    std::fprintf(stderr, "  Ctrl + k            Delete the sentence after the cursor\n");
    std::fprintf(stderr, "  Ctrl + u            Delete the sentence before the cursor\n");
    std::fprintf(stderr, "  Ctrl + w            Delete the word before the cursor\n\n");
    std::fprintf(stderr, "  Ctrl + l            Clear the screen\n");
    std::fprintf(stderr, "  Ctrl + g            Open default editor to compose a prompt\n");
    std::fprintf(stderr, "  Ctrl + c            Stop the model from responding\n");
    std::fprintf(stderr, "  Ctrl + d            Exit llmash (/bye)\n\n");
}

void usage_parameters() {
    std::fprintf(stderr, "Available Parameters:\n");
    std::fprintf(stderr, "  /set parameter seed <int>             Random number seed\n");
    std::fprintf(stderr, "  /set parameter num_predict <int>      Max number of tokens to predict\n");
    std::fprintf(stderr, "  /set parameter top_k <int>            Pick from top k num of tokens\n");
    std::fprintf(stderr, "  /set parameter top_p <float>          Pick token based on sum of probabilities\n");
    std::fprintf(stderr, "  /set parameter min_p <float>          Pick token based on top token probability * min_p\n");
    std::fprintf(stderr, "  /set parameter num_ctx <int>          Set the context size\n");
    std::fprintf(stderr, "  /set parameter temperature <float>    Set creativity level\n");
    std::fprintf(stderr, "  /set parameter repeat_penalty <float> How strongly to penalize repetitions\n");
    std::fprintf(stderr, "  /set parameter repeat_last_n <int>    Set how far back to look for repetitions\n");
    std::fprintf(stderr, "  /set parameter num_gpu <int>          The number of layers to send to the GPU\n");
    std::fprintf(stderr, "  /set parameter stop <string> <string> ...   Set the stop parameters\n\n");
}

std::vector<std::string> fields(const std::string & s) {
    std::vector<std::string> out;
    std::istringstream       iss(s);
    std::string              t;
    while (iss >> t) out.push_back(t);
    return out;
}

bool starts_with(const std::string & s, const std::string & p) {
    return s.compare(0, p.size(), p) == 0;
}

void generate_interactive(RunOptions o) {
    SimpleEditor ed;
    bool         think_set = !o.think.is_null();

    std::string sb;
    enum class Multiline { None, Prompt, System } multiline = Multiline::None;

    const auto set_system = [&](const std::string & text) {
        o.system = text;
        const Message nm = {{"role", "system"}, {"content", text}};
        if (!o.messages.empty() && j_str(o.messages.back(), "role") == "system") o.messages.back() = nm;
        else o.messages.push_back(nm);
    };

    // Bracketed paste is intentionally never turned on here: it wraps pasted
    // text in ESC[200~ / ESC[201~, and SimpleEditor (unlike readline.go's
    // raw-mode reader) doesn't strip those markers back out.
    for (;;) {
        const std::string cur_prompt = (multiline != Multiline::None) ? "... " : ">>> ";
        const auto [line, outcome]   = ed.readline(cur_prompt);
        if (outcome == ReadOutcome::Eof) return;

        if (multiline != Multiline::None) {
            std::string before  = line;
            bool        closed  = before.size() >= 3 && before.compare(before.size() - 3, 3, "\"\"\"") == 0;
            if (closed) before.resize(before.size() - 3);
            sb += before;
            if (!closed) {
                sb += "\n";
                continue;
            }
            if (multiline == Multiline::System) {
                set_system(sb);
                std::printf("Set system message.\n");
                sb.clear();
            }
            multiline = Multiline::None;
        } else if (starts_with(line, "\"\"\"")) {
            std::string rest   = line.substr(3);
            const bool  closed = rest.size() >= 3 && rest.compare(rest.size() - 3, 3, "\"\"\"") == 0;
            if (closed) rest.resize(rest.size() - 3);
            sb += rest;
            if (!closed) {
                sb += "\n";
                multiline  = Multiline::Prompt;
                ed.use_alt = true;
            }
        } else if (ed.pasting) {
            // Never true today (see SimpleEditor); kept so the branch order
            // matches run.go's if a paste-aware editor is wired in later.
            sb += line;
            sb += "\n";
            continue;
        } else if (starts_with(line, "/list")) {
            stand_in_list(); // cmds.go's cmdList also filters by a name prefix; ours doesn't
        } else if (starts_with(line, "/load")) {
            const auto args = fields(line);
            if (args.size() != 2) {
                std::printf("Usage:\n  /load <modelname>\n");
                continue;
            }
            RunOptions orig = o.copy();
            o.model         = args[1];
            o.messages.clear();
            o.loaded_messages.clear();
            std::printf("Loading model '%s'\n", o.model.c_str());
            auto [info, code] = show_model(o.model);
            if (code != 200) {
                std::printf("Couldn't find model '%s'\n", o.model.c_str());
                o = orig.copy();
                continue;
            }
            o.parent_model = j_str(j_sub(info, "details"), "parent_model");
            infer_thinking(info, o, think_set);
            o.multi_modal = has_cap(info, "vision") || has_cap(info, "audio");
            try {
                load_or_unload_model(o);
            } catch (const std::exception & e) {
                const std::string msg = e.what();
                if (msg.find("not found") != std::string::npos) {
                    std::printf("Couldn't find model '%s'\n", o.model.c_str());
                    o = orig.copy();
                    continue;
                }
                std::printf("error: %s\n", msg.c_str());
                continue;
            }
            continue;
        } else if (starts_with(line, "/save")) {
            const auto args = fields(line);
            if (args.size() != 2) {
                std::printf("Usage:\n  /save <modelname>\n");
                continue;
            }
            json msgs = json::array();
            for (const auto & m : o.loaded_messages) msgs.push_back(m);
            for (const auto & m : o.messages) msgs.push_back(m);
            json req = {{"model", args[1]}, {"from", first_of({o.parent_model, o.model})}};
            if (!o.system.empty()) req["system"] = o.system;
            if (!o.options.empty()) req["parameters"] = o.options;
            if (!msgs.empty()) req["messages"] = msgs;
            const std::string err = create_model(req);
            if (!err.empty()) {
                std::printf("error: %s\n", err.c_str());
                continue;
            }
            std::printf("Created new model '%s'\n", args[1].c_str());
            continue;
        } else if (starts_with(line, "/clear")) {
            o.messages.clear();
            if (!o.system.empty()) o.messages.push_back(Message{{"role", "system"}, {"content", o.system}});
            std::printf("Cleared session context\n");
            continue;
        } else if (starts_with(line, "/set")) {
            const auto args = fields(line);
            if (args.size() <= 1) {
                usage_set();
            } else {
                const std::string & sub = args[1];
                if (sub == "history") {
                    ed.history_enabled = true;
                } else if (sub == "nohistory") {
                    ed.history_enabled = false;
                } else if (sub == "wordwrap") {
                    o.word_wrap = true;
                    std::printf("Set 'wordwrap' mode.\n");
                } else if (sub == "nowordwrap") {
                    o.word_wrap = false;
                    std::printf("Set 'nowordwrap' mode.\n");
                } else if (sub == "verbose") {
                    o.verbose = true;
                    std::printf("Set 'verbose' mode.\n");
                } else if (sub == "quiet") {
                    o.verbose = false;
                    std::printf("Set 'quiet' mode.\n");
                } else if (sub == "think") {
                    const std::string level = args.size() > 2 ? args[2] : "";
                    if (!level.empty()) {
                        o.think = level;
                        std::printf("Set 'think' mode to '%s'.\n", level.c_str());
                    } else {
                        o.think = true;
                        std::printf("Set 'think' mode.\n");
                    }
                    think_set = true;
                    ensure_thinking_support(o.model);
                } else if (sub == "nothink") {
                    o.think   = false;
                    think_set = true;
                    ensure_thinking_support(o.model);
                    std::printf("Set 'nothink' mode.\n");
                } else if (sub == "format") {
                    if (args.size() < 3 || args[2] != "json") {
                        std::printf("Invalid or missing format. For 'json' mode use '/set format json'\n");
                    } else {
                        o.format = args[2];
                        std::printf("Set format to '%s' mode.\n", args[2].c_str());
                    }
                } else if (sub == "noformat") {
                    o.format.clear();
                    std::printf("Disabled format.\n");
                } else if (sub == "parameter") {
                    if (args.size() < 4) {
                        usage_parameters();
                        continue;
                    }
                    const std::vector<std::string> params(args.begin() + 3, args.end());
                    try {
                        const json v = format_param(args[2], params);
                        std::string joined;
                        for (size_t i = 0; i < params.size(); i++) {
                            if (i) joined += ", ";
                            joined += params[i];
                        }
                        std::printf("Set parameter '%s' to '%s'\n", args[2].c_str(), joined.c_str());
                        o.options[args[2]] = v;
                    } catch (const std::exception & e) {
                        std::printf("Couldn't set parameter: \"%s\"\n", e.what());
                        continue;
                    }
                } else if (sub == "system") {
                    if (args.size() < 3) {
                        usage_set();
                        continue;
                    }
                    std::string rest;
                    for (size_t i = 2; i < args.size(); i++) {
                        if (i > 2) rest += " ";
                        rest += args[i];
                    }
                    multiline = Multiline::System;
                    if (!starts_with(rest, "\"\"\"")) {
                        multiline = Multiline::None;
                    } else {
                        rest = rest.substr(3);
                        if (rest.size() >= 3 && rest.compare(rest.size() - 3, 3, "\"\"\"") == 0) {
                            rest.resize(rest.size() - 3);
                            multiline = Multiline::None;
                        }
                    }
                    sb += rest;
                    if (multiline != Multiline::None) {
                        ed.use_alt = true;
                        continue;
                    }
                    set_system(sb);
                    std::printf("Set system message.\n");
                    sb.clear();
                    continue;
                } else {
                    std::printf("Unknown command '/set %s'. Type /? for help\n", sub.c_str());
                }
            }
        } else if (starts_with(line, "/show")) {
            const auto args = fields(line);
            if (args.size() <= 1) {
                usage_show();
            } else {
                auto [info, code] = show_model(o.model);
                if (code != 200) {
                    std::printf("error: couldn't get model\n");
                    continue;
                }
                const std::string & sub = args[1];
                if (sub == "info") {
                    print_model_info(info);
                } else if (sub == "license") {
                    const std::string lic = j_str(info, "license");
                    if (lic.empty()) std::printf("No license was specified for this model.\n");
                    else std::printf("%s\n", lic.c_str());
                } else if (sub == "modelfile") {
                    std::printf("%s\n", j_str(info, "modelfile").c_str());
                } else if (sub == "parameters") {
                    std::printf("Model defined parameters:\n");
                    const std::string p = j_str(info, "parameters");
                    if (p.empty()) {
                        std::printf("  No additional parameters were specified for this model.\n");
                    } else {
                        std::istringstream iss(p);
                        std::string        ln;
                        while (std::getline(iss, ln)) std::printf("  %s\n", ln.c_str());
                    }
                    std::printf("\n");
                    if (!o.options.empty()) {
                        std::printf("User defined parameters:\n");
                        std::vector<std::string> keys;
                        for (auto it = o.options.begin(); it != o.options.end(); ++it) keys.push_back(it.key());
                        std::sort(keys.begin(), keys.end());
                        for (const auto & k : keys) {
                            std::printf("  %-30s %s\n", k.c_str(), o.options[k].dump().c_str());
                        }
                        std::printf("\n");
                    }
                } else if (sub == "system") {
                    if (!o.system.empty()) std::printf("%s\n\n", o.system.c_str());
                    else if (!j_str(info, "system").empty()) std::printf("%s\n\n", j_str(info, "system").c_str());
                    else std::printf("No system message was specified for this model.\n");
                } else if (sub == "template") {
                    const std::string t = j_str(info, "template");
                    if (!t.empty()) std::printf("%s\n", t.c_str());
                    else std::printf("No prompt template was specified for this model.\n");
                } else {
                    std::printf("Unknown command '/show %s'. Type /? for help\n", sub.c_str());
                }
            }
        } else if (starts_with(line, "/help") || starts_with(line, "/?")) {
            const auto args = fields(line);
            if (args.size() > 1) {
                if (args[1] == "set" || args[1] == "/set") usage_set();
                else if (args[1] == "show" || args[1] == "/show") usage_show();
                else if (args[1] == "shortcut" || args[1] == "shortcuts") usage_shortcuts();
            } else {
                usage_main(o.multi_modal);
            }
        } else if (starts_with(line, "/exit") || starts_with(line, "/bye")) {
            return;
        } else if (!line.empty() && line[0] == '/') {
            const auto args    = fields(line);
            bool       is_file = false;
            if (o.multi_modal && !args.empty()) {
                for (auto it = std::sregex_iterator(line.begin(), line.end(), file_re()); it != std::sregex_iterator();
                     ++it) {
                    if (starts_with(it->str(), args[0])) {
                        is_file = true;
                        break;
                    }
                }
            }
            if (!is_file) {
                std::printf("Unknown command '%s'. Type /? for help\n", args.empty() ? "" : args[0].c_str());
                continue;
            }
            sb += line;
        } else {
            sb += line;
        }

        if (!sb.empty() && multiline == Multiline::None) {
            Message nm = {{"role", "user"}, {"content", sb}};
            if (o.multi_modal) {
                const FileExtraction fe = extract_file_data(sb);
                if (!fe.ok) {
                    sb.clear();
                    continue;
                }
                nm["content"] = fe.text;
                if (!fe.images_b64.empty()) nm["images"] = fe.images_b64;
            }
            o.messages.push_back(nm);
            try {
                const Message assistant = chat_once(o);
                if (!assistant.is_null()) o.messages.push_back(assistant);
            } catch (const std::exception & e) {
                const std::string msg = e.what();
                if (msg.find("does not support thinking") != std::string::npos ||
                    msg.find("invalid think value") != std::string::npos) {
                    std::printf("error: %s\n", msg.c_str());
                } else {
                    std::fprintf(stderr, "error: %s\n", msg.c_str());
                }
                o.messages.pop_back();
                sb.clear();
                continue;
            }
            sb.clear();
        }
    }
}

} // namespace

// ================================================================== entry

int cmd_run(const RunArgs & args) {
    try {
        need_server();
        bool interactive = true;

        RunOptions opts;
        opts.model        = args.model;
        opts.word_wrap    = !args.nowordwrap;
        opts.show_connect = true;
        opts.format       = args.format;
        opts.hide_thinking = args.hidethinking;
        opts.verbose      = args.verbose;

        if (args.think_set) {
            if (args.think == "true") opts.think = true;
            else if (args.think == "false") opts.think = false;
            else opts.think = args.think;
        }
        if (args.ctx > 0) opts.options["num_ctx"] = args.ctx;
        if (args.temperature) opts.options["temperature"] = *args.temperature;
        if (!args.keepalive.empty()) opts.keep_alive = args.keepalive;

        std::string prompt = args.prompt;
        if (!is_console(stdin)) {
            std::ostringstream ss;
            ss << std::cin.rdbuf();
            std::string piped = ss.str();
            if (!piped.empty()) {
                while (!piped.empty() && (piped.back() == '\n' || piped.back() == '\r')) piped.pop_back();
                prompt          = piped + " " + prompt;
                const size_t a  = prompt.find_first_not_of(" \t\r\n");
                const size_t b  = prompt.find_last_not_of(" \t\r\n");
                prompt          = (a == std::string::npos) ? "" : prompt.substr(a, b - a + 1);
            }
            opts.show_connect = false;
            opts.word_wrap    = false;
            interactive       = false;
        }
        opts.prompt = prompt;
        if (!prompt.empty()) interactive = false;
        if (!is_console(stdout)) interactive = false;

        json info      = show_or_pull(opts);
        opts.parent_model = j_str(j_sub(info, "details"), "parent_model");
        infer_thinking(info, opts, args.think_set);
        opts.multi_modal = has_cap(info, "vision") || has_cap(info, "audio");

        if (has_cap(info, "embedding")) {
            if (opts.prompt.empty()) {
                die("Error: embedding models require input text. Usage: " + prog_name() + " run " + opts.model +
                    " \"your text here\"");
            }
            embed_once(opts, args.truncate, args.dimensions);
            return 0;
        }

        if (interactive) {
            try {
                load_or_unload_model(opts);
            } catch (const std::exception & e) {
                die("Error: " + std::string(e.what()));
            }
            for (const auto & m : opts.loaded_messages) {
                std::printf("%s: %s\n\n", j_str(m, "role").c_str(), j_str(m, "content").c_str());
            }
            generate_interactive(opts);
            return 0;
        }
        if (opts.multi_modal) {
            const FileExtraction fe = extract_file_data(opts.prompt);
            if (!fe.ok) return 1;
            opts.prompt = fe.text;
            opts.images = fe.images_b64;
        }
        generate_once(opts);
        return 0;
    } catch (const CliExit & e) {
        return e.code;
    }
}

} // namespace llmash
