#include "launch.h"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <conio.h>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <io.h>
#include <windows.h>

namespace fs = std::filesystem;
using json   = nlohmann::json;

namespace llmash {

namespace {

constexpr const char * kDim   = "\x1b[2m";
constexpr const char * kBold  = "\x1b[1m";
constexpr const char * kReset = "\x1b[0m";

bool file_exists(const std::string & p) {
    std::error_code ec;
    return fs::exists(p, ec) && !fs::is_directory(p, ec);
}

std::vector<std::string> split(const std::string & s, char sep) {
    std::vector<std::string> out;
    size_t start = 0;
    while (start <= s.size()) {
        const size_t pos = s.find(sep, start);
        const size_t end = pos == std::string::npos ? s.size() : pos;
        if (end > start) {
            out.push_back(s.substr(start, end - start));
        }
        if (pos == std::string::npos) {
            break;
        }
        start = pos + 1;
    }
    return out;
}

std::wstring utf8_to_wide(const std::string & s) {
    if (s.empty()) {
        return L"";
    }
    const int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), nullptr, 0);
    std::wstring out(n, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), out.data(), n);
    return out;
}

std::string wide_to_utf8(const std::wstring & s) {
    if (s.empty()) {
        return "";
    }
    const int n = WideCharToMultiByte(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), nullptr, 0, nullptr, nullptr);
    std::string out(n, '\0');
    WideCharToMultiByte(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), out.data(), n, nullptr, nullptr);
    return out;
}

// The documented CreateProcess/CommandLineToArgvW-compatible quoting rule.
std::wstring quote_arg(const std::wstring & arg) {
    if (!arg.empty() && arg.find_first_of(L" \t\n\v\"") == std::wstring::npos) {
        return arg;
    }
    std::wstring out = L"\"";
    for (auto it = arg.begin();; ++it) {
        size_t backslashes = 0;
        while (it != arg.end() && *it == L'\\') {
            ++it;
            ++backslashes;
        }
        if (it == arg.end()) {
            out.append(backslashes * 2, L'\\');
            break;
        }
        if (*it == L'"') {
            out.append(backslashes * 2 + 1, L'\\');
            out.push_back(*it);
        } else {
            out.append(backslashes, L'\\');
            out.push_back(*it);
        }
    }
    out.push_back(L'"');
    return out;
}

std::string win_error_message(DWORD code) {
    LPWSTR buf = nullptr;
    const DWORD n = FormatMessageW(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
                                        FORMAT_MESSAGE_IGNORE_INSERTS,
                                    nullptr, code, 0, reinterpret_cast<LPWSTR>(&buf), 0, nullptr);
    std::string msg = n > 0 ? wide_to_utf8(std::wstring(buf, n)) : "";
    if (buf != nullptr) {
        LocalFree(buf);
    }
    while (!msg.empty() && (msg.back() == '\n' || msg.back() == '\r')) {
        msg.pop_back();
    }
    return msg.empty() ? ("error " + std::to_string(code)) : msg;
}

// The current process's environment, one "NAME=value" entry per element.
std::vector<std::string> current_environment() {
    struct Block {
        LPWCH p;
        ~Block() {
            if (p) {
                FreeEnvironmentStringsW(p);
            }
        }
    } block{GetEnvironmentStringsW()};
    std::vector<std::string> out;
    if (!block.p) {
        return out;
    }
    for (const wchar_t * s = block.p; *s != L'\0';) {
        const size_t len = wcslen(s);
        if (s[0] != L'=') {
            out.push_back(wide_to_utf8(std::wstring(s, len)));
        }
        s += len + 1;
    }
    return out;
}

// Launches exe with args, giving the child our real console (stdin/stdout/
// stderr are inherited, not piped) so an interactive CLI works exactly as if
// the user typed it themselves; this is runInherit in the Go original.
[[noreturn]] void run_inherit(const std::string & exe, const std::vector<std::string> & args,
                               const std::vector<std::string> * env) {
    std::wstring cmdline = quote_arg(utf8_to_wide(exe));
    for (const auto & a : args) {
        cmdline += L' ';
        cmdline += quote_arg(utf8_to_wide(a));
    }

    std::wstring env_block;
    DWORD        extra_flags = 0;
    if (env != nullptr) {
        for (const auto & e : *env) {
            env_block += utf8_to_wide(e);
            env_block.push_back(L'\0');
        }
        env_block.push_back(L'\0');
        extra_flags = CREATE_UNICODE_ENVIRONMENT;
    }

    STARTUPINFOW        si{};
    si.cb = sizeof(si);
    PROCESS_INFORMATION pi{};
    std::vector<wchar_t> mutable_cmdline(cmdline.begin(), cmdline.end());
    mutable_cmdline.push_back(L'\0');

    const BOOL ok = CreateProcessW(nullptr, mutable_cmdline.data(), nullptr, nullptr, TRUE, extra_flags,
                                    env != nullptr ? env_block.data() : nullptr, nullptr, &si, &pi);
    if (!ok) {
        std::fprintf(stderr, "Error: %s\n", win_error_message(GetLastError()).c_str());
        std::exit(1);
    }
    CloseHandle(pi.hThread);
    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD code = 0;
    GetExitCodeProcess(pi.hProcess, &code);
    CloseHandle(pi.hProcess);
    std::exit(static_cast<int>(code));
}

// exec.LookPath's Windows rule: PATH combined with PATHEXT, current directory
// not searched (a plain integration name is never a relative path here).
std::string which(const std::string & exe) {
    if (exe.find('\\') != std::string::npos || exe.find('/') != std::string::npos) {
        return file_exists(exe) ? exe : "";
    }
    const bool has_ext = exe.rfind('.') != std::string::npos && exe.rfind('.') > exe.find_last_of("\\/");
    std::vector<std::string> exts;
    if (has_ext) {
        exts.push_back("");
    } else {
        for (auto & e : split(env_str("PATHEXT", ".COM;.EXE;.BAT;.CMD"), ';')) {
            exts.push_back(e);
        }
    }
    for (const auto & dir : split(env_str("PATH"), ';')) {
        for (const auto & ext : exts) {
            const std::string candidate = (fs::path(dir) / (exe + ext)).string();
            if (file_exists(candidate)) {
                return candidate;
            }
        }
    }
    return "";
}

std::vector<std::string> installed_models() {
    std::vector<std::string> out;
    httplib::Client cli(ollama_host());
    cli.set_connection_timeout(4, 0);
    cli.set_read_timeout(4, 0);
    const auto res = cli.Get("/api/tags");
    if (!res || res->status != 200) {
        return out;
    }
    const json d = json::parse(res->body, nullptr, false);
    if (d.is_discarded() || !d.is_object()) {
        return out;
    }
    for (const auto & m : d.value("models", json::array())) {
        if (m.is_object()) {
            const std::string n = m.value("name", std::string());
            if (!n.empty()) {
                out.push_back(n);
            }
        }
    }
    return out;
}

bool is_console() {
    return _isatty(_fileno(stdout)) != 0 && _isatty(_fileno(stdin)) != 0;
}

// One raw key, matching the Go menu's arrow/enter/escape vocabulary: 'U'/'D'
// for up/down, '\r' enter, 0x1b escape, plain characters pass through as-is.
int read_key() {
    int ch = _getch();
    if (ch == 0xE0 || ch == 0x00) {
        switch (_getch()) {
            case 'H':
                return 'U';
            case 'P':
                return 'D';
            default:
                return 0;
        }
    }
    return ch;
}

struct Row {
    std::string key, title, subtitle;
};

// Redraws in place: moves the cursor up over the previous frame and clears
// each line, mirroring the Go menu's "\x1b[F\x1b[2K" per line.
void draw(int & drawn, const std::vector<std::string> & lines) {
    for (int i = 0; i < drawn; i++) {
        std::fputs("\x1b[F\x1b[2K", stdout);
    }
    for (const auto & l : lines) {
        std::fputs(l.c_str(), stdout);
        std::fputc('\n', stdout);
    }
    std::fflush(stdout);
    drawn = static_cast<int>(lines.size());
}

std::string title_of(const std::string & key) {
    const auto it = integrations().find(key);
    return it != integrations().end() ? it->second.title : key;
}

std::string model_of(const nlohmann::json & prof, const std::string & key, const std::vector<std::string> & installed) {
    const std::string m = profile_model(prof, key);
    if (!m.empty()) {
        return m;
    }
    if (key == "claude" && !installed.empty()) {
        return installed.front();
    }
    return "";
}

void pick_model(const Config & cfg, nlohmann::json & prof, const std::string & key,
                 std::vector<std::string> & installed) {
    if (installed.empty()) {
        installed = installed_models();
    }
    int drawn = 0;
    if (installed.empty()) {
        draw(drawn, {std::string(kBold) + "model for " + title_of(key) + kReset, "",
                     "  couldn't reach llmash at " + ollama_host() + ". Start it, then press the arrow again", "",
                     std::string(kDim) + "press any key to go back" + kReset});
        _getch();
        return;
    }
    int sel = 0;
    const std::string cur = model_of(prof, key, installed);
    for (size_t i = 0; i < installed.size(); i++) {
        if (installed[i] == cur) {
            sel = static_cast<int>(i);
        }
    }
    for (;;) {
        std::vector<std::string> lines{std::string(kBold) + "model for " + title_of(key) + kReset, ""};
        for (size_t i = 0; i < installed.size(); i++) {
            const bool hl = static_cast<int>(i) == sel;
            lines.push_back(std::string(hl ? "\xE2\x96\xB8 " : "  ") + (hl ? kBold : "") + installed[i] + kReset);
        }
        lines.push_back("");
        lines.push_back(std::string(kDim) + "up/down navigate, enter select, esc back" + kReset);
        draw(drawn, lines);
        switch (const int k = read_key()) {
            case 'U':
                sel = (sel - 1 + static_cast<int>(installed.size())) % static_cast<int>(installed.size());
                break;
            case 'D':
                sel = (sel + 1) % static_cast<int>(installed.size());
                break;
            case '\r':
                if (!prof.contains(key) || !prof[key].is_object()) {
                    prof[key] = json::object();
                }
                prof[key]["model"] = installed[sel];
                save_profiles(cfg, prof);
                return;
            case 0x1b:
            case 'q':
                return;
            default:
                (void) k;
                break;
        }
    }
}

} // namespace

const std::map<std::string, Integration> & integrations() {
    static const std::map<std::string, Integration> m = {
        {"claude", {"Claude Code", "claude"}},
        {"chatgpt", {"ChatGPT", "chatgpt"}},
        {"hermes", {"Hermes Agent", "hermes"}},
        {"openclaw", {"OpenClaw", "openclaw"}},
        {"opencode", {"OpenCode", "opencode"}},
        {"codex", {"Codex", "codex"}},
        {"hermes-desktop", {"Hermes Desktop", "hermes-desktop"}},
        {"copilot", {"Copilot CLI", "copilot"}},
        {"omp", {"OMP", "omp"}},
        {"droid", {"Droid", "droid"}},
        {"kimi", {"Kimi Code CLI", "kimi"}},
        {"pi", {"Pi", "pi"}},
        {"pool", {"Pool", "pool"}},
        {"cline", {"Cline", "cline"}},
        {"qwen", {"Qwen Code", "qwen"}},
        {"vscode", {"VS Code", "code"}},
    };
    return m;
}

const std::map<std::string, std::string> & integration_aliases() {
    static const std::map<std::string, std::string> m = {
        {"codex-app", "chatgpt"}, {"codex-desktop", "chatgpt"}, {"codex-gui", "chatgpt"},
        {"clawdbot", "openclaw"}, {"moltbot", "openclaw"},
        {"copilot-cli", "copilot"}, {"code", "vscode"},
    };
    return m;
}

std::string resolve_integration(const std::string & name) {
    if (integrations().count(name) != 0) {
        return name;
    }
    const auto it = integration_aliases().find(name);
    return it != integration_aliases().end() ? it->second : "";
}

std::string profile_file(const Config & cfg) {
    return (fs::path(cfg.root) / "integrations.json").string();
}

nlohmann::json load_profiles(const Config & cfg) {
    std::ifstream in(profile_file(cfg), std::ios::binary);
    if (!in) {
        return json::object();
    }
    const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    const json        j = json::parse(text, nullptr, false);
    return j.is_discarded() || !j.is_object() ? json::object() : j;
}

void save_profiles(const Config & cfg, const nlohmann::json & profiles) {
    std::ofstream out(profile_file(cfg), std::ios::binary | std::ios::trunc);
    if (out) {
        out << profiles.dump(2);
    }
}

std::string profile_model(const nlohmann::json & profiles, const std::string & key) {
    const auto it = profiles.find(key);
    if (it == profiles.end() || !it->is_object()) {
        return "";
    }
    return it->value("model", std::string());
}

LaunchArgs parse_launch_args(const std::vector<std::string> & args, std::string & error) {
    error = "";
    std::vector<std::string> rest = args, extra;
    for (size_t i = 0; i < args.size(); i++) {
        if (args[i] == "--") {
            rest.assign(args.begin(), args.begin() + static_cast<long>(i));
            extra.assign(args.begin() + static_cast<long>(i) + 1, args.end());
            break;
        }
    }

    LaunchArgs out;
    for (size_t i = 0; i < rest.size(); i++) {
        const std::string & t = rest[i];
        if (t == "--config") {
            out.config = true;
        } else if (t == "--restore") {
            out.restore = true;
        } else if (t == "-y" || t == "--yes") {
            // accepted, no effect: llmash never prompts before a launch
        } else if (t == "--model") {
            if (i + 1 >= rest.size()) {
                error = "Error: flag needs an argument: --model";
                return {};
            }
            out.model = rest[++i];
        } else if (t.rfind("--model=", 0) == 0) {
            out.model = t.substr(std::string("--model=").size());
        } else if (!t.empty() && t[0] == '-') {
            error = "Error: unknown flag: '" + t + "'";
            return {};
        } else if (out.name.empty()) {
            out.name = t;
        } else {
            extra.push_back(t);
        }
    }
    out.extra = extra;

    if (out.name.empty() && (!extra.empty() || !out.model.empty() || out.config || out.restore)) {
        error = "Error: flags and extra arguments require an integration name";
        return {};
    }
    return out;
}

std::string ollama_host() {
    std::string h = env_str("OLLAMA_HOST");
    if (h.empty()) {
        h = "http://127.0.0.1:11434";
    }
    if (h.rfind("http", 0) != 0) {
        h = "http://" + h;
    }
    while (!h.empty() && h.back() == '/') {
        h.pop_back();
    }
    return h;
}

std::vector<std::string> generic_launch_env(const std::string & host, const std::string & model) {
    std::vector<std::string> env{"OPENAI_BASE_URL=" + host + "/v1", "OPENAI_API_KEY=ollama", "OLLAMA_HOST=" + host};
    if (!model.empty()) {
        env.push_back("OPENAI_MODEL=" + model);
        env.push_back("OLLAMA_MODEL=" + model);
    }
    return env;
}

void launch_claude(const Config & cfg, const std::string & model, const std::vector<std::string> & extra) {
    const std::string ps1 = (fs::path(cfg.root) / "Claude-on-Local.ps1").string();
    if (!file_exists(ps1)) {
        std::fprintf(stderr, "Error: launcher missing: %s\n", ps1.c_str());
        std::exit(1);
    }
    std::vector<std::string> args{"-NoProfile", "-ExecutionPolicy", "Bypass", "-File", ps1};
    if (!model.empty()) {
        args.push_back("-Model");
        args.push_back(model);
    }
    for (const auto & e : extra) {
        args.push_back(e);
    }
    run_inherit("powershell", args, nullptr);
}

void launch_generic(const std::string & key, const std::string & model, const std::vector<std::string> & extra) {
    const auto & it = integrations().at(key);
    const std::string path = which(it.exe);
    if (path.empty()) {
        std::fprintf(stderr, "Error: %s is not installed (no '%s' on PATH)\n", it.title.c_str(), it.exe.c_str());
        std::exit(1);
    }
    const std::string host = ollama_host();
    const std::string with = model.empty() ? "" : (" with " + model);
    std::printf("%slaunching %s against %s/v1%s%s\n", kDim, it.title.c_str(), host.c_str(), with.c_str(), kReset);

    std::vector<std::string> env = current_environment();
    for (auto & e : generic_launch_env(host, model)) {
        env.push_back(e);
    }
    run_inherit(path, extra, &env);
}

void launch_menu(const Config & cfg) {
    if (!is_console()) {
        std::printf("usage: llmash launch [name] [--model M] [--config] [--restore]\n");
        for (const auto & kv : integrations()) {
            std::printf("  %-16s %s\n", kv.first.c_str(), kv.second.title.c_str());
        }
        std::exit(0);
    }
    json prof = load_profiles(cfg);
    const std::vector<Row> rows = {
        {"claude", "Launch Claude Code", "Anthropic's coding tool with subagents"},
        {"opencode", "Launch OpenCode", "Anomaly's open-source coding agent"},
        {"hermes", "Launch Hermes Agent", "Self-improving AI agent built by Nous Research"},
        {"openclaw", "Launch OpenClaw", "Personal AI with 100+ skills"},
    };
    std::vector<std::string> installed;
    int                      sel   = 0;
    int                      drawn = 0;
    std::fputs("\x1b[?25l", stdout);
    for (;;) {
        std::vector<std::string> lines{std::string(kBold) + "llmash" + kReset, "", "  Chat, Code, & Work",
                                        std::string(kDim) +
                                            "    Chat with models, code, search the web, and delegate real work" +
                                            kReset,
                                        ""};
        for (size_t i = 0; i < rows.size(); i++) {
            const bool        hl = static_cast<int>(i) == sel;
            const std::string exe_installed = which(integrations().at(rows[i].key).exe).empty() ? " (install)" : "";
            std::string       suffix        = exe_installed;
            if (rows[i].key == "claude") {
                const std::string m = model_of(prof, rows[i].key, installed);
                suffix = m.empty() ? "" : (" (" + m + ")");
            }
            lines.push_back(std::string(hl ? "\xE2\x96\xB8 " : "  ") + (hl ? kBold : "") + rows[i].title + suffix +
                             kReset);
            lines.push_back(std::string(kDim) + "    " + rows[i].subtitle + kReset);
            lines.push_back("");
        }
        lines.push_back(std::string(kDim) + "up/down navigate, enter launch, right-arrow configure, esc quit" +
                         kReset);
        draw(drawn, lines);

        int ch = _getch();
        if (ch == 0xE0 || ch == 0x00) {
            switch (_getch()) {
                case 'H':
                    sel = (sel - 1 + static_cast<int>(rows.size())) % static_cast<int>(rows.size());
                    break;
                case 'P':
                    sel = (sel + 1) % static_cast<int>(rows.size());
                    break;
                case 'M':
                    pick_model(cfg, prof, rows[sel].key, installed);
                    break;
                default:
                    break;
            }
            continue;
        }
        if (ch == '\r') {
            std::fputs("\x1b[?25h\n", stdout);
            const std::string model = model_of(prof, rows[sel].key, installed);
            if (rows[sel].key == "claude") {
                launch_claude(cfg, model, {});
            }
            launch_generic(rows[sel].key, model, {});
        }
        if (ch == 0x1b || ch == 'q' || ch == 0x03) {
            std::fputc('\n', stdout);
            std::fputs("\x1b[?25h", stdout);
            std::exit(0);
        }
    }
}

void cmd_launch(const Config & cfg, const std::vector<std::string> & args) {
    std::string error;
    LaunchArgs  la = parse_launch_args(args, error);
    if (!error.empty()) {
        std::fprintf(stderr, "%s\n", error.c_str());
        std::exit(1);
    }
    if (la.name.empty()) {
        launch_menu(cfg);
    }

    const std::string key = resolve_integration(la.name);
    if (key.empty()) {
        std::fprintf(stderr, "Error: unknown integration \"%s\"\n", la.name.c_str());
        std::exit(1);
    }
    json prof = load_profiles(cfg);
    if (la.restore) {
        prof.erase(key);
        save_profiles(cfg, prof);
        std::printf("restored %s to its default profile\n", integrations().at(key).title.c_str());
        std::exit(0);
    }
    if (!la.model.empty()) {
        if (!prof.contains(key) || !prof[key].is_object()) {
            prof[key] = json::object();
        }
        prof[key]["model"] = la.model;
        save_profiles(cfg, prof);
    }
    std::string model = la.model.empty() ? profile_model(prof, key) : la.model;
    if (la.config) {
        const std::string shown = model.empty() ? "(default)" : model;
        const std::string hint  = model.empty() ? "  (set one with --model)" : "";
        std::printf("%s: model = %s%s\n", integrations().at(key).title.c_str(), shown.c_str(), hint.c_str());
        std::exit(0);
    }
    if (key == "claude") {
        launch_claude(cfg, model, la.extra);
    }
    launch_generic(key, model, la.extra);
}

} // namespace llmash
