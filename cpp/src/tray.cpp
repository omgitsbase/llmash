#include "cli_win.h"
#include "tray.h"
#include "tray_internal.h"

#include <httplib.h>

#include <windows.h>
#include <shellapi.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <functional>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;
using json   = nlohmann::json;

namespace llmash {

namespace {

// Go's tray used WM_USER (its FFI had no WM_APP constant handy); kept as-is
// since the window class is private to this process either way.
constexpr UINT     WM_TRAY_CALLBACK = WM_USER + 1;
constexpr UINT     WM_TRAY_REOPEN   = WM_USER + 2;
constexpr UINT_PTR ICON_UID         = 1;
const wchar_t *     kClassName      = L"llmashTray";

std::wstring to_wide(const std::string & s) {
    if (s.empty()) {
        return std::wstring();
    }
    const int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), nullptr, 0);
    std::wstring w(static_cast<size_t>(n), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), w.data(), n);
    return w;
}

std::string to_utf8(const std::wstring & s) {
    if (s.empty()) {
        return std::string();
    }
    const int n = WideCharToMultiByte(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), nullptr, 0, nullptr, nullptr);
    std::string out(static_cast<size_t>(n), '\0');
    WideCharToMultiByte(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), out.data(), n, nullptr, nullptr);
    return out;
}

std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

std::string trimmed(std::string s) {
    const auto not_space = [](unsigned char c) { return !std::isspace(c); };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
    s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
    return s;
}

std::string join(const std::vector<std::string> & parts, const std::string & sep) {
    std::string out;
    for (size_t i = 0; i < parts.size(); i++) {
        if (i) {
            out += sep;
        }
        out += parts[i];
    }
    return out;
}

bool file_exists(const std::string & p) {
    std::error_code ec;
    return fs::is_regular_file(fs::path(p), ec);
}

template <size_t N>
void copy_wide(WCHAR (&dst)[N], const std::wstring & src) {
    const size_t n = (std::min)(src.size(), N - 1);
    if (n) {
        std::wmemcpy(dst, src.data(), n);
    }
    dst[n] = L'\0';
}

std::string j_str(const json & m, const char * k) {
    const auto it = m.find(k);
    if (it == m.end() || it->is_null()) {
        return "";
    }
    return it->is_string() ? it->get<std::string>() : it->dump();
}

double j_num(const json & m, const char * k) {
    const auto it = m.find(k);
    return (it != m.end() && it->is_number()) ? it->get<double>() : 0.0;
}

double now_epoch() {
    return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
}

// RFC3339(Nano): "2026-09-08T18:30:00[.frac](Z|+HH:MM|-HH:MM)". Every
// timestamp the server actually emits uses a numeric +00:00 offset; Z is
// handled too since that is what the pinned sentinel would parse as.
bool parse_offset_time(const std::string & s, double & epoch_out) {
    if (s.size() < 19) {
        return false;
    }
    int y, mo, d, h, mi, se;
    if (std::sscanf(s.c_str(), "%4d-%2d-%2dT%2d:%2d:%2d", &y, &mo, &d, &h, &mi, &se) != 6) {
        return false;
    }
    size_t pos  = 19;
    double frac = 0.0;
    if (pos < s.size() && s[pos] == '.') {
        const size_t start = ++pos;
        while (pos < s.size() && std::isdigit(static_cast<unsigned char>(s[pos]))) {
            pos++;
        }
        if (pos > start) {
            frac = std::stod("0." + s.substr(start, pos - start));
        }
    }
    int offset_sec = 0;
    if (pos < s.size() && (s[pos] == '+' || s[pos] == '-')) {
        int oh = 0, om = 0;
        if (std::sscanf(s.c_str() + pos + 1, "%2d:%2d", &oh, &om) == 2) {
            offset_sec = oh * 3600 + om * 60;
            if (s[pos] == '-') {
                offset_sec = -offset_sec;
            }
        }
    }
    std::tm tm{};
    tm.tm_year = y - 1900;
    tm.tm_mon  = mo - 1;
    tm.tm_mday = d;
    tm.tm_hour = h;
    tm.tm_min  = mi;
    tm.tm_sec  = se;
    const std::time_t utc = _mkgmtime(&tm);
    if (utc == static_cast<std::time_t>(-1)) {
        return false;
    }
    epoch_out = static_cast<double>(utc - offset_sec) + frac;
    return true;
}

// -------------------------------------------------------- talking to it

class WinsockScope {
public:
    WinsockScope() : ok_(WSAStartup(MAKEWORD(2, 2), &data_) == 0) {}
    ~WinsockScope() {
        if (ok_) {
            WSACleanup();
        }
    }
    WinsockScope(const WinsockScope &)             = delete;
    WinsockScope & operator=(const WinsockScope &) = delete;

private:
    WSADATA data_{};
    bool    ok_ = false;
};

std::string api_host() {
    std::string h = env_str("OLLAMA_HOST", "http://127.0.0.1:11434");
    if (h.rfind("http", 0) != 0) {
        h = "http://" + h;
    }
    while (!h.empty() && h.back() == '/') {
        h.pop_back();
    }
    return h;
}

struct ApiReply {
    bool          reached = false; // a response came back at all, any status
    int           status  = 0;
    std::string   body;
    httplib::Error error  = httplib::Error::Success;
};

ApiReply api_call(const std::string & method, const std::string & path, const std::string & json_body,
                  std::chrono::milliseconds timeout) {
    ApiReply        reply;
    httplib::Client cli(api_host());
    cli.set_connection_timeout(timeout);
    cli.set_read_timeout(timeout);
    cli.set_write_timeout(timeout);

    const httplib::Result res = (method == "GET") ? cli.Get(path) : cli.Post(path, json_body, "application/json");
    reply.error                = res.error();
    if (!res) {
        return reply;
    }
    reply.reached = true;
    reply.status  = res->status;
    reply.body    = res->body;
    return reply;
}

bool server_up() { return api_call("GET", "/", "", std::chrono::milliseconds(1500)).reached; }

std::vector<json> loaded_models() {
    std::vector<json> out;
    const ApiReply     r = api_call("GET", "/api/ps", "", std::chrono::milliseconds(2000));
    if (!r.reached) {
        return out;
    }
    const json d = json::parse(r.body, nullptr, false);
    if (d.is_discarded() || !d.is_object()) {
        return out;
    }
    const auto it = d.find("models");
    if (it == d.end() || !it->is_array()) {
        return out;
    }
    for (const auto & m : *it) {
        if (m.is_object()) {
            out.push_back(m);
        }
    }
    return out;
}

// Returns the server's complaint, or "" when the model really was retimed.
std::string set_keep_alive(const std::string & model, int seconds) {
    const json     body = {{"model", model}, {"keep_alive", seconds}};
    const ApiReply r    = api_call("POST", "/api/keep_alive", body.dump(), std::chrono::milliseconds(20000));
    if (!r.reached) {
        return "llmash: " + httplib::to_string(r.error);
    }
    if (r.status >= 400) {
        const json d = json::parse(r.body, nullptr, false);
        if (!d.is_discarded() && d.is_object()) {
            const std::string e = j_str(d, "error");
            if (!e.empty()) {
                return e;
            }
        }
        return trimmed(r.body);
    }
    return "";
}

// ------------------------------------------------------- server control

bool server_process_exists(const Config & cfg) {
    const PowerShellRun r = run_hidden_powershell(process_exists_script(cfg.root));
    return std::atoi(trimmed(r.stdout_text).c_str()) > 0;
}

class ProcessHandles {
public:
    explicit ProcessHandles(const PROCESS_INFORMATION & pi) : process_(pi.hProcess), thread_(pi.hThread) {}
    ~ProcessHandles() {
        if (thread_) {
            CloseHandle(thread_);
        }
        if (process_) {
            CloseHandle(process_);
        }
    }
    ProcessHandles(const ProcessHandles &)             = delete;
    ProcessHandles & operator=(const ProcessHandles &) = delete;

private:
    HANDLE process_;
    HANDLE thread_;
};

std::wstring last_error_message() {
    const DWORD err = GetLastError();
    LPWSTR      buf = nullptr;
    const DWORD len = FormatMessageW(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
                                     FORMAT_MESSAGE_IGNORE_INSERTS,
                                     nullptr, err, 0, reinterpret_cast<LPWSTR>(&buf), 0, nullptr);
    std::wstring msg = (len && buf) ? std::wstring(buf, len) : std::wstring();
    if (buf) {
        LocalFree(buf);
    }
    return msg;
}

// Starts `exe arg` hidden, detached from us: closing our handles to it
// (below, via ProcessHandles) does not stop it running. `werr` gets the raw
// FormatMessage text on failure, the same text Windows gives for an
// antivirus quarantine (ERROR_VIRUS_INFECTED).
bool spawn_detached(const std::wstring & exe, const std::wstring & arg, const std::wstring & cwd, DWORD extra_flags,
                    std::wstring * werr) {
    std::wstring     cmdline = L"\"" + exe + L"\" " + arg;
    STARTUPINFOW     si{};
    si.cb          = sizeof(si);
    si.dwFlags     = STARTF_USESHOWWINDOW;
    si.wShowWindow = SW_HIDE;
    PROCESS_INFORMATION pi{};
    const BOOL ok = CreateProcessW(nullptr, cmdline.data(), nullptr, nullptr, FALSE, CREATE_NO_WINDOW | extra_flags,
                                   nullptr, cwd.empty() ? nullptr : cwd.c_str(), &si, &pi);
    if (!ok) {
        if (werr) {
            *werr = last_error_message();
        }
        return false;
    }
    ProcessHandles guard(pi);
    return true;
}

// After the 20s grace period an existing-but-unresponsive process is not
// waited on further: a fresh spawn is tried anyway, matching the original.
bool start_server_process(const Config & cfg) {
    using clock = std::chrono::steady_clock;
    if (server_up()) {
        return true;
    }
    if (server_process_exists(cfg)) {
        const auto t0 = clock::now();
        while (clock::now() - t0 < std::chrono::seconds(20)) {
            if (server_up()) {
                return true;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
    }
    const std::wstring exe = to_wide((fs::path(cfg.root) / "llmashw.exe").string());
    if (!spawn_detached(exe, L"serve", to_wide(cfg.root), 0, nullptr)) {
        return false;
    }
    const auto t0 = clock::now();
    while (clock::now() - t0 < std::chrono::seconds(30)) {
        if (server_up()) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(400));
    }
    return false;
}

// The engines go first, and this waits for the port: killing the server
// alone leaves any llama-server.exe it started holding their VRAM, and
// returning before it is really gone lets the next start race it.
void stop_server_process(const Config & cfg) {
    run_hidden_powershell(stop_script(cfg.root));
    using clock   = std::chrono::steady_clock;
    const auto t0 = clock::now();
    while (clock::now() - t0 < std::chrono::seconds(15)) {
        if (!server_up() && !server_process_exists(cfg)) {
            return;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
}

// --------------------------------------------------------- start at login

std::string tray_state_path(const Config & cfg) { return (fs::path(cfg.root) / "tray.json").string(); }

json read_tray_state(const Config & cfg) {
    std::ifstream in(tray_state_path(cfg), std::ios::binary);
    if (!in) {
        return json::object();
    }
    const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    const json         j = json::parse(text, nullptr, false);
    return (j.is_discarded() || !j.is_object()) ? json::object() : j;
}

void save_tray_state(const Config & cfg, const std::string & key, const json & value) {
    json d  = read_tray_state(cfg);
    d[key]  = value;
    std::ofstream out(tray_state_path(cfg), std::ios::binary | std::ios::trunc);
    if (out) {
        out << d.dump();
    }
}

bool startup_enabled() { return file_exists(startup_shortcut_path()); }

void enable_startup(const Config & cfg) {
    const std::string icon   = (fs::path(cfg.root) / "llmash.ico").string();
    const std::string script = enable_startup_script(cfg.root, startup_shortcut_path(), file_exists(icon) ? icon : "");
    run_hidden_powershell(script);
    save_tray_state(cfg, "startup", true);
}

void toggle_startup(const Config & cfg) {
    if (startup_enabled()) {
        std::error_code ec;
        fs::remove(startup_shortcut_path(), ec);
        save_tray_state(cfg, "startup", false);
    } else {
        enable_startup(cfg);
    }
}

// A first run turns "start at login" on; an explicit earlier "off" sticks.
void default_startup(const Config & cfg) {
    if (startup_enabled()) {
        return;
    }
    const json       state = read_tray_state(cfg);
    const auto       it    = state.find("startup");
    if (it != state.end() && it->is_boolean() && !it->get<bool>()) {
        return;
    }
    enable_startup(cfg);
}

// ---------------------------------------------------------- the window

struct TrayAction {
    std::function<void()> fn;
    bool                   wait = false; // finish the work first, so the reopened menu shows the result
    bool                   stop = false; // no reopen
};

class TrayApp {
public:
    explicit TrayApp(Config cfg) : cfg_(std::move(cfg)) {}
    ~TrayApp() { teardown(); }

    TrayApp(const TrayApp &)             = delete;
    TrayApp & operator=(const TrayApp &) = delete;

    bool create();
    int  run();

private:
    Config           cfg_;
    HWND             hwnd_             = nullptr;
    NOTIFYICONDATAW  nid_{};
    HICON            hicon_            = nullptr;
    bool             icon_added_       = false;
    bool             class_registered_ = false;

    std::mutex              mu_; // guards closing_
    std::condition_variable cv_;
    std::mutex              server_lock_; // one server start/stop at a time
    bool                    closing_   = false;
    bool                    menu_open_ = false;
    POINT                   anchor_{};
    std::thread             watch_thread_;

    std::unordered_map<UINT_PTR, TrayAction> actions_;
    UINT_PTR                                 next_id_ = 0;

    void teardown();
    void show_menu(POINT pt);
    void add_item(HMENU menu, UINT flags, const std::wstring & text, const TrayAction * act);
    void balloon(const std::wstring & title, const std::wstring & text);
    void kick();
    void restart();
    void quit();
    void watch();
    void retime(const std::string & name, int secs);

    friend LRESULT CALLBACK wnd_proc(HWND, UINT, WPARAM, LPARAM);
};

TrayApp * g_app = nullptr;

LRESULT CALLBACK wnd_proc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    switch (msg) {
    case WM_TRAY_CALLBACK:
        if (g_app && (lp == WM_LBUTTONUP || lp == WM_RBUTTONUP)) {
            POINT pt;
            GetCursorPos(&pt);
            g_app->show_menu(pt);
        }
        return 0;
    case WM_TRAY_REOPEN:
        if (g_app) {
            g_app->show_menu(g_app->anchor_);
        }
        return 0;
    case WM_DESTROY:
        if (g_app && g_app->icon_added_) {
            Shell_NotifyIconW(NIM_DELETE, &g_app->nid_);
            g_app->icon_added_ = false;
        }
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProcW(hwnd, msg, wp, lp);
}

bool TrayApp::create() {
    const HINSTANCE hinst = GetModuleHandleW(nullptr);
    WNDCLASSEXW      wc{};
    wc.cbSize        = sizeof(wc);
    wc.lpfnWndProc   = wnd_proc;
    wc.hInstance     = hinst;
    wc.lpszClassName = kClassName;
    if (!RegisterClassExW(&wc)) {
        return false;
    }
    class_registered_ = true;

    hwnd_ = CreateWindowExW(0, kClassName, L"llmash", 0, 0, 0, 0, 0, nullptr, nullptr, hinst, nullptr);
    if (!hwnd_) {
        return false;
    }
    g_app = this;

    const std::string icon_path = (fs::path(cfg_.root) / "llmash.ico").string();
    if (file_exists(icon_path)) {
        hicon_ = static_cast<HICON>(
            LoadImageW(nullptr, to_wide(icon_path).c_str(), IMAGE_ICON, 0, 0, LR_LOADFROMFILE | LR_DEFAULTSIZE));
    }

    nid_.cbSize           = sizeof(nid_);
    nid_.hWnd             = hwnd_;
    nid_.uID              = ICON_UID;
    nid_.uFlags           = NIF_MESSAGE | NIF_ICON | NIF_TIP;
    nid_.uCallbackMessage = WM_TRAY_CALLBACK;
    nid_.hIcon            = hicon_;
    copy_wide(nid_.szTip, L"llmash");
    icon_added_ = Shell_NotifyIconW(NIM_ADD, &nid_) != FALSE;
    return icon_added_;
}

void TrayApp::teardown() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        closing_ = true;
    }
    cv_.notify_all();
    if (watch_thread_.joinable()) {
        watch_thread_.join();
    }
    if (icon_added_) {
        Shell_NotifyIconW(NIM_DELETE, &nid_);
        icon_added_ = false;
    }
    if (hicon_) {
        DestroyIcon(hicon_);
        hicon_ = nullptr;
    }
    if (hwnd_) {
        DestroyWindow(hwnd_);
        hwnd_ = nullptr;
    }
    if (class_registered_) {
        UnregisterClassW(kClassName, GetModuleHandleW(nullptr));
        class_registered_ = false;
    }
    if (g_app == this) {
        g_app = nullptr;
    }
}

void TrayApp::balloon(const std::wstring & title, const std::wstring & text) {
    NOTIFYICONDATAW nid = nid_;
    nid.uFlags          = NIF_INFO;
    nid.dwInfoFlags     = NIIF_WARNING;
    copy_wide(nid.szInfo, text);
    copy_wide(nid.szInfoTitle, title);
    Shell_NotifyIconW(NIM_MODIFY, &nid);
}

void TrayApp::kick() {
    std::thread([this] {
        std::unique_lock<std::mutex> lk(server_lock_, std::try_to_lock);
        if (!lk.owns_lock()) {
            return;
        }
        if (!server_up() && !server_process_exists(cfg_)) {
            start_server_process(cfg_);
        }
    }).detach();
}

void TrayApp::restart() {
    std::thread([this] {
        std::lock_guard<std::mutex> lk(server_lock_);
        stop_server_process(cfg_);
        start_server_process(cfg_);
    }).detach();
}

void TrayApp::quit() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        closing_ = true;
    }
    cv_.notify_all();
    {
        std::lock_guard<std::mutex> lk(server_lock_);
        stop_server_process(cfg_);
    }
    PostMessageW(hwnd_, WM_DESTROY, 0, 0);
}

void TrayApp::watch() {
    std::unique_lock<std::mutex> lk(mu_);
    while (!cv_.wait_for(lk, std::chrono::seconds(5), [this] { return closing_; })) {
        lk.unlock();
        {
            std::unique_lock<std::mutex> sl(server_lock_, std::try_to_lock);
            if (sl.owns_lock() && !server_up() && !server_process_exists(cfg_)) {
                start_server_process(cfg_);
            }
        }
        lk.lock();
    }
}

void TrayApp::retime(const std::string & name, int secs) {
    const std::string e = set_keep_alive(name, secs);
    if (!e.empty()) {
        balloon(L"llmash", to_wide(name + ": " + e));
    }
}

void TrayApp::add_item(HMENU menu, UINT flags, const std::wstring & text, const TrayAction * act) {
    UINT_PTR id = 0;
    if (act) {
        id            = ++next_id_;
        actions_[id]  = *act;
    } else if (!(flags & MF_POPUP)) {
        flags |= MF_GRAYED;
    }
    AppendMenuW(menu, flags, id, text.c_str());
}

// Rebuilt from scratch every open so the loaded models and their timers are
// live; TrackPopupMenu is itself a nested message loop, so a click while one
// is already open must be ignored rather than nesting a second.
void TrayApp::show_menu(POINT pt) {
    if (menu_open_) {
        return;
    }
    menu_open_ = true;
    anchor_    = pt;

    const bool         up = server_up();
    std::vector<json>  models;
    if (up) {
        models = loaded_models();
    } else {
        kick();
    }
    actions_.clear();
    next_id_ = 0;

    const HMENU menu = CreatePopupMenu();
    add_item(menu, MF_STRING, L"llmash · " + std::wstring(up ? L"running" : L"starting…"), nullptr);
    AppendMenuW(menu, MF_SEPARATOR, 0, nullptr);

    if (!models.empty()) {
        for (const auto & m : models) {
            const std::string name   = j_str(m, "name");
            const bool         pinned = j_str(m, "expires_at").rfind("9999", 0) == 0;
            const HMENU         subm  = CreatePopupMenu();

            std::vector<std::string> parts;
            const std::string         sz = size_text(m);
            const std::string         ex = expires_text(m);
            if (!sz.empty()) {
                parts.push_back(sz);
            }
            if (!ex.empty()) {
                parts.push_back(ex);
            }
            const std::string head = parts.empty() ? "loaded" : join(parts, " · ");
            add_item(subm, MF_STRING, to_wide(head), nullptr);
            AppendMenuW(subm, MF_SEPARATOR, 0, nullptr);

            const TrayAction unload{[this, name] { retime(name, 0); }, true, false};
            add_item(subm, MF_STRING, L"Unload now", &unload);

            struct Choice {
                const wchar_t * label;
                int             secs;
            };
            for (const Choice & c : {Choice{L"Unload in 5 minutes", 300}, Choice{L"Unload in 15 minutes", 900},
                                     Choice{L"Unload in 1 hour", 3600}, Choice{L"Keep loaded", -1}}) {
                UINT flags = MF_STRING;
                if (c.secs == -1 && pinned) {
                    flags |= MF_CHECKED;
                }
                const int        secs = c.secs;
                const TrayAction act{[this, name, secs] { retime(name, secs); }, true, false};
                add_item(subm, flags, c.label, &act);
            }

            std::string label = name;
            if (!ex.empty()) {
                label += " · " + ex;
            }
            AppendMenuW(menu, MF_POPUP | MF_STRING, reinterpret_cast<UINT_PTR>(subm), to_wide(label).c_str());
        }
        const std::vector<json> all = models;
        const TrayAction        unload_all{
            [this, all] {
                for (const auto & m : all) {
                    retime(j_str(m, "name"), 0);
                }
            },
            true, false};
        add_item(menu, MF_STRING, L"Unload all (" + std::to_wstring(models.size()) + L")", &unload_all);
    } else if (up) {
        add_item(menu, MF_STRING, L"No models loaded", nullptr);
    } else {
        add_item(menu, MF_STRING, L"Server is starting", nullptr);
    }

    AppendMenuW(menu, MF_SEPARATOR, 0, nullptr);
    const TrayAction restart_act{[this] { restart(); }, false, false};
    add_item(menu, MF_STRING, L"Restart server", &restart_act);

    UINT su_flags = MF_STRING;
    if (startup_enabled()) {
        su_flags |= MF_CHECKED;
    }
    const TrayAction toggle_act{[this] { toggle_startup(cfg_); }, true, false};
    add_item(menu, su_flags, L"Start at login", &toggle_act);

    AppendMenuW(menu, MF_SEPARATOR, 0, nullptr);
    const TrayAction quit_act{[this] { quit(); }, false, true};
    add_item(menu, MF_STRING, L"Close llmash", &quit_act);

    SetForegroundWindow(hwnd_);
    const UINT_PTR cmd = static_cast<UINT_PTR>(
        TrackPopupMenu(menu, TPM_RETURNCMD | TPM_RIGHTBUTTON, pt.x, pt.y, 0, hwnd_, nullptr));
    // the well-known workaround so the menu reliably closes on an outside click
    PostMessageW(hwnd_, WM_NULL, 0, 0);
    DestroyMenu(menu);
    menu_open_ = false;

    const auto it = actions_.find(cmd);
    if (it == actions_.end()) {
        return;
    }
    const TrayAction act  = it->second;
    const HWND        hwnd = hwnd_;
    const auto        reopen = [hwnd] { PostMessageW(hwnd, WM_TRAY_REOPEN, 0, 0); };
    if (act.stop) {
        std::thread(act.fn).detach();
    } else if (act.wait) {
        std::thread([fn = act.fn, reopen] {
            fn();
            reopen();
        }).detach();
    } else {
        std::thread(act.fn).detach();
        reopen();
    }
}

int TrayApp::run() {
    if (!create()) {
        return 1;
    }
    std::thread([this] {
        std::lock_guard<std::mutex> lk(server_lock_);
        start_server_process(cfg_);
    }).detach();
    std::thread([this] { default_startup(cfg_); }).detach();
    watch_thread_ = std::thread([this] { watch(); });

    MSG msg;
    BOOL r;
    while ((r = GetMessageW(&msg, nullptr, 0, 0)) > 0) {
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }
    return r < 0 ? 1 : 0;
}

} // namespace

// ---------------------------------------------------------- public API


std::string stop_script(const std::string & root) {
    const std::string r    = ps_quote(root);
    const std::string mine = "($_.CommandLine -like '*" + r + "\\llmashw.exe*' -or $_.CommandLine -like '*" + r +
                             "\\llmash.exe*') -and $_.CommandLine -like '* serve*'";
    return "$s = @(Get-CimInstance Win32_Process -Filter \"Name='llmashw.exe' OR Name='llmash.exe'\" "
          "| Where-Object { " +
          mine +
          " }); "
          "$ids = @($s | ForEach-Object { $_.ProcessId }); "
          "if ($ids.Count) { Get-CimInstance Win32_Process -Filter \"Name='llama-server.exe'\" "
          "| Where-Object { $ids -contains $_.ParentProcessId } "
          "| ForEach-Object { Stop-Process -Id $_.ProcessId -Force } }; "
          "$s | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }";
}

std::string process_exists_script(const std::string & root) {
    const std::string r = ps_quote(root);
    return "@(Get-CimInstance Win32_Process -Filter \"Name='llmashw.exe' OR Name='llmash.exe'\" "
          "| Where-Object { ($_.CommandLine -like '*" +
          r + "\\llmashw.exe*' -or $_.CommandLine -like '*" + r +
          "\\llmash.exe*') "
          "-and $_.CommandLine -like '* serve*' }).Count";
}

std::string startup_shortcut_path() {
    return (fs::path(env_str("APPDATA")) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup" /
           "llmash.lnk")
        .string();
}

std::string enable_startup_script(const std::string & root, const std::string & shortcut_path,
                                  const std::string & icon_path) {
    std::string script = "$w = New-Object -ComObject WScript.Shell; $s = $w.CreateShortcut('" +
                         ps_quote(shortcut_path) + "'); $s.TargetPath = '" +
                         ps_quote((fs::path(root) / "llmashw.exe").string()) +
                         "'; $s.Arguments = 'tray'; $s.WorkingDirectory = '" + ps_quote(root) +
                         "'; $s.WindowStyle = 7; $s.Description = 'llmash'; ";
    if (!icon_path.empty()) {
        script += "$s.IconLocation = '" + ps_quote(icon_path) + "'; ";
    }
    script += "$s.Save()";
    return script;
}

std::string expires_text(const nlohmann::json & model) {
    const std::string raw = j_str(model, "expires_at");
    if (raw.empty()) {
        return "";
    }
    if (raw.rfind("9999", 0) == 0) {
        return "pinned";
    }
    double at = 0.0;
    if (!parse_offset_time(raw, at)) {
        return "";
    }
    const double left = at - now_epoch();
    char         buf[32];
    if (left <= 0) {
        return "unloading";
    }
    if (left < 90) {
        std::snprintf(buf, sizeof(buf), "%ds left", static_cast<int>(left));
        return buf;
    }
    if (left < 5400) {
        std::snprintf(buf, sizeof(buf), "%dm left", static_cast<int>(left / 60));
        return buf;
    }
    std::snprintf(buf, sizeof(buf), "%.1fh left", left / 3600.0);
    return buf;
}

std::string size_text(const nlohmann::json & model) {
    double b = j_num(model, "size_vram");
    if (b == 0) {
        b = j_num(model, "size");
    }
    if (b == 0) {
        return "";
    }
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.1f GB", b / static_cast<double>(1ull << 30));
    return buf;
}

bool cmd_tray(const Config & cfg, std::string & err) {
    const std::string exe = (fs::path(cfg.root) / "llmashw.exe").string();
    if (!file_exists(exe)) {
        err = "llmashw.exe was not found in " + cfg.root + "; reinstall llmash.";
        return false;
    }
    std::wstring werr;
    if (!spawn_detached(to_wide(exe), L"tray", to_wide(cfg.root), DETACHED_PROCESS, &werr)) {
        const std::string low = lower(to_utf8(werr));
        // ERROR_VIRUS_INFECTED (225): llmash is unsigned, and a freshly
        // built program with no reputation behind it is a common false
        // positive (see GitHub issue from user nigelp).
        if (low.find("virus") != std::string::npos || low.find("potentially unwanted") != std::string::npos) {
            err = "your antivirus blocked " + exe +
                 " from running.\n"
                 "The command line still works; only the tray is stopped.\n"
                 "Allow it in your antivirus (in Windows Security it is under Protection history), then run "
                 "`llmash tray` again.\n"
                 "Reporting it helps everyone else: https://www.microsoft.com/en-us/wdsi/filesubmission";
        } else {
            err = "could not start the tray: " + to_utf8(werr);
        }
        return false;
    }
    return true;
}

int tray_main(const Config & cfg) {
    const WinsockScope winsock;
    TrayApp             app(cfg);
    return app.run();
}

bool self_test_tray_icon() {
    const HINSTANCE hinst = GetModuleHandleW(nullptr);
    const wchar_t * cls   = L"llmashTraySelfTest";
    WNDCLASSEXW      wc{};
    wc.cbSize        = sizeof(wc);
    wc.lpfnWndProc   = DefWindowProcW;
    wc.hInstance     = hinst;
    wc.lpszClassName = cls;
    if (!RegisterClassExW(&wc)) {
        return false;
    }
    const HWND hwnd = CreateWindowExW(0, cls, L"llmash-self-test", 0, 0, 0, 0, 0, nullptr, nullptr, hinst, nullptr);
    bool       ok   = hwnd != nullptr;
    if (ok) {
        NOTIFYICONDATAW nid{};
        nid.cbSize           = sizeof(nid);
        nid.hWnd             = hwnd;
        nid.uID              = 1;
        nid.uFlags           = NIF_MESSAGE | NIF_ICON | NIF_TIP;
        nid.uCallbackMessage = WM_TRAY_CALLBACK;
        nid.hIcon            = LoadIconW(nullptr, IDI_APPLICATION); // shared system icon, never destroyed
        copy_wide(nid.szTip, L"llmash self test");
        ok = Shell_NotifyIconW(NIM_ADD, &nid) != FALSE;
        if (ok) {
            ok = Shell_NotifyIconW(NIM_DELETE, &nid) != FALSE;
        }
        DestroyWindow(hwnd);
    }
    UnregisterClassW(cls, hinst);
    return ok;
}

} // namespace llmash
