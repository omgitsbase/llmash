// Standalone verification for tray.cpp. Not wired into CMakeLists.txt (a
// protected file for this job); compiled and run directly against
// tray.cpp/tray_process.cpp/config.cpp, mirroring the Go package's own
// tray_test.go rather than exercising a live server or the real Startup
// folder.

#include "config.h"
#include "tray.h"

#include <nlohmann/json.hpp>

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <string>

namespace fs = std::filesystem;
using json   = nlohmann::json;
using namespace llmash;

namespace {

int failures = 0;

void check(bool ok, const char * what) {
    std::printf("%s  %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) {
        failures++;
    }
}

bool has_powershell() { return std::system("where powershell >NUL 2>&1") == 0; }

// Writes `script` into a PowerShell here-string so no quoting has to survive
// two process layers, then asks the engine to parse (never run) it.
bool script_parses(const std::string & script) {
    const fs::path ps1 = fs::temp_directory_path() / "llmash_tray_test_parse.ps1";
    std::ofstream  out(ps1, std::ios::binary | std::ios::trunc);
    if (!out) {
        return false;
    }
    out << "$ErrorActionPreference = 'Stop'\n$null = [ScriptBlock]::Create(@'\n" << script << "\n'@)\n";
    out.close();
    const std::string cmd = "powershell -NoProfile -File \"" + ps1.string() + "\" >NUL 2>&1";
    const int          rc  = std::system(cmd.c_str());
    std::error_code    ec;
    fs::remove(ps1, ec);
    return rc == 0;
}

std::string iso_utc(double epoch_seconds) {
    const std::time_t t = static_cast<std::time_t>(epoch_seconds);
    std::tm            tm{};
    gmtime_s(&tm, &t);
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%04d-%02d-%02dT%02d:%02d:%02d+00:00", tm.tm_year + 1900, tm.tm_mon + 1,
                 tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);
    return buf;
}

double now_epoch() { return static_cast<double>(std::time(nullptr)); }

// A leading integer from a string like "12m left"; -1 if there isn't one.
int leading_int(const std::string & s) {
    if (s.empty() || !std::isdigit(static_cast<unsigned char>(s[0]))) {
        return -1;
    }
    return std::atoi(s.c_str());
}

} // namespace

int main() {
    // ---- ps_quote / stop_script: the quoting that kills real processes,
    // so a stray quote here would kill nothing or kill somebody else's
    // server. Mirrors tray_test.go's TestStopScript exactly.
    {
        const std::string s = stop_script(R"(C:\Program Files\llmash)");
        check(s.find(R"('*C:\Program Files\llmash\llmashw.exe*')") != std::string::npos,
             "stop_script: this install's llmashw.exe only");
        check(s.find(R"('*C:\Program Files\llmash\llmash.exe*')") != std::string::npos,
             "stop_script: this install's llmash.exe too");
        check(s.find("'* serve*'") != std::string::npos, "stop_script: the server, not a tray or a client");
        check(s.find("Name='llama-server.exe'") != std::string::npos, "stop_script: the engines it started");
        check(s.find("ParentProcessId") != std::string::npos, "stop_script: and only its own");

        size_t quotes = 0;
        for (const char c : s) {
            quotes += c == '"';
        }
        check(quotes % 2 == 0, "stop_script: balanced double quotes");

        // the engines have to go before their parent, or they are orphaned
        check(s.find("llama-server.exe") < s.rfind("$s | ForEach-Object"),
             "stop_script: engines are stopped before the server that owns them");
    }

    // A single quote in the install path is what would break the quoting;
    // check PowerShell itself accepts what comes out (TestStopScriptParses).
    if (has_powershell()) {
        for (const std::string & root : {R"(C:\Program Files\llmash)", "C:\\Users\\someone\\It's Mine\\llmash",
                                         R"(B:\llmash)"}) {
            check(script_parses(stop_script(root)), ("stop_script parses for root: " + root).c_str());
        }
    } else {
        std::printf("skip  no powershell here\n");
    }

    // ---- process_exists_script: same quoting risk, same filter shape.
    {
        const std::string s = process_exists_script(R"(C:\Users\It's Mine\llmash)");
        check(s.find("Name='llmashw.exe' OR Name='llmash.exe'") != std::string::npos,
             "process_exists_script: looks for both exe names");
        check(s.find("'* serve*'") != std::string::npos, "process_exists_script: only a running server");
        check(s.find("''s Mine") != std::string::npos, "process_exists_script: an embedded quote is doubled");
        size_t quotes = 0;
        for (const char c : s) {
            quotes += c == '"';
        }
        check(quotes % 2 == 0, "process_exists_script: balanced double quotes");
        if (has_powershell()) {
            check(script_parses(s), "process_exists_script parses");
        }
    }

    // ---- enable_startup_script: the Startup-folder shortcut, not a
    // registry Run key (ported from the actual Go implementation).
    {
        const std::string s = enable_startup_script(R"(C:\llmash)", R"(C:\Startup\llmash.lnk)", "");
        check(s.find("New-Object -ComObject WScript.Shell") != std::string::npos,
             "enable_startup_script: uses the Startup-folder shortcut COM object");
        check(s.find(R"('C:\llmash\llmashw.exe')") != std::string::npos,
             "enable_startup_script: targets llmashw.exe, not the console build");
        check(s.find("$s.Arguments = 'tray'") != std::string::npos, "enable_startup_script: launches into tray mode");
        check(s.find("$s.WindowStyle = 7") != std::string::npos, "enable_startup_script: minimized window style");
        check(s.find("IconLocation") == std::string::npos, "enable_startup_script: no icon line when none is given");
        if (has_powershell()) {
            check(script_parses(s), "enable_startup_script parses");
        }
        const std::string with_icon = enable_startup_script(R"(C:\llmash)", R"(C:\Startup\llmash.lnk)",
                                                            R"(C:\llmash\llmash.ico)");
        check(with_icon.find(R"($s.IconLocation = 'C:\llmash\llmash.ico')") != std::string::npos,
             "enable_startup_script: sets the icon when one is given");
    }

    // ---- expires_text: the countdown shown on each model's submenu.
    {
        check(expires_text(json::object()).empty(), "expires_text: no expires_at at all");
        check(expires_text({{"expires_at", "9999-12-31T23:59:59Z"}}) == "pinned", "expires_text: pinned sentinel");
        check(expires_text({{"expires_at", "not-a-timestamp"}}).empty(), "expires_text: unparsable is empty, not a crash");
        check(expires_text({{"expires_at", iso_utc(now_epoch() - 5)}}) == "unloading",
             "expires_text: already past is 'unloading'");

        const std::string secs = expires_text({{"expires_at", iso_utc(now_epoch() + 30)}});
        check(secs.find("s left") != std::string::npos, "expires_text: seconds bucket reads 'Ns left'");
        check(leading_int(secs) >= 15 && leading_int(secs) <= 31, "expires_text: seconds bucket value is plausible");

        const std::string mins = expires_text({{"expires_at", iso_utc(now_epoch() + 200)}});
        check(mins.find("m left") != std::string::npos, "expires_text: minutes bucket reads 'Nm left'");

        const std::string hrs = expires_text({{"expires_at", iso_utc(now_epoch() + 10000)}});
        check(hrs.find("h left") != std::string::npos, "expires_text: hours bucket reads 'N.Nh left'");
    }

    // ---- size_text
    {
        check(size_text(json::object()).empty(), "size_text: neither field set is empty");
        check(size_text({{"size_vram", 3.5 * (1ull << 30)}}) == "3.5 GB", "size_text: size_vram in GB");
        check(size_text({{"size", 2.0 * (1ull << 30)}}) == "2.0 GB", "size_text: falls back to size");
        check(size_text({{"size_vram", 0}, {"size", 1.0 * (1ull << 30)}}) == "1.0 GB",
             "size_text: a zero size_vram still falls back");
    }

    // ---- the real Win32 sequence: add a notification icon, then remove
    // it. Run twice in a row: the second RegisterClassExW only succeeds if
    // the first call's UnregisterClassW (and DestroyWindow, and the
    // Shell_NotifyIconW NIM_DELETE before it) actually released everything.
    check(self_test_tray_icon(), "Shell_NotifyIconW: add then remove succeeds");
    check(self_test_tray_icon(), "Shell_NotifyIconW: second cycle proves the first left nothing behind");

    std::printf("\n%s\n", failures == 0 ? "all passed" : "FAILURES");
    return failures == 0 ? 0 : 1;
}
