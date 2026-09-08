// Standalone verification for cmd_doctor.cpp / cmd_models.cpp / help.cpp.

#define WIN32_LEAN_AND_MEAN
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#pragma comment(lib, "ws2_32.lib")

#include "cli_util.h"
#include "cmd_models.h"
#include "config.h"
#include "help.h"
#include "registry.h"

#include <nlohmann/json.hpp>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace llmash;
using namespace llmash::clidoc;

namespace {

int failures = 0;

void check(bool ok, const char * what) {
    std::printf("%s  %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) {
        failures++;
    }
}

// Same recipe test_main.cpp uses for the finished modules: a header with one
// architecture key and one tensor, just enough for read_gguf to accept it.
void write_gguf(const fs::path & p, const std::string & arch, const char * tensor) {
    std::ofstream out(p, std::ios::binary);
    const auto    u32 = [&](uint32_t v) { out.write(reinterpret_cast<const char *>(&v), 4); };
    const auto    u64 = [&](uint64_t v) { out.write(reinterpret_cast<const char *>(&v), 8); };
    const auto    str = [&](const std::string & s) {
        u64(s.size());
        out.write(s.data(), s.size());
    };
    out.write("GGUF", 4);
    u32(3);
    u64(1); // tensors
    u64(1); // kv pairs
    str("general.architecture");
    u32(8);
    str(arch);
    str(tensor);
    u32(1);
    u64(16);
    u32(0);
    u64(0);
}

// A port nothing is listening on right now: bind ephemeral, read back what
// the OS gave us, close it.
int free_port_for_test() {
    WSADATA wsa;
    WSAStartup(MAKEWORD(2, 2), &wsa);
    SOCKET s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port        = 0;
    bind(s, reinterpret_cast<sockaddr *>(&addr), sizeof(addr));
    int len = sizeof(addr);
    getsockname(s, reinterpret_cast<sockaddr *>(&addr), &len);
    const int port = ntohs(addr.sin_port);
    closesocket(s);
    WSACleanup();
    return port;
}

std::string self_path() {
    wchar_t buf[MAX_PATH];
    const DWORD n = GetModuleFileNameW(nullptr, buf, MAX_PATH);
    return fs::path(std::wstring(buf, n)).string();
}

} // namespace

int main(int argc, char ** argv) {
    // A child-process mode used below to exercise cmd_models' die()/exit(1)
    // path in a process that is not this test's own.
    if (argc >= 3 && std::string(argv[1]) == "--models-set") {
        cmd_models({"set", argv[2]});
        return 0;
    }

    // Safety first: redirect every HTTP call this test (and anything it
    // spawns) makes away from whatever might really be listening on 11434.
    const int dead_port = free_port_for_test();
    _putenv_s("OLLAMA_HOST", ("127.0.0.1:" + std::to_string(dead_port)).c_str());

    const fs::path dir = fs::temp_directory_path() / "llmash_cpp_doctor_test";
    fs::remove_all(dir);
    fs::create_directories(dir / "store" / "manifests");
    fs::create_directories(dir / "loose");
    fs::create_directories(dir / "neither");

    // ------------------------------------------------- folder classification

    Config   cfg0;
    cfg0.root        = dir.string();
    cfg0.models_root = (dir / "store").string();
    Registry reg0(cfg0);
    check(reg0.is_ollama_store((dir / "store").string()), "a folder with manifests/ inside is an Ollama store");
    check(!reg0.is_ollama_store((dir / "loose").string()), "a folder without manifests/ is not a store");
    check(!reg0.is_ollama_store((dir / "neither").string()), "an empty folder is not a store either");

    write_gguf(dir / "loose" / "Real-Model-Q8_0.gguf", "qwen3", "blk.0.attn_norm.weight");
    write_gguf(dir / "loose" / "Real-Model-Q8_0.mtp.gguf", "qwen3", "blk.0.nextn.weight"); // a sidecar, excluded
    {
        std::ofstream bad(dir / "loose" / "corrupt.gguf", std::ios::binary);
        bad << "not a real gguf header at all";
    }

    check(walk_gguf((dir / "loose").string()).size() == 3, "walk_gguf finds every .gguf, sidecar and corrupt included");

    const bool loose_is_gguf_folder = dir_exists((dir / "loose").string()) &&
                                     !reg0.is_ollama_store((dir / "loose").string()) &&
                                     !walk_gguf((dir / "loose").string()).empty();
    const bool neither_is_gguf_folder = dir_exists((dir / "neither").string()) &&
                                       !reg0.is_ollama_store((dir / "neither").string()) &&
                                       !walk_gguf((dir / "neither").string()).empty();
    check(loose_is_gguf_folder, "a folder of GGUFs with no manifests/ classifies as a loose GGUF folder");
    check(!neither_is_gguf_folder, "an empty folder classifies as neither a store nor a loose folder");

    const auto model_counts = count_library((dir / "loose").string());
    check(model_counts.first == 1, "count_library counts the one real model as clean");
    check(model_counts.second == 1, "count_library counts the corrupt file as skipped, not clean or silently dropped");

    // ------------------------------------------- doctor's disk-space check

    const double free_gb = free_disk_gb(dir.root_path().string());
    check(free_gb > 0, "free_disk_gb reads a real positive number for a real drive");
    check(free_ram_gb() > 0, "free_ram_gb reads a real positive number");

    // ------------------------------------- doctor's model-count building block

    Config   cfg1;
    cfg1.root        = dir.string();
    cfg1.models_root = (dir / "loose").string();
    Registry reg1(cfg1);
    check(reg1.all().size() == 1, "a fake registry over the loose folder reads exactly the one real model");

    // -------------------------------------------------------- formatting

    check(human_bytes(500) == "500 B", "human_bytes: plain bytes");
    check(human_bytes(1500) == "1.5 KB", "human_bytes: fractional KB");
    check(human_bytes(2000000000ULL) == "2 GB", "human_bytes: a whole GB");
    check(human_time_iso("", "never") == "never", "human_time_iso: an empty string falls back to zero_value");
    check(human_time_iso("not-a-timestamp", "never") == "never", "human_time_iso: unparsable falls back too");
    {
        char now_iso[64];
        const time_t now = std::time(nullptr) - 3600; // an hour ago
        std::tm      tmv;
        gmtime_s(&tmv, &now);
        std::strftime(now_iso, sizeof(now_iso), "%Y-%m-%dT%H:%M:%SZ", &tmv);
        const std::string got = human_time_iso(now_iso, "never");
        check(got.find("hour") != std::string::npos || got.find("minute") != std::string::npos,
             "human_time_iso reads an hour-old RFC3339 timestamp as elapsed time");
    }

    // -------------------------------------------------------- local.json

    {
        const fs::path root = dir / "install";
        fs::create_directories(root);
        nlohmann::json j;
        j["models_root"] = "C:/models";
        j["extra_roots"] = nlohmann::json::array({"C:/old-store"});
        std::string err;
        check(write_local_json(root.string(), j, err), "write_local_json succeeds");
        const nlohmann::json back = read_local_json(root.string());
        check(back.value("models_root", std::string()) == "C:/models", "read_local_json round-trips models_root");
        check(back.contains("extra_roots") && back["extra_roots"].size() == 1 && back["extra_roots"][0] == "C:/old-store",
             "read_local_json round-trips extra_roots");
    }
    check(read_local_json((dir / "does-not-exist").string()).is_object(), "read_local_json on a missing file is an empty object, not an error");

    // --------------------------------------- a timed subprocess, and its RAII

    {
        DWORD hc_before = 0, hc_after = 0;
        GetProcessHandleCount(GetCurrentProcess(), &hc_before);
        bool first_ok = false;
        for (int i = 0; i < 15; i++) {
            const auto result = run_with_timeout("cmd", {"/c", "echo hello-from-subprocess"}, 5000);
            if (i == 0) {
                first_ok = result.second && result.first.find("hello-from-subprocess") != std::string::npos;
            }
        }
        check(first_ok, "run_with_timeout captures a real subprocess's real stdout");
        GetProcessHandleCount(GetCurrentProcess(), &hc_after);
        check(hc_after < hc_before + 20,
             "15 runs of a real subprocess leak no meaningful number of process handles (RAII actually releases them)");
    }
    {
        const auto result = run_with_timeout("this-command-does-not-exist-anywhere-xyz", {}, 1000);
        check(!result.second && result.first.empty(), "run_with_timeout fails cleanly for a missing executable; nothing is spawned");
    }
    {
        // A real timeout: `cmd /c` with a longer sleep than the timeout given.
        const auto started = std::chrono::steady_clock::now();
        const auto result  = run_with_timeout("cmd", {"/c", "ping -n 6 127.0.0.1 >nul"}, 500);
        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - started).count();
        check(!result.second, "run_with_timeout reports failure when the process outlives its timeout");
        check(elapsed < 4000, "run_with_timeout actually terminates the process instead of waiting it out");
    }

    // -------------------------------------------------------- which() / PATH

    check(which("this-command-does-not-exist-xyz").empty(), "which() returns empty for a command that isn't on PATH");
    check(!which("cmd").empty(), "which() finds a real command that is on PATH");
    check(dir_exists(dir.string()), "dir_exists is true for a real directory");
    check(!dir_exists((dir / "nope-nope-nope").string()), "dir_exists is false for a directory that doesn't exist");
    check(!file_exists(dir.string()), "file_exists is false for something that is actually a directory");

    // -------------------------------------------------- cmd_models, for real

    {
        // No args: prints the current dirs. Just needs to not crash or exit.
        cmd_models({});
        check(true, "cmd_models() with no arguments returns instead of exiting");
    }
    {
        const fs::path exe_dir_path = fs::path(self_path()).parent_path();
        fs::remove(exe_dir_path / "local.json");
        cmd_models({"set", (dir / "loose").string()});
        const nlohmann::json local = read_local_json(exe_dir_path.string());
        check(same_dir(local.value("models_root", std::string()), (dir / "loose").string()),
             "`models set <loose GGUF folder>` writes that folder into local.json");
        fs::remove(exe_dir_path / "local.json");
    }
    {
        // The die()/exit(1) path, exercised in a child process so a failure
        // there cannot take this test process down with it.
        const auto result = run_with_timeout(self_path(), {"--models-set", (dir / "neither").string()}, 5000);
        check(!result.second, "`models set <folder with no GGUF, not a store>` exits non-zero");
        check(result.first.find("has no GGUF") != std::string::npos, "...and says why, the way the Go original does");
    }

    // ------------------------------------------------------------- help.cpp

    check(help_text().find("doctor       Check this machine over") != std::string::npos,
         "help_text lists doctor among the available commands");
    check(help_text().find("models       Show where models are read from") != std::string::npos,
         "help_text lists models among the available commands");
    check(command_help("doctor") != nullptr, "command_help finds the doctor topic");
    check(command_help("nonexistent-topic-xyz") == nullptr, "command_help returns nullptr for an unknown topic");
    check(command_help("models")->find("llmash models set DIR") != std::string::npos,
         "the models help text documents the set subcommand");
    {
        const std::string t = prog_text(help_text(), "llmash");
        // The top-level help lists bare command names ("  serve   Start Ollama"),
        // so the "\n  ollama " rewrite only bites in a subcommand's usage block.
        const std::string * sub = command_help("serve");
        check(sub && prog_text(*sub, "llmash").find("\n  llmash serve") != std::string::npos,
              "prog_text re-addresses a subcommand usage line to the typed name");
        check(t.find("Use \"llmash [command]") != std::string::npos, "prog_text re-addresses the footer line too");
        check(prog_text(help_text(), "ollama") == help_text(), "prog_text is a no-op when the typed name really is ollama");
    }

    std::printf("\n%s\n", failures == 0 ? "all passed" : "FAILURES");
    return failures == 0 ? 0 : 1;
}
