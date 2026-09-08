#pragma once

// Small pieces shared by cmd_doctor.cpp and cmd_models.cpp: process launches
// with a timeout, the local server's HTTP API, local.json, and a couple of
// formatters.

#include <nlohmann/json.hpp>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace llmash {
namespace clidoc {

// ------------------------------------------------------------- formatting

// table.go's humanBytes: 1000-based, one decimal below 10 units.
std::string human_bytes(uint64_t b);

// table.go's humanTimeISO: an RFC3339(-nano) instant as "N units ago/from
// now", or zero_value for an empty or unparsable string.
std::string human_time_iso(const std::string & iso, const std::string & zero_value);

// -------------------------------------------------------- filesystem / PATH

bool        file_exists(const std::string & path);
bool        dir_exists(const std::string & path);

// win.go's which: exec.LookPath, PATHEXT included. "" when not found.
std::string which(const std::string & exe);

// GetDiskFreeSpaceExW's free bytes on the volume holding path, in GB (0 on
// any failure, matching doctor.go's freeDiskGB).
double free_disk_gb(const std::string & path);

double free_ram_gb();

// -------------------------------------------------------------- the server

// http.go's host: OLLAMA_HOST, defaulted to 127.0.0.1:11434 and normalized.
// LLMASH_PORT does not affect this, matching the Go original.
std::string server_host();

bool server_up();

// http.go's callJSON: only a transport failure or a non-JSON, non-empty body
// counts as an error.
bool call_json(const std::string & method, const std::string & path, const nlohmann::json * body,
               int timeout_s, nlohmann::json & out, std::string & err);

// -------------------------------------------------------------- local.json

// config.go's readLocal, as a generic object so a command that only changes
// one or two keys does not clobber the rest.
nlohmann::json read_local_json(const std::string & root);

// config.go's writeLocal: UTF-8, no BOM, indented, trailing newline.
bool write_local_json(const std::string & root, const nlohmann::json & j, std::string & err);

// ------------------------------------------------------- a timed subprocess

// doctor.go's run(): the command's combined output, trimmed, and whether it
// exited zero inside timeout_ms.
std::pair<std::string, bool> run_with_timeout(const std::string & exe, const std::vector<std::string> & args,
                                              int timeout_ms);

// ------------------------------------------------------- GitHub's releases

std::string repo_slug();

struct Release {
    std::string tag;
    bool        draft = false;
};

// update.go's release.version(): the tag with a leading "v"/"." and any
// "-suffix" removed.
std::string release_version(const std::string & tag);

// update.go's latestRelease, over WinHTTP so no SSL library needs vendoring;
// Windows does the TLS. err carries the same wording callers act on: no
// releases yet, GitHub is rate limiting, or an unexpected status.
bool latest_release(const std::string & slug, Release & out, std::string & err);

// A best-effort guess at what the user typed (llmash, ollama, llamash),
// since only main.cpp's argv[0] handling knows for certain: LLMASH_PROG if
// set, else this executable's own file name.
std::string prog_name();

} // namespace clidoc
} // namespace llmash
