#pragma once

#include "config.h"
#include "registry.h"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <cstdint>
#include <functional>
#include <istream>
#include <optional>
#include <string>
#include <vector>

// Port of cmd/llmash/pull.go: /api/pull for both the Ollama registry (into
// the blob store) and Hugging Face (hf:owner/repo[@QUANT], into the loose
// GGUF folder), and the parallel resumable download engine both use.

namespace llmash {

// Implemented in pull.cpp, a separate module ported from pull.go (~970
// lines of Ollama-registry and Hugging Face client logic: manifest and blob
// fetching, range-probing a remote GGUF header to pick a quant, downloading
// with resumable progress). None of that is HTTP route wiring, so it does
// not belong in api.cpp any more than the chat proxy does; register_routes()
// only wires these three paths to it. Not implemented here.
void handle_pull(const httplib::Request & req, httplib::Response & res, Config & cfg, Registry & reg);
void handle_quants(const httplib::Request & req, httplib::Response & res, Config & cfg, Registry & reg);
void handle_resolve(const httplib::Request & req, httplib::Response & res, Config & cfg, Registry & reg);

// ---------------------------------------------------------------------
// Everything below is the logic those three handlers are wired to, kept
// free of httplib types so it can be exercised directly (draft.cpp calls
// into the HTTP/GGUF primitives here too, for the same reason).
// ---------------------------------------------------------------------

// -------------------------------------------------------------- transport
//
// cpp-httplib's HTTPS client only compiles in with CPPHTTPLIB_OPENSSL_SUPPORT
// defined and OpenSSL linked; neither is true of this build (see the CMake
// file), and OpenSSL is not among the vendored dependencies this project is
// allowed to add. huggingface.co and the Ollama registry are HTTPS-only, so
// outbound requests go through WinHTTP instead, which does TLS itself and
// needs nothing beyond what Windows already ships (linked via a pragma in
// pull.cpp, so the shared CMakeLists does not need a winhttp.lib entry).
struct HttpResult {
    int         status = 0;
    std::string body;
    int64_t     content_length = -1; // from the response header; -1 if absent
    std::string error;               // set only when the request itself failed
};

// A single buffered GET/HEAD. `range` is a Range header value ("bytes=0-99"),
// or empty for the whole body. `headers` are extra "Name: Value" lines.
// Redirects are followed automatically.
HttpResult http_request(const std::string & url, const std::string & method = "GET",
                         const std::string & range = "",
                         const std::vector<std::string> & headers = {});

// GGUF value type ids, as the format itself defines them.
enum GGUFValueType : uint32_t {
    GGUF_U8 = 0, GGUF_I8 = 1, GGUF_U16 = 2, GGUF_I16 = 3, GGUF_U32 = 4, GGUF_I32 = 5,
    GGUF_F32 = 6, GGUF_BOOL = 7, GGUF_STRING = 8, GGUF_ARRAY = 9, GGUF_U64 = 10,
    GGUF_I64 = 11, GGUF_F64 = 12,
};

// The low-level cursor gguf.h keeps to itself. read_gguf() exposes a fixed
// field set (arch/quant/params/tensors/mtp/vision) read from a local file;
// this module needs the embedding size, block count, vocabulary size and
// tensor shapes of a byte window fetched from the hub, none of which
// read_gguf reports, and it has no local file at all to point read_gguf at.
// Both probe_gguf_header() here (a registry build's own header, to see
// whether llama.cpp can load it) and draft.cpp's scan_gguf() (a drafter's
// compatibility with its target) read a header with this instead of
// re-deriving the byte layout a second time each.
class GGUFReader {
public:
    explicit GGUFReader(std::istream & in) : in_(in) {}

    bool        bad() const { return bad_; }
    uint32_t    u32();
    uint64_t    u64();
    std::string str();
    void        skip(int64_t n);
    void        skip_value(uint32_t type);
    // Numeric types only; false (still consumed) for float/bool/string/array.
    bool        read_int(uint32_t type, int64_t & out);

private:
    std::istream & in_;
    bool bad_ = false;
};

// -------------------------------------------------------- draft-file names
//
// Shared with draft.cpp: pull.go's own build picker has to leave a drafter's
// files out of "the model", which is the same classification draft.go uses
// to spot a drafter in the first place.
struct DraftKind {
    std::string name;     // what llmash calls it, and the sidecar file's suffix
    std::string spec_arg; // llama.cpp's --spec-type
    int         rank = 0; // higher wins when several exist
    std::vector<std::string> words;
};

const std::vector<DraftKind> & draft_kinds();
const DraftKind *              kind_of(const std::string & text);

// ------------------------------------------------------------- HF listing
struct HfFile {
    std::string name;
    int64_t     size = 0;
};

// `err` (when given) is set only for a request that failed outright (a
// transport error or unparseable JSON); a repository that answers with no
// GGUF files, or answers 404, comes back as an empty list with no error,
// matching hfFiles' own "not found is not a failure" reading of the API.
std::vector<HfFile>   hf_files(const std::string & repo, std::string * err = nullptr);
std::vector<HfFile>   pick_gguf(const std::vector<HfFile> & files, const std::string & quant);
std::optional<HfFile> pick_mmproj(const std::vector<HfFile> & files);
std::string           quant_tag(const std::string & name);
std::string           hf_download_url(const std::string & repo, const std::string & file);

struct QuantInfo {
    std::string name;
    int64_t     size  = 0;
    int         files = 0;
};
std::vector<QuantInfo> quants_of(const std::vector<HfFile> & files);

// ----------------------------------------------------------- hub search
// huggingface.co/api, used to find a drafter for a model already on disk.
struct HubModel {
    std::string              id;
    std::vector<std::string> tags;
    int                      downloads = 0;
    std::vector<std::string> base_models;
};
std::vector<HubModel> hub_search(const std::string & query, int limit);
bool                  hub_info(const std::string & repo, HubModel & out);

struct HubFile {
    std::string path;
    std::string type;
    int64_t     size = 0;
};
std::vector<HubFile> hub_files(const std::string & repo);

// ------------------------------------------------------------ downloading
using ProgressFn = std::function<void(int64_t)>;

int     dl_streams();
int64_t dl_block();

// dest already holding exactly `total` bytes: nothing to download. total==0
// (size unknown) counts an existing file of any size as already fetched,
// matching pull.go's own reading of a zero Content-Length.
bool already_have(const std::string & dest, int64_t total);

// Several ranged connections at once, with a `<tmp>.idx` block map beside
// the file so an interrupted download resumes instead of restarting.
bool fetch_blocks(const std::string & url, const std::string & tmp, int64_t total,
                   const ProgressFn & progress, std::string & err);

// One connection when the server does not honour Range, fetch_blocks when
// it does and the file is worth splitting.
bool fetch_blob(const std::string & url, const std::string & tmp, int64_t total,
                 const ProgressFn & progress, std::string & err);

// ----------------------------------------------------------------- pulls
using Emit = std::function<void(const nlohmann::json &)>;

std::string loose_dir(const Config & cfg);

// aliases.json in the loose-GGUF folder: the display name a Hugging Face
// pull was given. registry.h's Registry does not read this file yet (see
// concerns_about_contracts), so a pull recorded here does not surface under
// that name from Registry::all()/find() until it grows a reader for it.
void set_alias(const Config & cfg, const std::string & file, const std::string & name);

// Downloads a Hugging Face repo's chosen build (and its mmproj, if any) into
// the loose-GGUF folder. Returns the path of the first file pulled, or ""
// when it failed (emit carries the reason).
std::string hf_pull(const std::string & repo, const std::string & quant, const std::string & as,
                     const Config & cfg, Registry & reg, const Emit & emit);

// Records the name a Hugging Face pull was asked for, and fetches an MTP
// head from the same repository when one was chosen.
bool finish_hf(const std::string & repo, const std::string & first, const std::string & as,
               const std::string & mtp, const Config & cfg, Registry & reg, const Emit & emit);

// -------------------------------------------------------- registry pulls
struct RegistryLayer {
    std::string media_type;
    std::string digest;
    int64_t     size = 0;
};

struct RegistryManifest {
    std::string                  host, repo, tag;
    std::vector<RegistryLayer>   layers;
    std::optional<RegistryLayer> config_layer;
    std::string                  raw;

    std::string name() const; // the ref, "library/" trimmed for the Ollama host
    std::string base() const; // https://<host>/v2/<repo>
    std::string manifest_path(const Config & cfg) const;
};

void        split_ref(const std::string & ref, std::string & host, std::string & repo, std::string & tag);
bool        fetch_manifest(const std::string & ref, RegistryManifest & out, std::string & err);
std::string blob_path(const Config & cfg, const std::string & digest);

// Drops a registry model: its manifest, and every blob no other manifest
// still names. registry.h's Registry keeps no such method (it only reads);
// this walks the same manifest folders registry.cpp scans to find what
// still points at a blob before removing it, then invalidates `reg`.
void remove_manifest_model(const Config & cfg, Registry & reg, const std::string & manifest_path);

// registryBuild: what makes a registry build unreadable for llama.cpp (Ollama
// packs vision/audio encoders and mllama models the vendored server rejects),
// and the Hugging Face build to take instead, at the same quantisation.
struct RegistryBuild {
    std::string unloadable;
    std::string hf_repo;
    std::string quant;
};
RegistryBuild inspect_registry_build(const RegistryManifest & m);

void registry_pull(const std::string & ref, const Config & cfg, Registry & reg, const Emit & emit);

// The logic behind handle_pull/handle_resolve/handle_quants, kept free of
// httplib types. `body` is the parsed JSON POST body of /api/pull (model,
// quant, as, mtp, from_ollama). api_resolve's and api_quants' "error" key,
// when present, is the 404/502 case; anything else is 200.
void           run_pull(const nlohmann::json & body, const Config & cfg, Registry & reg, const Emit & emit);
nlohmann::json api_resolve(const std::string & ref, const Config & cfg);
nlohmann::json api_quants(const std::string & repo_arg);

std::string human_bytes(int64_t b);

} // namespace llmash
