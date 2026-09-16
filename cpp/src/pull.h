#pragma once

#include "config.h"
#include "registry.h"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <cstdint>
#include <functional>
#include <istream>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace llmash {

void handle_pull(const httplib::Request & req, httplib::Response & res, Config & cfg, Registry & reg);
void handle_quants(const httplib::Request & req, httplib::Response & res, Config & cfg, Registry & reg);
void handle_resolve(const httplib::Request & req, httplib::Response & res, Config & cfg, Registry & reg);

struct HttpResult {
    int         status = 0;
    std::string body;
    int64_t     content_length = -1; // from the response header; -1 if absent
    std::string error;               // set only when the request itself failed
};

// A single buffered GET/HEAD. `range` is a Range header value
// ("bytes=0-99"), or empty for the whole body.
HttpResult http_request(const std::string & url, const std::string & method = "GET",
                         const std::string & range = "",
                         const std::vector<std::string> & headers = {},
                         int timeout_s = 0);

// GGUF value type ids, as the format itself defines them.
enum GGUFValueType : uint32_t {
    GGUF_U8 = 0, GGUF_I8 = 1, GGUF_U16 = 2, GGUF_I16 = 3, GGUF_U32 = 4, GGUF_I32 = 5,
    GGUF_F32 = 6, GGUF_BOOL = 7, GGUF_STRING = 8, GGUF_ARRAY = 9, GGUF_U64 = 10,
    GGUF_I64 = 11, GGUF_F64 = 12,
};

// The low-level cursor gguf.h keeps to itself.
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

// `others` collects the extensions of the repo's non-GGUF files, so a repo
// that holds none can say what it does hold.
std::vector<HfFile>   hf_files(const std::string & repo, std::string * err = nullptr,
                               std::set<std::string> * others = nullptr);

// "safetensors and MLX weights" from those extensions, empty when there are none
std::string other_formats_text(const std::set<std::string> & exts);
std::vector<HfFile>   pick_gguf(const std::vector<HfFile> & files, const std::string & quant);
std::optional<HfFile> pick_mmproj(const std::vector<HfFile> & files);
std::string           quant_tag(const std::string & name);
std::string           hf_download_url(const std::string & repo, const std::string & file);

struct QuantInfo {
    std::string name;
    int64_t     size  = 0;  // what the finished model weighs
    int         files = 0;
    int64_t     fetch = 0;  // what has to come down for it, when that differs
};
std::vector<QuantInfo> quants_of(const std::vector<HfFile> & files);

// Bits per weight a quantisation name implies: BF16 is 16, not 1, and a
// custom build carries its own budget.
double bits_of_quant(const std::string & quant);

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

// dest already holding exactly `total` bytes: nothing to download.
bool already_have(const std::string & dest, int64_t total);

// Several ranged connections at once, with a `<tmp>.idx` block map beside
// the file so an interrupted download resumes instead of restarting.
bool fetch_blocks(const std::string & url, const std::string & tmp, int64_t total,
                   const ProgressFn & progress, std::string & err);

// One connection when the server does not honour Range, fetch_blocks when
// it does and the file is worth splitting.
bool fetch_blob(const std::string & url, const std::string & tmp, int64_t total,
                 const ProgressFn & progress, std::string & err);

// One stretch of a remote file, written into a local file at `into`. No
// resume map: the caller is building scratch it will throw away.
struct RangeJob {
    std::string url;
    int64_t     from  = 0; // offset in the remote file
    int64_t     bytes = 0;
    int64_t     into  = 0; // offset in the local file
};
bool fetch_ranges(const std::string & dest, const std::vector<RangeJob> & jobs, const ProgressFn & progress,
                  std::string & err);

// ----------------------------------------------------------------- pulls
using Emit = std::function<void(const nlohmann::json &)>;

std::string loose_dir(const Config & cfg);

// aliases.json in the loose-GGUF folder: the display name a Hugging Face
// pull was given.
void set_alias(const Config & cfg, const std::string & file, const std::string & name);

// Downloads a Hugging Face repo's chosen build (and its mmproj, if any) into
// the loose-GGUF folder.
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
// still names.
void remove_manifest_model(const Config & cfg, Registry & reg, const std::string & manifest_path);

struct RegistryBuild {
    std::string unloadable;
    std::string hf_repo;  // where this model's other builds are, or ""
    std::string quant;
    int64_t     size = 0; // what the registry's own build would pull
};
RegistryBuild inspect_registry_build(const RegistryManifest & m);

void registry_pull(const std::string & ref, const Config & cfg, Registry & reg, const Emit & emit);

// The logic behind handle_pull/handle_resolve/handle_quants, kept free of
// httplib types.
void           run_pull(const nlohmann::json & body, const Config & cfg, Registry & reg, const Emit & emit);
nlohmann::json api_resolve(const std::string & ref, const Config & cfg);
// A quality llmash assembles rather than downloads.
std::string rco_quant_name(double bpw);
double      rco_bpw_of_quant(const std::string & quant);
// From a GGUF already here. `imatrix_from` is a file or repo, or empty to read
// it off the source's own metadata.
std::string rco_convert(const std::string & path, double bpw, const std::string & imatrix_from, const Config & cfg,
                        Registry & reg, const Emit & emit);

std::string rco_pull(const std::string & repo, double bpw, const std::string & as, const Config & cfg,
                     Registry & reg, const Emit & emit);

nlohmann::json api_quants(const std::string & repo_arg);

std::string human_bytes(int64_t b);

} // namespace llmash
