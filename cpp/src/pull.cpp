#include "pull.h"

#include "gguf.h"
#include "gsq.h"

#ifdef _WIN32
#include <windows.h>
#include <winhttp.h>
#else
#include <curl/curl.h>
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <regex>
#include <sstream>
#include <thread>

// TLS without a vendored SSL library: Windows already has one.
#ifdef _WIN32
#pragma comment(lib, "winhttp.lib")
#endif

namespace fs = std::filesystem;
using json   = nlohmann::json;

namespace llmash {

namespace {

const char * const OLLAMA_REGISTRY = "registry.ollama.ai";
const char * const HF_BASE         = "https://huggingface.co";
const char * const HUB_API         = "https://huggingface.co/api";

const std::vector<std::pair<std::string, std::pair<std::string, std::string>>> & hf_replacements() {
    static const std::vector<std::pair<std::string, std::pair<std::string, std::string>>> m = {
        {"gemma4:e4b", {"unsloth/gemma-4-E4B-it-GGUF", "Q4_K_M"}},
        {"gemma4:26b", {"unsloth/gemma-4-26B-A4B-it-GGUF", "Q4_K_M"}},
        {"gemma4:31b", {"unsloth/gemma-4-31B-it-GGUF", "Q4_K_M"}},
        {"Qwen3.6:27B", {"unsloth/Qwen3.6-27B-GGUF", "Q4_K_M"}},
        {"qwen3.5:35b", {"unsloth/Qwen3.5-35B-A3B-GGUF", "Q4_K_M"}},
        {"glm-4.7-flash:latest", {"lmstudio-community/GLM-4.7-Flash-GGUF", "Q4_K_M"}},
        {"gpt-oss:120b", {"lmstudio-community/gpt-oss-120b-GGUF", "MXFP4"}},
    };
    return m;
}

const std::vector<std::string> & hf_prefixes() {
    static const std::vector<std::string> p = {"hf:", "hf.co/", "huggingface.co/"};
    return p;
}

// ------------------------------------------------------------- small utils

std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

std::string upper(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
    return s;
}

bool ends_with(const std::string & s, const std::string & suffix) {
    return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

bool starts_with(const std::string & s, const std::string & prefix) {
    return s.rfind(prefix, 0) == 0;
}

bool contains(const std::string & hay, const std::string & needle) {
    return hay.find(needle) != std::string::npos;
}

bool equal_fold(const std::string & a, const std::string & b) { return lower(a) == lower(b); }

std::string base_name(const std::string & p) { return fs::path(p).filename().string(); }

std::string stem_of(const std::string & p) { return fs::path(p).stem().string(); }

const std::regex & shard_suffix_re() {
    static const std::regex re(R"(-\d+-of-\d+$)");
    return re;
}

std::string strip_shard(const std::string & stem) {
    return std::regex_replace(stem, shard_suffix_re(), "");
}

std::string short12(const std::string & digest) {
    std::string d = digest;
    if (starts_with(d, "sha256:")) {
        d = d.substr(7);
    }
    return d.size() > 12 ? d.substr(0, 12) : d;
}

json error_obj(const std::string & msg) { return json{{"error", msg}}; }

std::string first_non_empty(const std::string & a, const std::string & b) { return a.empty() ? b : a; }

std::string j_str(const json & j, const char * key) {
    const auto it = j.find(key);
    if (it == j.end() || it->is_null()) {
        return "";
    }
    if (it->is_string()) {
        return it->get<std::string>();
    }
    return it->dump();
}

int64_t j_int(const json & j, const char * key) {
    const auto it = j.find(key);
    if (it == j.end() || !it->is_number()) {
        return 0;
    }
    return it->get<int64_t>();
}

// ------------------------------------------------------------- WinHTTP
//
// One request at a time, read in chunks. Both platforms answer to the same
// two names: a Stream the caller drains with read_chunk, and open_stream to
// start one.

#ifdef _WIN32

std::wstring widen(const std::string & s) {
    if (s.empty()) {
        return std::wstring();
    }
    const int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), nullptr, 0);
    std::wstring w(static_cast<size_t>(n), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), &w[0], n);
    return w;
}

class Handle {
public:
    Handle() = default;
    explicit Handle(HINTERNET h) : h_(h) {}
    ~Handle() { reset(); }
    Handle(const Handle &)             = delete;
    Handle & operator=(const Handle &) = delete;
    Handle(Handle && o) noexcept : h_(o.h_) { o.h_ = nullptr; }
    Handle & operator=(Handle && o) noexcept {
        if (this != &o) {
            reset(o.h_);
            o.h_ = nullptr;
        }
        return *this;
    }
    void      reset(HINTERNET h = nullptr) {
        if (h_ != nullptr && h_ != h) {
            WinHttpCloseHandle(h_);
        }
        h_ = h;
    }
    HINTERNET get() const { return h_; }
    explicit  operator bool() const { return h_ != nullptr; }

private:
    HINTERNET h_ = nullptr;
};

// One in-flight response. Every handle it owns is closed by its destructor,
// on the error paths too.
struct Stream {
    Handle  session, connect, request;
    int     status         = 0;
    int64_t content_length = -1;

    bool read_chunk(char * buf, size_t cap, size_t & got) {
        DWORD n = 0;
        got     = 0;
        if (WinHttpReadData(request.get(), buf, static_cast<DWORD>(cap), &n) == FALSE) {
            return false;
        }
        got = n;
        return true;
    }
};

std::string winhttp_error(const char * what) {
    const DWORD e = GetLastError();
    char        buf[160];
    std::snprintf(buf, sizeof(buf), "%s failed (WinHTTP error %lu)", what, static_cast<unsigned long>(e));
    return buf;
}

bool open_stream(const std::string & url, const std::string & method, const std::string & range,
                 const std::vector<std::string> & headers, Stream & st, std::string & err, int timeout_s = 0) {
    const std::wstring wurl = widen(url);

    URL_COMPONENTS uc{};
    uc.dwStructSize = sizeof(uc);
    std::wstring   host(256, L'\0'), path(8192, L'\0'), extra(4096, L'\0');
    uc.lpszHostName     = &host[0];
    uc.dwHostNameLength = static_cast<DWORD>(host.size() - 1);
    uc.lpszUrlPath      = &path[0];
    uc.dwUrlPathLength  = static_cast<DWORD>(path.size() - 1);
    uc.lpszExtraInfo    = &extra[0];
    uc.dwExtraInfoLength = static_cast<DWORD>(extra.size() - 1);
    if (!WinHttpCrackUrl(wurl.c_str(), static_cast<DWORD>(wurl.size()), 0, &uc)) {
        err = "could not parse the URL " + url;
        return false;
    }
    const std::wstring host_s(uc.lpszHostName, uc.dwHostNameLength);
    std::wstring       target(uc.lpszUrlPath, uc.dwUrlPathLength);
    target.append(uc.lpszExtraInfo, uc.dwExtraInfoLength);

    st.session.reset(WinHttpOpen(L"llmash", WINHTTP_ACCESS_TYPE_AUTOMATIC_PROXY, WINHTTP_NO_PROXY_NAME,
                                 WINHTTP_NO_PROXY_BYPASS, 0));
    if (!st.session) {
        st.session.reset(WinHttpOpen(L"llmash", WINHTTP_ACCESS_TYPE_DEFAULT_PROXY, WINHTTP_NO_PROXY_NAME,
                                     WINHTTP_NO_PROXY_BYPASS, 0));
    }
    if (!st.session) {
        err = winhttp_error("WinHttpOpen");
        return false;
    }
    if (timeout_s > 0) {
        const int ms = timeout_s * 1000;
        WinHttpSetTimeouts(st.session.get(), ms, ms, ms, ms);
    } else {
        WinHttpSetTimeouts(st.session.get(), 30000, 30000, 60000, 120000);
    }

    st.connect.reset(WinHttpConnect(st.session.get(), host_s.c_str(), uc.nPort, 0));
    if (!st.connect) {
        err = winhttp_error("WinHttpConnect");
        return false;
    }

    const DWORD flags = (uc.nScheme == INTERNET_SCHEME_HTTPS) ? WINHTTP_FLAG_SECURE : 0;
    st.request.reset(WinHttpOpenRequest(st.connect.get(), widen(method).c_str(), target.c_str(), nullptr,
                                        WINHTTP_NO_REFERER, WINHTTP_DEFAULT_ACCEPT_TYPES, flags));
    if (!st.request) {
        err = winhttp_error("WinHttpOpenRequest");
        return false;
    }

    std::string extra_headers;
    for (const auto & h : headers) {
        extra_headers += h + "\r\n";
    }
    if (!range.empty()) {
        extra_headers += "Range: " + range + "\r\n";
    }
    if (!extra_headers.empty()) {
        const std::wstring w = widen(extra_headers);
        WinHttpAddRequestHeaders(st.request.get(), w.c_str(), static_cast<DWORD>(w.size()),
                                 WINHTTP_ADDREQ_FLAG_ADD | WINHTTP_ADDREQ_FLAG_REPLACE);
    }

    if (!WinHttpSendRequest(st.request.get(), WINHTTP_NO_ADDITIONAL_HEADERS, 0, WINHTTP_NO_REQUEST_DATA, 0, 0, 0)) {
        err = winhttp_error("WinHttpSendRequest");
        return false;
    }
    if (!WinHttpReceiveResponse(st.request.get(), nullptr)) {
        err = winhttp_error("WinHttpReceiveResponse");
        return false;
    }

    DWORD code = 0, len = sizeof(code);
    if (WinHttpQueryHeaders(st.request.get(), WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER,
                            WINHTTP_HEADER_NAME_BY_INDEX, &code, &len, WINHTTP_NO_HEADER_INDEX)) {
        st.status = static_cast<int>(code);
    }

    wchar_t clen[64] = {};
    DWORD   clen_len = sizeof(clen);
    if (WinHttpQueryHeaders(st.request.get(), WINHTTP_QUERY_CONTENT_LENGTH, WINHTTP_HEADER_NAME_BY_INDEX, clen,
                            &clen_len, WINHTTP_NO_HEADER_INDEX)) {
        st.content_length = static_cast<int64_t>(_wcstoi64(clen, nullptr, 10));
    }
    return true;
}

#else

// ---------------------------------------------------------------- libcurl

// curl pushes bytes at a write callback, so the multi interface turns that
// round: perform until the callback has left something in `pending`, then
// hand it out. One perform delivers at most a few writes, so the buffer
// stays small without pausing the transfer.
struct Stream {
    CURLM *      multi = nullptr;
    CURL *       easy  = nullptr;
    curl_slist * hdrs  = nullptr;
    std::string  pending;
    bool         done           = false;
    int          status         = 0;
    int64_t      content_length = -1;

    Stream() = default;
    ~Stream() {
        if (multi != nullptr && easy != nullptr) {
            curl_multi_remove_handle(multi, easy);
        }
        if (easy != nullptr) {
            curl_easy_cleanup(easy);
        }
        if (multi != nullptr) {
            curl_multi_cleanup(multi);
        }
        if (hdrs != nullptr) {
            curl_slist_free_all(hdrs);
        }
    }
    Stream(const Stream &)             = delete;
    Stream & operator=(const Stream &) = delete;

    bool read_chunk(char * buf, size_t cap, size_t & got) {
        got = 0;
        while (pending.empty() && !done) {
            int running = 0;
            if (curl_multi_perform(multi, &running) != CURLM_OK) {
                return false;
            }
            if (running == 0) {
                done = true;
                break;
            }
            if (pending.empty()) {
                curl_multi_poll(multi, nullptr, 0, 200, nullptr);
            }
        }
        const size_t n = std::min(cap, pending.size());
        std::memcpy(buf, pending.data(), n);
        pending.erase(0, n);
        got = n;
        return true;
    }
};

size_t curl_sink(char * data, size_t size, size_t nmemb, void * user) {
    const size_t n = size * nmemb;
    static_cast<Stream *>(user)->pending.append(data, n);
    return n;
}

bool open_stream(const std::string & url, const std::string & method, const std::string & range,
                 const std::vector<std::string> & headers, Stream & st, std::string & err, int timeout_s = 0) {
    st.multi = curl_multi_init();
    st.easy  = curl_easy_init();
    if (st.multi == nullptr || st.easy == nullptr) {
        err = "could not start a request";
        return false;
    }
    curl_easy_setopt(st.easy, CURLOPT_URL, url.c_str());
    curl_easy_setopt(st.easy, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(st.easy, CURLOPT_WRITEFUNCTION, curl_sink);
    curl_easy_setopt(st.easy, CURLOPT_WRITEDATA, &st);
    curl_easy_setopt(st.easy, CURLOPT_USERAGENT, "llmash");
    curl_easy_setopt(st.easy, CURLOPT_ACCEPT_ENCODING, "");
    curl_easy_setopt(st.easy, CURLOPT_CONNECTTIMEOUT, static_cast<long>(timeout_s > 0 ? timeout_s : 30));
    if (method == "HEAD") {
        curl_easy_setopt(st.easy, CURLOPT_NOBODY, 1L);
    } else if (method != "GET") {
        curl_easy_setopt(st.easy, CURLOPT_CUSTOMREQUEST, method.c_str());
    }
    if (!range.empty()) {
        const std::string bytes = starts_with(range, "bytes=") ? range.substr(6) : range;
        curl_easy_setopt(st.easy, CURLOPT_RANGE, bytes.c_str());
    }
    for (const std::string & h : headers) {
        st.hdrs = curl_slist_append(st.hdrs, h.c_str());
    }
    if (st.hdrs != nullptr) {
        curl_easy_setopt(st.easy, CURLOPT_HTTPHEADER, st.hdrs);
    }
    if (curl_multi_add_handle(st.multi, st.easy) != CURLM_OK) {
        err = "could not start a request";
        return false;
    }

    // Perform until the response line has arrived, which is when the status
    // code becomes readable.
    for (;;) {
        int running = 0;
        if (curl_multi_perform(st.multi, &running) != CURLM_OK) {
            err = "the request failed";
            return false;
        }
        long code = 0;
        curl_easy_getinfo(st.easy, CURLINFO_RESPONSE_CODE, &code);
        if (code != 0) {
            st.status = static_cast<int>(code);
            break;
        }
        if (running == 0) {
            st.done = true;
            int      left = 0;
            CURLMsg * m   = curl_multi_info_read(st.multi, &left);
            err = (m != nullptr && m->data.result != CURLE_OK) ? curl_easy_strerror(m->data.result)
                                                               : "the request failed";
            return false;
        }
        curl_multi_poll(st.multi, nullptr, 0, 200, nullptr);
    }
    curl_off_t len = -1;
    if (curl_easy_getinfo(st.easy, CURLINFO_CONTENT_LENGTH_DOWNLOAD_T, &len) == CURLE_OK && len >= 0) {
        st.content_length = static_cast<int64_t>(len);
    }
    return true;
}

#endif

// ------------------------------------------------- GGUF header, from a URL

// pull.go's own quantisation table.
std::string gguf_file_type(int64_t ft) {
    switch (ft) {
        case 0:  return "F32";
        case 1:  return "F16";
        case 2:  return "Q4_0";
        case 3:  return "Q4_1";
        case 7:  return "Q8_0";
        case 8:  return "Q5_0";
        case 9:  return "Q5_1";
        case 10: return "Q2_K";
        case 11: return "Q3_K_S";
        case 12: return "Q3_K_M";
        case 13: return "Q3_K_L";
        case 14: return "Q4_K_S";
        case 15: return "Q4_K_M";
        case 16: return "Q5_K_S";
        case 17: return "Q5_K_M";
        case 18: return "Q6_K";
        case 19: return "IQ2_XXS";
        case 20: return "IQ2_XS";
        case 21: return "Q2_K_S";
        case 22: return "IQ3_XS";
        case 23: return "IQ3_XXS";
        case 24: return "IQ1_S";
        case 25: return "IQ4_NL";
        case 26: return "IQ3_S";
        case 27: return "IQ3_M";
        case 28: return "IQ2_S";
        case 29: return "IQ2_M";
        case 30: return "IQ4_XS";
        case 31: return "IQ1_M";
        case 32: return "BF16";
        case 38: return "MXFP4";
        default: return "";
    }
}

// Only what unloadableBuild/quantOfFileType/hfEquivalent ask of a header;
// ggufMetaFrom's full any-typed map has no other caller here.
struct HeaderMeta {
    std::string              arch;
    std::string              size_label;
    int64_t                  file_type = -1;
    int64_t                  embd      = 0; // <arch>.embedding_length
    std::vector<std::string> keys;
};

HeaderMeta read_header_meta(std::istream & in) {
    HeaderMeta m;
    char       magic[4] = {};
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
        const std::string key  = g.str();
        const uint32_t    type = g.u32();
        if (g.bad()) {
            break;
        }
        if (key == "general.architecture" && type == GGUF_STRING) {
            const std::string v = g.str();
            if (g.bad()) break;
            m.arch = v;
        } else if (key == "general.size_label" && type == GGUF_STRING) {
            const std::string v = g.str();
            if (g.bad()) break;
            m.size_label = v;
        } else if (key == "general.file_type") {
            int64_t v = 0;
            if (g.read_int(type, v) && !g.bad()) {
                m.file_type = v;
            }
            if (g.bad()) break;
        } else if (ends_with(key, ".embedding_length")) {
            int64_t v = 0;
            if (g.read_int(type, v) && !g.bad()) {
                m.embd = v;
            }
            if (g.bad()) break;
        } else {
            g.skip_value(type);
            if (g.bad()) break;
        }
        m.keys.push_back(key);
    }
    return m;
}

// The first few MB of a remote GGUF: every key that precedes the tokenizer,
// which is what tells one build from another.
HeaderMeta probe_gguf_header(const std::string & url) {
    const HttpResult r = http_request(url, "GET", "bytes=0-4194303");
    if (!r.error.empty() || (r.status != 200 && r.status != 206)) {
        return HeaderMeta{};
    }
    std::istringstream in(r.body, std::ios::binary);
    return read_header_meta(in);
}

// What makes a registry build unreadable for llama.cpp, or "" for a plain
// text-model GGUF.
std::string unloadable_build(const HeaderMeta & m) {
    if (m.arch.empty()) {
        return "";
    }
    if (m.arch == "mllama") {
        return "is packaged for Ollama's own runtime";
    }
    const std::string vp = m.arch + ".vision.", ap = m.arch + ".audio.";
    bool              vision = false, audio = false;
    for (const auto & k : m.keys) {
        if (starts_with(k, vp)) vision = true;
        if (starts_with(k, ap)) audio = true;
    }
    if (vision && audio) return "bundles its vision and audio encoders into the one file";
    if (vision) return "bundles its vision encoder into the one file";
    if (audio) return "bundles its audio encoder into the one file";
    return "";
}

std::string quant_of_file_type(const HeaderMeta & m) {
    const std::string q = m.file_type >= 0 ? gguf_file_type(m.file_type) : std::string();
    return q.empty() ? "Q4_K_M" : q;
}

// -------------------------------------------------------------- hub client

// draft.go's hubGet: no account, no token, and a limited body.
bool hub_get(const std::string & url, json & out, std::string & err) {
    const HttpResult r = http_request(url, "GET", "", {"User-Agent: llmash", "Accept: application/json"});
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
    out = json::parse(r.body, nullptr, false);
    if (out.is_discarded()) {
        err = "hub answered with something that is not JSON";
        return false;
    }
    return true;
}

std::string url_query_escape(const std::string & s) {
    static const char * hex = "0123456789ABCDEF";
    std::string         out;
    for (const unsigned char c : s) {
        if (std::isalnum(c) != 0 || c == '-' || c == '_' || c == '.' || c == '~') {
            out += static_cast<char>(c);
        } else if (c == ' ') {
            out += '+';
        } else {
            out += '%';
            out += hex[c >> 4];
            out += hex[c & 0xF];
        }
    }
    return out;
}

// ------------------------------------------------------------ block maps

int64_t block_count(int64_t total) {
    const int64_t n = (total + dl_block() - 1) / dl_block();
    return n < 1 ? 1 : n;
}

std::vector<unsigned char> load_block_map(const std::string & idx_path, int64_t nblocks) {
    std::vector<unsigned char> have(static_cast<size_t>(nblocks), 0);
    std::ifstream              in(idx_path, std::ios::binary);
    if (!in) {
        return have;
    }
    const std::string b((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (static_cast<int64_t>(b.size()) == nblocks) {
        std::memcpy(have.data(), b.data(), b.size());
    }
    return have;
}

bool save_block_map(const std::string & idx_path, const std::vector<unsigned char> & have) {
    std::ofstream out(idx_path, std::ios::binary | std::ios::trunc);
    if (!out) {
        return false;
    }
    out.write(reinterpret_cast<const char *>(have.data()), static_cast<std::streamsize>(have.size()));
    return out.good();
}

struct BlockPlan {
    std::vector<int64_t> todo;
    int64_t              done = 0;
};

BlockPlan plan_blocks(const std::vector<unsigned char> & have, int64_t total) {
    BlockPlan p;
    for (int64_t i = 0; i < static_cast<int64_t>(have.size()); i++) {
        if (have[static_cast<size_t>(i)] == 1) {
            p.done += dl_block();
        } else {
            p.todo.push_back(i);
        }
    }
    if (p.done > total) {
        p.done = total;
    }
    return p;
}

// A .part left by an older single-stream download has no block map, so the
// whole blocks it already holds are marked before the streams start.
bool seed_block_map(const std::string & tmp, int64_t total, int64_t & prefix) {
    prefix = 0;
    const std::string idx = tmp + ".idx";
    std::error_code   ec;
    if (!fs::is_regular_file(tmp, ec) || fs::is_regular_file(idx, ec)) {
        return false;
    }
    const uintmax_t sz = fs::file_size(tmp, ec);
    if (ec) {
        return false;
    }
    prefix = static_cast<int64_t>(sz);
    const int64_t              n = block_count(total);
    std::vector<unsigned char> have(static_cast<size_t>(n), 0);
    for (int64_t i = 0; i < prefix / dl_block() && i < n; i++) {
        have[static_cast<size_t>(i)] = 1;
    }
    save_block_map(idx, have);
    return true;
}

// ------------------------------------------------------------- aliases

bool write_alias(const Config & cfg, const std::string & file, const std::string & name, std::string & err) {
    const fs::path  dir = loose_dir(cfg);
    std::error_code ec;
    fs::create_directories(dir, ec);
    const fs::path path = dir / "aliases.json";

    json data = json::object();
    {
        std::ifstream in(path, std::ios::binary);
        if (in) {
            json parsed = json::parse(in, nullptr, false);
            if (!parsed.is_discarded() && parsed.is_object()) {
                data = std::move(parsed);
            }
        }
    }
    data[file] = name;

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        err = "could not open " + path.string();
        return false;
    }
    out << data.dump(2);
    if (!out.good()) {
        err = "could not write " + path.string();
        return false;
    }
    return true;
}

} // namespace

// ============================================================== transport

HttpResult http_request(const std::string & url, const std::string & method, const std::string & range,
                        const std::vector<std::string> & headers, int timeout_s) {
    HttpResult out;
    Stream     st;
    if (!open_stream(url, method, range, headers, st, out.error, timeout_s)) {
        return out;
    }
    out.status         = st.status;
    out.content_length = st.content_length;

    std::vector<char> buf(64 * 1024);
    size_t            got = 0;
    // Beyond this a caller wanted a file, and those go through fetch_blob.
    const size_t cap = 64u * 1024u * 1024u;
    while (st.read_chunk(buf.data(), buf.size(), got) && got > 0) {
        out.body.append(buf.data(), got);
        if (out.body.size() >= cap) {
            break;
        }
    }
    return out;
}

// ============================================================ GGUFReader

uint32_t GGUFReader::u32() {
    uint32_t v = 0;
    if (!in_.read(reinterpret_cast<char *>(&v), 4)) {
        bad_ = true;
        return 0;
    }
    return v;
}

uint64_t GGUFReader::u64() {
    uint64_t v = 0;
    if (!in_.read(reinterpret_cast<char *>(&v), 8)) {
        bad_ = true;
        return 0;
    }
    return v;
}

std::string GGUFReader::str() {
    const uint64_t n = u64();
    if (bad_ || n > (64ull << 20)) {
        bad_ = true;
        return "";
    }
    std::string s(static_cast<size_t>(n), '\0');
    if (n != 0 && !in_.read(&s[0], static_cast<std::streamsize>(n))) {
        bad_ = true;
        return "";
    }
    return s;
}

void GGUFReader::skip(int64_t n) {
    if (n < 0) {
        bad_ = true;
        return;
    }
    in_.seekg(n, std::ios::cur);
    if (!in_) {
        bad_ = true;
    }
}

void GGUFReader::skip_value(uint32_t type) {
    switch (type) {
        case GGUF_U8:
        case GGUF_I8:
        case GGUF_BOOL: skip(1); return;
        case GGUF_U16:
        case GGUF_I16: skip(2); return;
        case GGUF_U32:
        case GGUF_I32:
        case GGUF_F32: skip(4); return;
        case GGUF_U64:
        case GGUF_I64:
        case GGUF_F64: skip(8); return;
        case GGUF_STRING: str(); return;
        case GGUF_ARRAY: break;
        default: bad_ = true; return;
    }
    const uint32_t et = u32();
    const uint64_t n  = u64();
    if (bad_) {
        return;
    }
    // The skip has to be exact: land one byte off and every key after the
    // array is read from the middle of it.
    switch (et) {
        case GGUF_U8:
        case GGUF_I8:
        case GGUF_BOOL: skip(static_cast<int64_t>(n)); return;
        case GGUF_U16:
        case GGUF_I16: skip(static_cast<int64_t>(n) * 2); return;
        case GGUF_U32:
        case GGUF_I32:
        case GGUF_F32: skip(static_cast<int64_t>(n) * 4); return;
        case GGUF_U64:
        case GGUF_I64:
        case GGUF_F64: skip(static_cast<int64_t>(n) * 8); return;
        case GGUF_STRING:
            for (uint64_t i = 0; i < n && !bad_; i++) {
                const uint64_t ln = u64();
                if (bad_) return;
                skip(static_cast<int64_t>(ln));
            }
            return;
        default:
            for (uint64_t i = 0; i < n && !bad_; i++) {
                skip_value(et);
            }
            return;
    }
}

bool GGUFReader::read_int(uint32_t type, int64_t & out) {
    switch (type) {
        case GGUF_U8: {
            uint8_t v = 0;
            if (!in_.read(reinterpret_cast<char *>(&v), 1)) { bad_ = true; return false; }
            out = v;
            return true;
        }
        case GGUF_I8: {
            int8_t v = 0;
            if (!in_.read(reinterpret_cast<char *>(&v), 1)) { bad_ = true; return false; }
            out = v;
            return true;
        }
        case GGUF_U16: {
            uint16_t v = 0;
            if (!in_.read(reinterpret_cast<char *>(&v), 2)) { bad_ = true; return false; }
            out = v;
            return true;
        }
        case GGUF_I16: {
            int16_t v = 0;
            if (!in_.read(reinterpret_cast<char *>(&v), 2)) { bad_ = true; return false; }
            out = v;
            return true;
        }
        case GGUF_U32: out = static_cast<int64_t>(u32()); return !bad_;
        case GGUF_I32: out = static_cast<int32_t>(u32()); return !bad_;
        case GGUF_U64:
        case GGUF_I64: out = static_cast<int64_t>(u64()); return !bad_;
        default: skip_value(type); return false;
    }
}

// ========================================================= draft-file names

const std::vector<DraftKind> & draft_kinds() {
    static const std::vector<DraftKind> kinds = {
        {"mtp", "draft-mtp", 50, {}},
        {"eagle3", "draft-eagle3", 40, {"eagle3", "eagle-3", "eagle_3"}},
        {"dspark", "draft-dspark", 30, {"dspark", "d-spark"}},
        {"dflash", "draft-dflash", 20, {"dflash", "d-flash"}},
        {"draft", "draft-simple", 10, {"draft", "speculator", "speculative"}},
    };
    return kinds;
}

const DraftKind * kind_of(const std::string & text) {
    const std::string low  = lower(text);
    std::string       base = low;
    const size_t      i    = base.find_last_of("/\\");
    if (i != std::string::npos) {
        base = base.substr(i + 1);
    }
    const auto & kinds = draft_kinds();
    // an MTP head is published next to its model as mtp-<model>.gguf; the
    // word alone also names models that carry a head of their own
    if (starts_with(base, "mtp-") || starts_with(base, "mtp_") || contains(base, ".mtp.")) {
        return &kinds[0];
    }
    for (size_t k = 0; k < kinds.size(); k++) {
        for (const auto & w : kinds[k].words) {
            if (contains(low, w)) {
                return &kinds[k];
            }
        }
    }
    return nullptr;
}

// ============================================================== HF listing

std::string quant_tag(const std::string & name) {
    static const std::regex re(
        R"((?:^|[-_.])((?:UD-)?(?:IQ|Q|TQ)[1-8](?:_[0-9A-Z]+)*|BF16|F16|F32|MXFP4(?:_MOE)?|NVFP4)(?:[-_.]|$))",
        std::regex::icase);
    std::smatch m;
    if (!std::regex_search(name, m, re)) {
        return "";
    }
    return upper(m[1].str());
}

std::string hf_download_url(const std::string & repo, const std::string & file) {
    return std::string(HF_BASE) + "/" + repo + "/resolve/main/" + file;
}

namespace {

// The GGUF entries of /api/models/<repo>?blobs=true. False only for a body
// that is not the JSON object the API documents.
bool parse_hf_siblings(const std::string & body, std::vector<HfFile> & out) {
    const json d = json::parse(body, nullptr, false);
    if (d.is_discarded() || !d.is_object()) {
        return false;
    }
    const auto it = d.find("siblings");
    if (it == d.end() || !it->is_array()) {
        return true;
    }
    for (const auto & f : *it) {
        if (!f.is_object()) {
            continue;
        }
        HfFile h;
        h.name = j_str(f, "rfilename");
        h.size = j_int(f, "size");
        if (ends_with(lower(h.name), ".gguf")) {
            out.push_back(std::move(h));
        }
    }
    return true;
}

} // namespace

std::vector<HfFile> hf_files(const std::string & repo, std::string * err) {
    if (err != nullptr) {
        err->clear();
    }
    const HttpResult r = http_request(std::string(HF_BASE) + "/api/models/" + repo + "?blobs=true");
    if (!r.error.empty()) {
        if (err != nullptr) *err = r.error;
        return {};
    }
    if (r.status != 200) {
        return {};
    }
    std::vector<HfFile> out;
    if (!parse_hf_siblings(r.body, out)) {
        if (err != nullptr) *err = "the Hugging Face API answered with something that is not JSON";
        return {};
    }
    return out;
}

std::vector<HfFile> pick_gguf(const std::vector<HfFile> & files, const std::string & quant) {
    std::vector<HfFile> builds;
    for (const auto & f : files) {
        const std::string low = lower(f.name);
        if (ends_with(low, ".gguf") && !contains(low, "mmproj") && kind_of(f.name) == nullptr) {
            builds.push_back(f);
        }
    }
    const std::string   q = lower(quant);
    std::vector<HfFile> cand;
    for (const auto & f : builds) {
        if (equal_fold(quant_tag(f.name), quant) || (!q.empty() && contains(lower(f.name), q))) {
            cand.push_back(f);
        }
    }
    if (cand.empty()) {
        // the nearest build to the one asked for; a full-precision file is
        // the last resort, never the fallback
        for (const char * nearest : {"Q4_K_M", "Q4_K_S", "Q4_0", "IQ4_XS", "Q5_K_M", "Q6_K", "Q8_0", "Q4_1", "Q5_K_S",
                                  "Q3_K_M"}) {
            for (const auto & f : builds) {
                if (equal_fold(quant_tag(f.name), nearest)) {
                    cand.push_back(f);
                }
            }
            if (!cand.empty()) {
                break;
            }
        }
    }
    if (cand.empty()) {
        cand = builds;
    }
    std::vector<HfFile> shards;
    for (const auto & f : cand) {
        if (contains(f.name, "-of-")) {
            shards.push_back(f);
        }
    }
    if (!shards.empty()) {
        std::sort(shards.begin(), shards.end(), [](const HfFile & a, const HfFile & b) { return a.name < b.name; });
        return shards;
    }
    if (cand.empty()) {
        return {};
    }
    std::sort(cand.begin(), cand.end(), [](const HfFile & a, const HfFile & b) { return a.size < b.size; });
    return {cand.front()};
}

std::optional<HfFile> pick_mmproj(const std::vector<HfFile> & files) {
    std::vector<HfFile> mm;
    for (const auto & f : files) {
        if (contains(lower(f.name), "mmproj")) {
            mm.push_back(f);
        }
    }
    if (mm.empty()) {
        return std::nullopt;
    }
    const auto smallest = [](std::vector<HfFile> xs) {
        std::sort(xs.begin(), xs.end(), [](const HfFile & a, const HfFile & b) { return a.size < b.size; });
        return xs.front();
    };
    for (const char * want : {"-f16", "f16", "-bf16", "bf16"}) {
        std::vector<HfFile> hit;
        for (const auto & f : mm) {
            if (contains(lower(f.name), want)) {
                hit.push_back(f);
            }
        }
        if (!hit.empty()) {
            return smallest(hit);
        }
    }
    return smallest(mm);
}

double bits_of_quant(const std::string & quant) {
    const std::string up = upper(quant);
    if (up == "F32" || up == "FP32") {
        return 32;
    }
    if (up == "F16" || up == "BF16" || up == "FP16") {
        return 16;
    }
    if (const double g = gsq::bpw_of_quant(quant); g > 0) {
        return g;
    }
    static const std::regex re(R"((?:^|[^0-9])([1-8])(?:_|$|[A-Za-z]))");
    std::smatch             m;
    if (!std::regex_search(up, m, re)) {
        return 0;
    }
    return (m[1].str()[0] - '0') + (contains(up, "_K") || contains(up, "XL") ? 0.5 : 0.0);
}

std::vector<QuantInfo> quants_of(const std::vector<HfFile> & files) {
    std::vector<QuantInfo> out; // insertion order, then a stable sort by size
    for (const auto & f : files) {
        const std::string low = lower(f.name);
        if (!ends_with(low, ".gguf") || contains(low, "mmproj") || kind_of(f.name) != nullptr) {
            continue; // a projector or a draft head is not a build of the model
        }
        const std::string q = quant_tag(f.name);
        if (q.empty()) {
            continue;
        }
        auto it = std::find_if(out.begin(), out.end(), [&](const QuantInfo & qi) { return qi.name == q; });
        if (it == out.end()) {
            out.push_back(QuantInfo{q, 0, 0});
            it = out.end() - 1;
        }
        it->size += f.size;
        it->files++;
    }
    std::stable_sort(out.begin(), out.end(), [](const QuantInfo & a, const QuantInfo & b) { return a.size < b.size; });
    return out;
}

// =============================================================== hub search

std::vector<HubModel> hub_search(const std::string & query, int limit) {
    const std::string url = std::string(HUB_API) + "/models?search=" + url_query_escape(query) +
                            "&limit=" + std::to_string(limit) + "&sort=downloads&direction=-1";
    json        doc;
    std::string err;
    if (!hub_get(url, doc, err) || !doc.is_array()) {
        return {};
    }
    std::vector<HubModel> out;
    for (const auto & e : doc) {
        if (!e.is_object()) {
            continue;
        }
        HubModel m;
        m.id        = j_str(e, "id");
        m.downloads = static_cast<int>(j_int(e, "downloads"));
        const auto tags = e.find("tags");
        if (tags != e.end() && tags->is_array()) {
            for (const auto & t : *tags) {
                if (t.is_string()) {
                    m.tags.push_back(t.get<std::string>());
                }
            }
        }
        const auto card = e.find("cardData");
        if (card != e.end() && card->is_object()) {
            const auto bm = card->find("base_model");
            if (bm != card->end()) {
                if (bm->is_string()) {
                    m.base_models.push_back(bm->get<std::string>());
                } else if (bm->is_array()) {
                    for (const auto & b : *bm) {
                        m.base_models.push_back(b.is_string() ? b.get<std::string>() : b.dump());
                    }
                }
            }
        }
        out.push_back(std::move(m));
    }
    return out;
}

bool hub_info(const std::string & repo, HubModel & out) {
    json        doc;
    std::string err;
    if (!hub_get(std::string(HUB_API) + "/models/" + repo, doc, err) || !doc.is_object()) {
        return false;
    }
    out    = HubModel{};
    out.id = j_str(doc, "id");
    out.downloads = static_cast<int>(j_int(doc, "downloads"));
    const auto tags = doc.find("tags");
    if (tags != doc.end() && tags->is_array()) {
        for (const auto & t : *tags) {
            if (t.is_string()) {
                out.tags.push_back(t.get<std::string>());
            }
        }
    }
    const auto card = doc.find("cardData");
    if (card != doc.end() && card->is_object()) {
        const auto bm = card->find("base_model");
        if (bm != card->end()) {
            if (bm->is_string()) {
                out.base_models.push_back(bm->get<std::string>());
            } else if (bm->is_array()) {
                for (const auto & b : *bm) {
                    out.base_models.push_back(b.is_string() ? b.get<std::string>() : b.dump());
                }
            }
        }
    }
    return true;
}

std::vector<HubFile> hub_files(const std::string & repo) {
    json        doc;
    std::string err;
    if (!hub_get(std::string(HUB_API) + "/models/" + repo + "/tree/main?recursive=true", doc, err) ||
        !doc.is_array()) {
        return {};
    }
    std::vector<HubFile> out;
    for (const auto & e : doc) {
        if (!e.is_object()) {
            continue;
        }
        HubFile f;
        f.path = j_str(e, "path");
        f.type = j_str(e, "type");
        f.size = j_int(e, "size");
        out.push_back(std::move(f));
    }
    return out;
}

// =============================================================== downloading

int dl_streams() {
    static const int n = env_int("LLMASH_DL_STREAMS", 8);
    return n;
}

int64_t dl_block() {
    static const int64_t n = env_int("LLMASH_DL_BLOCK", 64 * 1024 * 1024);
    return n;
}

bool already_have(const std::string & dest, int64_t total) {
    std::error_code ec;
    if (!fs::is_regular_file(dest, ec)) {
        return false;
    }
    const uintmax_t sz = fs::file_size(dest, ec);
    if (ec) {
        return false;
    }
    return total == 0 || static_cast<int64_t>(sz) == total;
}

bool fetch_blocks(const std::string & url, const std::string & tmp, int64_t total, const ProgressFn & progress,
                  std::string & err) {
    const int64_t     nblocks  = block_count(total);
    const std::string idx_path = tmp + ".idx";

    std::vector<unsigned char> have = load_block_map(idx_path, nblocks);

    std::error_code ec;
    if (!fs::exists(tmp, ec)) {
        std::ofstream create(tmp, std::ios::binary);
        if (!create) {
            err = "could not create " + tmp;
            return false;
        }
    }
    fs::resize_file(tmp, static_cast<uintmax_t>(total), ec);
    if (ec) {
        err = "could not size " + tmp + ": " + ec.message();
        return false;
    }

    const BlockPlan plan = plan_blocks(have, total);
    std::mutex      mu;
    std::string     first_err;
    size_t          next = 0;
    int64_t         done = plan.done;

    const auto take = [&](int64_t & block) {
        std::lock_guard<std::mutex> lk(mu);
        if (next >= plan.todo.size() || !first_err.empty()) {
            return false;
        }
        block = plan.todo[next++];
        return true;
    };

    int workers = dl_streams();
    if (static_cast<int64_t>(workers) > nblocks) {
        workers = static_cast<int>(nblocks);
    }
    if (workers < 1) {
        workers = 1;
    }

    std::vector<std::thread> pool;
    pool.reserve(static_cast<size_t>(workers));
    for (int w = 0; w < workers; w++) {
        pool.emplace_back([&] {
            std::fstream fh(tmp, std::ios::binary | std::ios::in | std::ios::out);
            if (!fh) {
                std::lock_guard<std::mutex> lk(mu);
                if (first_err.empty()) {
                    first_err = "could not open " + tmp;
                }
                return;
            }
            std::vector<char> buf(1 << 20);
            int64_t           i = 0;
            while (take(i)) {
                const int64_t start = i * dl_block();
                int64_t       end   = start + dl_block();
                if (end > total) {
                    end = total;
                }
                std::string last_err;
                for (int attempt = 0; attempt < 4; attempt++) {
                    last_err.clear();
                    char range[64];
                    std::snprintf(range, sizeof(range), "bytes=%lld-%lld", static_cast<long long>(start),
                                  static_cast<long long>(end - 1));
                    Stream stm;
                    if (!open_stream(url, "GET", range, {}, stm, last_err)) {
                        // last_err carries the reason
                    } else if (stm.status != 200 && stm.status != 206) {
                        last_err = "HTTP " + std::to_string(stm.status);
                    } else {
                        int64_t off = start;
                        size_t  got = 0;
                        bool    ok  = true;
                        while (true) {
                            if (!stm.read_chunk(buf.data(), buf.size(), got)) {
                                last_err = "the connection dropped";
                                ok       = false;
                                break;
                            }
                            if (got == 0) {
                                break;
                            }
                            fh.seekp(off, std::ios::beg);
                            fh.write(buf.data(), got);
                            if (!fh) {
                                last_err = "could not write to " + tmp;
                                ok       = false;
                                break;
                            }
                            off += got;
                            std::lock_guard<std::mutex> lk(mu);
                            done += got;
                            if (progress) {
                                progress(done);
                            }
                        }
                        if (ok) {
                            fh.flush();
                            break;
                        }
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(1500 * (attempt + 1)));
                }
                if (!last_err.empty()) {
                    std::lock_guard<std::mutex> lk(mu);
                    if (first_err.empty()) {
                        first_err = last_err;
                    }
                    return;
                }
                std::lock_guard<std::mutex> lk(mu);
                have[static_cast<size_t>(i)] = 1;
                save_block_map(idx_path, have);
            }
        });
    }
    for (auto & t : pool) {
        t.join();
    }
    if (!first_err.empty()) {
        err = first_err;
        return false;
    }
    fs::remove(idx_path, ec);
    return true;
}

bool fetch_ranges(const std::string & dest, const std::vector<RangeJob> & jobs, const ProgressFn & progress,
                  std::string & err) {
    // Split into blocks so several connections share the work, the way a
    // whole-file download does.
    struct Block {
        std::string url;
        int64_t     from = 0, bytes = 0, into = 0;
    };
    std::vector<Block> blocks;
    for (const RangeJob & j : jobs) {
        for (int64_t at = 0; at < j.bytes; at += dl_block()) {
            const int64_t n = (std::min)(dl_block(), j.bytes - at);
            blocks.push_back(Block{j.url, j.from + at, n, j.into + at});
        }
    }
    if (blocks.empty()) {
        return true;
    }

    std::mutex  mu;
    std::string first_err;
    size_t      next = 0;
    int64_t     done = 0;

    int workers = dl_streams();
    if (static_cast<size_t>(workers) > blocks.size()) {
        workers = static_cast<int>(blocks.size());
    }
    if (workers < 1) {
        workers = 1;
    }

    std::vector<std::thread> pool;
    for (int w = 0; w < workers; w++) {
        pool.emplace_back([&] {
            std::fstream fh(dest, std::ios::binary | std::ios::in | std::ios::out);
            if (!fh) {
                std::lock_guard<std::mutex> lk(mu);
                if (first_err.empty()) {
                    first_err = "could not open " + dest;
                }
                return;
            }
            std::vector<char> buf(1 << 20);
            for (;;) {
                Block b;
                {
                    std::lock_guard<std::mutex> lk(mu);
                    if (next >= blocks.size() || !first_err.empty()) {
                        return;
                    }
                    b = blocks[next++];
                }
                std::string last_err;
                for (int attempt = 0; attempt < 4; attempt++) {
                    last_err.clear();
                    char range[64];
                    std::snprintf(range, sizeof(range), "bytes=%lld-%lld", static_cast<long long>(b.from),
                                  static_cast<long long>(b.from + b.bytes - 1));
                    Stream stm;
                    if (!open_stream(b.url, "GET", range, {}, stm, last_err)) {
                        // last_err carries the reason
                    } else if (stm.status != 200 && stm.status != 206) {
                        last_err = "HTTP " + std::to_string(stm.status);
                    } else {
                        int64_t off  = b.into;
                        int64_t left = b.bytes;
                        size_t  got  = 0;
                        bool    ok   = true;
                        while (left > 0) {
                            if (!stm.read_chunk(buf.data(), (std::min)(buf.size(), static_cast<size_t>(left)), got)) {
                                last_err = "the connection dropped";
                                ok       = false;
                                break;
                            }
                            if (got == 0) {
                                break;
                            }
                            fh.seekp(off, std::ios::beg);
                            fh.write(buf.data(), static_cast<std::streamsize>(got));
                            if (!fh) {
                                last_err = "could not write to " + dest;
                                ok       = false;
                                break;
                            }
                            off += static_cast<int64_t>(got);
                            left -= static_cast<int64_t>(got);
                            std::lock_guard<std::mutex> lk(mu);
                            done += static_cast<int64_t>(got);
                            if (progress) {
                                progress(done);
                            }
                        }
                        if (ok && left == 0) {
                            fh.flush();
                            break;
                        }
                        if (ok) {
                            last_err = "the connection ended early";
                        }
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(1500 * (attempt + 1)));
                }
                if (!last_err.empty()) {
                    std::lock_guard<std::mutex> lk(mu);
                    if (first_err.empty()) {
                        first_err = last_err;
                    }
                    return;
                }
            }
        });
    }
    for (auto & t : pool) {
        t.join();
    }
    if (!first_err.empty()) {
        err = first_err;
        return false;
    }
    return true;
}

bool fetch_blob(const std::string & url, const std::string & tmp, int64_t total, const ProgressFn & progress,
                std::string & err) {
    if (total > 2 * dl_block()) {
        int  code = 0;
        bool probed = false;
        {
            Stream      probe;
            std::string perr;
            probed = open_stream(url, "GET", "bytes=0-0", {}, probe, perr);
            code   = probe.status;
        }
        if (probed && code == 206) {
            return fetch_blocks(url, tmp, total, progress, err);
        }
    }

    Stream st;
    if (!open_stream(url, "GET", "", {}, st, err)) {
        return false;
    }
    if (st.status != 200) {
        err = "HTTP " + std::to_string(st.status);
        return false;
    }
    std::ofstream fh(tmp, std::ios::binary | std::ios::trunc);
    if (!fh) {
        err = "could not create " + tmp;
        return false;
    }
    int64_t           done = 0;
    auto              last = std::chrono::steady_clock::now() - std::chrono::hours(1);
    std::vector<char> buf(1 << 20);
    size_t            got = 0;
    while (true) {
        if (!st.read_chunk(buf.data(), buf.size(), got)) {
            err = "the connection dropped";
            return false;
        }
        if (got == 0) {
            break;
        }
        fh.write(buf.data(), got);
        if (!fh) {
            err = "could not write to " + tmp;
            return false;
        }
        done += got;
        const auto now = std::chrono::steady_clock::now();
        if (now - last > std::chrono::milliseconds(200)) {
            last = now;
            if (progress) {
                progress(done);
            }
        }
    }
    fh.flush();
    if (!fh.good()) {
        err = "could not write to " + tmp;
        return false;
    }
    return true;
}

// ===================================================================== pulls


void set_alias(const Config & cfg, const std::string & file, const std::string & name) {
    std::string err;
    write_alias(cfg, file, name, err);
}

// ----------------------------------------------------------------- GSQ
// A chunk at a time, so the source and the result are never both whole.

namespace {

// The highest-quality build in the repo, as the requantization source.
std::vector<HfFile> pick_gsq_source(const std::vector<HfFile> & files) {
    for (const std::string & want : gsq::source_preference()) {
        std::vector<HfFile> cand;
        for (const HfFile & f : files) {
            const std::string low = lower(f.name);
            if (contains(low, want) && !contains(low, "mmproj") && kind_of(f.name) == nullptr &&
                ends_with(low, ".gguf")) {
                cand.push_back(f);
            }
        }
        if (cand.empty()) {
            continue;
        }
        std::vector<HfFile> shards;
        for (const HfFile & f : cand) {
            if (contains(f.name, "-of-")) {
                shards.push_back(f);
            }
        }
        if (!shards.empty()) {
            std::sort(shards.begin(), shards.end(), [](const HfFile & a, const HfFile & b) { return a.name < b.name; });
            return shards;
        }
        std::sort(cand.begin(), cand.end(), [](const HfFile & a, const HfFile & b) { return a.size < b.size; });
        return {cand.back()};
    }
    return {};
}

std::string mmss(double seconds) {
    const int s = static_cast<int>(seconds);
    char      buf[32];
    std::snprintf(buf, sizeof(buf), "%d:%02d", s / 60, s % 60);
    return buf;
}

std::string pad_right(const std::string & s, size_t n) {
    return s.size() >= n ? s : s + std::string(n - s.size(), ' ');
}

// The repository's own file, fetched into the cache once.
bool cache_file(const std::string & repo, const std::string & rel, const Config & cfg, std::string & path,
                std::string & err) {
    const fs::path  dir = fs::path(cfg.root) / "cache" / "gsq";
    std::error_code ec;
    fs::create_directories(dir, ec);
    const fs::path dest = dir / base_name(rel);
    if (fs::is_regular_file(dest, ec) && fs::file_size(dest, ec) > 0) {
        path = dest.string();
        return true;
    }
    const std::string url  = hf_download_url(repo, rel);
    const HttpResult  head = http_request(url, "HEAD");
    const int64_t     n    = head.content_length;
    if (n > 0 && n > (64ll << 20)) {
        // the imatrix is tens of megabytes; anything larger is not one
        if (!fetch_blob(url, dest.string() + ".part", n, [](int64_t) {}, err)) {
            return false;
        }
        fs::rename(dest.string() + ".part", dest, ec);
        path = dest.string();
        return true;
    }
    const HttpResult r = http_request(url);
    if (!r.error.empty() || r.status != 200) {
        err = rel + " came back " + (r.error.empty() ? std::to_string(r.status) : r.error);
        return false;
    }
    std::ofstream out(dest, std::ios::binary | std::ios::trunc);
    out << r.body;
    if (!out) {
        err = "could not write " + dest.string();
        return false;
    }
    out.close();
    path = dest.string();
    return true;
}

} // namespace

// Returns the finished model's path, or "".
std::string gsq_pull(const std::string & repo, double bpw, const std::string & as, const Config & cfg, Registry & reg,
                     const Emit & emit) {
    const std::string dest_dir = loose_dir(cfg);
    std::error_code   ec;
    fs::create_directories(dest_dir, ec);

    emit(json{{"status", "looking up " + repo + " on Hugging Face"}});
    std::string               ferr;
    const std::vector<HfFile> files = hf_files(repo, &ferr);
    if (files.empty()) {
        emit(error_obj(ferr.empty() ? "no GGUF files in " + repo : ferr));
        return "";
    }
    const std::vector<HfFile> src = pick_gsq_source(files);
    if (src.empty()) {
        emit(error_obj("no high-quality build in " + repo + " to requantize from"));
        return "";
    }

    // The first shard's header names the architecture; the rest add tensors.
    const std::string first_url = hf_download_url(repo, src.front().name);
    const HttpResult  probe     = http_request(first_url, "GET", "bytes=0-33554431");
    if (probe.error.empty() && probe.status != 200 && probe.status != 206) {
        emit(error_obj("could not read the header of " + base_name(src.front().name)));
        return "";
    }
    gsq::Layout layout = gsq::layout_from(probe.body);
    if (!layout.error.empty()) {
        emit(error_obj(base_name(src.front().name) + ": " + layout.error));
        return "";
    }
    std::istringstream hin(probe.body, std::ios::binary);
    const HeaderMeta   meta = read_header_meta(hin);
    const gsq::Allocation * alloc = gsq::for_model(meta.arch, meta.embd, bpw);
    if (alloc == nullptr) {
        emit(error_obj("no GSQ-RCO allocation is published for " + (meta.arch.empty() ? "this model" : meta.arch)));
        return "";
    }
    for (size_t i = 1; i < src.size(); i++) {
        const HttpResult p = http_request(hf_download_url(repo, src[i].name), "GET", "bytes=0-33554431");
        if (!gsq::append_part(layout, p.body)) {
            emit(error_obj(base_name(src[i].name) + ": " + layout.error));
            return "";
        }
    }

    const std::string exe = gsq::quantize_exe(cfg);
    if (exe.empty()) {
        emit(error_obj("a GSQ build is assembled by llama-quantize, which is not in " +
                       fs::path(cfg.llama_bin).parent_path().string() +
                       ". Run `llmash update` to replace the runtime with one that has it."));
        return "";
    }
    std::string alloc_path, imatrix_path, cerr;
    if (!cache_file(alloc->repo, alloc->alloc, cfg, alloc_path, cerr) ||
        !cache_file(alloc->repo, alloc->imatrix, cfg, imatrix_path, cerr)) {
        emit(error_obj(cerr));
        return "";
    }
    std::ifstream     af(alloc_path, std::ios::binary);
    const std::string text((std::istreambuf_iterator<char>(af)), std::istreambuf_iterator<char>());
    const gsq::Plan   plan = gsq::plan_types(layout, gsq::parse_allocation(text));
    if (plan.total > 0 && plan.covered * 10 < plan.total * 9) {
        emit(error_obj("that allocation covers only " + std::to_string(plan.covered) + " of " +
                       std::to_string(plan.total) + " tensors here, so it was built for another architecture"));
        return "";
    }

    int64_t src_bytes = 0;
    for (const HfFile & f : src) {
        src_bytes += f.size;
    }
    emit(json{{"status", ""}});
    emit(json{{"status", "  " + alloc->name}});
    emit(json{{"status", "  one quantization type per tensor under a whole-file bit budget,"}});
    emit(json{{"status", "  searched by IST-DASLab and assembled here."}});
    emit(json{{"status", ""}});
    emit(json{{"status", "  " + pad_right("source", 12) + base_name(src.front().name)}});
    emit(json{{"status", "  " + pad_right("", 12) + human_bytes(src_bytes) +
                             (src.size() > 1 ? ", " + std::to_string(src.size()) + " shards" : "")}});
    emit(json{{"status", "  " + pad_right("allocation", 12) + base_name(alloc->alloc)}});
    emit(json{{"status", "  " + pad_right("", 12) + std::to_string(plan.covered) + " of " +
                             std::to_string(plan.total) + " tensors, " + std::to_string(plan.lines.size()) +
                             " to requantize"}});
    {
        std::vector<std::pair<std::string, int>> top(plan.by_type.begin(), plan.by_type.end());
        std::sort(top.begin(), top.end(), [](const auto & a, const auto & b) { return a.second > b.second; });
        std::string shown;
        for (size_t i = 0; i < top.size() && i < 5; i++) {
            shown += (i ? "  " : "") + top[i].first + " x" + std::to_string(top[i].second);
        }
        if (top.size() > 5) {
            shown += "  and " + std::to_string(top.size() - 5) + " more";
        }
        emit(json{{"status", "  " + pad_right("", 12) + shown}});
    }

    // Scratch space stays a fraction of the model.
    const int64_t target = (std::max)(static_cast<int64_t>(2) << 30, src_bytes / 8);
    const std::vector<gsq::Chunk> chunks = gsq::plan_chunks(layout, target);
    emit(json{{"status", "  " + pad_right("chunks", 12) + std::to_string(chunks.size()) + " of about " +
                             human_bytes(target) + ", each fetched, requantized and freed in turn"}});
    emit(json{{"status", ""}});

    const fs::path work = fs::path(cfg.root) / "cache" / "gsq" / "work";
    fs::remove_all(work, ec);
    fs::create_directories(work, ec);
    const std::string type_file = (work / "types.txt").string();
    {
        std::ofstream tf(type_file, std::ios::binary | std::ios::trunc);
        for (const std::string & l : plan.lines) {
            tf << l << "\n";
        }
    }

    const auto               t0 = std::chrono::steady_clock::now();
    std::vector<std::string> made;
    for (size_t ci = 0; ci < chunks.size(); ci++) {
        const gsq::Chunk & c   = chunks[ci];
        const std::string  raw = (work / ("part" + std::to_string(ci) + ".gguf")).string();
        const std::string  out = (work / ("part" + std::to_string(ci) + ".q.gguf")).string();
        const std::string  tag = "chunk " + std::to_string(ci + 1) + "/" + std::to_string(chunks.size());

        std::string err;
        const int64_t head_bytes = gsq::write_chunk_header(raw, layout, c, err);
        if (head_bytes < 0) {
            emit(error_obj(err));
            return "";
        }
        std::vector<RangeJob> jobs;
        for (const gsq::Piece & p : gsq::chunk_pieces(layout, c, head_bytes)) {
            jobs.push_back(RangeJob{hf_download_url(repo, src[p.part].name), p.from, p.bytes, p.into});
        }
        {
            // size it up front so every connection writes into place
            std::ofstream sz(raw, std::ios::binary | std::ios::in | std::ios::out);
        }
        fs::resize_file(raw, static_cast<uintmax_t>(head_bytes + c.bytes), ec);

        std::atomic<int64_t> seen{0};
        std::atomic<bool>    done{false};
        bool                 ok = false;
        std::thread          worker([&] {
            ok = fetch_ranges(raw, jobs, [&](int64_t n) { seen.store(n); }, err);
            done.store(true);
        });
        while (!done.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
            emit(json{{"status", "fetching " + tag},
                      {"digest", "gsq-fetch-" + std::to_string(ci)},
                      {"total", c.bytes},
                      {"completed", (std::min)(seen.load(), c.bytes)}});
        }
        worker.join();
        if (!ok) {
            emit(error_obj(err));
            return "";
        }
        emit(json{{"status", "fetching " + tag},
                  {"digest", "gsq-fetch-" + std::to_string(ci)},
                  {"total", c.bytes},
                  {"completed", c.bytes}});

        std::atomic<int> at{0}, of{0};
        std::atomic<bool> qdone{false};
        bool              qok = false;
        std::string       qerr;
        std::thread       quant([&] {
            qok = gsq::quantize(exe, raw, out, type_file, imatrix_path, alloc->fallback,
                                [&](int a, int b) { at.store(a); of.store(b); }, qerr);
            qdone.store(true);
        });
        while (!qdone.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
            if (of.load() > 0) {
                emit(json{{"status", "requantizing " + tag},
                          {"digest", "gsq-quant-" + std::to_string(ci)},
                          {"total", of.load()},
                          {"completed", at.load()}});
            }
        }
        quant.join();
        fs::remove(raw, ec); // the source chunk has done its job
        if (!qok) {
            emit(error_obj(qerr));
            return "";
        }
        emit(json{{"status", "requantizing " + tag},
                  {"digest", "gsq-quant-" + std::to_string(ci)},
                  {"total", (std::max)(of.load(), 1)},
                  {"completed", (std::max)(of.load(), 1)}});
        made.push_back(out);
    }

    std::string stem = strip_shard(stem_of(src.front().name));
    for (const std::string & q : gsq::source_preference()) {
        const std::string up = upper(q);
        for (const std::string & sep : {"-", "."}) {
            const size_t at = lower(stem).find(lower(sep + up));
            if (at != std::string::npos) {
                stem.erase(at, sep.size() + up.size());
                break;
            }
        }
    }
    const std::string dest = (fs::path(dest_dir) / (stem + "-" + gsq::quant_name(bpw) + ".gguf")).string();
    int64_t           written = 0;
    for (const std::string & m : made) {
        written += static_cast<int64_t>(fs::file_size(m, ec));
    }
    std::string aerr;
    {
        std::atomic<int64_t> seen{0};
        std::atomic<bool>    done{false};
        bool                 ok = false;
        std::thread          worker([&] {
            ok = gsq::assemble(made, dest, [&](int64_t n) { seen.store(n); }, aerr);
            done.store(true);
        });
        while (!done.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
            emit(json{{"status", "assembling"},
                      {"digest", "gsq-assemble"},
                      {"total", written},
                      {"completed", (std::min)(seen.load(), written)}});
        }
        worker.join();
        if (!ok) {
            emit(error_obj(aerr));
            return "";
        }
    }
    fs::remove_all(work, ec);

    const int64_t out_bytes = static_cast<int64_t>(fs::file_size(dest, ec));
    emit(json{{"status", ""}});
    emit(json{{"status", "  " + pad_right("built", 12) + base_name(dest)}});
    emit(json{{"status", "  " + pad_right("", 12) + human_bytes(out_bytes) + ", down from " +
                             human_bytes(src_bytes) + " (" +
                             std::to_string(100 - 100 * out_bytes / (std::max<int64_t>)(src_bytes, 1)) +
                             "% smaller)"}});
    emit(json{{"status", "  " + pad_right("", 12) + "took " + mmss(std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count()) +
                             "; nothing but the result is left on disk"}});
    emit(json{{"status", ""}});

    reg.invalidate();
    if (as.empty()) {
        emit(json{{"status", loose_name(dest) + " ready, " + human_bytes(out_bytes)}});
    }
    return dest;
}

std::string hf_pull(const std::string & repo, const std::string & quant, const std::string & as, const Config & cfg,
                    Registry & reg, const Emit & emit) {
    if (const double bpw = gsq::bpw_of_quant(quant); bpw > 0) {
        return gsq_pull(repo, bpw, as, cfg, reg, emit);
    }
    const std::string dest_dir = loose_dir(cfg);
    std::error_code   ec;
    fs::create_directories(dest_dir, ec);

    emit(json{{"status", "looking up " + repo + " on Hugging Face"}});
    std::string               err;
    const std::vector<HfFile> files = hf_files(repo, &err);
    if (!err.empty()) {
        emit(error_obj(err));
        return "";
    }
    if (files.empty()) {
        emit(error_obj("no GGUF files in " + repo));
        return "";
    }
    const std::vector<HfFile> want = pick_gguf(files, quant);
    if (want.empty()) {
        emit(error_obj("no " + quant + " build in " + repo));
        return "";
    }

    struct Job {
        HfFile      f;
        std::string save_as;
    };
    std::vector<Job> jobs;
    for (const auto & f : want) {
        jobs.push_back(Job{f, base_name(f.name)});
    }
    if (const std::optional<HfFile> proj = pick_mmproj(files)) {
        const std::string stem = strip_shard(stem_of(want.front().name));
        jobs.push_back(Job{*proj, stem + ".mmproj.gguf"});
    }

    for (const auto & j : jobs) {
        int64_t           total = j.f.size;
        const std::string dest  = (fs::path(dest_dir) / j.save_as).string();
        if (already_have(dest, total)) {
            emit(json{{"status", "already have " + j.save_as}, {"total", total}, {"completed", total}});
            continue;
        }
        const std::string tmp = dest + ".part";
        const std::string url = std::string(HF_BASE) + "/" + repo + "/resolve/main/" + j.f.name;
        if (total == 0) {
            const HttpResult h = http_request(url, "HEAD");
            if (h.content_length > 0) {
                total = h.content_length;
            }
        }
        if (total == 0) {
            emit(error_obj("could not determine the size of " + j.f.name));
            return "";
        }
        const std::string idx_path = tmp + ".idx";
        int64_t           prefix   = 0;
        if (seed_block_map(tmp, total, prefix)) {
            char gb[32];
            std::snprintf(gb, sizeof(gb), "%.1f", static_cast<double>(prefix) / static_cast<double>(1ll << 30));
            emit(json{{"status", "resuming " + j.save_as + " from " + gb + " GB"}});
        }

        std::atomic<int64_t> seen{0};
        std::atomic<bool>    finished{false};
        std::string          derr;
        bool                 ok = false;
        std::thread          worker([&] {
            ok = fetch_blocks(url, tmp, total, [&](int64_t n) { seen.store(n); }, derr);
            finished.store(true);
        });
        while (true) {
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
            if (finished.load()) {
                break;
            }
            int64_t n = seen.load();
            if (n > total) {
                n = total;
            }
            emit(json{{"status", "pulling " + j.save_as},
                      {"digest", j.save_as},
                      {"total", total},
                      {"completed", n}});
        }
        worker.join();
        if (!ok) {
            fs::remove(tmp, ec);
            fs::remove(idx_path, ec);
            emit(error_obj(derr));
            return "";
        }
        // the name goes on before the file is visible, or a list rendered in
        // between shows the file's own name
        if (!as.empty() && j.f.name == want.front().name) {
            set_alias(cfg, j.save_as, as);
        }
        fs::rename(tmp, dest, ec);
        emit(json{{"status", "pulling " + j.save_as},
                  {"digest", j.save_as},
                  {"total", total},
                  {"completed", total}});
    }

    reg.invalidate();
    const std::string first = (fs::path(dest_dir) / base_name(want.front().name)).string();
    if (read_gguf(first).arch.empty()) {
        emit(error_obj(base_name(first) + " does not read as a model"));
        return "";
    }
    int64_t pulled = 0;
    for (const auto & j : jobs) {
        const uintmax_t sz = fs::file_size(fs::path(dest_dir) / j.save_as, ec);
        if (!ec) {
            pulled += static_cast<int64_t>(sz);
        }
    }
    if (as.empty()) {
        emit(json{{"status", loose_name(first) + " ready, " + human_bytes(pulled)}});
    }
    return first;
}

bool finish_hf(const std::string & repo, const std::string & first, const std::string & as, const std::string & mtp,
               const Config & cfg, Registry & reg, const Emit & emit) {
    std::error_code ec;
    if (!as.empty()) {
        std::string err;
        if (!write_alias(cfg, base_name(first), as, err)) {
            emit(error_obj("could not record the name: " + err));
            return false;
        }
    }
    if (!mtp.empty()) {
        const std::string stem = strip_shard(stem_of(first));
        const std::string dest = (fs::path(first).parent_path() / (stem + ".mtp.gguf")).string();
        if (!fs::is_regular_file(dest, ec)) {
            const std::string url   = std::string(HF_BASE) + "/" + repo + "/resolve/main/" + mtp;
            const HttpResult  head  = http_request(url, "HEAD");
            const int64_t     total = head.content_length > 0 ? head.content_length : 0;
            if (total == 0) {
                emit(error_obj("could not determine the size of " + mtp));
                return false;
            }
            const std::string tmp = dest + ".part";
            fs::remove(tmp, ec);
            fs::remove(tmp + ".idx", ec);
            std::string derr;
            const bool  ok = fetch_blocks(url, tmp, total,
                                          [&](int64_t n) {
                                             emit(json{{"status", "pulling " + mtp},
                                                       {"digest", mtp},
                                                       {"total", total},
                                                       {"completed", n}});
                                         },
                                          derr);
            fs::remove(tmp + ".idx", ec);
            if (!ok) {
                fs::remove(tmp, ec);
                emit(error_obj(mtp + ": " + derr));
                return false;
            }
            fs::rename(tmp, dest, ec);
            emit(json{{"status", "pulling " + mtp}, {"digest", mtp}, {"total", total}, {"completed", total}});
        }
    }
    reg.invalidate();
    if (!as.empty()) {
        emit(json{{"status", as + " ready"}});
    }
    return true;
}

// ========================================================== registry pulls

std::string RegistryManifest::name() const {
    std::string n = repo + ":" + tag;
    if (host == OLLAMA_REGISTRY && starts_with(n, "library/")) {
        n = n.substr(std::strlen("library/"));
    }
    return n;
}

std::string RegistryManifest::base() const { return "https://" + host + "/v2/" + repo; }

std::string RegistryManifest::manifest_path(const Config & cfg) const {
    fs::path p = fs::path(cfg.models_root) / "manifests" / host;
    size_t   i = 0;
    while (i < repo.size()) {
        const size_t j = repo.find('/', i);
        p /= repo.substr(i, j == std::string::npos ? std::string::npos : j - i);
        if (j == std::string::npos) {
            break;
        }
        i = j + 1;
    }
    p /= tag;
    return p.string();
}

void split_ref(const std::string & ref, std::string & host, std::string & repo, std::string & tag) {
    host              = OLLAMA_REGISTRY;
    const size_t colon = ref.find(':');
    std::string  name  = colon == std::string::npos ? ref : ref.substr(0, colon);
    tag                = colon == std::string::npos ? "" : ref.substr(colon + 1);
    if (tag.empty()) {
        tag = "latest";
    }
    std::vector<std::string> parts;
    size_t                   i = 0;
    while (true) {
        const size_t j = name.find('/', i);
        parts.push_back(name.substr(i, j == std::string::npos ? std::string::npos : j - i));
        if (j == std::string::npos) {
            break;
        }
        i = j + 1;
    }
    if (parts.size() == 1) {
        repo = "library/" + parts[0];
    } else if (parts.size() == 2) {
        repo = name;
    } else {
        host = parts[0];
        repo.clear();
        for (size_t k = 1; k < parts.size(); k++) {
            if (k > 1) {
                repo += "/";
            }
            repo += parts[k];
        }
    }
}

bool fetch_manifest(const std::string & ref, RegistryManifest & out, std::string & err) {
    split_ref(ref, out.host, out.repo, out.tag);
    const HttpResult r = http_request(out.base() + "/manifests/" + out.tag, "GET", "",
                                      {"Accept: application/vnd.docker.distribution.manifest.v2+json"});
    if (!r.error.empty()) {
        err = r.error;
        return false;
    }
    out.raw = r.body;
    if (r.status != 200) {
        err = "manifest " + std::to_string(r.status) + " for " + ref;
        return false;
    }
    const json j = json::parse(out.raw, nullptr, false);
    if (j.is_discarded() || !j.is_object()) {
        err = "bad manifest for " + ref;
        return false;
    }
    const auto layers = j.find("layers");
    if (layers != j.end() && layers->is_array()) {
        for (const auto & l : *layers) {
            if (!l.is_object()) {
                continue;
            }
            RegistryLayer rl;
            rl.media_type = j_str(l, "mediaType");
            rl.digest     = j_str(l, "digest");
            rl.size       = j_int(l, "size");
            out.layers.push_back(std::move(rl));
        }
    }
    const auto cfg_layer = j.find("config");
    if (cfg_layer != j.end() && cfg_layer->is_object()) {
        RegistryLayer rl;
        rl.media_type    = j_str(*cfg_layer, "mediaType");
        rl.digest        = j_str(*cfg_layer, "digest");
        rl.size          = j_int(*cfg_layer, "size");
        out.config_layer = rl;
    }
    return true;
}

std::string blob_path(const Config & cfg, const std::string & digest) {
    std::string name = digest;
    std::replace(name.begin(), name.end(), ':', '-');
    std::error_code ec;
    const fs::path  p = fs::path(cfg.models_root) / "blobs" / name;
    if (fs::is_regular_file(p, ec)) {
        return p.string();
    }
    for (const auto & e : cfg.extra_roots) {
        const fs::path q = fs::path(e) / "blobs" / name;
        if (fs::is_regular_file(q, ec)) {
            return q.string();
        }
    }
    return p.string();
}

namespace {

// Every manifest file this install reads, the root store's copy of a name
// winning over an extra store's, matching Registry.manifestFiles.
std::vector<std::string> manifest_files(const Config & cfg) {
    std::vector<std::string> roots{cfg.models_root};
    roots.insert(roots.end(), cfg.extra_roots.begin(), cfg.extra_roots.end());

    std::vector<std::string> out, seen;
    std::error_code          ec;
    for (const auto & root : roots) {
        const fs::path dir = fs::path(root) / "manifests";
        if (!fs::is_directory(dir, ec)) {
            continue;
        }
        for (auto it = fs::recursive_directory_iterator(dir, fs::directory_options::skip_permission_denied, ec);
             it != fs::recursive_directory_iterator(); it.increment(ec)) {
            if (ec) {
                ec.clear();
                continue;
            }
            if (!it->is_regular_file(ec)) {
                continue;
            }
            const std::string rel = lower(fs::relative(it->path(), dir, ec).generic_string());
            if (ec || std::find(seen.begin(), seen.end(), rel) != seen.end()) {
                ec.clear();
                continue;
            }
            seen.push_back(rel);
            out.push_back(it->path().string());
        }
    }
    return out;
}

std::vector<std::string> manifest_digests(const std::string & path, bool & ok) {
    ok = false;
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return {};
    }
    const json j = json::parse(in, nullptr, false);
    if (j.is_discarded() || !j.is_object()) {
        return {};
    }
    ok = true;
    std::vector<std::string> out;
    const auto               layers = j.find("layers");
    if (layers != j.end() && layers->is_array()) {
        for (const auto & l : *layers) {
            if (l.is_object()) {
                out.push_back(j_str(l, "digest"));
            }
        }
    }
    const auto conf = j.find("config");
    if (conf != j.end() && conf->is_object()) {
        out.push_back(j_str(*conf, "digest"));
    }
    return out;
}

} // namespace

void remove_manifest_model(const Config & cfg, Registry & reg, const std::string & manifest_path) {
    bool                           ok    = false;
    const std::vector<std::string> gone  = manifest_digests(manifest_path, ok);
    if (!ok) {
        return;
    }
    std::error_code ec;
    const fs::path  target = fs::path(manifest_path).lexically_normal();

    std::vector<std::string> used;
    for (const auto & other : manifest_files(cfg)) {
        if (fs::path(other).lexically_normal() == target) {
            continue;
        }
        bool other_ok = false;
        for (const auto & d : manifest_digests(other, other_ok)) {
            used.push_back(d);
        }
    }
    fs::remove(manifest_path, ec);
    for (const auto & d : gone) {
        if (std::find(used.begin(), used.end(), d) == used.end()) {
            fs::remove(blob_path(cfg, d), ec);
        }
    }
    reg.invalidate();
}

namespace {

const std::regex & size_tag_junk_re() {
    static const std::regex re(R"(^(i?q\d|f16|bf16|f32|fp16|mxfp4|instruct|it|chat|text|latest))",
                               std::regex::icase);
    return re;
}

std::string normalise_id(const std::string & s) {
    std::string out;
    for (const char c : lower(s)) {
        if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9')) {
            out += c;
        }
    }
    return out;
}

// The Hugging Face GGUF repository carrying the same model as a registry
// reference: same family, same size, from a quantiser whose files load.
std::string hf_equivalent(const std::string & repo, const std::string & tag, const HeaderMeta & meta,
                          const std::string & quant) {
    std::string  family = repo;
    const size_t slash  = family.find_last_of('/');
    if (slash != std::string::npos) {
        family = family.substr(slash + 1);
    }

    std::vector<std::string> size_parts;
    size_t                   i = 0;
    while (i <= tag.size()) {
        const size_t      j    = tag.find('-', i);
        const std::string part = tag.substr(i, j == std::string::npos ? std::string::npos : j - i);
        if (!part.empty() && !std::regex_search(part, size_tag_junk_re())) {
            size_parts.push_back(part);
        }
        if (j == std::string::npos) {
            break;
        }
        i = j + 1;
    }
    std::string joined;
    for (const auto & p : size_parts) {
        joined += p;
    }
    std::string size = normalise_id(joined);
    if (size.empty()) {
        size = normalise_id(meta.size_label);
    }
    if (size.empty()) {
        return ""; // a bare family name matches every size the family comes in
    }

    static const std::regex family_digits(R"(([a-zA-Z])(\d))");
    const std::string       spaced = std::regex_replace(family, family_digits, "$1-$2");
    std::string             size_text;
    for (size_t k = 0; k < size_parts.size(); k++) {
        if (k > 0) {
            size_text += "-";
        }
        size_text += size_parts[k];
    }
    const std::vector<std::string> queries{family + " " + size_text, spaced + " " + size_text, spaced};
    const std::vector<std::pair<std::string, int>> org_rank{
        {"ggml-org", 5}, {"unsloth", 4}, {"bartowski", 3}, {"lmstudio-community", 2}};
    const char * markers[] = {"abliterat", "uncensored", "heretic", "distill",
                              "merge",     "roleplay",   "mobile",  "caption"};

    struct Scored {
        std::string id;
        int         score;
    };
    std::vector<std::string> seen;
    std::vector<Scored>      cands;
    const std::string        want_family = normalise_id(family);
    for (const auto & q : queries) {
        for (const auto & hit : hub_search(q, 40)) {
            if (std::find(seen.begin(), seen.end(), hit.id) != seen.end() || !contains(hit.id, "/")) {
                continue;
            }
            seen.push_back(hit.id);
            const std::string low = lower(hit.id);
            const std::string n   = normalise_id(hit.id);
            if (!contains(low, "gguf") || !contains(n, want_family) || !contains(n, size)) {
                continue;
            }
            bool foreign = false;
            for (const char * m : markers) {
                if (contains(low, m)) {
                    foreign = true;
                }
            }
            if (foreign) {
                continue;
            }
            const std::string org = low.substr(0, low.find('/'));
            int               score = 0;
            for (const auto & r : org_rank) {
                if (r.first == org) {
                    score = r.second * 100;
                }
            }
            if (contains(low, "-it") || contains(low, "instruct")) {
                score += 30;
            }
            if (contains(low, "qat")) {
                score -= 20;
            }
            score += static_cast<int>(std::log10(static_cast<double>(hit.downloads) + 1.0)) * 3;
            cands.push_back(Scored{hit.id, score});
        }
    }
    if (cands.empty()) {
        return "";
    }
    std::stable_sort(cands.begin(), cands.end(), [](const Scored & a, const Scored & b) { return a.score > b.score; });

    // among the well-ranked, the one that has the build asked for
    for (size_t k = 0; k < cands.size() && k < 4; k++) {
        std::string err;
        const auto  files = hf_files(cands[k].id, &err);
        if (!err.empty()) {
            continue;
        }
        for (const auto & f : files) {
            if (equal_fold(quant_tag(f.name), quant) && kind_of(f.name) == nullptr &&
                !contains(lower(f.name), "mmproj")) {
                return cands[k].id;
            }
        }
    }
    return cands.front().id;
}

} // namespace

RegistryBuild inspect_registry_build(const RegistryManifest & m) {
    RegistryBuild b;
    for (const auto & layer : m.layers) {
        if (!ends_with(layer.media_type, ".model")) {
            continue;
        }
        const HeaderMeta meta = probe_gguf_header(m.base() + "/blobs/" + layer.digest);
        b.unloadable          = unloadable_build(meta);
        b.size                = layer.size;
        b.quant               = quant_of_file_type(meta);
        // The registry serves one build per tag and publishes no list of
        // them, so the other builds of a model are the ones the Hugging Face
        // repository behind it carries.
        b.hf_repo = hf_equivalent(m.repo, m.tag, meta, b.quant);
        return b;
    }
    return b;
}

void registry_pull(const std::string & ref, const Config & cfg, Registry & reg, const Emit & emit) {
    std::string host, repo, tag;
    split_ref(ref, host, repo, tag);
    emit(json{{"status", "looking up " + repo + " on " + host}});

    RegistryManifest manifest;
    std::string      err;
    if (!fetch_manifest(ref, manifest, err)) {
        emit(error_obj(err));
        return;
    }
    const std::string name    = manifest.name();
    const std::string base    = manifest.base();
    const std::string mf_path = manifest.manifest_path(cfg);

    const RegistryBuild b = inspect_registry_build(manifest);
    if (!b.unloadable.empty()) {
        emit(json{{"status", "the registry build of " + name + " " + b.unloadable + ", which llama.cpp does not load"}});
        if (b.hf_repo.empty()) {
            emit(error_obj("no Hugging Face build of " + name +
                           " was found to take instead; pull one directly with `llmash pull hf:<org>/<repo>`"));
            return;
        }
        emit(json{{"status", "taking " + b.hf_repo + " from Hugging Face instead, as " + name}});
        const std::string first = hf_pull(b.hf_repo, b.quant, name, cfg, reg, emit);
        if (first.empty()) {
            return;
        }
        remove_manifest_model(cfg, reg, mf_path);
        finish_hf(b.hf_repo, first, name, "", cfg, reg, emit);
        return;
    }

    std::vector<RegistryLayer> layers = manifest.layers;
    if (manifest.config_layer) {
        layers.push_back(*manifest.config_layer);
    }
    std::error_code ec;
    for (const auto & layer : layers) {
        const int64_t     total = layer.size;
        const std::string dest  = blob_path(cfg, layer.digest);
        const std::string shrt  = short12(layer.digest);
        if (fs::is_regular_file(dest, ec) && static_cast<int64_t>(fs::file_size(dest, ec)) == total) {
            emit(json{{"status", "pulling " + shrt},
                      {"digest", layer.digest},
                      {"total", total},
                      {"completed", total}});
            continue;
        }
        fs::create_directories(fs::path(dest).parent_path(), ec);
        const std::string tmp = dest + ".partial";
        const std::string url = base + "/blobs/" + layer.digest;
        std::string       ferr;
        const bool        ok = fetch_blob(url, tmp, total,
                                          [&](int64_t done) {
                                       emit(json{{"status", "pulling " + shrt},
                                                 {"digest", layer.digest},
                                                 {"total", total},
                                                 {"completed", done}});
                                   },
                                          ferr);
        if (!ok) {
            fs::remove(tmp, ec);
            emit(error_obj(shrt + ": " + ferr));
            return;
        }
        fs::rename(tmp, dest, ec);
        emit(json{{"status", "pulling " + shrt}, {"digest", layer.digest}, {"total", total}, {"completed", total}});
    }
    fs::create_directories(fs::path(mf_path).parent_path(), ec);
    {
        std::ofstream out(mf_path, std::ios::binary | std::ios::trunc);
        out << manifest.raw;
    }
    reg.invalidate();
    int64_t pulled = 0;
    for (const auto & l : manifest.layers) {
        pulled += l.size;
    }
    emit(json{{"status", name + " ready, " + human_bytes(pulled)}});
}

// ======================================================== the API handlers

void run_pull(const json & body, const Config & cfg, Registry & reg, const Emit & emit) {
    const std::string ref   = first_non_empty(j_str(body, "model"), j_str(body, "name"));
    const std::string quant = j_str(body, "quant");
    const std::string as    = j_str(body, "as");
    const std::string mtp   = j_str(body, "mtp");

    for (const auto & p : hf_prefixes()) {
        if (!starts_with(ref, p)) {
            continue;
        }
        const std::string spec = ref.substr(p.size());
        const size_t      at   = spec.find('@');
        const std::string repo = at == std::string::npos ? spec : spec.substr(0, at);
        std::string       q    = at == std::string::npos ? "" : spec.substr(at + 1);
        if (!quant.empty()) {
            q = quant;
        }
        if (q.empty()) {
            q = "Q4_K_M";
        }
        const std::string first = hf_pull(repo, q, as, cfg, reg, emit);
        if (!first.empty()) {
            if (!as.empty()) {
                RegistryManifest m;
                std::string      err;
                if (fetch_manifest(as, m, err)) {
                    remove_manifest_model(cfg, reg, m.manifest_path(cfg));
                }
            }
            finish_hf(repo, first, as, mtp, cfg, reg, emit);
        }
        return;
    }

    const auto from_ollama_it = body.find("from_ollama");
    const bool from_ollama =
        from_ollama_it != body.end() && from_ollama_it->is_boolean() && from_ollama_it->get<bool>();
    for (const auto & rep : hf_replacements()) {
        if (rep.first == ref && !from_ollama) {
            const std::string first =
                hf_pull(rep.second.first, first_non_empty(quant, rep.second.second), as, cfg, reg, emit);
            if (!first.empty()) {
                finish_hf(rep.second.first, first, as, mtp, cfg, reg, emit);
            }
            return;
        }
    }
    registry_pull(ref, cfg, reg, emit);
}

json api_resolve(const std::string & ref, const Config & cfg) {
    (void) cfg;
    for (const auto & p : hf_prefixes()) {
        if (starts_with(ref, p)) {
            const std::string spec = ref.substr(p.size());
            const size_t      at   = spec.find('@');
            return json{{"source", "hf"}, {"repo", at == std::string::npos ? spec : spec.substr(0, at)}};
        }
    }
    RegistryManifest m;
    std::string      err;
    if (!fetch_manifest(ref, m, err)) {
        return error_obj(err);
    }
    const RegistryBuild b = inspect_registry_build(m);
    if (b.unloadable.empty()) {
        return json{{"source", "registry"}, {"quant", b.quant}, {"size", b.size}, {"repo", b.hf_repo}};
    }
    if (b.hf_repo.empty()) {
        return error_obj("the registry build of " + m.name() + " " + b.unloadable +
                         ", which llama.cpp does not load, and no Hugging Face build was found to take instead");
    }
    return json{{"source", "hf"}, {"repo", b.hf_repo}, {"reason", b.unloadable}, {"quant", b.quant}};
}

json api_quants(const std::string & repo_arg) {
    std::string repo = repo_arg;
    while (!repo.empty() && (repo.front() == ' ' || repo.front() == '\t')) repo.erase(repo.begin());
    while (!repo.empty() && (repo.back() == ' ' || repo.back() == '\t')) repo.pop_back();
    for (const auto & p : hf_prefixes()) {
        if (starts_with(repo, p)) {
            repo = repo.substr(p.size());
        }
    }
    const size_t at = repo.find('@');
    if (at != std::string::npos) {
        repo = repo.substr(0, at);
    }

    std::string err;
    const auto  files = hf_files(repo, &err);
    if (!err.empty()) {
        return error_obj(err);
    }
    std::vector<QuantInfo> quants = quants_of(files);

    // Assembled here rather than downloaded, so it is offered whenever an
    // allocation exists for this architecture.
    if (const std::vector<HfFile> src = pick_gsq_source(files); !src.empty() && !quants.empty()) {
        const HttpResult p = http_request(hf_download_url(repo, src.front().name), "GET", "bytes=0-4194303");
        std::istringstream in(p.body, std::ios::binary);
        const HeaderMeta   meta = read_header_meta(in);
        const double       ref  = bits_of_quant(quants.back().name);
        for (const gsq::Allocation & a : gsq::all_for_arch(meta.arch, meta.embd)) {
            int64_t size = 0;
            if (ref > 0) {
                size = static_cast<int64_t>(static_cast<double>(quants.back().size) * a.bpw / ref);
            }
            quants.push_back(QuantInfo{gsq::quant_name(a.bpw), size, 1});
        }
    }

    std::vector<QuantInfo> heads;
    for (const auto & f : files) {
        const DraftKind * k = kind_of(f.name);
        if (k != nullptr && k->name == "mtp" && ends_with(lower(f.name), ".gguf")) {
            heads.push_back(QuantInfo{f.name, f.size, 1});
        }
    }
    std::stable_sort(heads.begin(), heads.end(),
                     [](const QuantInfo & a, const QuantInfo & b) { return a.size < b.size; });

    const auto to_json = [](const std::vector<QuantInfo> & v) {
        json arr = json::array();
        for (const auto & q : v) {
            arr.push_back(json{{"name", q.name}, {"size", q.size}, {"files", q.files}});
        }
        return arr;
    };
    std::stable_sort(quants.begin(), quants.end(),
                     [](const QuantInfo & a, const QuantInfo & b) { return a.size < b.size; });
    return json{{"repo", repo},
                {"quants", to_json(quants)},
                {"mtp", to_json(heads)},
                {"vision", pick_mmproj(files).has_value()}};
}

// -------------------------------------------------------------- httplib

void handle_pull(const httplib::Request & req, httplib::Response & res, Config & cfg, Registry & reg) {
    json body = json::parse(req.body, nullptr, false);
    if (body.is_discarded() || !body.is_object()) {
        body = json::object();
    }
    res.status = 200;
    res.set_chunked_content_provider("application/x-ndjson", [body, &cfg, &reg](size_t, httplib::DataSink & sink) {
        run_pull(body, cfg, reg, [&sink](const json & ev) {
            const std::string line = ev.dump() + "\n";
            sink.write(line.data(), line.size());
        });
        sink.done();
        return true;
    });
}

void handle_quants(const httplib::Request & req, httplib::Response & res, Config & cfg, Registry & reg) {
    (void) cfg;
    (void) reg;
    const json out = api_quants(req.get_param_value("repo"));
    res.status     = out.contains("error") ? 502 : 200;
    res.set_content(out.dump(), "application/json");
}

void handle_resolve(const httplib::Request & req, httplib::Response & res, Config & cfg, Registry & reg) {
    (void) reg;
    std::string ref = req.get_param_value("model");
    while (!ref.empty() && (ref.front() == ' ' || ref.front() == '\t')) ref.erase(ref.begin());
    while (!ref.empty() && (ref.back() == ' ' || ref.back() == '\t')) ref.pop_back();
    const json out = api_resolve(ref, cfg);
    res.status     = out.contains("error") ? 404 : 200;
    res.set_content(out.dump(), "application/json");
}

} // namespace llmash
