#include "caps.h"

#include "pull.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <regex>
#include <set>

namespace fs = std::filesystem;
using json   = nlohmann::json;

namespace llmash {
namespace {

std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

const std::map<std::string, std::vector<std::string>> & pipeline_caps() {
    static const std::map<std::string, std::vector<std::string>> m = {
        {"any-to-any", {"vision", "audio", "video"}},
        {"image-text-to-text", {"vision", "video"}},
        {"video-text-to-text", {"vision", "video"}},
        {"visual-question-answering", {"vision"}},
        {"image-to-text", {"vision"}},
        {"audio-text-to-text", {"audio"}},
        {"automatic-speech-recognition", {"audio"}},
        {"audio-classification", {"audio"}},
        {"text-generation", {}},
        {"text2text-generation", {}},
    };
    return m;
}

bool is_embedding(const GGUFInfo & g) {
    static const std::set<std::string> embed_arch = {"bert",   "nomic-bert", "nomic-bert-moe", "jina-bert-v2",
                                                     "xlm-roberta", "mpnet",  "gte",           "t5encoder"};
    return g.has_pooling || embed_arch.count(lower(g.arch)) != 0;
}

std::mutex                                        g_mem_mu;
std::map<std::string, std::pair<std::vector<std::string>, bool>> g_mem;

} // namespace

std::vector<std::string> projector_caps(const std::string & mmproj_path) {
    const GGUFInfo g = read_gguf(mmproj_path);
    if (!g.ok) {
        return {"vision"};
    }
    std::vector<std::string> caps;
    if (g.has_vision_encoder) {
        caps.push_back("vision");
    }
    if (g.has_audio_encoder) {
        caps.push_back("audio");
    }
    return caps.empty() ? std::vector<std::string>{"vision"} : caps;
}

std::string hf_repo_of(const GGUFInfo & g) {
    static const std::regex re(R"(huggingface\.co/([^/\s]+/[^/\s]+))");
    for (const std::string & url : {g.base_repo_url, g.repo}) {
        std::smatch m;
        if (!url.empty() && std::regex_search(url, m, re)) {
            std::string repo = m[1].str();
            while (!repo.empty() && repo.back() == '/') {
                repo.pop_back();
            }
            return repo;
        }
    }
    if (!g.base_org.empty() && !g.base_name.empty()) {
        std::string name = g.base_name;
        std::replace(name.begin(), name.end(), ' ', '-');
        return g.base_org + "/" + name;
    }
    return "";
}

std::vector<std::string> hf_caps(const std::string & repo, const std::string & cache_dir, bool & known) {
    known = false;
    if (repo.empty()) {
        return {};
    }
    {
        std::lock_guard<std::mutex> lock(g_mem_mu);
        const auto                  it = g_mem.find(repo);
        if (it != g_mem.end()) {
            known = it->second.second;
            return it->second.first;
        }
    }

    const fs::path cache = fs::path(cache_dir) / "hf_caps.json";
    json           disk  = json::object();
    {
        std::ifstream in(cache, std::ios::binary);
        if (in) {
            json j = json::parse(in, nullptr, false);
            if (j.is_object()) {
                disk = j;
            }
        }
    }
    if (disk.contains(repo)) {
        std::vector<std::string> caps;
        const json &             v = disk[repo];
        known                      = !v.is_null();
        if (v.is_array()) {
            for (const auto & c : v) {
                if (c.is_string()) {
                    caps.push_back(c.get<std::string>());
                }
            }
        }
        std::lock_guard<std::mutex> lock(g_mem_mu);
        g_mem[repo] = {caps, known};
        return caps;
    }

    // Short timeout: this runs inside the first `show` of a newly pulled
    // model, and a slow hub should cost a beat, not six seconds.
    const HttpResult      r = http_request("https://huggingface.co/api/models/" + repo, "GET", "", {}, 2);
    std::vector<std::string> caps;
    bool                     settled = false;
    if (r.status == 200) {
        const json d = json::parse(r.body, nullptr, false);
        if (d.is_object()) {
            std::set<std::string> tags;
            if (d.contains("tags") && d["tags"].is_array()) {
                for (const auto & t : d["tags"]) {
                    if (t.is_string()) {
                        tags.insert(lower(t.get<std::string>()));
                    }
                }
            }
            const std::string pipeline = lower(d.value("pipeline_tag", std::string()));
            std::set<std::string> set;
            for (const auto & [key, val] : pipeline_caps()) {
                if (pipeline == key || tags.count(key) != 0) {
                    set.insert(val.begin(), val.end());
                }
            }
            known   = !set.empty() || pipeline_caps().count(pipeline) != 0;
            settled = true;
            caps.assign(set.begin(), set.end());
        }
    } else if (r.status == 401 || r.status == 403 || r.status == 404 || r.status == 410) {
        settled = true;
    }

    {
        std::lock_guard<std::mutex> lock(g_mem_mu);
        g_mem[repo] = {caps, known};
    }
    if (settled) {
        disk[repo] = known ? json(caps) : json(nullptr);
        std::error_code ec;
        fs::create_directories(cache_dir, ec);
        std::ofstream out(cache, std::ios::binary | std::ios::trunc);
        if (out) {
            out << disk.dump(1, ' ') << "\n";
        }
    }
    return caps;
}

std::vector<std::string> caps_for(const GGUFInfo & g, const std::string & projector,
                                  const std::string & cache_dir) {
    if (is_embedding(g)) {
        return {"embedding"};
    }
    std::vector<std::string> caps{"completion"};

    std::set<std::string> installed;
    if (!projector.empty()) {
        for (const std::string & c : projector_caps(projector)) {
            installed.insert(c);
        }
    }
    // Only a model with a sidecar can gain from the hub's answer: with
    // nothing installed the intersection is empty either way.
    bool                     known = false;
    std::vector<std::string> published;
    if (!installed.empty()) {
        published = hf_caps(hf_repo_of(g), cache_dir, known);
    }

    std::vector<std::string> extra;
    if (!known) {
        extra.assign(installed.begin(), installed.end());
    } else {
        for (const std::string & c : published) {
            if (installed.count(c) != 0) {
                extra.push_back(c);
            }
        }
        const bool has_video = std::find(published.begin(), published.end(), "video") != published.end();
        if (has_video && installed.count("vision") != 0 &&
            std::find(extra.begin(), extra.end(), "video") == extra.end()) {
            extra.push_back("video");
        }
    }
    std::sort(extra.begin(), extra.end());
    caps.insert(caps.end(), extra.begin(), extra.end());

    const std::string tl  = lower(g.chat_template);
    const bool        oss = g.arch == "gpt-oss";
    if (oss || tl.find("tool") != std::string::npos || tl.find("function") != std::string::npos) {
        caps.push_back("tools");
    }
    if (oss || tl.find("think") != std::string::npos || tl.find("reason") != std::string::npos) {
        caps.push_back("thinking");
    }
    return caps;
}

} // namespace llmash
