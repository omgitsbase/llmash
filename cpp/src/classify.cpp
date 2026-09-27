#include "classify.h"

#include "api_logic.h"
#include "caps.h"
#include "gguf_io.h"
#include "log.h"
#include "pull.h"

#include <httplib.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <regex>
#include <thread>

namespace fs = std::filesystem;
using json   = nlohmann::json;
using ojson  = nlohmann::ordered_json;

namespace llmash {

namespace {

// ------------------------------------------------------------------ small pieces

std::string stem_of(const std::string & p) { return fs::path(p).stem().string(); }

std::string strip_shard(const std::string & stem) {
    static const std::regex re(R"(-\d{5}-of-\d{5}$)");
    return std::regex_replace(stem, re, "");
}

std::string sidecar(const std::string & gguf, const char * suffix) {
    const fs::path  c = fs::path(gguf).parent_path() / (strip_shard(stem_of(gguf)) + suffix);
    std::error_code ec;
    return fs::is_regular_file(c, ec) ? c.string() : std::string();
}

std::string sidecar_dest(const std::string & gguf, const char * suffix) {
    return (fs::path(gguf).parent_path() / (strip_shard(stem_of(gguf)) + suffix)).string();
}

std::string trim(const std::string & s) {
    const size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) {
        return "";
    }
    const size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

bool starts_with(const std::string & s, const std::string & p) { return s.rfind(p, 0) == 0; }
bool ends_with(const std::string & s, const std::string & p) {
    return s.size() >= p.size() && s.compare(s.size() - p.size(), p.size(), p) == 0;
}

std::string replace_all(std::string s, const std::string & from, const std::string & to) {
    for (size_t at = s.find(from); at != std::string::npos; at = s.find(from, at + to.size())) {
        s.replace(at, from.size(), to);
    }
    return s;
}

std::vector<std::string> split_lines(const std::string & s) {
    std::vector<std::string> out;
    std::string              cur;
    for (const char c : s) {
        if (c == '\n') {
            out.push_back(cur);
            cur.clear();
        } else if (c != '\r') {
            cur += c;
        }
    }
    out.push_back(cur);
    return out;
}

std::string text_of(const json & v) { return v.is_string() ? v.get<std::string>() : v.is_null() ? "" : v.dump(); }

float f16_to_f32(uint16_t h) {
    const uint32_t sign = (h >> 15) & 1, exp = (h >> 10) & 0x1f, man = h & 0x3ff;
    uint32_t       bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign << 31;
        } else {
            int      e = -1;
            uint32_t m = man;
            do {
                e++;
                m <<= 1;
            } while ((m & 0x400) == 0);
            bits = (sign << 31) | ((127 - 15 - e) << 23) | ((m & 0x3ff) << 13);
        }
    } else if (exp == 31) {
        bits = (sign << 31) | 0x7f800000 | (man << 13);
    } else {
        bits = (sign << 31) | ((exp + 127 - 15) << 23) | (man << 13);
    }
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

float gelu(float x) { return 0.5f * x * (1.0f + std::erf(x * 0.70710678118654752f)); }

void parallel_for(int64_t n, const std::function<void(int64_t, int64_t)> & fn) {
    const int threads = static_cast<int>(std::min<int64_t>(n, std::max(1u, std::min(32u, std::thread::hardware_concurrency()))));
    if (threads <= 1) {
        fn(0, n);
        return;
    }
    std::vector<std::thread> pool;
    const int64_t            per = (n + threads - 1) / threads;
    for (int t = 0; t < threads; t++) {
        const int64_t a = t * per, b = std::min(n, a + per);
        if (a < b) {
            pool.emplace_back(fn, a, b);
        }
    }
    for (auto & t : pool) {
        t.join();
    }
}

// y[L][out] = x[L][in] . W^T + b, W [out][in] row-major as PyTorch keeps a Linear
void linear(const std::vector<float> & x, int64_t L, int64_t in, const std::vector<float> & w, const std::vector<float> & b,
            int64_t out, std::vector<float> & y) {
    y.assign(static_cast<size_t>(L * out), 0.0f);
    parallel_for(L * out, [&](int64_t a, int64_t e) {
        for (int64_t t = a; t < e; t++) {
            const int64_t r  = t / out, o = t % out;
            const float * xr = x.data() + r * in;
            const float * wr = w.data() + o * in;
            float         s  = b.empty() ? 0.0f : b[static_cast<size_t>(o)];
            for (int64_t i = 0; i < in; i++) {
                s += xr[i] * wr[i];
            }
            y[static_cast<size_t>(t)] = s;
        }
    });
}

void layer_norm(const std::vector<float> & x, int64_t L, int64_t d, const std::vector<float> & w, const std::vector<float> & b,
                std::vector<float> & y, float eps = 1e-5f) {
    y.resize(x.size());
    for (int64_t r = 0; r < L; r++) {
        const float * xr   = x.data() + r * d;
        double        mean = 0;
        for (int64_t i = 0; i < d; i++) {
            mean += xr[i];
        }
        mean /= static_cast<double>(d);
        double var = 0;
        for (int64_t i = 0; i < d; i++) {
            const double c = xr[i] - mean;
            var += c * c;
        }
        var /= static_cast<double>(d);
        const float inv = 1.0f / std::sqrt(static_cast<float>(var) + eps);
        float *     yr  = y.data() + r * d;
        for (int64_t i = 0; i < d; i++) {
            yr[i] = static_cast<float>((xr[i] - mean) * inv) * w[static_cast<size_t>(i)] + (b.empty() ? 0.0f : b[static_cast<size_t>(i)]);
        }
    }
}

void softmax_inplace(float * z, int64_t n) {
    float m = z[0];
    for (int64_t i = 1; i < n; i++) {
        m = std::max(m, z[i]);
    }
    double s = 0;
    for (int64_t i = 0; i < n; i++) {
        z[i] = std::exp(z[i] - m);
        s += z[i];
    }
    for (int64_t i = 0; i < n; i++) {
        z[i] = static_cast<float>(z[i] / s);
    }
}

double round4(double v) { return std::round(v * 10000.0) / 10000.0; }

// 1 - entropy / log K: 1 with all the mass on one option, 0 when flat
double confidence(const std::vector<float> & p) {
    if (p.size() < 2) {
        return 1.0;
    }
    double ent = 0;
    for (const float v : p) {
        ent -= v * std::log(std::max<double>(v, 1e-12));
    }
    return std::max(0.0, 1.0 - ent / std::log(static_cast<double>(p.size())));
}

void py_dump(const ojson & v, std::string & out) {
    if (v.is_object()) {
        out += "{";
        bool first = true;
        for (const auto & [k, val] : v.items()) {
            if (!first) {
                out += ", ";
            }
            first = false;
            py_dump(ojson(k), out);
            out += ": ";
            py_dump(val, out);
        }
        out += "}";
    } else if (v.is_array()) {
        out += "[";
        for (size_t i = 0; i < v.size(); i++) {
            if (i) {
                out += ", ";
            }
            py_dump(v[i], out);
        }
        out += "]";
    } else if (v.is_string()) {
        out += "\"";
        for (const unsigned char c : v.get<std::string>()) {
            switch (c) {
                case '"': out += "\\\""; break;
                case '\\': out += "\\\\"; break;
                case '\n': out += "\\n"; break;
                case '\r': out += "\\r"; break;
                case '\t': out += "\\t"; break;
                case '\b': out += "\\b"; break;
                case '\f': out += "\\f"; break;
                default:
                    if (c < 0x20) {
                        char buf[8];
                        std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                        out += buf;
                    } else {
                        out += static_cast<char>(c);
                    }
            }
        }
        out += "\"";
    } else if (v.is_boolean()) {
        out += v.get<bool>() ? "true" : "false";
    } else if (v.is_null()) {
        out += "null";
    } else {
        out += v.dump();
    }
}

std::string without_mask(std::string s) {
    for (size_t at = s.find("[MASK]"); at != std::string::npos; at = s.find("[MASK]", at)) {
        s.replace(at, 6, " ");
    }
    return s;
}

}  // namespace

std::string py_dumps(const ojson & v) {
    if (v.is_string()) {
        return v.get<std::string>();
    }
    std::string s;
    py_dump(v, s);
    return s;
}

// ------------------------------------------------------------------ recipes

Recipe letters_recipe(const std::string & flavour) {
    Recipe r;
    r.kind = "letters";
    if (flavour == "decider") {
        // a state, then the question with its options in parentheses, read at the letter after "Answer: ("
        r.tmpl            = "Context:\n{state}\n\nQuestion: {question}\nOptions:\n{options}\n\nAnswer: (";
        r.option          = "({letter}) {option}";
        r.letter          = "{letter}";
        r.option_style    = "plain";
        r.isolated_levels = true;
        return r;
    }
    // the decision-function prompt: a state, the question, lettered options, read at the letter after "Answer:"
    r.tmpl         = "You are a decision function. Read the state, then answer the question by choosing exactly one option.\n\n"
                     "[State]\n{state}\n\n[Question]\n{question}\n\n[Options]\n{options}\n\nAnswer:";
    r.option       = "{letter}. {option}";
    r.letter       = " {letter}";
    r.option_style = "plain";
    return r;
}

Recipe read_recipe(const std::string & path, std::string & err) {
    Recipe r;
    err.clear();
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        err = "could not read " + path;
        return r;
    }
    const json j = json::parse(in, nullptr, false);
    if (!j.is_object()) {
        err = path + " is not JSON";
        return r;
    }
    r.kind = j.value("kind", "");
    if (r.kind == "letters") {
        r = letters_recipe(j.value("layout", "jev"));
    }
    r.source   = j.value("source", "");
    r.pipeline = j.value("pipeline_tag", "");
    if (j.contains("template") && j["template"].is_string()) {
        r.tmpl = j["template"].get<std::string>();
    }
    if (j.contains("option") && j["option"].is_string()) {
        r.option = j["option"].get<std::string>();
    }
    if (j.contains("letter") && j["letter"].is_string()) {
        r.letter = j["letter"].get<std::string>();
    }
    if (j.contains("option_style") && j["option_style"].is_string()) {
        r.option_style = j["option_style"].get<std::string>();
    }
    if (j.contains("isolated_levels") && j["isolated_levels"].is_boolean()) {
        r.isolated_levels = j["isolated_levels"].get<bool>();
    }
    if (const auto t = j.find("temperature"); t != j.end()) {
        if (t->is_number()) {
            r.temperature[""] = t->get<double>();
            r.T               = t->get<double>();
        } else if (t->is_object()) {
            for (const auto & [k, v] : t->items()) {
                if (v.is_number()) {
                    r.temperature[k] = v.get<double>();
                }
            }
            if (const auto g = r.temperature.find(""); g != r.temperature.end()) {
                r.T = g->second;
            }
        }
    }
    if (const auto c = j.find("clamp"); c != j.end() && c->is_array() && c->size() == 2) {
        r.T_lo = (*c)[0].get<double>();
        r.T_hi = (*c)[1].get<double>();
    }
    if (const auto g = j.find("groups"); g != j.end() && g->is_object()) {
        for (const auto & [k, v] : g->items()) {
            if (v.is_number()) {
                r.groups[k] = v.get<double>();
            }
        }
    }
    for (const auto & [key, into] : {std::pair<const char *, std::string *>{"yes", &r.yes}, {"no", &r.no}, {"slot", &r.slot}}) {
        if (j.contains(key) && j[key].is_string()) {
            *into = j[key].get<std::string>();
        }
    }
    if (r.kind != "head" && r.kind != "verdict" && r.kind != "letters" && r.kind != "labels") {
        err = path + " names no kind this build reads (head, verdict, letters or labels)";
    }
    return r;
}

json recipe_json(const Recipe & r) {
    ojson j;
    j["kind"]         = r.kind;
    j["source"]       = r.source;
    j["pipeline_tag"] = r.pipeline;
    if (r.kind == "letters") {
        j["template"]        = r.tmpl;
        j["option"]          = r.option;
        j["letter"]          = r.letter;
        j["option_style"]    = r.option_style;
        j["isolated_levels"] = r.isolated_levels;
        ojson t = ojson::object();
        for (const auto & [k, v] : r.temperature) {
            t[k] = v;
        }
        j["temperature"] = t;
    } else if (r.kind == "verdict") {
        j["yes"]         = r.yes;
        j["no"]          = r.no;
        j["slot"]        = r.slot;
        j["temperature"] = r.T;
        j["clamp"]       = json::array({r.T_lo, r.T_hi});
        ojson g = ojson::object();
        for (const auto & [k, v] : r.groups) {
            g[k] = v;
        }
        j["groups"] = g;
    }
    return j;
}

// ------------------------------------------------------------------ questions

bool parse_question(const json & q, Question & out, std::string & err) {
    out = Question();
    if (!q.is_object()) {
        err = "a question is an object with type, instructions and criteria";
        return false;
    }
    out.type = q.value("type", "choice");
    if (out.type == "bool" || out.type == "yes/no") {
        out.type = "noul";
    }
    if (out.type != "choice" && out.type != "score" && out.type != "noul") {
        err = "type must be choice, score or noul";
        return false;
    }
    const auto ins = q.contains("instructions") ? q.find("instructions") : q.find("question");
    if (ins != q.end() && !ins->is_null()) {
        out.text = text_of(*ins);
    }
    const auto crit = q.contains("criteria") ? q.find("criteria") : q.find("options");
    if (out.type == "choice") {
        if (crit == q.end() || (!crit->is_object() && !crit->is_array()) || crit->size() < 2) {
            err = "a choice question lists two or more options in criteria";
            return false;
        }
        if (crit->is_array()) {
            for (const auto & c : *crit) {
                out.criteria.emplace_back(text_of(c), "");
            }
        } else {
            for (const auto & [k, v] : crit->items()) {
                out.criteria.emplace_back(k, v.is_null() ? "" : text_of(v));
            }
        }
    } else if (out.type == "score") {
        if (crit == q.end() || (!crit->is_array() && !crit->is_object()) || crit->size() < 2 || crit->size() > 10) {
            err = "a score question lists its 2 to 10 levels in criteria, lowest first";
            return false;
        }
        if (crit->is_array()) {
            for (const auto & c : *crit) {
                out.criteria.emplace_back(text_of(c), "");
            }
        } else {
            std::vector<std::pair<double, std::string>> levels;
            for (const auto & [k, v] : crit->items()) {
                levels.emplace_back(std::atof(k.c_str()), text_of(v));
            }
            std::sort(levels.begin(), levels.end());
            for (const auto & l : levels) {
                out.criteria.emplace_back(l.second, "");
            }
        }
    } else {
        std::string f, t;
        if (crit != q.end() && crit->is_object()) {
            f = text_of(crit->value("false", json()));
            t = text_of(crit->value("true", json()));
        }
        if (out.text.empty() && f.empty() && t.empty()) {
            err = "a yes/no question needs instructions, or criteria describing true or false";
            return false;
        }
        out.criteria.emplace_back("false", f);
        out.criteria.emplace_back("true", t);
    }
    if (out.text.empty()) {
        if (out.type != "noul") {
            err = "a question needs instructions";
            return false;
        }
        out.text = "Which answer fits the context?";
    }
    return true;
}

std::vector<std::string> render_options(const Question & q, const std::string & style) {
    std::vector<std::string> out;
    const bool               typed = style == "typed";
    if (q.type == "choice") {
        for (const auto & [k, v] : q.criteria) {
            out.push_back(v.empty() ? k : k + ": " + v);
        }
    } else if (q.type == "score") {
        for (size_t i = 0; i < q.criteria.size(); i++) {
            out.push_back((typed ? "level " : "") + std::to_string(i) + ": " + q.criteria[i].first);
        }
    } else {
        const std::string f = q.criteria.size() > 0 ? q.criteria[0].second : "", t = q.criteria.size() > 1 ? q.criteria[1].second : "";
        if (typed) {
            out.push_back("false: " + (f.empty() ? "no, the statement does not hold" : f));
            out.push_back("true: " + (t.empty() ? "yes, the statement holds" : t));
        } else {
            out.push_back(f.empty() ? "no" : "no: " + f);
            out.push_back(t.empty() ? "yes" : "yes: " + t);
        }
    }
    return out;
}

std::vector<std::string> option_names(const Question & q) {
    std::vector<std::string> out;
    if (q.type == "choice") {
        for (const auto & c : q.criteria) {
            out.push_back(c.first);
        }
    } else if (q.type == "score") {
        for (size_t i = 0; i < q.criteria.size(); i++) {
            out.push_back(std::to_string(i));
        }
    } else {
        out = {"false", "true"};
    }
    return out;
}

bool question_line(const std::string & raw, Question & out) {
    out = Question();
    std::string line = trim(raw);
    if (line.empty() || line.size() > 2000 || starts_with(line, "/")) {
        return false;
    }
    std::string type;
    for (const auto & [prefix, t] : {std::pair<const char *, const char *>{"choice:", "choice"}, {"score:", "score"},
                                     {"noul:", "noul"}, {"yes/no:", "noul"}, {"bool:", "noul"}}) {
        if (starts_with(lower(line), prefix)) {
            type = t;
            line = trim(line.substr(std::strlen(prefix)));
            break;
        }
    }
    std::vector<std::string> opts;
    if (ends_with(line, "]")) {
        const size_t open = line.rfind('[');
        if (open == std::string::npos) {
            return false;
        }
        const std::string inside = line.substr(open + 1, line.size() - open - 2);
        const char        sep    = inside.find('|') != std::string::npos ? '|' : ',';
        std::string       cur;
        for (const char c : inside + sep) {
            if (c == sep) {
                if (const std::string o = trim(cur); !o.empty()) {
                    opts.push_back(o);
                }
                cur.clear();
            } else {
                cur += c;
            }
        }
        line = trim(line.substr(0, open));
        if (type.empty()) {
            type = "choice";
        }
    } else if (type.empty()) {
        if (!ends_with(line, "?")) {
            return false;
        }
        type = "noul";
    }
    if (line.empty()) {
        return false;
    }
    out.type = type;
    out.text = line;
    if (type == "noul") {
        out.criteria.emplace_back("false", opts.size() == 2 ? opts[0] : "");
        out.criteria.emplace_back("true", opts.size() == 2 ? opts[1] : "");
        return true;
    }
    if (opts.size() < 2 || (type == "score" && opts.size() > 10)) {
        return false;
    }
    for (const std::string & o : opts) {
        const size_t colon = o.find(": ");
        if (type == "choice" && colon != std::string::npos && colon > 0) {
            out.criteria.emplace_back(trim(o.substr(0, colon)), trim(o.substr(colon + 2)));
        } else {
            out.criteria.emplace_back(o, "");
        }
    }
    return true;
}

ParsedText parse_text(const std::string & text) {
    ParsedText                     out;
    const std::vector<std::string> lines = split_lines(text);
    std::vector<bool>              isq(lines.size(), false);
    std::vector<Question>          qs(lines.size());
    for (size_t i = 0; i < lines.size(); i++) {
        isq[i] = question_line(lines[i], qs[i]);
    }
    // the run of question lines at the end, or else at the start; a blank line ends either. A text whose own
    // first line asks something is not mistaken for a question when the questions follow it.
    size_t tail = lines.size();
    while (tail > 0 && trim(lines[tail - 1]).empty()) {
        tail--;
    }
    size_t tq = tail;
    while (tq > 0 && isq[tq - 1]) {
        tq--;
    }
    size_t head = 0;
    while (tq == tail && head < tq && isq[head]) {
        head++;
    }
    std::vector<size_t> picked;
    for (size_t i = 0; i < head; i++) {
        picked.push_back(i);
    }
    for (size_t i = tq; i < tail; i++) {
        picked.push_back(i);
    }
    std::string state;
    for (size_t i = head; i < tq; i++) {
        state += lines[i] + "\n";
    }
    out.state = trim(state);
    for (const size_t i : picked) {
        out.questions.emplace_back("q" + std::to_string(out.questions.size() + 1), qs[i]);
    }
    return out;
}

ojson questions_json(const Questions & qs) {
    ojson out;
    for (const auto & [id, q] : qs) {
        ojson spec;
        spec["type"]         = q.type;
        spec["instructions"] = q.text;
        if (q.type == "choice") {
            ojson crit;
            for (const auto & [k, v] : q.criteria) {
                crit[k] = v.empty() ? ojson() : ojson(v);
            }
            spec["criteria"] = crit;
        } else if (q.type == "score") {
            ojson crit = ojson::array();
            for (const auto & c : q.criteria) {
                crit.push_back(c.first);
            }
            spec["criteria"] = crit;
        } else {
            spec["criteria"] = ojson{{"false", q.criteria.size() > 0 ? q.criteria[0].second : ""},
                                     {"true", q.criteria.size() > 1 ? q.criteria[1].second : ""}};
        }
        out[id] = spec;
    }
    return out;
}

// ------------------------------------------------------------------ answers

json answer_json(const Question & q, const std::vector<float> & probs) {
    ojson                          a;
    const std::vector<std::string> names = option_names(q);
    a["type"]                            = q.type;
    if (q.type == "choice") {
        size_t best = 0;
        for (size_t i = 1; i < probs.size(); i++) {
            if (probs[i] > probs[best]) {
                best = i;
            }
        }
        a["choice"] = names[best];
        ojson p;
        for (size_t i = 0; i < probs.size() && i < names.size(); i++) {
            p[names[i]] = round4(probs[i]);
        }
        a["probabilities"] = p;
        a["confidence"]    = round4(confidence(probs));
    } else if (q.type == "score") {
        double s = 0;
        ojson  legend, p;
        for (size_t i = 0; i < probs.size() && i < q.criteria.size(); i++) {
            s += static_cast<double>(i) * probs[i];
            legend[std::to_string(i)] = q.criteria[i].first;
            p[std::to_string(i)]      = round4(probs[i]);
        }
        a["score"]         = round4(s);
        a["legend"]        = legend;
        a["probabilities"] = p;
        a["confidence"]    = round4(confidence(probs));
    } else {
        a["noul"]          = round4(probs.size() > 1 ? probs[1] : 0.0);
        a["probabilities"] = ojson{{"false", round4(probs.size() > 0 ? probs[0] : 1.0)}, {"true", round4(probs.size() > 1 ? probs[1] : 0.0)}};
    }
    return a;
}

namespace {

std::string pct(double p) {
    char buf[16];
    std::snprintf(buf, sizeof(buf), p >= 0.095 || p == 0 ? "%.0f%%" : "%.1f%%", p * 100.0);
    return buf;
}

std::string one_answer(const Question & q, const json & a) {
    if (!a.is_object()) {
        return "?";
    }
    const json p = a.value("probabilities", json::object());
    if (q.type == "noul") {
        const double yes = a.value("noul", 0.0);
        return yes >= 0.5 ? "yes " + pct(yes) : "no " + pct(1.0 - yes);
    }
    std::vector<std::pair<std::string, double>> rows;
    for (const auto & [k, v] : p.items()) {
        rows.emplace_back(k, v.is_number() ? v.get<double>() : 0.0);
    }
    std::string out;
    if (q.type == "score") {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%.1f of 0-%d", a.value("score", 0.0), static_cast<int>(q.criteria.size()) - 1);
        out = buf;
        for (size_t i = 0; i < q.criteria.size(); i++) {
            const auto it = std::find_if(rows.begin(), rows.end(), [&](const auto & r) { return r.first == std::to_string(i); });
            out += "  ·  " + q.criteria[i].first + " " + pct(it != rows.end() ? it->second : 0.0);
        }
        return out;
    }
    std::stable_sort(rows.begin(), rows.end(), [](const auto & x, const auto & y) { return x.second > y.second; });
    for (const auto & [k, v] : rows) {
        out += (out.empty() ? "" : "  ·  ") + k + " " + pct(v);
    }
    return out;
}

}  // namespace

std::string answers_text(const Questions & qs, const json & answers) {
    static const std::regex generic(R"(^q\d+$)");
    std::vector<std::pair<std::string, std::string>> rows;
    size_t                                           width = 0;
    for (const auto & [id, q] : qs) {
        std::string label = std::regex_match(id, generic) ? q.text : id;
        if (label.size() > 48) {
            label = label.substr(0, 45) + "...";
        }
        width = std::max(width, label.size());
        rows.emplace_back(label, one_answer(q, answers.is_object() ? answers.value(id, json()) : json()));
    }
    std::string out;
    for (const auto & [label, text] : rows) {
        out += label + std::string(width - label.size() + 2, ' ') + text + "\n";
    }
    return out;
}

// ------------------------------------------------------------------ readouts, the pure parts

std::string letters_prompt(const Recipe & r, const std::string & state, const Question & q, std::vector<std::string> & pieces) {
    static const char * letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    pieces.clear();
    const std::vector<std::string> opts = render_options(q, r.option_style);
    if (opts.size() > 26) {
        return "";
    }
    std::string lines;
    for (size_t k = 0; k < opts.size(); k++) {
        const std::string L(1, letters[k]);
        lines += (k ? "\n" : "") + replace_all(replace_all(r.option, "{letter}", L), "{option}", opts[k]);
        pieces.push_back(replace_all(r.letter, "{letter}", L));
    }
    return replace_all(replace_all(replace_all(r.tmpl, "{state}", state), "{question}", q.text), "{options}", lines);
}

std::vector<std::string> verdict_prompts(const Recipe & r, const std::string & state, const Question & q) {
    const std::vector<std::string> opts = render_options(q, "typed");
    std::string                    head = "State:\n" + state + "\n\nQuestion [" + q.type + "]: " + q.text + "\nOptions:\n";
    for (const std::string & o : opts) {
        head += "- " + o + "\n";
    }
    head += "Judge each option:\n";
    std::vector<std::string> out;
    std::string              sofar;
    for (const std::string & o : opts) {
        out.push_back(head + sofar + o + r.slot);
        sofar += o + r.slot + "\n";
    }
    return out;
}

std::vector<double> piece_logprobs(const json & completion, const std::vector<std::string> & pieces) {
    std::vector<double> out(pieces.size(), -1e9);
    const json          probs = completion.value("completion_probabilities", json::array());
    if (!probs.is_array() || probs.empty() || !probs[0].is_object()) {
        return out;
    }
    const json & first = probs[0];
    const json   top   = first.contains("top_logprobs") ? first["top_logprobs"] : first.value("top_probs", json::array());
    for (const auto & t : top) {
        if (!t.is_object()) {
            continue;
        }
        const std::string tok = t.value("token", "");
        for (size_t k = 0; k < pieces.size(); k++) {
            if (tok == pieces[k]) {
                out[k] = t.contains("logprob") ? t.value("logprob", -1e9) : std::log(std::max(t.value("prob", 0.0), 1e-300));
            }
        }
    }
    return out;
}

std::vector<float> softmax_scaled(const std::vector<double> & scores, double T) {
    std::vector<float> p(scores.size());
    if (scores.empty()) {
        return p;
    }
    const double t = T > 0 ? T : 1.0;
    for (size_t i = 0; i < scores.size(); i++) {
        p[i] = static_cast<float>(scores[i] / t);
    }
    softmax_inplace(p.data(), static_cast<int64_t>(p.size()));
    return p;
}

HeadSeq head_sequence(const std::function<std::vector<int>(const std::string &)> & tok, const std::string & state,
                      const Question & q, int max_len, int head_max_len, int cls, int sep, int mask) {
    HeadSeq s;
    s.qtype                                = q.type == "choice" ? 0 : q.type == "score" ? 1 : 2;
    std::vector<int>              head_ids = tok(q.type + " question: " + without_mask(q.text));
    std::vector<std::vector<int>> opts;
    for (const std::string & o : render_options(q, "typed")) {
        std::vector<int> ids = tok(" " + without_mask(o));
        if (ids.size() > 48) {
            ids.resize(48);
        }
        ids.insert(ids.begin(), mask);
        opts.push_back(std::move(ids));
    }
    const auto opt_total = [&] {
        size_t n = 0;
        for (const auto & o : opts) {
            n += o.size();
        }
        return static_cast<int>(n);
    };
    int budget = head_max_len - opt_total();
    if (budget < 16) {  // too many or too long options: every option shrunk evenly
        const size_t per = static_cast<size_t>(std::max(4, (head_max_len - 16) / std::max<int>(1, static_cast<int>(opts.size()))));
        for (auto & o : opts) {
            if (o.size() > per) {
                o.resize(per);
            }
        }
        budget = head_max_len - opt_total();
    }
    if (head_ids.size() > static_cast<size_t>(std::max(8, budget))) {
        head_ids.resize(static_cast<size_t>(std::max(8, budget)));
    }
    s.ids.push_back(cls);
    s.ids.insert(s.ids.end(), head_ids.begin(), head_ids.end());
    s.ids.push_back(sep);
    for (const auto & o : opts) {
        s.markers.push_back(static_cast<int>(s.ids.size()));
        s.ids.insert(s.ids.end(), o.begin(), o.end());
    }
    s.ids.push_back(sep);
    const int        room = std::max(0, max_len - static_cast<int>(s.ids.size()) - 1);
    std::vector<int> st   = tok(without_mask(state));
    if (st.size() > static_cast<size_t>(room)) {
        st.resize(static_cast<size_t>(room));
    }
    s.ids.insert(s.ids.end(), st.begin(), st.end());
    s.ids.push_back(sep);
    if (s.ids.size() > static_cast<size_t>(max_len)) {
        s.ids.resize(static_cast<size_t>(max_len));
    }
    std::vector<int> kept;
    for (const int m : s.markers) {
        if (m < max_len) {
            kept.push_back(m);
        }
    }
    s.markers = kept;
    return s;
}

std::string temp_bucket(const std::string & type, int k) {
    const std::string size = k <= 2 ? "2" : k <= 5 ? "3-5" : k <= 10 ? "6-10" : "11+";
    return type + ":" + size;
}

// ------------------------------------------------------------------ the encoder head

namespace {

void put_u32(std::string & h, uint32_t v) { h.append(reinterpret_cast<const char *>(&v), 4); }
void put_u64(std::string & h, uint64_t v) { h.append(reinterpret_cast<const char *>(&v), 8); }
void put_f32(std::string & h, float v) { h.append(reinterpret_cast<const char *>(&v), 4); }
void put_str(std::string & h, const std::string & s) {
    put_u64(h, s.size());
    h += s;
}
ggufio::KvEntry kv_string(const std::string & k, const std::string & v) {
    std::string raw;
    put_str(raw, k);
    put_u32(raw, 8);
    put_str(raw, v);
    return {k, raw};
}
ggufio::KvEntry kv_u32(const std::string & k, uint32_t v) {
    std::string raw;
    put_str(raw, k);
    put_u32(raw, 4);
    put_u32(raw, v);
    return {k, raw};
}
ggufio::KvEntry kv_f32_array(const std::string & k, const std::vector<float> & v) {
    std::string raw;
    put_str(raw, k);
    put_u32(raw, 9);
    put_u32(raw, 6);
    put_u64(raw, v.size());
    for (const float f : v) {
        put_f32(raw, f);
    }
    return {k, raw};
}

// The values back out of a raw entry: the type, then what follows the key.
struct KvView {
    uint32_t     type = 0;
    const char * p    = nullptr;
    size_t       n    = 0;
};
KvView kv_view(const ggufio::KvEntry & e) {
    KvView       v;
    const size_t at = 8 + e.key.size();
    if (e.raw.size() < at + 4) {
        return v;
    }
    std::memcpy(&v.type, e.raw.data() + at, 4);
    v.p = e.raw.data() + at + 4;
    v.n = e.raw.size() - at - 4;
    return v;
}
std::string kv_as_string(const ggufio::KvEntry & e) {
    const KvView v = kv_view(e);
    if (v.type != 8 || v.n < 8) {
        return "";
    }
    uint64_t n = 0;
    std::memcpy(&n, v.p, 8);
    return std::string(v.p + 8, std::min<size_t>(n, v.n - 8));
}
uint32_t kv_as_u32(const ggufio::KvEntry & e) {
    const KvView v = kv_view(e);
    uint32_t     x = 0;
    if ((v.type == 4 || v.type == 5) && v.n >= 4) {
        std::memcpy(&x, v.p, 4);
    }
    return x;
}
std::vector<float> kv_as_f32s(const ggufio::KvEntry & e) {
    const KvView       v = kv_view(e);
    std::vector<float> out;
    if (v.type != 9 || v.n < 12) {
        return out;
    }
    uint32_t et = 0;
    uint64_t n  = 0;
    std::memcpy(&et, v.p, 4);
    std::memcpy(&n, v.p + 4, 8);
    if (et != 6) {
        return out;
    }
    for (uint64_t i = 0; i < n && 12 + 4 * (i + 1) <= v.n; i++) {
        float f;
        std::memcpy(&f, v.p + 12 + 4 * i, 4);
        out.push_back(f);
    }
    return out;
}

// A safetensors header on the hub: the tensors, where the data starts, and the file's own metadata.
struct StTensor {
    std::string          name, dtype;
    std::vector<int64_t> shape;
    int64_t              begin = 0, end = 0;
};

bool st_header(const std::string & url, std::vector<StTensor> & out, int64_t & data_start, json & meta, std::string & err) {
    const HttpResult r = http_request(url, "GET", "bytes=0-7", {"User-Agent: llmash"});
    if (!r.error.empty() || (r.status != 200 && r.status != 206) || r.body.size() < 8) {
        err = r.error.empty() ? "could not read the checkpoint's header" : r.error;
        return false;
    }
    uint64_t n = 0;
    std::memcpy(&n, r.body.data(), 8);
    if (n == 0 || n > (64u << 20)) {
        err = "not a safetensors file";
        return false;
    }
    const HttpResult h = http_request(url, "GET", "bytes=8-" + std::to_string(7 + n), {"User-Agent: llmash"});
    if (!h.error.empty() || h.body.size() < n) {
        err = h.error.empty() ? "could not read the checkpoint's header" : h.error;
        return false;
    }
    const json doc = json::parse(h.body.substr(0, static_cast<size_t>(n)), nullptr, false);
    if (!doc.is_object()) {
        err = "the checkpoint's header is not JSON";
        return false;
    }
    data_start = static_cast<int64_t>(8 + n);
    meta       = doc.value("__metadata__", json::object());
    for (const auto & [k, v] : doc.items()) {
        if (k == "__metadata__" || !v.is_object()) {
            continue;
        }
        StTensor t;
        t.name  = k;
        t.dtype = v.value("dtype", "");
        for (const auto & d : v.value("shape", json::array())) {
            t.shape.push_back(d.get<int64_t>());
        }
        const auto off = v.value("data_offsets", json::array());
        if (off.size() == 2) {
            t.begin = off[0].get<int64_t>();
            t.end   = off[1].get<int64_t>();
        }
        out.push_back(std::move(t));
    }
    return true;
}

bool is_head_tensor(const std::string & name) { return !starts_with(name, "encoder."); }

// The head's tensors and settings, as <stem>.classifier.gguf. The checkpoint is either the whole model, whose head
// is everything outside encoder.*, or the head on its own.
bool fetch_head(const std::string & url, const std::string & gguf, const json & cfg, const std::string & source,
                const std::function<void(int64_t, int64_t)> & progress, std::string & err) {
    std::vector<StTensor> ts;
    int64_t               start = 0;
    json                  meta;
    if (!st_header(url, ts, start, meta, err)) {
        return false;
    }
    json conf = cfg.is_object() ? cfg : json::object();
    if (meta.is_object() && meta.contains("laya.config") && meta["laya.config"].is_string()) {
        if (const json m = json::parse(meta["laya.config"].get<std::string>(), nullptr, false); m.is_object()) {
            for (const auto & [k, v] : m.items()) {
                conf[k] = v;
            }
        }
    }
    std::vector<StTensor> head;
    int64_t               total = 0;
    for (const auto & t : ts) {
        if (is_head_tensor(t.name) && t.name != "temperature") {
            head.push_back(t);
            total += t.end - t.begin;
        }
    }
    if (head.empty()) {
        err = "the checkpoint carries no decision head";
        return false;
    }

    ggufio::Layout l;
    l.version = 3;
    l.kv.push_back(kv_string("general.architecture", "laya-head"));
    l.kv.push_back(kv_string("general.name", conf.value("model_name", "decision head")));
    l.kv.push_back(kv_string("classifier.source", source));
    l.kv.push_back(kv_u32("classifier.max_len", conf.value("max_len", 1024u)));
    l.kv.push_back(kv_u32("classifier.head_max_len", conf.value("head_max_len", 256u)));
    std::vector<float> temps{1.0f, 1.0f, 1.0f};
    if (conf.contains("temperature") && conf["temperature"].is_array() && conf["temperature"].size() >= 3) {
        temps.clear();
        for (const auto & t : conf["temperature"]) {
            temps.push_back(t.get<float>());
        }
    }
    l.kv.push_back(kv_f32_array("classifier.temperature", temps));
    l.kv.push_back(kv_string("classifier.temperature_by_options",
                             conf.contains("temperature_by_options") ? conf["temperature_by_options"].dump() : "{}"));
    for (const auto & t : head) {
        ggufio::TensorEntry e;
        e.name = t.name;
        // GGUF orders dimensions innermost first
        for (auto it = t.shape.rbegin(); it != t.shape.rend(); ++it) {
            e.dims.push_back(*it);
        }
        e.type  = t.dtype == "F16" ? 1u : t.dtype == "BF16" ? 30u : 0u;
        e.bytes = t.end - t.begin;
        l.tensors.push_back(e);
    }
    const std::string dest = sidecar_dest(gguf, ".classifier.gguf");
    const std::string tmp  = dest + ".part";
    if (ggufio::write_header(tmp, l, err) < 0) {
        return false;
    }
    std::ofstream out(tmp, std::ios::binary | std::ios::app);
    int64_t       done = 0;
    for (const auto & t : head) {
        const int64_t     n = t.end - t.begin;
        std::vector<char> buf(static_cast<size_t>(n));
        if (!hub_span(url, start + t.begin, n, buf.data(), err)) {
            out.close();
            std::error_code ec;
            fs::remove(tmp, ec);
            return false;
        }
        out.write(buf.data(), static_cast<std::streamsize>(n));
        const int64_t pad = (n + l.align - 1) / l.align * l.align - n;
        for (int64_t k = 0; k < pad; k++) {
            out.put('\0');
        }
        done += n;
        if (progress) {
            progress(done, total);
        }
    }
    out.close();
    std::error_code ec;
    fs::rename(tmp, dest, ec);
    if (ec) {
        err = ec.message();
        return false;
    }
    return true;
}

struct Head {
    struct Layer {
        std::vector<float> in_w, in_b, out_w, out_b, l1_w, l1_b, l2_w, l2_b, n1_w, n1_b, n2_w, n2_b;
    };
    int64_t                      d = 0, ff = 0, nhead = 0;
    std::vector<Layer>           layers;
    std::vector<float>           type_emb;                                // [3][d]
    std::vector<float>           s_ln_w, s_ln_b, s1_w, s1_b, s3_w, s3_b;  // the scorer
    std::vector<float>           a0_w, a0_b, a2_w, a2_b;                  // the act head
    std::vector<float>           temperature;                             // per question type
    std::map<std::string, float> temperature_by_options;
    int                          max_len = 1024, head_max_len = 256;
    int                          cls = -1, sep = -1, mask = -1;
};

std::vector<float> tensor_f32(std::ifstream & in, const ggufio::Layout & l, const ggufio::TensorEntry & t) {
    std::vector<float> out;
    const int64_t      n = (t.type == 1 || t.type == 30) ? t.bytes / 2 : t.bytes / 4;
    out.resize(static_cast<size_t>(n));
    in.seekg(l.file_offset(t), std::ios::beg);
    if (t.type == 0) {
        in.read(reinterpret_cast<char *>(out.data()), t.bytes);
    } else {
        std::vector<uint16_t> raw(static_cast<size_t>(n));
        in.read(reinterpret_cast<char *>(raw.data()), t.bytes);
        for (int64_t i = 0; i < n; i++) {
            if (t.type == 1) {
                out[static_cast<size_t>(i)] = f16_to_f32(raw[static_cast<size_t>(i)]);
            } else {
                const uint32_t bits = static_cast<uint32_t>(raw[static_cast<size_t>(i)]) << 16;
                std::memcpy(&out[static_cast<size_t>(i)], &bits, 4);
            }
        }
    }
    return out;
}

std::mutex                                   g_heads_mutex;
std::map<std::string, std::unique_ptr<Head>> g_heads;

const Head * load_head(const std::string & sidecar, const GGUFInfo & model, std::string & err) {
    std::lock_guard<std::mutex> lock(g_heads_mutex);
    if (const auto it = g_heads.find(sidecar); it != g_heads.end()) {
        return it->second.get();
    }
    const ggufio::Layout l = ggufio::read_layout(sidecar);
    if (!l.error.empty()) {
        err = l.error;
        return nullptr;
    }
    std::ifstream in(sidecar, std::ios::binary);
    if (!in) {
        err = "could not open " + sidecar;
        return nullptr;
    }
    auto                                               h = std::make_unique<Head>();
    std::map<std::string, const ggufio::TensorEntry *> by_name;
    for (const auto & t : l.tensors) {
        by_name[t.name] = &t;
    }
    const auto take = [&](const std::string & name, std::vector<float> & into) {
        const auto it = by_name.find(name);
        if (it == by_name.end()) {
            if (err.empty()) {
                err = "the head has no " + name;
            }
            return;
        }
        into = tensor_f32(in, l, *it->second);
    };
    take("type_emb.weight", h->type_emb);
    if (!err.empty()) {
        return nullptr;
    }
    h->d = static_cast<int64_t>(h->type_emb.size() / 3);
    for (int i = 0;; i++) {
        const std::string p = "head.layers." + std::to_string(i) + ".";
        if (by_name.find(p + "linear1.weight") == by_name.end()) {
            break;
        }
        Head::Layer L;
        take(p + "self_attn.in_proj_weight", L.in_w);
        take(p + "self_attn.in_proj_bias", L.in_b);
        take(p + "self_attn.out_proj.weight", L.out_w);
        take(p + "self_attn.out_proj.bias", L.out_b);
        take(p + "linear1.weight", L.l1_w);
        take(p + "linear1.bias", L.l1_b);
        take(p + "linear2.weight", L.l2_w);
        take(p + "linear2.bias", L.l2_b);
        take(p + "norm1.weight", L.n1_w);
        take(p + "norm1.bias", L.n1_b);
        take(p + "norm2.weight", L.n2_w);
        take(p + "norm2.bias", L.n2_b);
        h->ff = static_cast<int64_t>(L.l1_b.size());
        h->layers.push_back(std::move(L));
    }
    take("scorer.0.weight", h->s_ln_w);
    take("scorer.0.bias", h->s_ln_b);
    take("scorer.1.weight", h->s1_w);
    take("scorer.1.bias", h->s1_b);
    take("scorer.3.weight", h->s3_w);
    take("scorer.3.bias", h->s3_b);
    take("act_head.0.weight", h->a0_w);
    take("act_head.0.bias", h->a0_b);
    take("act_head.2.weight", h->a2_w);
    take("act_head.2.bias", h->a2_b);
    if (!err.empty()) {
        return nullptr;
    }
    h->nhead = std::max<int64_t>(1, h->d / 64);
    for (const auto & e : l.kv) {
        if (e.key == "classifier.max_len") {
            h->max_len = static_cast<int>(kv_as_u32(e));
        } else if (e.key == "classifier.head_max_len") {
            h->head_max_len = static_cast<int>(kv_as_u32(e));
        } else if (e.key == "classifier.temperature") {
            h->temperature = kv_as_f32s(e);
        } else if (e.key == "classifier.temperature_by_options") {
            const json m = json::parse(kv_as_string(e), nullptr, false);
            if (m.is_object()) {
                for (const auto & [k, v] : m.items()) {
                    if (v.is_number()) {
                        h->temperature_by_options[k] = v.get<float>();
                    }
                }
            }
        }
    }
    if (h->temperature.size() < 3) {
        h->temperature.assign(3, 1.0f);
    }
    h->cls  = model.cls_id >= 0 ? model.cls_id : model.bos_id;
    h->sep  = model.sep_id >= 0 ? model.sep_id : model.eos_id;
    h->mask = model.mask_id;
    if (h->cls < 0 || h->sep < 0 || h->mask < 0) {
        err = "the model's tokenizer names no [CLS], [SEP] or [MASK] token";
        return nullptr;
    }
    const Head * out = h.get();
    g_heads[sidecar] = std::move(h);
    return out;
}

struct HeadResult {
    std::vector<float> probs;  // over the options, calibrated
    float              act = 0;
};

// The head over the encoder's per-token states, one row of `rows` per token.
HeadResult run_head(const Head & head, const std::vector<std::vector<float>> & rows, const HeadSeq & seq, const std::string & type) {
    const int64_t      L = static_cast<int64_t>(rows.size()), d = head.d, nh = head.nhead, dk = d / nh;
    std::vector<float> x(static_cast<size_t>(L * d));
    for (int64_t r = 0; r < L; r++) {
        for (int64_t i = 0; i < d; i++) {
            x[static_cast<size_t>(r * d + i)] = rows[static_cast<size_t>(r)][static_cast<size_t>(i)] +
                                                head.type_emb[static_cast<size_t>(seq.qtype * d + i)];
        }
    }
    std::vector<float> y, qkv, ctx, o, f1, f2;
    for (const auto & Lr : head.layers) {
        // x = x + attention(norm1(x))
        layer_norm(x, L, d, Lr.n1_w, Lr.n1_b, y);
        linear(y, L, d, Lr.in_w, Lr.in_b, 3 * d, qkv);
        ctx.assign(static_cast<size_t>(L * d), 0.0f);
        const float scale = 1.0f / std::sqrt(static_cast<float>(dk));
        parallel_for(nh * L, [&](int64_t a, int64_t e) {
            std::vector<float> sc(static_cast<size_t>(L));
            for (int64_t t = a; t < e; t++) {
                const int64_t hh = t / L, i = t % L;
                const float * qi = qkv.data() + i * 3 * d + hh * dk;
                for (int64_t j = 0; j < L; j++) {
                    const float * kj = qkv.data() + j * 3 * d + d + hh * dk;
                    float         s  = 0;
                    for (int64_t c = 0; c < dk; c++) {
                        s += qi[c] * kj[c];
                    }
                    sc[static_cast<size_t>(j)] = s * scale;
                }
                softmax_inplace(sc.data(), L);
                float * out = ctx.data() + i * d + hh * dk;
                for (int64_t j = 0; j < L; j++) {
                    const float * vj = qkv.data() + j * 3 * d + 2 * d + hh * dk;
                    const float   p  = sc[static_cast<size_t>(j)];
                    for (int64_t c = 0; c < dk; c++) {
                        out[c] += p * vj[c];
                    }
                }
            }
        });
        linear(ctx, L, d, Lr.out_w, Lr.out_b, d, o);
        for (size_t i = 0; i < x.size(); i++) {
            x[i] += o[i];
        }
        // x = x + linear2(relu(linear1(norm2(x))))
        layer_norm(x, L, d, Lr.n2_w, Lr.n2_b, y);
        linear(y, L, d, Lr.l1_w, Lr.l1_b, head.ff, f1);
        for (float & v : f1) {
            v = std::max(0.0f, v);
        }
        linear(f1, L, head.ff, Lr.l2_w, Lr.l2_b, d, f2);
        for (size_t i = 0; i < x.size(); i++) {
            x[i] += f2[i];
        }
    }
    // each option's marker through the scorer
    const int64_t      K = static_cast<int64_t>(seq.markers.size());
    std::vector<float> m(static_cast<size_t>(K * d));
    for (int64_t k = 0; k < K; k++) {
        std::copy_n(x.data() + seq.markers[static_cast<size_t>(k)] * d, d, m.data() + k * d);
    }
    std::vector<float> ln, s1, s3;
    layer_norm(m, K, d, head.s_ln_w, head.s_ln_b, ln);
    linear(ln, K, d, head.s1_w, head.s1_b, d, s1);
    for (float & v : s1) {
        v = gelu(v);
    }
    linear(s1, K, d, head.s3_w, head.s3_b, 1, s3);
    HeadResult r;
    // the act head: the sequence's first state and a summary of the answer distribution
    std::vector<float> p = s3;
    softmax_inplace(p.data(), K);
    const float kf  = static_cast<float>(std::max<int64_t>(2, K));
    double      ent = 0;
    for (const float v : p) {
        ent -= v * std::log(std::max(v, 1e-9f));
    }
    std::vector<float> sorted = p;
    std::sort(sorted.begin(), sorted.end(), std::greater<float>());
    const float        top1 = sorted[0], top2 = sorted.size() > 1 ? sorted[1] : 0.0f;
    std::vector<float> in(x.begin(), x.begin() + d);
    in.push_back(top1);
    in.push_back(top1 - top2);
    in.push_back(static_cast<float>(ent / std::log(kf)));
    in.push_back(kf / 255.0f);
    std::vector<float> a0, a2;
    linear(in, 1, d + 4, head.a0_w, head.a0_b, static_cast<int64_t>(head.a0_b.size()), a0);
    for (float & v : a0) {
        v = gelu(v);
    }
    linear(a0, 1, static_cast<int64_t>(a0.size()), head.a2_w, head.a2_b, static_cast<int64_t>(head.a2_b.size()), a2);
    softmax_inplace(a2.data(), static_cast<int64_t>(a2.size()));
    r.act = a2[0];
    // calibrated
    const auto  it = head.temperature_by_options.find(temp_bucket(type, static_cast<int>(K)));
    const float T  = it != head.temperature_by_options.end() ? it->second : head.temperature[static_cast<size_t>(seq.qtype)];
    r.probs        = s3;
    for (float & v : r.probs) {
        v /= T;
    }
    softmax_inplace(r.probs.data(), K);
    return r;
}

// ------------------------------------------------------------------ the runtime

bool backend_post(int port, const std::string & path, const json & body, json & out, std::string & err) {
    httplib::Client cli("127.0.0.1", port);
    cli.set_connection_timeout(10, 0);
    cli.set_read_timeout(300, 0);
    cli.set_write_timeout(60, 0);
    const auto r = cli.Post(path, body.dump(), "application/json");
    if (!r) {
        err = "the runtime did not answer";
        return false;
    }
    if (r->status != 200) {
        err = "the runtime answered " + std::to_string(r->status) + ": " + r->body.substr(0, 200);
        return false;
    }
    out = json::parse(r->body, nullptr, false);
    if (out.is_discarded()) {
        err = "the runtime's answer was not JSON";
        return false;
    }
    return true;
}

struct Run {
    Instance *   inst = nullptr;
    Recipe       recipe;
    const Head * head   = nullptr;
    int64_t      tokens = 0;
    std::string  err;
};

double letters_T(const Recipe & r, const std::string & type) {
    if (const auto it = r.temperature.find(type); it != r.temperature.end()) {
        return it->second;
    }
    if (const auto it = r.temperature.find(""); it != r.temperature.end()) {
        return it->second;
    }
    return 1.0;
}

// One prompt, one token: the runtime's likeliest next tokens, with the pieces asked for read out of them.
bool next_token_logprobs(Run & run, const std::string & prompt, const std::vector<std::string> & pieces, std::vector<double> & lp) {
    const json body = {{"prompt", prompt},          {"n_predict", 1}, {"n_probs", std::max<int>(64, 4 * static_cast<int>(pieces.size()))},
                       {"temperature", 0},          {"cache_prompt", true}};
    json       out;
    if (!backend_post(run.inst->port, "/completion", body, out, run.err)) {
        return false;
    }
    run.inst->touch();
    run.tokens += out.value("tokens_evaluated", 0) + out.value("tokens_cached", 0) + 1;
    lp = piece_logprobs(out, pieces);
    return true;
}

bool letters_row(Run & run, const std::string & state, const Question & q, const std::string & type, std::vector<float> & probs) {
    std::vector<std::string> pieces;
    const std::string        prompt = letters_prompt(run.recipe, state, q, pieces);
    if (prompt.empty()) {
        run.err = "a question may have up to 26 options";
        return false;
    }
    std::vector<double> lp;
    if (!next_token_logprobs(run, prompt, pieces, lp)) {
        return false;
    }
    if (std::all_of(lp.begin(), lp.end(), [](double v) { return v <= -1e8; })) {
        run.err = "the model put none of the option letters among its likely next tokens; it may not be a decision model";
        return false;
    }
    probs = softmax_scaled(lp, letters_T(run.recipe, type));
    return true;
}

bool probs_letters(Run & run, const std::string & state, const Question & q, std::vector<float> & probs) {
    if (q.type != "score" || !run.recipe.isolated_levels) {
        return letters_row(run, state, q, q.type, probs);
    }
    // every level judged on its own, yes or no, and the yeses shared out
    std::vector<double> fit;
    for (const auto & level : q.criteria) {
        Question row;
        row.type     = "choice";
        row.text     = q.text + "\nProposed answer: " + level.first + "\nDoes the proposed answer fit?";
        row.criteria = {{"no", ""}, {"yes", ""}};
        std::vector<float> p;
        if (!letters_row(run, state, row, "score", p)) {
            return false;
        }
        fit.push_back(p.size() > 1 ? p[1] : 0.0);
    }
    double total = 0;
    for (const double f : fit) {
        total += f;
    }
    probs.clear();
    for (const double f : fit) {
        probs.push_back(static_cast<float>(total > 0 ? f / total : 1.0 / static_cast<double>(fit.size())));
    }
    return true;
}

bool probs_verdict(Run & run, const std::string & state, const Question & q, std::vector<float> & probs) {
    std::vector<double> scores;
    for (const std::string & prompt : verdict_prompts(run.recipe, state, q)) {
        std::vector<double> lp;
        if (!next_token_logprobs(run, prompt, {run.recipe.yes, run.recipe.no}, lp)) {
            return false;
        }
        if (lp[0] <= -1e8 && lp[1] <= -1e8) {
            run.err = "the model put neither yes nor no among its likely next tokens at an option's slot";
            return false;
        }
        scores.push_back(std::max(lp[0], -30.0) - std::max(lp[1], -30.0));
    }
    probs = softmax_scaled(scores, std::min(run.recipe.T_hi, std::max(run.recipe.T_lo, run.recipe.T)));
    return true;
}

bool probs_head(Run & run, const std::string & state, const Question & q, std::vector<float> & probs, float & act) {
    const int  port = run.inst->port;
    const auto tok  = [&](const std::string & text) {
        json             out;
        std::vector<int> ids;
        if (backend_post(port, "/tokenize", json{{"content", text}, {"add_special", false}}, out, run.err)) {
            for (const auto & t : out.value("tokens", json::array())) {
                if (t.is_number_integer()) {
                    ids.push_back(t.get<int>());
                }
            }
        }
        return ids;
    };
    const Head &  h   = *run.head;
    const HeadSeq seq = head_sequence(tok, state, q, h.max_len, h.head_max_len, h.cls, h.sep, h.mask);
    if (!run.err.empty()) {
        return false;
    }
    if (seq.markers.size() != render_options(q, "typed").size()) {
        run.err = "the options do not fit in " + std::to_string(h.head_max_len) + " tokens";
        return false;
    }
    json emb;
    if (!backend_post(port, "/embedding", json{{"content", seq.ids}}, emb, run.err)) {
        return false;
    }
    run.inst->touch();
    const json &                    rows_j = emb.is_array() && !emb.empty() ? emb[0]["embedding"] : emb["embedding"];
    std::vector<std::vector<float>> rows;
    for (const auto & row : rows_j) {
        rows.emplace_back(row.begin(), row.end());
    }
    if (rows.size() != seq.ids.size() || rows.empty() || static_cast<int64_t>(rows[0].size()) != h.d) {
        run.err = "the runtime returned " + std::to_string(rows.size()) + " states for " + std::to_string(seq.ids.size()) +
                  " tokens; the model has to run with --embeddings --pooling none";
        return false;
    }
    const HeadResult r = run_head(h, rows, seq, q.type);
    probs              = r.probs;
    act                = r.act;
    run.tokens += static_cast<int64_t>(seq.ids.size());
    return true;
}

bool probs_labels(Run & run, const std::string & state, const Question & q, std::vector<float> & probs) {
    json body, out;
    if (q.type == "noul") {
        // does the text answer the question: one score, through the sigmoid
        body = {{"query", q.text}, {"documents", json::array({state})}};
        if (!backend_post(run.inst->port, "/rerank", body, out, run.err)) {
            return false;
        }
        const json  results = out.value("results", json::array());
        const double s      = results.is_array() && !results.empty() ? results[0].value("relevance_score", 0.0) : 0.0;
        const float  p      = static_cast<float>(1.0 / (1.0 + std::exp(-s)));
        probs               = {1.0f - p, p};
    } else {
        // which option the text fits best: each scored against it
        const std::vector<std::string> opts = render_options(q, "plain");
        body                                = {{"query", state}, {"documents", opts}};
        if (!backend_post(run.inst->port, "/rerank", body, out, run.err)) {
            return false;
        }
        std::vector<double> scores(opts.size(), -30.0);
        for (const auto & r : out.value("results", json::array())) {
            const size_t i = r.value("index", 0);
            if (i < scores.size()) {
                scores[i] = r.value("relevance_score", 0.0);
            }
        }
        probs = softmax_scaled(scores, 1.0);
    }
    run.inst->touch();
    run.tokens += out.value("usage", json::object()).value("prompt_tokens", 0);
    return true;
}

json fail(int & status, int code, const std::string & msg) {
    status = code;
    return json{{"error", msg}};
}

}  // namespace

// ------------------------------------------------------------------ what a model is

std::string classifier_recipe_path(const std::string & gguf) { return sidecar(gguf, ".classifier.json"); }
std::string classifier_head_path(const std::string & gguf) { return sidecar(gguf, ".classifier.gguf"); }

std::string classifier_kind(const std::string & gguf, const GGUFInfo & g) {
    if (!classifier_head_path(gguf).empty()) {
        return "head";
    }
    if (const std::string p = classifier_recipe_path(gguf); !p.empty()) {
        std::string  err;
        const Recipe r = read_recipe(p, err);
        if (err.empty()) {
            return r.kind;
        }
    }
    if (g.has_cls) {
        return "labels";
    }
    return "";
}

std::vector<std::string> classifier_launch_flags(const Model & m) {
    // a sequence is one batch for an encoder, so the batch has to hold the longest one
    if (m.classifier == "head") {
        return {"--embeddings", "--pooling", "none", "-b", "2048", "-ub", "2048"};
    }
    if (m.classifier == "labels") {
        return {"--embeddings", "--pooling", "rank", "-b", "2048", "-ub", "2048"};
    }
    return {};
}

// ------------------------------------------------------------------ the run

json classify(const ojson & request, Config & cfg, Manager & mgr, Registry & reg, int & status) {
    status = 200;
    if (!request.is_object()) {
        return fail(status, 400, "the body is JSON with model, state and questions");
    }
    const std::string name = request.value("model", "");
    if (name.empty()) {
        return fail(status, 400, "model is needed");
    }
    std::string state;
    if (const auto st = request.find("state"); st != request.end()) {
        state = py_dumps(*st);
    } else if (const auto in = request.find("input"); in != request.end()) {
        state = py_dumps(*in);
    } else {
        return fail(status, 400, "state is needed: the text, or an object, the questions are about");
    }
    Questions qs;
    if (const auto q = request.find("questions"); q != request.end() && q->is_object()) {
        for (const auto & [id, spec] : q->items()) {
            Question    parsed;
            std::string qerr;
            if (!parse_question(json(spec), parsed, qerr)) {
                return fail(status, 400, "question " + id + ": " + qerr);
            }
            qs.emplace_back(id, parsed);
        }
    } else if (const auto l = request.find("labels"); l != request.end() && l->is_array() && l->size() >= 2) {
        Question q;
        q.type = "choice";
        q.text = request.value("question", std::string("Which label fits the text?"));
        for (const auto & c : *l) {
            q.criteria.emplace_back(text_of(json(c)), "");
        }
        qs.emplace_back("label", q);
    }
    if (qs.empty()) {
        return fail(status, 400, "questions is needed: {id: {type: choice|score|noul, instructions, criteria}}, or labels: [...]");
    }
    (void) reg;
    std::string err;
    const json  ka   = request.contains("keep_alive") ? json(request["keep_alive"]) : json();
    Instance *  inst = mgr.get(name, 0, parse_keep_alive(ka, cfg.keep_alive), false, err);
    if (inst == nullptr) {
        return fail(status, 404, err.empty() ? "could not load " + name : err);
    }
    Run run;
    run.inst = inst;
    const std::string kind = inst->model.classifier;
    if (const std::string p = classifier_recipe_path(inst->model.path); !p.empty()) {
        run.recipe = read_recipe(p, err);
        if (!err.empty()) {
            return fail(status, 500, err);
        }
    } else if (kind == "head" || kind == "labels") {
        run.recipe.kind = kind;
    } else {
        // a chat model asked to decide: the common layout, read at the letter
        run.recipe = letters_recipe("jev");
    }
    if (kind == "head") {
        run.head = load_head(classifier_head_path(inst->model.path), read_gguf(inst->model.path), err);
        if (run.head == nullptr) {
            return fail(status, 500, "the decision head did not load: " + err);
        }
    }
    ojson answers;
    for (const auto & [id, q] : qs) {
        std::vector<float> probs;
        float              act = -1;
        bool               ok  = false;
        if (run.recipe.kind == "head") {
            ok = probs_head(run, state, q, probs, act);
        } else if (run.recipe.kind == "verdict") {
            ok = probs_verdict(run, state, q, probs);
        } else if (run.recipe.kind == "labels") {
            ok = probs_labels(run, state, q, probs);
        } else {
            ok = probs_letters(run, state, q, probs);
        }
        if (!ok) {
            return fail(status, 502, "question " + id + ": " + run.err);
        }
        json a = answer_json(q, probs);
        if (act >= 0) {
            a["rl_agent"] = json{{"act_probability", round4(act)}};
        }
        answers[id] = a;
    }
    ojson out;
    out["model"]   = name;
    out["answers"] = answers;
    out["usage"]   = ojson{{"input_tokens", run.tokens}, {"output_tokens", 0}};
    return out;
}

void handle_classify(const httplib::Request & req, httplib::Response & res, Config & cfg, Manager & mgr, Registry & reg) {
    const ojson body = ojson::parse(req.body, nullptr, false);
    int         status = 200;
    const json  out    = body.is_discarded() ? fail(status, 400, "invalid JSON") : classify(body, cfg, mgr, reg, status);
    res.status         = status;
    res.set_content(out.dump(), "application/json");
}

namespace {

std::string message_text(const json & content) {
    if (content.is_string()) {
        return content.get<std::string>();
    }
    std::string out;
    if (content.is_array()) {
        for (const auto & part : content) {
            if (part.is_object() && part.value("type", "") == "text") {
                out += part.value("text", "");
            } else if (part.is_string()) {
                out += part.get<std::string>();
            }
        }
    }
    return out;
}

const char * const kHowToAsk =
    "is a classifier: it answers questions about a text and does not chat. Put the text first and the questions "
    "after it, one per line:\n\n"
    "  Which team should handle this? [billing, technical, sales]\n"
    "  score: How urgent is it? [not at all, slightly, very]\n"
    "  Is the customer angry?\n\n"
    "A choice lists its options in brackets, \"name: description\" describes one, score: lists ordered levels, and a "
    "question with no options is answered yes or no. The questions can also sit in the system prompt, as such lines "
    "or as JSON {\"questions\": {...}}, and POST /api/classify takes them as JSON.";

}  // namespace

bool classify_chat(const json & body, httplib::Response & res, Config & cfg, Manager & mgr, Registry & reg, const std::string & shape) {
    const std::string          name = body.value("model", "");
    const std::optional<Model> m    = reg.find(name);
    if (!m || m->classifier.empty()) {
        return false;
    }
    std::string system, user;
    if (shape == "generate") {
        system = body.value("system", "");
        user   = message_text(body.value("prompt", json()));
    } else {
        for (const auto & msg : body.value("messages", json::array())) {
            if (!msg.is_object()) {
                continue;
            }
            const std::string role = msg.value("role", "");
            if (role == "system") {
                system += message_text(msg.value("content", json())) + "\n";
            } else if (role == "user") {
                user = message_text(msg.value("content", json()));
            }
        }
    }
    Questions qs;
    if (const std::string s = trim(system); starts_with(s, "{")) {
        const ojson j = ojson::parse(s, nullptr, false);
        if (j.is_object() && j.contains("questions") && j["questions"].is_object()) {
            for (const auto & [id, spec] : j["questions"].items()) {
                Question    q;
                std::string qerr;
                if (parse_question(json(spec), q, qerr)) {
                    qs.emplace_back(id, q);
                }
            }
        }
    } else {
        qs = parse_text(system).questions;
    }
    ParsedText pu = parse_text(user);
    for (const auto & [id, q] : pu.questions) {
        qs.emplace_back("q" + std::to_string(qs.size() + 1), q);
    }
    std::string text;
    json        answers;
    int         status = 200;
    if (qs.empty()) {
        text = name + " " + kHowToAsk;
    } else if (pu.state.empty()) {
        text = "Nothing to judge: put the text before the questions.";
    } else {
        ojson req;
        req["model"]     = name;
        req["state"]     = pu.state;
        req["questions"] = questions_json(qs);
        if (body.contains("keep_alive")) {
            req["keep_alive"] = ojson(body["keep_alive"]);
        }
        const json out = classify(req, cfg, mgr, reg, status);
        if (status != 200) {
            res.status = status;
            res.set_content(out.dump(), "application/json");
            return true;
        }
        answers = out["answers"];
        text    = answers_text(qs, answers);
        while (!text.empty() && text.back() == '\n') {
            text.pop_back();
        }
    }
    const bool stream = body.value("stream", shape != "openai");
    if (shape == "openai") {
        const auto chunk = [&](const json & delta, const json & finish) {
            return json{{"id", "chatcmpl-classify"}, {"object", "chat.completion.chunk"}, {"created", static_cast<int64_t>(now_unix())},
                        {"model", name}, {"choices", json::array({json{{"index", 0}, {"delta", delta}, {"finish_reason", finish}}})}};
        };
        if (stream) {
            std::string sse = "data: " + chunk(json{{"role", "assistant"}, {"content", text}}, nullptr).dump() + "\n\n" +
                              "data: " + chunk(json::object(), "stop").dump() + "\n\n" + "data: [DONE]\n\n";
            res.status = 200;
            res.set_content(sse, "text/event-stream");
            return true;
        }
        json out = {{"id", "chatcmpl-classify"}, {"object", "chat.completion"}, {"created", static_cast<int64_t>(now_unix())}, {"model", name},
                    {"choices", json::array({json{{"index", 0}, {"message", json{{"role", "assistant"}, {"content", text}}}, {"finish_reason", "stop"}}})},
                    {"usage", json{{"prompt_tokens", 0}, {"completion_tokens", 0}, {"total_tokens", 0}}}};
        if (!answers.is_null()) {
            out["answers"] = answers;
        }
        res.status = 200;
        res.set_content(out.dump(), "application/json");
        return true;
    }
    json out;
    out["model"]      = name;
    out["created_at"] = iso(now_unix());
    if (shape == "generate") {
        out["response"] = text;
    } else {
        out["message"] = json{{"role", "assistant"}, {"content", text}};
    }
    out["done"]        = true;
    out["done_reason"] = "stop";
    if (!answers.is_null()) {
        out["answers"] = answers;
    }
    res.status = 200;
    res.set_content(out.dump() + (stream ? "\n" : ""), stream ? "application/x-ndjson" : "application/json");
    return true;
}

// ------------------------------------------------------------------ at pull

namespace {

bool has_file(const std::vector<std::string> & files, const std::string & name) {
    return std::find(files.begin(), files.end(), name) != files.end();
}

json hub_json(const std::string & repo, const std::string & file) {
    const HttpResult r = http_request(hf_download_url(repo, file), "GET", "", {"User-Agent: llmash"}, 20);
    return r.status == 200 ? json::parse(r.body, nullptr, false) : json();
}

bool write_recipe(const std::string & gguf, const Recipe & r, std::string & err) {
    const std::string path = sidecar_dest(gguf, ".classifier.json");
    std::ofstream     out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        err = "could not write " + path;
        return false;
    }
    out << recipe_json(r).dump(2) << "\n";
    return out.good();
}

}  // namespace

void install_classifier(const std::string & repo, const std::string & from, const std::string & gguf, const std::string & pipeline,
                        const std::function<void(const json &)> & emit) {
    if (pipeline != "text-classification" || !classifier_head_path(gguf).empty() || !classifier_recipe_path(gguf).empty()) {
        return;
    }
    const GGUFInfo g = read_gguf(gguf);
    // The repository, then the models it says it was made from: a GGUF repository often leaves the pieces there.
    std::vector<std::pair<std::string, std::vector<std::string>>> repos;
    repos.emplace_back(repo, hf_repo_files(repo));
    for (const std::string & o : {from, hf_base_model(repo), hf_repo_of(g)}) {
        const bool seen = std::any_of(repos.begin(), repos.end(), [&](const auto & r) { return r.first == o; });
        if (!o.empty() && !seen) {
            repos.emplace_back(o, hf_repo_files(o));
        }
    }
    const auto holder = [&](const std::string & file) {
        for (const auto & [name, files] : repos) {
            if (has_file(files, file)) {
                return name;
            }
        }
        return std::string();
    };
    Recipe r;
    r.source   = repo;
    r.pipeline = pipeline;
    std::string err;

    // a decoder read at one slot per option
    if (const std::string at = holder("readout_config.json"); !at.empty()) {
        const json c = hub_json(at, "readout_config.json");
        if (c.is_object() && c.value("format", "") == "macjev-readout-v1" && c.value("readout", "") == "verdict") {
            r.kind = "verdict";
            const json st = c.value("slot_tokens", json::object());
            r.yes         = st.value("yes", json::object()).value("text", " yes");
            r.no          = st.value("no", json::object()).value("text", " no");
            r.slot        = st.value("verdict_slot", json::object()).value("text", " ->");
            const json t  = c.value("temperatures", json::object());
            r.T           = t.value("global", 1.0);
            if (const auto cl = t.find("clamp"); cl != t.end() && cl->is_array() && cl->size() == 2) {
                r.T_lo = (*cl)[0].get<double>();
                r.T_hi = (*cl)[1].get<double>();
            }
            const json groups = t.value("groups", json::object());
            for (const auto & [k, v] : groups.items()) {
                if (v.is_object() && v.contains("T")) {
                    r.groups[k] = v["T"].get<double>();
                }
            }
            if (!write_recipe(gguf, r, err)) {
                emit(json{{"status", "could not record how to read it: " + err}});
            } else {
                emit(json{{"status", "a classifier: its answers are read at one slot per option"}});
            }
            return;
        }
    }

    // a decoder read at the answer letter
    if (const std::string at = holder("decider_config.json"); !at.empty()) {
        const json c = hub_json(at, "decider_config.json");
        r            = letters_recipe("decider");
        r.source     = repo;
        r.pipeline   = pipeline;
        if (c.is_object()) {
            if (c.contains("temperature") && c["temperature"].is_number()) {
                r.temperature[""] = c["temperature"].get<double>();
            }
            const json by_type = c.value("temperature_by_type", json::object());
            for (const auto & [k, v] : by_type.items()) {
                if (v.is_number()) {
                    r.temperature[k] = v.get<double>();
                }
            }
            r.isolated_levels = c.value("isolated_levels", true);
        }
        if (!write_recipe(gguf, r, err)) {
            emit(json{{"status", "could not record how to read it: " + err}});
        } else {
            emit(json{{"status", "a classifier: its answers are read at the answer letter"}});
        }
        return;
    }

    // a scorer inside the GGUF
    if (g.has_cls) {
        r.kind = "labels";
        write_recipe(gguf, r, err);
        emit(json{{"status", "a classifier: it scores a text against a question"}});
        return;
    }

    // an encoder with a decision head published beside it, or in the checkpoint it came from
    if (is_embedding(g)) {
        std::string head_repo, head_file;
        for (const auto & [name, files] : repos) {
            for (const std::string & f : files) {
                if (ends_with(f, ".safetensors") && lower(f).find("head") != std::string::npos) {
                    head_repo = name;
                    head_file = f;
                }
            }
            if (!head_file.empty()) {
                break;
            }
        }
        for (const auto & [name, files] : repos) {
            if (!head_file.empty()) {
                break;
            }
            if (!has_file(files, "model.safetensors")) {
                continue;
            }
            std::vector<StTensor> ts;
            int64_t               start = 0;
            json                  meta;
            if (!st_header(hf_download_url(name, "model.safetensors"), ts, start, meta, err)) {
                emit(json{{"status", name + "/model.safetensors: " + err}});
                continue;
            }
            const bool scorer = std::any_of(ts.begin(), ts.end(), [](const StTensor & t) { return t.name == "scorer.0.weight"; });
            const bool types  = std::any_of(ts.begin(), ts.end(), [](const StTensor & t) { return t.name == "type_emb.weight"; });
            if (scorer && types) {
                head_repo = name;
                head_file = "model.safetensors";
            }
        }
        if (head_file.empty()) {
            emit(json{{"status", "tagged text-classification on the hub, but nothing published beside it reads answers; it runs as an embedding model"}});
            return;
        }
        json cfg;
        if (const std::string at = holder("rl_agent_config.json"); !at.empty()) {
            cfg = hub_json(at, "rl_agent_config.json");
        }
        emit(json{{"status", "taking the decision head from " + head_repo}});
        const bool ok = fetch_head(hf_download_url(head_repo, head_file), gguf, cfg, head_repo,
                                   [&](int64_t done, int64_t total) {
                                       emit(json{{"status", "pulling the decision head"}, {"digest", "head"}, {"total", total}, {"completed", done}});
                                   },
                                   err);
        if (!ok) {
            emit(json{{"status", "the decision head could not be taken: " + err}});
            return;
        }
        r.kind = "head";
        write_recipe(gguf, r, err);
        emit(json{{"status", "a classifier: its answers come from the decision head"}});
        return;
    }

    // a decoder with only the tag: the common layout, read at the letter
    r          = letters_recipe("jev");
    r.source   = repo;
    r.pipeline = pipeline;
    if (!write_recipe(gguf, r, err)) {
        emit(json{{"status", "could not record how to read it: " + err}});
    } else {
        emit(json{{"status", "a classifier: its answers are read at the answer letter, in the common layout (edit " +
                                 fs::path(sidecar_dest(gguf, ".classifier.json")).filename().string() + " for another)"}});
    }
}

bool ensure_classifier(const Model & m, const Config & cfg, Registry & reg) {
    (void) cfg;
    if (!m.classifier.empty() || !m.manifest.empty() || m.incomplete || m.path.empty()) {
        return false;
    }
    const std::string dir  = fs::path(m.path).parent_path().string();
    const std::string file = fs::path(m.path).filename().string();
    SourceRecord      src  = read_source(dir, file);
    if (!src.checked) {
        const GGUFInfo g = read_gguf(m.path);
        if (src.repo.empty()) {
            src.repo = hf_repo_of(g);
        }
        std::string tag = src.repo.empty() ? "" : hf_pipeline_tag(src.repo);
        if (tag != "text-classification") {
            // the header names nothing, or only the model it was built from: the file itself, on the hub
            if (const std::string holder = hf_find_repo_by_file(file, static_cast<int64_t>(m.size)); !holder.empty()) {
                const std::string t = hf_pipeline_tag(holder);
                if (t == "text-classification" || src.repo.empty()) {
                    src.repo = holder;
                    tag      = t;
                }
            }
        }
        if (src.from.empty()) {
            src.from = src.repo;
        }
        src.pipeline_tag = tag;
        src.checked      = true;
        write_source(dir, file, src);
    }
    if (src.pipeline_tag != "text-classification" || src.repo.empty()) {
        return false;
    }
    log_line(m.name + ": tagged text-classification on the hub, taking what reads its answers from " + src.repo);
    install_classifier(src.repo, src.from, m.path, src.pipeline_tag, [&](const json & j) {
        if (j.contains("status") && j["status"].is_string() && !j.contains("completed")) {
            log_line(m.name + ": " + j["status"].get<std::string>());
        }
    });
    reg.invalidate();
    return !classifier_kind(m.path, read_gguf(m.path)).empty();
}

}  // namespace llmash
