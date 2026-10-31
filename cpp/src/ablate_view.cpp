#include "ablate_view.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace llmash {
namespace {

const char * const kSpin[] = {"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"};
const char * const kSpark[] = {"▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"};

std::string repeat(const char * unit, int n) {
    std::string s;
    for (int i = 0; i < n; i++) {
        s += unit;
    }
    return s;
}

// columns a string takes on screen: one a code point, none for an escape sequence
int columns(const std::string & s) {
    int n = 0;
    for (size_t i = 0; i < s.size(); i++) {
        const unsigned char c = static_cast<unsigned char>(s[i]);
        if (c == 0x1b && i + 1 < s.size() && s[i + 1] == '[') {
            for (i += 2; i < s.size() && (s[i] < 0x40 || s[i] > 0x7e); i++) {
            }
            continue;
        }
        if ((c & 0xc0) != 0x80) {
            n++;
        }
    }
    return n;
}

std::string clock(double s) {
    const long long t = std::llround(std::max(0.0, s));
    char            buf[32];
    if (t >= 3600) {
        std::snprintf(buf, sizeof(buf), "%lldh%02lldm", t / 3600, (t % 3600) / 60);
    } else if (t >= 60) {
        std::snprintf(buf, sizeof(buf), "%lldm%02llds", t / 60, t % 60);
    } else {
        std::snprintf(buf, sizeof(buf), "%llds", t);
    }
    return buf;
}

std::string fixed(double v, int digits) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.*f", digits, v);
    return buf;
}

}  // namespace

std::string ablate_field(const std::string & s, const char * key) {
    const size_t p = s.find(key);
    if (p == std::string::npos) {
        return "";
    }
    size_t a = p + std::strlen(key), b = a;
    while (b < s.size() && s[b] != ' ') {
        b++;
    }
    return s.substr(a, b - a);
}

void AblateView::feed(const std::string & s) {
    const auto num = [&](const char * key) { return std::atof(ablate_field(s, key).c_str()); };
    if (s.rfind("capture ", 0) == 0) {
        stage     = kDirection;
        cap_done  = static_cast<size_t>(num("done="));
        cap_total = static_cast<size_t>(num("total="));
    } else if (s.rfind("prompts ", 0) == 0) {
        stage = kDirection;
    } else if (s.rfind("direction layer=", 0) == 0) {
        layer      = static_cast<int>(num("layer="));
        n_layer    = static_cast<int>(num("of="));
        separation = num("separation=");
        cap_done   = cap_total;
        stage      = kBaseline;
    } else if (s.rfind("baseline ", 0) == 0) {
        stage = kSearch;
    } else if (s.rfind("trial ", 0) == 0) {
        stage  = kSearch;
        passes = static_cast<int>(num("n="));
        kl.push_back(num("kl="));
        best_kl = num("best_kl=");
    } else if (s.rfind("scan ", 0) == 0 || s.rfind("markers ", 0) == 0 || s.rfind("wrote ", 0) == 0) {
        stage = kCheck;
    } else if (s.rfind("done ", 0) == 0) {
        stage   = kDone;
        best_kl = num("kl=");
        seconds = num("seconds=");
    }
}

std::vector<std::string> AblateView::rows(int width, int frame, double elapsed, bool color) const {
    const char * dim   = color ? "\x1b[2m" : "";
    const char * bold  = color ? "\x1b[1m" : "";
    const char * green = color ? "\x1b[38;5;42m" : "";
    const char * cyan  = color ? "\x1b[38;5;44m" : "";
    const char * reset = color ? "\x1b[0m" : "";
    const std::string spin = std::string(cyan) + kSpin[static_cast<size_t>(frame) % 10] + reset;
    const std::string tick = std::string(green) + "✓" + reset;

    std::vector<std::string> out;
    const double took = stage == kDone && seconds > 0 ? seconds : elapsed;
    out.push_back("  " + (stage == kDone ? tick + " " + bold + "abliterated " : std::string(bold) + "abliterating ") + model +
                  reset + "  " + dim + clock(took) + reset);

    // a stage's row: its mark (done, running, waiting), its name, what it has to show
    const auto row = [&](int s, const char * name, const std::string & body) {
        const std::string mark = stage > s ? tick : stage == s ? spin : std::string(dim) + "·" + reset;
        std::string       name_col = name;
        name_col.resize(12, ' ');
        std::string line = "  " + mark + " " + (stage >= s ? std::string() : std::string(dim)) + name_col + reset + body;
        out.push_back(line);
    };
    const auto bar = [&](double frac, int cells) {
        const int n = std::clamp(static_cast<int>(std::lround(frac * cells)), 0, cells);
        return std::string(dim) + "▕" + reset + green + repeat("█", n) + reset + std::string(static_cast<size_t>(cells - n), ' ') +
               dim + "▏" + reset;
    };

    row(kLoad, "load", stage > kLoad ? std::string(dim) + "on the card" + reset : "");

    std::string dir;
    if (layer >= 0 && n_layer > 0) {
        // the model's layers left to right, the one the direction was read from lit
        const int cells = std::clamp(width - 64, 8, std::min(n_layer, 64));
        const int lit   = std::min(cells - 1, layer * cells / n_layer);
        const std::string strip = std::string(dim) + repeat("▁", lit) + reset + green + "█" + reset + dim +
                                  repeat("▁", cells - 1 - lit) + reset;
        dir = std::to_string(cap_total) + " prompts  " + strip + "  layer " + std::to_string(layer) + " of " +
              std::to_string(n_layer) + std::string(dim) + "  separation " + fixed(separation, 2) + reset;
    } else if (cap_total > 0) {
        dir = bar(static_cast<double>(cap_done) / static_cast<double>(cap_total), 20) + " " + std::to_string(cap_done) + "/" +
              std::to_string(cap_total) + " prompts";
    }
    row(kDirection, "direction", dir);

    row(kBaseline, "baseline", stage > kBaseline ? std::string(dim) + "the untouched model's answers" + reset : "");

    std::string search;
    if (!kl.empty()) {
        // each pass's KL on its own scale, the kept one lit
        const int    show = std::clamp(width - 70, 8, 32);
        const size_t from = kl.size() > static_cast<size_t>(show) ? kl.size() - static_cast<size_t>(show) : 0;
        double lo = kl[from], hi = kl[from];
        for (size_t i = from; i < kl.size(); i++) {
            lo = std::min(lo, kl[i]);
            hi = std::max(hi, kl[i]);
        }
        std::string spark;
        bool        lit = false;
        for (size_t i = from; i < kl.size(); i++) {
            const int  level = hi > lo ? static_cast<int>(std::lround((kl[i] - lo) / (hi - lo) * 7.0)) : 3;
            const bool best  = !lit && best_kl >= 0 && std::fabs(kl[i] - best_kl) < 5e-5;
            lit              = lit || best;
            spark += (best ? std::string(green) : std::string(cyan)) + kSpark[level] + reset;
        }
        const int n = std::max(passes, static_cast<int>(kl.size()));
        search = (stage == kSearch ? bar(static_cast<double>(kl.size()) / std::max(1, n), 16) + " " : std::string()) + "pass " +
                 std::to_string(kl.size()) + "/" + std::to_string(n) + "  KL " + spark +
                 (best_kl >= 0 ? std::string(dim) + "  best " + reset + green + fixed(best_kl, 3) + reset : std::string());
    }
    row(kSearch, "search", search);

    row(kCheck, "check", stage > kCheck ? std::string(dim) + "vector written, applied on every load" + reset : "");

    // a row wider than the terminal would wrap and break the redraw: it loses its tail, escapes kept whole
    for (std::string & line : out) {
        if (columns(line) <= width) {
            continue;
        }
        std::string kept;
        int         cols = 0;
        for (size_t i = 0; i < line.size();) {
            const unsigned char c = static_cast<unsigned char>(line[i]);
            if (c == 0x1b && i + 1 < line.size() && line[i + 1] == '[') {
                size_t e = i + 2;
                while (e < line.size() && (line[e] < 0x40 || line[e] > 0x7e)) {
                    e++;
                }
                kept += line.substr(i, e + 1 - i);
                i = e + 1;
                continue;
            }
            size_t e = i + 1;
            while (e < line.size() && (static_cast<unsigned char>(line[e]) & 0xc0) == 0x80) {
                e++;
            }
            if (cols < width) {
                kept += line.substr(i, e - i);
                cols++;
            }
            i = e;
        }
        line = kept + reset;
    }
    return out;
}

}  // namespace llmash
