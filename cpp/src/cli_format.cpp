#include "cli_format.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <sstream>

namespace llmash {

namespace {

// Decodes one UTF-8 code point starting at i, advancing i past it. Returns
// U+FFFD for a malformed sequence so a bad byte still counts as one column.
uint32_t next_rune(const std::string & s, size_t & i) {
    const unsigned char c0 = static_cast<unsigned char>(s[i]);
    int                 n  = 0;
    uint32_t            cp = 0;
    if (c0 < 0x80) {
        i += 1;
        return c0;
    } else if ((c0 & 0xE0) == 0xC0) {
        n  = 1;
        cp = c0 & 0x1F;
    } else if ((c0 & 0xF0) == 0xE0) {
        n  = 2;
        cp = c0 & 0x0F;
    } else if ((c0 & 0xF8) == 0xF0) {
        n  = 3;
        cp = c0 & 0x07;
    } else {
        i += 1;
        return 0xFFFD;
    }
    size_t j = i + 1;
    for (int k = 0; k < n; k++) {
        if (j >= s.size() || (static_cast<unsigned char>(s[j]) & 0xC0) != 0x80) {
            i += 1;
            return 0xFFFD;
        }
        cp = (cp << 6) | (static_cast<unsigned char>(s[j]) & 0x3F);
        j++;
    }
    i = j;
    return cp;
}

bool is_combining(uint32_t r) {
    // The Mn/Me ranges table.go's unicode.Is checks reduce to, for the
    // combining marks actually seen in model names and status text.
    return (r >= 0x0300 && r <= 0x036F) || (r >= 0x1AB0 && r <= 0x1AFF) ||
           (r >= 0x1DC0 && r <= 0x1DFF) || (r >= 0x20D0 && r <= 0x20FF) ||
           (r >= 0xFE20 && r <= 0xFE2F);
}

bool is_wide(uint32_t r) {
    if (r < 0x1100) {
        return false;
    }
    return r <= 0x115f || r == 0x2329 || r == 0x232a ||
           (r >= 0x2e80 && r <= 0xa4cf && r != 0x303f) ||
           (r >= 0xac00 && r <= 0xd7a3) ||
           (r >= 0xf900 && r <= 0xfaff) ||
           (r >= 0xfe30 && r <= 0xfe6f) ||
           (r >= 0xff00 && r <= 0xff60) ||
           (r >= 0xffe0 && r <= 0xffe6) ||
           (r >= 0x1f300 && r <= 0x1f64f) ||
           (r >= 0x1f900 && r <= 0x1f9ff) ||
           (r >= 0x20000 && r <= 0x3fffd);
}

} // namespace

int disp_width(const std::string & s) {
    int    n = 0;
    size_t i = 0;
    while (i < s.size()) {
        const uint32_t r = next_rune(s, i);
        if (r == 0) {
            continue;
        }
        if (is_combining(r)) {
            continue;
        }
        n += is_wide(r) ? 2 : 1;
    }
    return n;
}

std::vector<std::string> split_lines(const std::string & s) {
    std::vector<std::string> out;
    std::string              cur;
    for (const char c : s) {
        if (c == '\n') {
            out.push_back(cur);
            cur.clear();
        } else {
            cur.push_back(c);
        }
    }
    out.push_back(cur);
    return out;
}

std::vector<std::string> split_fields(const std::string & s) {
    std::vector<std::string> out;
    std::string              cur;
    for (const char c : s) {
        if (std::isspace(static_cast<unsigned char>(c))) {
            if (!cur.empty()) {
                out.push_back(cur);
                cur.clear();
            }
        } else {
            cur.push_back(c);
        }
    }
    if (!cur.empty()) {
        out.push_back(cur);
    }
    return out;
}

std::string join(const std::vector<std::string> & v, const std::string & sep) {
    std::string out;
    for (size_t i = 0; i < v.size(); i++) {
        if (i > 0) {
            out += sep;
        }
        out += v[i];
    }
    return out;
}

std::string trim(const std::string & s) {
    const size_t b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) {
        return "";
    }
    const size_t e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

namespace {

std::string pad_right(const std::string & s, int width) {
    const int n = width - disp_width(s);
    return n > 0 ? s + std::string(static_cast<size_t>(n), ' ') : s;
}

} // namespace

// Greedy fill; a word longer than the limit widens the column instead of
// being broken (table.go's wrapString).
std::vector<std::string> wrap_string(const std::string & s, int lim) {
    std::string flat = s;
    std::replace(flat.begin(), flat.end(), '\n', ' ');
    const std::vector<std::string> words = split_fields(flat);
    for (const auto & w : words) {
        lim = std::max(lim, disp_width(w));
    }
    std::vector<std::string> lines;
    std::vector<std::string> cur;
    int                      n = 0;
    for (const auto & w : words) {
        const int width = disp_width(w);
        if (n > 0 && n + 1 + width > lim) {
            lines.push_back(join(cur, " "));
            cur.clear();
            n = 0;
        }
        if (n > 0) {
            n++;
        }
        cur.push_back(w);
        n += width;
    }
    if (!cur.empty()) {
        lines.push_back(join(cur, " "));
    }
    if (lines.empty()) {
        lines.push_back("");
    }
    return lines;
}

std::string tw_title(const std::string & name) {
    std::string rs = name;
    for (size_t i = 0; i < rs.size(); i++) {
        if (rs[i] == '_') {
            rs[i] = ' ';
        } else if (rs[i] == '.') {
            const bool prevDigit = i != 0 && std::isdigit(static_cast<unsigned char>(rs[i - 1]));
            const bool nextDigit = i != rs.size() - 1 && std::isdigit(static_cast<unsigned char>(rs[i + 1]));
            if ((i != 0 && !prevDigit) || (i != rs.size() - 1 && !nextDigit)) {
                rs[i] = ' ';
            }
        }
    }
    size_t b = rs.find_first_not_of(' ');
    size_t e = rs.find_last_not_of(' ');
    std::string out = (b == std::string::npos) ? "" : rs.substr(b, e - b + 1);
    std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
    if (out.empty() && !name.empty()) {
        return " ";
    }
    return out;
}

Table::Table(std::vector<std::string> header) : header_(std::move(header)) {}

void Table::add(std::vector<std::string> cells) { rows_.push_back(std::move(cells)); }

std::vector<std::string> Table::split(const std::string & s) const {
    const std::vector<std::string> raw = split_lines(s);
    int                             max = 0;
    for (const auto & ln : raw) {
        max = std::max(max, disp_width(ln));
    }
    if (wrap_at <= 0 || max <= wrap_at) {
        return raw;
    }
    std::vector<std::string> out;
    for (size_t i = 0; i < raw.size(); i++) {
        if (i > 0) {
            out.push_back(" ");
        }
        for (auto & ln : wrap_string(raw[i], wrap_at)) {
            out.push_back(ln);
        }
    }
    return out;
}

namespace {
std::string line_at(const std::vector<std::string> & lines, size_t i) {
    return i < lines.size() ? lines[i] : std::string("  ");
}
} // namespace

std::string Table::str() const {
    size_t cols = header_.size();
    for (const auto & r : rows_) {
        cols = std::max(cols, r.size());
    }
    std::vector<int>                      widths(cols, 0);
    std::vector<std::vector<std::string>> head(cols);
    std::vector<std::vector<std::vector<std::string>>> body(rows_.size());

    const auto measure = [&](const std::vector<std::string> & lines, size_t c) {
        for (const auto & ln : lines) {
            widths[c] = std::max(widths[c], disp_width(ln));
        }
    };
    for (size_t c = 0; c < cols; c++) {
        const std::string cell = c < header_.size() ? tw_title(header_[c]) : "";
        head[c]                = split(cell);
        measure(head[c], c);
    }
    for (size_t i = 0; i < rows_.size(); i++) {
        body[i].resize(cols);
        for (size_t c = 0; c < cols; c++) {
            const std::string cell = c < rows_[i].size() ? rows_[i][c] : "";
            body[i][c]             = split(cell);
            measure(body[i][c], c);
        }
    }

    const std::string pad = "    ";
    std::ostringstream sb;
    if (!header_.empty()) {
        size_t lines = 1;
        for (const auto & h : head) {
            lines = std::max(lines, h.size());
        }
        for (size_t x = 0; x < lines; x++) {
            for (size_t c = 0; c < cols; c++) {
                sb << pad_right(line_at(head[c], x), widths[c]);
                sb << (c == cols - 1 ? " " : pad);
            }
            sb << "\n";
        }
    }
    for (size_t i = 0; i < rows_.size(); i++) {
        size_t lines = 1;
        for (const auto & c : body[i]) {
            lines = std::max(lines, c.size());
        }
        for (size_t x = 0; x < lines; x++) {
            for (size_t c = 0; c < cols; c++) {
                sb << pad_right(line_at(body[i][c], x), widths[c]) << pad;
            }
            sb << "\n";
        }
    }
    return sb.str();
}

std::string human_bytes(int64_t b) {
    constexpr double kb = 1000, mb = 1000.0 * 1000, gb = 1000.0 * 1000 * 1000, tb = gb * 1000;
    double            value;
    const char *      unit;
    if (b >= static_cast<int64_t>(tb)) {
        value = b / tb;
        unit  = "TB";
    } else if (b >= static_cast<int64_t>(gb)) {
        value = b / gb;
        unit  = "GB";
    } else if (b >= static_cast<int64_t>(mb)) {
        value = b / mb;
        unit  = "MB";
    } else if (b >= static_cast<int64_t>(kb)) {
        value = b / kb;
        unit  = "KB";
    } else {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%lld B", static_cast<long long>(b));
        return buf;
    }
    char buf[32];
    if (value >= 10) {
        std::snprintf(buf, sizeof(buf), "%d %s", static_cast<int>(value), unit);
    } else if (value != std::trunc(value)) {
        std::snprintf(buf, sizeof(buf), "%.1f %s", value, unit);
    } else {
        std::snprintf(buf, sizeof(buf), "%d %s", static_cast<int>(value), unit);
    }
    return buf;
}

std::string human_number(uint64_t b) {
    char buf[32];
    if (b >= 1000000000ULL) {
        const double n = static_cast<double>(b) / 1e9;
        std::snprintf(buf, sizeof(buf), n == std::floor(n) ? "%.0fB" : "%.1fB", n);
    } else if (b >= 1000000ULL) {
        const double n = static_cast<double>(b) / 1e6;
        std::snprintf(buf, sizeof(buf), n == std::floor(n) ? "%.0fM" : "%.2fM", n);
    } else if (b >= 1000ULL) {
        std::snprintf(buf, sizeof(buf), "%.0fK", static_cast<double>(b) / 1000);
    } else {
        std::snprintf(buf, sizeof(buf), "%llu", static_cast<unsigned long long>(b));
    }
    return buf;
}

std::string human_duration(double seconds) {
    const int secs = static_cast<int>(seconds);
    char      buf[32];
    if (secs < 1) {
        return "Less than a second";
    }
    if (secs == 1) {
        return "1 second";
    }
    if (secs < 60) {
        std::snprintf(buf, sizeof(buf), "%d seconds", secs);
        return buf;
    }
    const int minutes = static_cast<int>(seconds / 60);
    if (minutes == 1) {
        return "About a minute";
    }
    if (minutes < 60) {
        std::snprintf(buf, sizeof(buf), "%d minutes", minutes);
        return buf;
    }
    const int hours = static_cast<int>(std::round(seconds / 3600.0));
    if (hours == 1) {
        return "About an hour";
    }
    if (hours < 48) {
        std::snprintf(buf, sizeof(buf), "%d hours", hours);
        return buf;
    }
    if (hours < 24 * 7 * 2) {
        std::snprintf(buf, sizeof(buf), "%d days", hours / 24);
        return buf;
    }
    if (hours < 24 * 30 * 2) {
        std::snprintf(buf, sizeof(buf), "%d weeks", hours / 24 / 7);
        return buf;
    }
    if (hours < 24 * 365 * 2) {
        std::snprintf(buf, sizeof(buf), "%d months", hours / 24 / 30);
        return buf;
    }
    std::snprintf(buf, sizeof(buf), "%d years", hours / 24 / 365);
    return buf;
}

std::string human_time(std::time_t t, bool is_zero, const std::string & zero_value, double now) {
    if (is_zero) {
        return zero_value;
    }
    const double delta = now - static_cast<double>(t);
    if (delta < 0 && (-delta) / 86400.0 / 365 > 20) {
        return "Forever";
    }
    if (delta < 0) {
        return human_duration(-delta) + " from now";
    }
    return human_duration(delta) + " ago";
}

double parse_rfc3339(const std::string & s) {
    int    y, mo, d, h, mi, se;
    double frac  = 0;
    int    tzh = 0, tzm = 0;
    char   tzsign = 0;
    const int n = std::sscanf(s.c_str(), "%4d-%2d-%2dT%2d:%2d:%2d", &y, &mo, &d, &h, &mi, &se);
    if (n != 6) {
        return -1;
    }
    size_t pos = 19; // after seconds
    if (pos < s.size() && s[pos] == '.') {
        size_t start = pos;
        pos++;
        while (pos < s.size() && std::isdigit(static_cast<unsigned char>(s[pos]))) {
            pos++;
        }
        frac = std::atof(s.substr(start).c_str());
    }
    if (pos < s.size() && (s[pos] == '+' || s[pos] == '-')) {
        tzsign = s[pos];
        std::sscanf(s.c_str() + pos + 1, "%2d:%2d", &tzh, &tzm);
    }

    // Days since the epoch via Howard Hinnant's civil_from_days inverse
    // (days_from_civil), so this needs no libc timezone/DST behaviour.
    const int64_t yy = y - (mo <= 2 ? 1 : 0);
    const int64_t era = (yy >= 0 ? yy : yy - 399) / 400;
    const unsigned yoe = static_cast<unsigned>(yy - era * 400);
    const unsigned mp = static_cast<unsigned>((mo + 9) % 12);
    const unsigned doy = (153 * mp + 2) / 5 + static_cast<unsigned>(d) - 1;
    const unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    const int64_t days = era * 146097 + static_cast<int64_t>(doe) - 719468;

    double secs = static_cast<double>(days) * 86400.0 + h * 3600.0 + mi * 60.0 + se + frac;
    if (tzsign == '+') {
        secs -= (tzh * 3600 + tzm * 60);
    } else if (tzsign == '-') {
        secs += (tzh * 3600 + tzm * 60);
    }
    return secs;
}

std::string human_time_iso(const std::string & s, const std::string & zero_value, double now) {
    const double t = parse_rfc3339(s);
    if (t < 0) {
        return zero_value;
    }
    return human_time(static_cast<std::time_t>(t), false, zero_value, now);
}

std::string human(double n) {
    static const char * units[] = {"B", "KB", "MB", "GB", "TB"};
    size_t              i       = 0;
    for (; i < 4; i++) {
        if (n < 1024) {
            break;
        }
        n /= 1024;
    }
    char buf[32];
    if (i >= 4) {
        std::snprintf(buf, sizeof(buf), "%.1f PB", n);
    } else if (i == 0 || i == 1) {
        std::snprintf(buf, sizeof(buf), "%.0f %s", n, units[i]);
    } else {
        std::snprintf(buf, sizeof(buf), "%.1f %s", n, units[i]);
    }
    return buf;
}

} // namespace llmash
