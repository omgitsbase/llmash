#include "cli_console.h"
#include "cli_process.h"
#include "terminal.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>

namespace llmash {

bool is_console_stdin() { return stdin_is_terminal(); }

bool driven_by_agent() {
    // What the harnesses put in the environment. AI_AGENT is the generic one;
    // the rest are the tools seen setting something of their own.
    static const char * const markers[] = {
        "AI_AGENT",      "CLAUDECODE",       "CLAUDE_CODE_ENTRYPOINT", "CURSOR_TRACE_ID",
        "AIDER_MODEL",   "GITHUB_COPILOT_AGENT", "OPENAI_AGENT",       "CODEX_SANDBOX",
        "REPLIT_AGENT",  "DEVIN_SESSION_ID", "CI",
    };
    for (const char * m : markers) {
        if (const char * v = std::getenv(m); v != nullptr && *v != '\0' && std::string(v) != "0") {
            return true;
        }
    }
    return false;
}
bool is_console_stdout() { return stdout_is_terminal(); }

int raw_getch() { return raw_key(); }

int read_pick(const ByteSource & next_byte) {
    const int ch = next_byte();
    if (ch == 0xE0 || ch == 0x00) {
        switch (next_byte()) {
            case 0x48: return kPickUp;
            case 0x50: return kPickDown;
            case 0x4D: return kPickRight;
            case 0x4B: return kPickLeft;
        }
        return 0;
    }
    if (ch == '\r' || ch == '\n') {
        return kPickEnter;
    }
    if (ch == 0x1b || ch == 0x03) {
        return kPickEsc;
    }
    return ch;
}

// ------------------------------------------------------------------ menus

namespace {

const char kDot[]    = " \xc2\xb7 ";      // " · "
const char kCursor[] = "\xe2\x9d\xaf";    // ❯
const char kTick[]   = "\xe2\x9c\x93";    // ✓
const char kCross[]  = "\xe2\x9c\x97";    // ✗
const char kUp[]     = "\xe2\x86\x91";    // ↑
const char kDown[]   = "\xe2\x86\x93";    // ↓

std::string key_name(int key) {
    switch (key) {
        case kPickRight: return "\xe2\x86\x92";
        case kPickLeft:  return "\xe2\x86\x90";
        default:         return std::string(1, static_cast<char>(key));
    }
}

std::string question_line(const std::string & q) {
    return std::string(kCyan) + "?" + kReset + " " + kBold + q + kReset;
}

// The row with its padding runs closed up, for the folded line.
std::string squeeze(const std::string & s) {
    std::string out;
    for (const char c : s) {
        if (c == ' ' && (out.empty() || out.back() == ' ')) {
            continue;
        }
        out += c;
    }
    while (!out.empty() && out.back() == ' ') {
        out.pop_back();
    }
    return out;
}

// At most `cols` columns of a line, escape sequences kept whole; a cut line ends with a reset.
std::string clip_line(const std::string & s, int cols) {
    if (cols <= 0 || display_width(s) <= cols) {
        return s;
    }
    std::string out;
    int         w = 0;
    for (size_t i = 0; i < s.size(); i++) {
        const unsigned char c = static_cast<unsigned char>(s[i]);
        if (c == 0x1b && i + 1 < s.size() && s[i + 1] == '[') {
            size_t j = i + 2;
            while (j < s.size() && !(s[j] >= 0x40 && s[j] <= 0x7e)) {
                j++;
            }
            out.append(s, i, j - i + 1);
            i = j;
            continue;
        }
        if ((c & 0xC0) != 0x80) {
            if (w == cols) {
                break;
            }
            w++;
        }
        out += static_cast<char>(c);
    }
    return out + kReset;
}

bool same_key(int pressed, int key) {
    if (pressed == key) {
        return true;
    }
    return pressed > 0 && pressed < 0x80 && key > 0 && key < 0x80 && std::tolower(pressed) == std::tolower(key);
}

} // namespace

int display_width(const std::string & s) {
    int w = 0;
    for (size_t i = 0; i < s.size(); i++) {
        const unsigned char c = static_cast<unsigned char>(s[i]);
        if (c == 0x1b && i + 1 < s.size() && s[i + 1] == '[') {
            i += 2;
            while (i < s.size() && !(s[i] >= 0x40 && s[i] <= 0x7e)) {
                i++;
            }
            continue;
        }
        if ((c & 0xC0) != 0x80) {
            w++;
        }
    }
    return w;
}

std::string key_hint(const std::vector<MenuKey> & keys, const std::string & esc) {
    std::string h;
    for (const MenuKey & k : keys) {
        h += (h.empty() ? "" : kDot) + key_name(k.key) + " " + k.what;
    }
    if (!esc.empty()) {
        h += (h.empty() ? "" : kDot) + std::string("esc ") + esc;
    }
    return h;
}

std::vector<std::string> menu_frame(const Menu & m, int cursor, int height) {
    std::vector<std::string> out{question_line(m.question)};
    const int                n      = static_cast<int>(m.rows.size());
    int                      widest = 0;
    for (const std::string & r : m.rows) {
        widest = std::max(widest, display_width(r));
    }
    // the question and the hint take two lines; a list longer than the rest scrolls with the cursor
    int       first = 0;
    int       last  = n;
    const int room  = std::max(3, height - 2);
    if (n > room) {
        const int shown = std::max(1, room - 2);
        first           = std::clamp(cursor - shown / 2, 0, n - shown);
        last            = first + shown;
    }
    if (first > 0) {
        out.push_back(std::string("    ") + kDim + kUp + " " + std::to_string(first) + " more" + kReset);
    }
    for (int i = first; i < last; i++) {
        const std::string & row  = m.rows[static_cast<size_t>(i)];
        std::string         line = i == cursor ? std::string("  ") + kCyan + kCursor + kReset + " " + kBold + row + kReset
                                               : "    " + row;
        if (static_cast<size_t>(i) < m.notes.size() && !m.notes[static_cast<size_t>(i)].empty()) {
            line += std::string(static_cast<size_t>(widest - display_width(row) + 3), ' ') + kDim +
                    m.notes[static_cast<size_t>(i)] + kReset;
        }
        out.push_back(line);
    }
    if (last < n) {
        out.push_back(std::string("    ") + kDim + kDown + " " + std::to_string(n - last) + " more" + kReset);
    }
    const std::string extra = key_hint(m.keys, m.esc);
    out.push_back(std::string("  ") + kDim + kUp + kDown + " move" + kDot + "enter select" +
                  (extra.empty() ? "" : kDot + extra) + kReset);
    return out;
}

std::string menu_folded(const std::string & question, const std::string & answer, bool cancelled) {
    if (cancelled) {
        return std::string(kDim) + kCross + " " + question + " " + answer + kReset;
    }
    return std::string(kGreen) + kTick + kReset + " " + question + " " + kCyan + answer + kReset;
}

int choose(const Menu & m, int * at) {
    const int n = static_cast<int>(m.rows.size());
    if (at != nullptr) {
        *at = std::clamp(m.cursor, 0, std::max(0, n - 1));
    }
    if (n == 0) {
        return -1;
    }
    if (!is_console_stdin() || !is_console_stdout()) {
        std::printf("%s\n", question_line(m.question).c_str());
        for (size_t i = 0; i < m.rows.size(); i++) {
            const bool note = i < m.notes.size() && !m.notes[i].empty();
            std::printf("    %s%s%s\n", m.rows[i].c_str(), note ? "   " : "", note ? m.notes[i].c_str() : "");
        }
        std::printf("  %sno terminal to answer from, so %s%s\n", kDim, m.esc.c_str(), kReset);
        return -1;
    }
    int        cursor = std::clamp(m.cursor, 0, n - 1);
    int        drawn  = 0;
    const auto erase  = [&]() {
        if (drawn > 0) {
            std::printf("\x1b[%dF\x1b[J", drawn);
        }
    };
    const auto draw = [&]() {
        const std::vector<std::string> lines = menu_frame(m, cursor, terminal_rows() - 1);
        const int                      cols  = terminal_columns() - 1;
        erase();
        for (const std::string & l : lines) {
            std::printf("%s\n", clip_line(l, cols).c_str());
        }
        std::fflush(stdout);
        drawn = static_cast<int>(lines.size());
    };
    const auto done = [&](const std::string & folded) {
        erase();
        if (!folded.empty()) {
            std::printf("%s\n", folded.c_str());
        }
        std::printf("\x1b[?25h");
        std::fflush(stdout);
    };
    std::printf("\x1b[?25l");
    draw();
    for (;;) {
        const int k = read_pick([]() { return raw_getch(); });
        if (k == kPickUp || k == kPickDown) {
            cursor = (cursor + (k == kPickUp ? n - 1 : 1)) % n;
            if (at != nullptr) {
                *at = cursor;
            }
            draw();
        } else if (k == kPickEnter) {
            const size_t      i      = static_cast<size_t>(cursor);
            const std::string answer = i < m.answers.size() && !m.answers[i].empty() ? m.answers[i] : squeeze(m.rows[i]);
            done(menu_folded(m.question, answer, false));
            return cursor;
        } else if (k == kPickEsc) {
            done(menu_folded(m.question, m.esc, true));
            return -1;
        } else {
            for (const MenuKey & mk : m.keys) {
                if (same_key(k, mk.key)) {
                    // the caller asks again in the same place, so nothing is left behind
                    done("");
                    return -mk.key;
                }
            }
        }
    }
}

bool ask_yes(const std::string & question, bool yes_by_default) {
    const char * hint = yes_by_default ? "(Y/n)" : "(y/N)";
    if (!is_console_stdin()) {
        std::printf("%s %sno, with no terminal to answer from%s\n", question_line(question).c_str(), kDim, kReset);
        return false;
    }
    std::printf("%s %s%s%s ", question_line(question).c_str(), kDim, hint, kReset);
    std::fflush(stdout);
    for (;;) {
        const int k   = read_pick([]() { return raw_getch(); });
        int       yes = -1;
        if (k == 'y' || k == 'Y') {
            yes = 1;
        } else if (k == 'n' || k == 'N' || k == kPickEsc) {
            yes = 0;
        } else if (k == kPickEnter) {
            yes = yes_by_default ? 1 : 0;
        }
        if (yes < 0) {
            continue;
        }
        std::printf("\r\x1b[K%s\n", menu_folded(question, yes ? "yes" : "no", false).c_str());
        std::fflush(stdout);
        return yes == 1;
    }
}

// --------------------------------------------------------------- drafters

bool drafter_offer_due(const nlohmann::json & local, const std::string & model) {
    if (!local.is_object()) {
        return true;
    }
    if (local.value("ask_drafter", true) == false) {
        return false;
    }
    const auto it = local.find("drafter_asked");
    if (it != local.end() && it->is_array()) {
        for (const auto & n : *it) {
            if (n.is_string() && n.get<std::string>() == model) {
                return false;
            }
        }
    }
    return true;
}

void drafter_offer_answered(nlohmann::json & local, const std::string & model, int answer) {
    if (!local.is_object()) {
        local = nlohmann::json::object();
    }
    if (answer == 2) {
        local["ask_drafter"] = false;
        return;
    }
    nlohmann::json & asked = local["drafter_asked"];
    if (!asked.is_array()) {
        asked = nlohmann::json::array();
    }
    for (const auto & n : asked) {
        if (n == model) {
            return;
        }
    }
    asked.push_back(model);
}

bool clip_copy(const std::string & text) {
    const ProcessResult r = run_hidden({"clip"}, &text, false, true);
    return r.started && r.exit_code == 0;
}

} // namespace llmash
