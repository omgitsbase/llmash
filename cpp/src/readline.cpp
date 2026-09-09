#include "readline.h"

#include "config.h"

#ifdef _WIN32
#include <windows.h>
#endif

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <ostream>
#include <random>
#include <sstream>

namespace fs = std::filesystem;

namespace llmash {

namespace {

// -------------------------------------------------------------- key codes
// Same names and values as readline.go: raw control bytes read with
// ENABLE_VIRTUAL_TERMINAL_INPUT on, so arrows and Home/End/Delete arrive as
// the escape sequences a Linux terminal would send.

constexpr char32_t charNull      = 0;
constexpr char32_t charLineStart = 1;
constexpr char32_t charBackward  = 2;
constexpr char32_t charInterrupt = 3;
constexpr char32_t charDelete    = 4;
constexpr char32_t charLineEnd   = 5;
constexpr char32_t charForward   = 6;
constexpr char32_t charBell      = 7;
constexpr char32_t charCtrlH     = 8;
constexpr char32_t charTab       = 9;
constexpr char32_t charCtrlJ     = 10;
constexpr char32_t charKill      = 11;
constexpr char32_t charCtrlL     = 12;
constexpr char32_t charEnter     = 13;
constexpr char32_t charNext      = 14;
constexpr char32_t charPrev      = 16;
constexpr char32_t charCtrlU     = 21;
constexpr char32_t charCtrlW     = 23;
constexpr char32_t charEsc       = 27;
constexpr char32_t charSpace     = 32;
constexpr char32_t charEscapeEx  = 91;
constexpr char32_t charBackspace = 127;

constexpr char32_t keyDel   = 51;
constexpr char32_t keyUp    = 65;
constexpr char32_t keyDown  = 66;
constexpr char32_t keyRight = 67;
constexpr char32_t keyLeft  = 68;
constexpr char32_t metaEnd   = 70;
constexpr char32_t metaStart = 72;

constexpr char32_t pasteMarker = 50;

const char * const colorGrey    = "\x1b[38;5;245m";
const char * const colorDefault = "\x1b[0m";

bool is_word_rune(char32_t r) {
    return r != U' ' && r != U'\t';
}

std::vector<std::string> split_lines(const std::string & s) {
    std::vector<std::string> out;
    size_t                   start = 0;
    for (;;) {
        const size_t p = s.find('\n', start);
        if (p == std::string::npos) {
            out.push_back(s.substr(start));
            return out;
        }
        out.push_back(s.substr(start, p - start));
        start = p + 1;
    }
}

std::string join(const std::vector<std::string> & v, const std::string & sep) {
    std::string out;
    for (size_t i = 0; i < v.size(); i++) {
        if (i) {
            out += sep;
        }
        out += v[i];
    }
    return out;
}

// The combining-mark blocks actually seen in typed text.
bool is_combining_mark(char32_t r) {
    return (r >= 0x0300 && r <= 0x036F) || (r >= 0x1AB0 && r <= 0x1AFF) ||
           (r >= 0x1DC0 && r <= 0x1DFF) || (r >= 0x20D0 && r <= 0x20FF) ||
           (r >= 0xFE20 && r <= 0xFE2F);
}

bool is_wide(char32_t r) {
    return r >= 0x1100 &&
           (r <= 0x115f || r == 0x2329 || r == 0x232a ||
            (r >= 0x2e80 && r <= 0xa4cf && r != 0x303f) ||
            (r >= 0xac00 && r <= 0xd7a3) || (r >= 0xf900 && r <= 0xfaff) ||
            (r >= 0xfe30 && r <= 0xfe6f) || (r >= 0xff00 && r <= 0xff60) ||
            (r >= 0xffe0 && r <= 0xffe6) || (r >= 0x1f300 && r <= 0x1f64f) ||
            (r >= 0x1f900 && r <= 0x1f9ff) || (r >= 0x20000 && r <= 0x3fffd));
}

int term_width() {
    CONSOLE_SCREEN_BUFFER_INFO info;
    HANDLE                     h = GetStdHandle(STD_OUTPUT_HANDLE);
    if (h == INVALID_HANDLE_VALUE || h == nullptr || !GetConsoleScreenBufferInfo(h, &info)) {
        return 100;
    }
    const int w = info.srWindow.Right - info.srWindow.Left + 1;
    return w > 0 ? w : 100;
}

int editor_width() {
    const int w = term_width();
    return w < 10 ? 80 : w;
}

// Raw-mode console state, restored on destruction; a no-op guard when stdin
// is not a real console (GetConsoleMode fails, e.g.
class RawModeGuard {
public:
    RawModeGuard() {
        HANDLE h = GetStdHandle(STD_INPUT_HANDLE);
        DWORD  mode = 0;
        if (h == INVALID_HANDLE_VALUE || h == nullptr || !GetConsoleMode(h, &mode)) {
            return;
        }
        mode_ = mode;
        on_   = true;
        // The literal readline.go mask ORs in a 4th flag that is the same
        // bit as ENABLE_PROCESSED_INPUT (0x0001) applied to an input
        // handle's mode; harmless, kept for a 1:1 mask.
        DWORD raw = mode & ~static_cast<DWORD>(ENABLE_ECHO_INPUT | ENABLE_PROCESSED_INPUT | ENABLE_LINE_INPUT | 0x0001);
        raw |= ENABLE_VIRTUAL_TERMINAL_INPUT;
        SetConsoleMode(h, raw);
        cp_ = GetConsoleCP();
        SetConsoleCP(65001); // typed text arrives as UTF-8
    }
    ~RawModeGuard() {
        if (!on_) {
            return;
        }
        HANDLE h = GetStdHandle(STD_INPUT_HANDLE);
        if (h != INVALID_HANDLE_VALUE && h != nullptr) {
            SetConsoleMode(h, mode_);
        }
        if (cp_ != 0) {
            SetConsoleCP(cp_);
        }
    }
    RawModeGuard(const RawModeGuard &)             = delete;
    RawModeGuard & operator=(const RawModeGuard &) = delete;

private:
    DWORD mode_ = 0;
    UINT  cp_   = 0;
    bool  on_   = false;
};

std::vector<std::string> split_on(const std::string & s, char sep) {
    std::vector<std::string> out;
    size_t                   start = 0;
    for (;;) {
        const size_t p = s.find(sep, start);
        out.push_back(s.substr(start, p == std::string::npos ? std::string::npos : p - start));
        if (p == std::string::npos) {
            return out;
        }
        start = p + 1;
    }
}

std::vector<std::string> split_fields(const std::string & s) {
    std::vector<std::string> out;
    size_t                   i = 0, n = s.size();
    while (i < n) {
        while (i < n && std::isspace(static_cast<unsigned char>(s[i]))) {
            i++;
        }
        if (i >= n) {
            break;
        }
        const size_t start = i;
        while (i < n && !std::isspace(static_cast<unsigned char>(s[i]))) {
            i++;
        }
        out.push_back(s.substr(start, i - start));
    }
    return out;
}

// The documented Windows argv quoting rules (as used by CommandLineToArgvW
// and Go's syscall.EscapeArg).
std::string quote_arg(const std::string & arg) {
    if (!arg.empty() && arg.find_first_of(" \t\n\v\"") == std::string::npos) {
        return arg;
    }
    std::string out = "\"";
    for (size_t i = 0; i < arg.size();) {
        size_t backslashes = 0;
        while (i < arg.size() && arg[i] == '\\') {
            backslashes++;
            i++;
        }
        if (i == arg.size()) {
            out.append(backslashes * 2, '\\');
            break;
        } else if (arg[i] == '"') {
            out.append(backslashes * 2 + 1, '\\');
            out.push_back('"');
            i++;
        } else {
            out.append(backslashes, '\\');
            out.push_back(arg[i]);
            i++;
        }
    }
    out.push_back('"');
    return out;
}

bool file_exists_not_dir(const std::string & p) {
    std::error_code ec;
    return fs::is_regular_file(p, ec);
}

std::vector<std::string> path_exts() {
    const std::string        x = env_str("PATHEXT");
    std::vector<std::string> exts;
    if (x.empty()) {
        return {".com", ".exe", ".bat", ".cmd"};
    }
    for (std::string e : split_on(x, ';')) {
        if (e.empty()) {
            continue;
        }
        std::transform(e.begin(), e.end(), e.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (e[0] != '.') {
            e = "." + e;
        }
        exts.push_back(e);
    }
    return exts;
}

std::optional<std::string> find_executable(const std::string & file) {
    if (fs::path(file).has_extension() && file_exists_not_dir(file)) {
        return file;
    }
    for (const auto & e : path_exts()) {
        std::string f = file + e;
        if (file_exists_not_dir(f)) {
            return f;
        }
    }
    return std::nullopt;
}

// A reasonable equivalent of exec.LookPath on Windows: PATH + PATHEXT, not
// a byte-identical reimplementation of the Go runtime's version.
std::optional<std::string> look_path(const std::string & name) {
    if (name.find_first_of(":\\/") != std::string::npos) {
        return find_executable(name);
    }
    if (auto f = find_executable("./" + name)) {
        return f;
    }
    const std::string path = env_str("PATH", env_str("Path"));
    for (const auto & dir : split_on(path, ';')) {
        if (dir.empty()) {
            continue;
        }
        if (auto f = find_executable((fs::path(dir) / name).string())) {
            return f;
        }
    }
    return std::nullopt;
}

std::wstring utf8_to_wide(const std::string & s) {
    if (s.empty()) {
        return L"";
    }
    const int len = MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), nullptr, 0);
    std::wstring out(len, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), out.data(), len);
    return out;
}

// Closes both PROCESS_INFORMATION handles no matter how the caller returns.
class ProcessGuard {
public:
    ProcessGuard(HANDLE proc, HANDLE thread) : proc_(proc), thread_(thread) {}
    ~ProcessGuard() {
        if (proc_ != nullptr) {
            CloseHandle(proc_);
        }
        if (thread_ != nullptr) {
            CloseHandle(thread_);
        }
    }
    ProcessGuard(const ProcessGuard &)             = delete;
    ProcessGuard & operator=(const ProcessGuard &) = delete;
    HANDLE         process() const { return proc_; }

private:
    HANDLE proc_;
    HANDLE thread_;
};

} // namespace

// -------------------------------------------------------------- utf-8

std::u32string utf8_decode(const std::string & s) {
    std::u32string out;
    size_t         i = 0, n = s.size();
    while (i < n) {
        const unsigned char c = static_cast<unsigned char>(s[i]);
        char32_t             cp;
        int                  len;
        if (c < 0x80) {
            cp  = c;
            len = 1;
        } else if ((c & 0xE0) == 0xC0) {
            cp  = c & 0x1F;
            len = 2;
        } else if ((c & 0xF0) == 0xE0) {
            cp  = c & 0x0F;
            len = 3;
        } else if ((c & 0xF8) == 0xF0) {
            cp  = c & 0x07;
            len = 4;
        } else {
            out.push_back(0xFFFD);
            i++;
            continue;
        }
        if (i + len > n) {
            out.push_back(0xFFFD);
            i++;
            continue;
        }
        bool ok = true;
        for (int k = 1; k < len && ok; k++) {
            const unsigned char cc = static_cast<unsigned char>(s[i + k]);
            if ((cc & 0xC0) != 0x80) {
                ok = false;
                break;
            }
            cp = (cp << 6) | (cc & 0x3F);
        }
        if (!ok) {
            out.push_back(0xFFFD);
            i++;
            continue;
        }
        out.push_back(cp);
        i += len;
    }
    return out;
}

std::string utf8_encode(const std::u32string & s) {
    std::string out;
    for (char32_t r : s) {
        if (r < 0x80) {
            out.push_back(static_cast<char>(r));
        } else if (r < 0x800) {
            out.push_back(static_cast<char>(0xC0 | (r >> 6)));
            out.push_back(static_cast<char>(0x80 | (r & 0x3F)));
        } else if (r < 0x10000) {
            out.push_back(static_cast<char>(0xE0 | (r >> 12)));
            out.push_back(static_cast<char>(0x80 | ((r >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (r & 0x3F)));
        } else {
            out.push_back(static_cast<char>(0xF0 | (r >> 18)));
            out.push_back(static_cast<char>(0x80 | ((r >> 12) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | ((r >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (r & 0x3F)));
        }
    }
    return out;
}

int disp_width(const std::u32string & s) {
    int n = 0;
    for (char32_t r : s) {
        if (r == 0 || is_combining_mark(r)) {
            continue;
        }
        n += is_wide(r) ? 2 : 1;
    }
    return n;
}


// -------------------------------------------------------------- rune sources

VectorRuneSource::VectorRuneSource(std::vector<char32_t> runes) : runes_(std::move(runes)) {}
VectorRuneSource::VectorRuneSource(const std::string & utf8) {
    const std::u32string d = utf8_decode(utf8);
    runes_.assign(d.begin(), d.end());
}

std::optional<char32_t> VectorRuneSource::next() {
    if (pos_ >= runes_.size()) {
        return std::nullopt;
    }
    return runes_[pos_++];
}

namespace {
bool read_console_byte(unsigned char & b) {
    HANDLE h = GetStdHandle(STD_INPUT_HANDLE);
    DWORD  n = 0;
    if (h == INVALID_HANDLE_VALUE || h == nullptr) {
        return false;
    }
    return ReadFile(h, &b, 1, &n, nullptr) && n == 1;
}
int utf8_lead_len(unsigned char c) {
    if (c < 0x80) return 1;
    if ((c & 0xE0) == 0xC0) return 2;
    if ((c & 0xF0) == 0xE0) return 3;
    if ((c & 0xF8) == 0xF0) return 4;
    return 1;
}
} // namespace

std::optional<char32_t> ConsoleRuneSource::next() {
    unsigned char lead;
    if (!read_console_byte(lead)) {
        return std::nullopt;
    }
    const int len = utf8_lead_len(lead);
    if (len == 1) {
        return static_cast<char32_t>(lead);
    }
    char32_t cp = lead & (0xFFu >> (len + 1));
    for (int k = 1; k < len; k++) {
        unsigned char c;
        if (!read_console_byte(c) || (c & 0xC0) != 0x80) {
            return static_cast<char32_t>(0xFFFD);
        }
        cp = (cp << 6) | (c & 0x3F);
    }
    return cp;
}

// -------------------------------------------------------------- history

std::string default_history_path() {
    const std::string home = env_str("USERPROFILE", env_str("HOME"));
    if (home.empty()) {
        return "";
    }
    return (fs::path(home) / ".ollama" / "history").string();
}

History::History() : History(default_history_path()) {}

History::History(std::string path) : path_(std::move(path)) {
    if (!path_.empty()) {
        std::ifstream in(path_, std::ios::binary);
        if (in) {
            std::string raw((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
            std::string norm;
            norm.reserve(raw.size());
            for (size_t i = 0; i < raw.size(); i++) {
                if (raw[i] == '\r' && i + 1 < raw.size() && raw[i + 1] == '\n') {
                    continue;
                }
                norm.push_back(raw[i]);
            }
            for (auto & ln : split_on(norm, '\n')) {
                if (!ln.empty()) {
                    lines.push_back(ln);
                }
            }
        }
    }
    pos = lines.size();
}

void History::add(const std::string & s) {
    pos = lines.size();
    if (!enabled) {
        return;
    }
    if (pos > 0 && lines[pos - 1] == s) {
        pos = lines.size();
        return;
    }
    lines.push_back(s);
    if (lines.size() > limit) {
        lines.erase(lines.begin(), lines.begin() + (lines.size() - limit));
    }
    pos = lines.size();
    save();
}

void History::save() {
    if (path_.empty()) {
        return;
    }
    std::error_code ec;
    fs::create_directories(fs::path(path_).parent_path(), ec);
    std::ofstream out(path_, std::ios::binary | std::ios::trunc);
    if (!out) {
        return;
    }
    for (const auto & l : lines) {
        out << l << "\n";
    }
}

// -------------------------------------------------------------- editor

Editor::Editor() : hist_() {}
Editor::Editor(History hist) : hist_(std::move(hist)) {}

std::string Editor::cur_prompt() const {
    return (use_alt_ || pasting_) ? alt_prompt : prompt;
}

std::string Editor::cur_placeholder() const {
    return use_alt_ ? alt_placeholder : placeholder;
}

void Editor::insert(char32_t r) {
    buf_.insert(buf_.begin() + pos_, r);
    pos_++;
}

void Editor::replace(const std::u32string & s) {
    buf_ = s;
    pos_ = buf_.size();
}

size_t Editor::left_word() const {
    size_t i = pos_;
    while (i > 0 && !is_word_rune(buf_[i - 1])) {
        i--;
    }
    while (i > 0 && is_word_rune(buf_[i - 1])) {
        i--;
    }
    return i;
}

size_t Editor::right_word() const {
    size_t i = pos_;
    while (i < buf_.size() && !is_word_rune(buf_[i])) {
        i++;
    }
    while (i < buf_.size() && is_word_rune(buf_[i])) {
        i++;
    }
    return i;
}

void Editor::draw(std::ostream & out) {
    const int         w      = editor_width();
    const std::string p      = cur_prompt();
    const int         pw     = disp_width(p);
    if (rows_ > 0) {
        out << "\x1b[" << rows_ << "A";
    }
    out << "\r" << p << utf8_encode(buf_) << "\x1b[0J";
    const int total   = pw + disp_width(buf_);
    const int end_row = total / w;
    const int end_col = total % w;
    if (end_col == 0 && total > 0) {
        out << " \r"; // force the deferred wrap so the cursor is on the new row
    }
    if (buf_.empty()) {
        const std::string ph = cur_placeholder();
        if (!ph.empty() && (!pasting_ || use_alt_)) {
            out << colorGrey << ph << colorDefault;
            out << "\r";
            if (pw > 0) {
                out << "\x1b[" << pw << "C";
            }
            rows_ = 0;
            return;
        }
    }
    const int target = pw + disp_width(buf_.substr(0, pos_));
    const int t_row   = target / w;
    const int t_col   = target % w;
    if (end_row > t_row) {
        out << "\x1b[" << (end_row - t_row) << "A";
    }
    out << "\r";
    if (t_col > 0) {
        out << "\x1b[" << t_col << "C";
    }
    rows_ = t_row;
}

void Editor::pasted_line(std::ostream & out) {
    pasted_lines_.push_back(utf8_encode(buf_));
    buf_.clear();
    pos_  = 0;
    rows_ = 0;
    out << "\n";
    use_alt_ = true;
    out << alt_prompt;
}

StepResult Editor::step(RuneSource & in, std::ostream & out) {
    const auto ro = in.next();
    if (!ro) {
        out << "\n";
        return {true, "", ReadStatus::Eof};
    }
    const char32_t r = *ro;

    if (escex_) {
        escex_ = false;
        switch (r) {
            case keyUp:
                if (hist_.pos > 0) {
                    if (hist_.pos == hist_.lines.size()) {
                        saved_ = buf_;
                    }
                    hist_.pos--;
                    replace(utf8_decode(hist_.lines[hist_.pos]));
                }
                break;
            case keyDown:
                if (hist_.pos < hist_.lines.size()) {
                    hist_.pos++;
                    if (hist_.pos == hist_.lines.size()) {
                        replace(saved_);
                    } else {
                        replace(utf8_decode(hist_.lines[hist_.pos]));
                    }
                }
                break;
            case keyLeft:
                if (pos_ > 0) {
                    pos_--;
                }
                break;
            case keyRight:
                if (pos_ < buf_.size()) {
                    pos_++;
                }
                break;
            case pasteMarker: {
                std::u32string code;
                for (int i = 0; i < 3; i++) {
                    const auto c = in.next();
                    if (!c) {
                        return {true, "", ReadStatus::Eof};
                    }
                    code += *c;
                }
                if (code == U"00~") {
                    pasting_ = true;
                } else if (code == U"01~") {
                    pasting_ = false;
                }
                break;
            }
            case keyDel:
                if (pos_ < buf_.size()) {
                    buf_.erase(buf_.begin() + pos_);
                }
                metaDel_ = true;
                break;
            case metaStart:
                pos_ = 0;
                break;
            case metaEnd:
                pos_ = buf_.size();
                break;
            default:
                return {false, "", ReadStatus::Ok};
        }
        draw(out);
        return {false, "", ReadStatus::Ok};
    }

    if (esc_) {
        esc_ = false;
        switch (r) {
            case U'b':
                pos_ = left_word();
                break;
            case U'f':
                pos_ = right_word();
                break;
            case charBackspace: {
                const size_t i = left_word();
                buf_.erase(buf_.begin() + i, buf_.begin() + pos_);
                pos_ = i;
                break;
            }
            case charEscapeEx:
                escex_ = true;
                return {false, "", ReadStatus::Ok};
            default:
                break;
        }
        draw(out);
        return {false, "", ReadStatus::Ok};
    }

    switch (r) {
        case charNull:
            return {false, "", ReadStatus::Ok};
        case charEsc:
            esc_ = true;
            return {false, "", ReadStatus::Ok};
        case charInterrupt:
            pasted_lines_.clear();
            use_alt_ = false;
            return {true, "", ReadStatus::Interrupt};
        case charPrev:
            if (hist_.pos > 0) {
                if (hist_.pos == hist_.lines.size()) {
                    saved_ = buf_;
                }
                hist_.pos--;
                replace(utf8_decode(hist_.lines[hist_.pos]));
            }
            break;
        case charNext:
            if (hist_.pos < hist_.lines.size()) {
                hist_.pos++;
                if (hist_.pos == hist_.lines.size()) {
                    replace(saved_);
                } else {
                    replace(utf8_decode(hist_.lines[hist_.pos]));
                }
            }
            break;
        case charLineStart:
            pos_ = 0;
            break;
        case charLineEnd:
            pos_ = buf_.size();
            break;
        case charBackward:
            if (pos_ > 0) {
                pos_--;
            }
            break;
        case charForward:
            if (pos_ < buf_.size()) {
                pos_++;
            }
            break;
        case charBackspace:
        case charCtrlH:
            if (buf_.empty() && !pasted_lines_.empty()) {
                const std::string last = pasted_lines_.back();
                pasted_lines_.pop_back();
                out << "\r\x1b[K\x1b[A\r\x1b[K";
                if (pasted_lines_.empty()) {
                    use_alt_ = false;
                }
                out << cur_prompt();
                rows_ = 0;
                replace(utf8_decode(last));
            } else if (pos_ > 0) {
                buf_.erase(buf_.begin() + pos_ - 1);
                pos_--;
            }
            break;
        case charTab:
            for (int i = 0; i < 8; i++) {
                insert(U' ');
            }
            break;
        case charDelete:
            if (!buf_.empty()) {
                if (pos_ < buf_.size()) {
                    buf_.erase(buf_.begin() + pos_);
                }
            } else {
                out << "\n";
                return {true, "", ReadStatus::Eof};
            }
            break;
        case charKill:
            buf_.resize(pos_);
            break;
        case charCtrlU:
            buf_.erase(buf_.begin(), buf_.begin() + pos_);
            pos_ = 0;
            break;
        case charCtrlL:
            out << "\x1b[2J\x1b[0;0f";
            rows_ = 0;
            out << cur_prompt();
            break;
        case charCtrlW: {
            const size_t i = left_word();
            buf_.erase(buf_.begin() + i, buf_.begin() + pos_);
            pos_ = i;
            break;
        }
        case charBell: {
            std::string text = utf8_encode(buf_);
            if (!pasted_lines_.empty()) {
                text = join(pasted_lines_, "\n") + "\n" + text;
                pasted_lines_.clear();
            }
            out << "\r\x1b[0J";
            use_alt_ = false;
            return {true, text, ReadStatus::EditPrompt};
        }
        case charCtrlJ:
            pasted_line(out);
            break;
        case charEnter: {
            std::string text = utf8_encode(buf_);
            if (!pasted_lines_.empty()) {
                text = join(pasted_lines_, "\n") + "\n" + text;
                pasted_lines_.clear();
            }
            pos_ = buf_.size();
            draw(out);
            out << "\n";
            use_alt_ = false;
            if (!text.empty()) {
                hist_.add(text);
            }
            return {true, text, ReadStatus::Ok};
        }
        default:
            if (metaDel_) {
                metaDel_ = false;
                return {false, "", ReadStatus::Ok};
            }
            if (r >= charSpace) {
                insert(r);
            }
            break;
    }
    draw(out);
    return {false, "", ReadStatus::Ok};
}

LineResult Editor::read_line(RuneSource & in, std::ostream & out) {
    RawModeGuard guard;
    rows_ = 0;
    buf_.clear();
    pos_ = 0;
    esc_ = escex_ = metaDel_ = false;
    saved_.clear();
    out << cur_prompt();

    if (!prefill_.empty()) {
        const std::string pf = prefill_;
        prefill_.clear();
        const auto lines = split_lines(pf);
        for (size_t i = 0; i < lines.size(); i++) {
            for (char32_t r : utf8_decode(lines[i])) {
                insert(r);
            }
            if (i + 1 < lines.size()) {
                pasted_line(out);
            }
        }
    }
    draw(out);

    for (;;) {
        StepResult sr = step(in, out);
        if (sr.done) {
            return {sr.text, sr.status};
        }
    }
}

LineResult Editor::read_line() {
    ConsoleRuneSource src;
    return read_line(src, std::cout);
}

// -------------------------------------------------------------- temp file

TempFile::TempFile(const std::string & prefix, const std::string & suffix) {
    std::error_code ec;
    fs::path        dir = fs::temp_directory_path(ec);
    if (ec) {
        dir = ".";
    }
    std::random_device rd;
    std::mt19937_64     rng(rd());
    for (int attempt = 0; attempt < 100; attempt++) {
        std::ostringstream name;
        name << prefix << std::hex << rng() << suffix;
        const fs::path   candidate = dir / name.str();
        const std::wstring wpath   = utf8_to_wide(candidate.string());
        HANDLE            h = CreateFileW(wpath.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_NEW,
                                          FILE_ATTRIBUTE_NORMAL, nullptr);
        if (h != INVALID_HANDLE_VALUE) {
            handle_ = h;
            path_   = candidate.string();
            open_   = true;
            return;
        }
    }
}

TempFile::~TempFile() {
    release();
}

TempFile::TempFile(TempFile && other) noexcept
    : path_(std::move(other.path_)), handle_(other.handle_), open_(other.open_) {
    other.handle_ = nullptr;
    other.open_   = false;
    other.path_.clear();
}

TempFile & TempFile::operator=(TempFile && other) noexcept {
    if (this != &other) {
        release();
        path_         = std::move(other.path_);
        handle_       = other.handle_;
        open_         = other.open_;
        other.handle_ = nullptr;
        other.open_   = false;
        other.path_.clear();
    }
    return *this;
}

bool TempFile::write(const std::string & content) {
    if (!open_) {
        return false;
    }
    DWORD written = 0;
    return WriteFile(static_cast<HANDLE>(handle_), content.data(), static_cast<DWORD>(content.size()), &written,
                      nullptr) &&
           written == content.size();
}

void TempFile::close() {
    if (open_) {
        CloseHandle(static_cast<HANDLE>(handle_));
        handle_ = nullptr;
        open_   = false;
    }
}

void TempFile::release() {
    close();
    if (!path_.empty()) {
        DeleteFileW(utf8_to_wide(path_).c_str());
        path_.clear();
    }
}

// -------------------------------------------------------------- external editor

EditorResult edit_in_external_editor(const std::string & content) {
    const std::string ed = env_str("OLLAMA_EDITOR", env_str("VISUAL", env_str("EDITOR", "notepad")));
    const auto         args = split_fields(ed);
    if (args.empty()) {
        return {"", "no editor configured, set OLLAMA_EDITOR to the path of your preferred editor"};
    }
    const auto resolved = look_path(args[0]);
    if (!resolved) {
        return {"", "editor \"" + args[0] + "\" not found, set OLLAMA_EDITOR to the path of your preferred editor"};
    }

    TempFile tmp("llmash-prompt-", ".txt");
    if (!tmp.ok()) {
        return {"", "creating temp file: could not create a temporary file"};
    }
    if (!content.empty() && !tmp.write(content)) {
        return {"", "creating temp file: could not write the temporary file"};
    }
    tmp.close();

    std::vector<std::string> cmd_args;
    cmd_args.push_back(*resolved);
    for (size_t i = 1; i < args.size(); i++) {
        cmd_args.push_back(args[i]);
    }
    cmd_args.push_back(tmp.path());

    std::string cmdline;
    for (size_t i = 0; i < cmd_args.size(); i++) {
        if (i) {
            cmdline += " ";
        }
        cmdline += quote_arg(cmd_args[i]);
    }

    STARTUPINFOW si;
    ZeroMemory(&si, sizeof(si));
    si.cb = sizeof(si);
    PROCESS_INFORMATION pi;
    ZeroMemory(&pi, sizeof(pi));

    std::wstring       wcmd = utf8_to_wide(cmdline);
    std::vector<wchar_t> buf(wcmd.begin(), wcmd.end());
    buf.push_back(L'\0');

    if (!CreateProcessW(nullptr, buf.data(), nullptr, nullptr, TRUE, 0, nullptr, nullptr, &si, &pi)) {
        return {"", "editor exited with error: could not launch " + args[0]};
    }
    ProcessGuard guard(pi.hProcess, pi.hThread);
    WaitForSingleObject(guard.process(), INFINITE);
    DWORD code = 0;
    GetExitCodeProcess(guard.process(), &code);
    if (code != 0) {
        return {"", "editor exited with error: exit status " + std::to_string(code)};
    }

    std::ifstream in(tmp.path(), std::ios::binary);
    if (!in) {
        return {"", "reading temp file: could not read the edited file"};
    }
    const std::string raw((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    std::string       norm;
    norm.reserve(raw.size());
    for (size_t i = 0; i < raw.size(); i++) {
        if (raw[i] == '\r' && i + 1 < raw.size() && raw[i + 1] == '\n') {
            continue;
        }
        norm.push_back(raw[i]);
    }
    const size_t end = norm.find_last_not_of('\n');
    norm             = (end == std::string::npos) ? "" : norm.substr(0, end + 1);
    return {norm, ""};
}

} // namespace llmash
