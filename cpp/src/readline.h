#pragma once

#include <cstdint>
#include <iosfwd>
#include <optional>
#include <string>
#include <vector>

namespace llmash {

// One raw key at a time, decoded to a Unicode code point.
class RuneSource {
public:
    virtual ~RuneSource()                    = default;
    virtual std::optional<char32_t> next()   = 0;
};

// A fixed sequence of synthetic key events, for driving the editor without a
// real console: what the tests use.
class VectorRuneSource : public RuneSource {
public:
    explicit VectorRuneSource(std::vector<char32_t> runes);
    explicit VectorRuneSource(const std::string & utf8);

    std::optional<char32_t> next() override;

private:
    std::vector<char32_t> runes_;
    size_t                pos_ = 0;
};

// The real thing: raw bytes off STD_INPUT_HANDLE, UTF-8 decoded.
class ConsoleRuneSource : public RuneSource {
public:
    std::optional<char32_t> next() override;
};

// ~/.ollama/history (the same file and format ollama's own CLI uses).
struct History {
    std::vector<std::string> lines;
    size_t                   pos     = 0;
    bool                     enabled = true;
    size_t                   limit   = 100;

    History();                       // loads from the default path
    explicit History(std::string path); // for tests: a path of your own

    void add(const std::string & s);

private:
    std::string path_;
    void        save();
};

std::string default_history_path();

enum class ReadStatus { Ok, Interrupt, EditPrompt, Eof };

struct LineResult {
    std::string text;
    ReadStatus  status = ReadStatus::Ok;
};

// The result of one key: either "keep reading" or the finished line.
struct StepResult {
    bool        done = false;
    std::string text;
    ReadStatus  status = ReadStatus::Ok;
};

// The prompt line for `run`: raw console input with VT escape sequences, the
// same keys and history file ollama uses, and its grey placeholder.
class Editor {
public:
    Editor();
    explicit Editor(History hist);

    // Blocks on the real console.
    LineResult read_line();

    // The same loop, driven by any RuneSource and writing to any ostream:
    // what makes it testable without a console.
    LineResult read_line(RuneSource & in, std::ostream & out);

    // One key. Public so the state machine can be driven and inspected a
    // step at a time in tests, independent of the blocking console read.
    StepResult step(RuneSource & in, std::ostream & out);

    void set_prefill(std::string s) { prefill_ = std::move(s); }

    const std::u32string &          buffer() const { return buf_; }
    size_t                          cursor() const { return pos_; }
    bool                            using_alt_prompt() const { return use_alt_; }
    bool                            pasting() const { return pasting_; }
    const std::vector<std::string> & pasted_lines() const { return pasted_lines_; }
    History &                       history() { return hist_; }

    std::string prompt        = ">>> ";
    std::string alt_prompt    = "... ";
    std::string placeholder   = "Send a message (/? for help)";
    std::string alt_placeholder = "Press Enter to send";

private:
    std::string prefill_;
    History     hist_;

    bool                     use_alt_ = false;
    bool                     pasting_ = false;
    std::u32string           buf_;
    size_t                   pos_  = 0;
    int                      rows_ = 0; // rows below the first that the last redraw used
    std::vector<std::string> pasted_lines_; // lines already committed inside this one prompt

    bool           esc_    = false;
    bool           escex_  = false;
    bool           metaDel_ = false;
    std::u32string saved_;

    std::string cur_prompt() const;
    std::string cur_placeholder() const;
    void        draw(std::ostream & out);
    void        insert(char32_t r);
    void        replace(const std::u32string & s);
    size_t      left_word() const;
    size_t      right_word() const;
    void        pasted_line(std::ostream & out);
};

// Terminal cell width: two columns for the wide CJK and emoji blocks, none
// for combining marks (a fixed set of the common combining-mark blocks, not
// the full Unicode Mn/Me category tables the Go original gets from its
// standard library).
int disp_width(const std::u32string & s);
int disp_width(const std::string & utf8);

std::u32string utf8_decode(const std::string & s);
std::string    utf8_encode(const std::u32string & s);

struct EditorResult {
    std::string text;
    std::string error; // empty on success
};

// Opens content in $OLLAMA_EDITOR/$VISUAL/$EDITOR (or notepad), waits for it
// to exit, and returns what it left behind.
EditorResult edit_in_external_editor(const std::string & content);

// A file under the temp directory, removed on destruction. Exposed for its
// own test; edit_in_external_editor is its only real caller.
class TempFile {
public:
    TempFile(const std::string & prefix, const std::string & suffix);
    ~TempFile();
    TempFile(const TempFile &)             = delete;
    TempFile & operator=(const TempFile &) = delete;
    TempFile(TempFile && other) noexcept;
    TempFile & operator=(TempFile && other) noexcept;

    bool               ok() const { return path_.empty() == false && open_; }
    const std::string & path() const { return path_; }
    bool               write(const std::string & content);
    void               close();

private:
    std::string   path_;
    void *        handle_ = nullptr; // HANDLE, opaque here so windows.h stays out of the header
    bool          open_   = false;

    void release();
};

} // namespace llmash
