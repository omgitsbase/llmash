#pragma once

// Console plumbing the commands call directly: is-a-console tests, raw key
// reads, the pick-menu key decoder, and the colours `link` prints with.

#include <functional>
#include <string>
#include <vector>

namespace llmash {

constexpr const char * kDim    = "\x1b[2m";
constexpr const char * kBold   = "\x1b[1m";
constexpr const char * kReset  = "\x1b[0m";
constexpr const char * kOrange = "\x1b[38;5;208m";
constexpr const char * kGreen  = "\x1b[38;5;42m";
constexpr const char * kCyan   = "\x1b[38;5;44m";

bool is_console_stdin();
bool is_console_stdout();

// _getch(): one raw key, no echo. Real console input only; not called by
// anything this module's own test exercises.
int raw_getch();

// A key from the console: arrows arrive as a 0xE0/0x00 prefix and a scan
// code, CR/LF as Enter, Esc/Ctrl-C as Escape, everything else as itself.
enum PickKey { kPickUp = 0x101, kPickDown = 0x102, kPickEnter = 0x103, kPickEsc = 0x104 };

using ByteSource = std::function<int()>;
int read_pick(const ByteSource & next_byte);

// One step of askNumber's digit-buffer state machine.
struct AskNumberStep {
    bool done     = false;
    bool reprompt = false;
    int  value    = 0;
};
AskNumberStep ask_number_feed(int ch, std::string & buf, int def, int max);

// y/n confirm's own tiny state machine: given a raw byte, either a final
// answer or "keep reading".
struct ConfirmStep {
    bool decided = false;
    bool yes     = false;
};
ConfirmStep confirm_feed(int ch);

bool confirm(const std::string & question);
int  ask_number(const std::string & prompt, int def, int max);

// Draws a list with a cursor, moving it with the arrow keys; Enter returns
// the row, Escape -1, and a key in extra its negative code.
int pick_menu(const std::string & title, const std::vector<std::string> & rows, int cursor,
              const std::string & hint, const std::string & extra);

bool clip_copy(const std::string & text);

} // namespace llmash
