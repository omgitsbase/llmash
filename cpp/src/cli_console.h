#pragma once

// Console plumbing the commands call directly: is-a-console tests, raw key
// reads, the one menu every question is asked with, and the colours.

#include <functional>
#include <nlohmann/json.hpp>

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

// Whether an agent harness or CI is driving rather than a person.
bool driven_by_agent();
bool is_console_stdout();

// _getch(): one raw key, no echo. Real console input only; not called by
// anything this module's own test exercises.
int raw_getch();

// A key from the console: arrows arrive as a 0xE0/0x00 prefix and a scan
// code, CR/LF as Enter, Esc/Ctrl-C as Escape, everything else as itself.
enum PickKey {
    kPickUp    = 0x101,
    kPickDown  = 0x102,
    kPickEnter = 0x103,
    kPickEsc   = 0x104,
    kPickRight = 0x105,
    kPickLeft  = 0x106,
};

using ByteSource = std::function<int()>;
int read_pick(const ByteSource & next_byte);

// Every question the CLI asks has one look:
//
//   ? Pull which build?
//     ❯ medium   Q4_K_M    5.0 GB
//       large    Q8_0      8.2 GB
//     ↑↓ move · enter select · a all builds · esc cancel
//
// and once answered it folds to the question and its answer:  ✓ Pull which build? Q4_K_M
struct MenuKey {
    int         key;  // a character, or kPickRight / kPickLeft
    std::string what; // what it does, for the hint
};
struct Menu {
    std::string              question;
    std::vector<std::string> rows;
    std::vector<std::string> notes;   // dim text after each row, lined up; may be empty
    std::vector<std::string> answers; // what the folded line shows for each row; the row itself when empty
    int                      cursor = 0;
    std::string              esc    = "cancel"; // what Escape does
    std::vector<MenuKey>     keys;              // keys beyond the arrows; pressing one returns minus the key
};

// The row chosen, -1 for Escape, or minus an extra key, with the row the cursor was on in `at`. Without a terminal
// it prints the question and returns -1.
int choose(const Menu & m, int * at = nullptr);
// The same look for a yes or no: "? Install it? (Y/n)", Enter takes the default and Escape is no.
bool ask_yes(const std::string & question, bool yes_by_default = true);

// What choose() draws, as lines: the question, the rows that fit in `height` lines around the cursor, and the hint.
std::vector<std::string> menu_frame(const Menu & m, int cursor, int height);
std::string              menu_folded(const std::string & question, const std::string & answer, bool cancelled);
std::string              key_hint(const std::vector<MenuKey> & keys, const std::string & esc);
// Columns a string takes on screen: escape sequences take none, a UTF-8 character one.
int display_width(const std::string & s);

// The drafter offer's memory in local.json: never for any model (ask_drafter: false), or never again for a model it
// was asked about (drafter_asked). The answer is 0 look now, 1 no for this model, 2 no for any model.
bool drafter_offer_due(const nlohmann::json & local, const std::string & model);
void drafter_offer_answered(nlohmann::json & local, const std::string & model, int answer);

bool clip_copy(const std::string & text);

} // namespace llmash
