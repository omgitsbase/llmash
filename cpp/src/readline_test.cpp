// The line editor fed key sequences: ctrl and alt with the arrows jump words, the control sequences for delete,
// home, end and bracketed paste are read whole, and a redraw is one write with the cursor hidden meanwhile.
#include "readline.h"

#include <cstdio>
#include <sstream>

using namespace llmash;

namespace {

int failures = 0;

void check(bool ok, const char * what) {
    if (!ok) {
        failures++;
        std::printf("FAIL %s\n", what);
    }
}

std::string typed(const std::string & keys) {
    Editor             ed;
    VectorRuneSource   in(keys);
    std::ostringstream out;
    return ed.read_line(in, out).text;
}

}  // namespace

int main() {
    check(typed("hello world\x1b[1;5DX\r") == "hello Xworld", "ctrl+left jumps to the start of the word");
    check(typed("hello world\x1b[1;5D\x1b[1;5D\x1b[1;5CX\r") == "helloX world", "ctrl+right jumps to the end of the word");
    check(typed("hello world\x1b[1;3DX\r") == "hello Xworld", "alt+left jumps a word too");
    check(typed("hello world\x1b[DX\r") == "hello worlXd", "a plain arrow moves one character");
    check(typed("abc\x1b[D\x1b[3~\r") == "ab", "delete takes the character under the cursor and nothing after it");
    check(typed("abc\x1b[HX\x1b[FY\r") == "XabcY", "home and end");
    check(typed("abc\x1b[1;5HX\r") == "Xabc", "ctrl+home is home");

    {
        Editor             ed;
        VectorRuneSource   in(std::string("\x1b[200~pasted\x1b[201~\r"));
        std::ostringstream out;
        bool               seen = false;
        for (;;) {
            const StepResult sr = ed.step(in, out);
            seen = seen || ed.pasting();
            if (sr.done) {
                check(sr.text == "pasted", "a bracketed paste arrives as text");
                break;
            }
        }
        check(seen && !ed.pasting(), "pasting is on between the markers and off after");
    }
    {
        Editor             ed;
        VectorRuneSource   in(std::string("a"));
        std::ostringstream out;
        ed.step(in, out);
        const std::string s = out.str();
        check(s.find("\x1b[?25l") != std::string::npos && s.rfind("\x1b[?25h") == s.size() - 6, "a redraw hides the cursor and shows it last");
    }
    if (failures == 0) {
        std::printf("readline_test: all passed\n");
    }
    return failures == 0 ? 0 : 1;
}
