#include "cli_console.h"
#include "cli_process.h"
#include "terminal.h"


#include <cstdio>
#include <cstdlib>

namespace llmash {

bool is_console_stdin() { return stdin_is_terminal(); }
bool is_console_stdout() { return stdout_is_terminal(); }

int raw_getch() { return raw_key(); }

int read_pick(const ByteSource & next_byte) {
    const int ch = next_byte();
    if (ch == 0xE0 || ch == 0x00) {
        switch (next_byte()) {
            case 0x48: return kPickUp;
            case 0x50: return kPickDown;
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

AskNumberStep ask_number_feed(int ch, std::string & buf, int def, int max) {
    AskNumberStep step;
    if (ch == '\r' || ch == '\n') {
        int n = def;
        if (!buf.empty()) {
            n = std::atoi(buf.c_str());
        }
        if (n >= 0 && n <= max) {
            step.done  = true;
            step.value = n;
            return step;
        }
        buf.clear();
        step.reprompt = true;
        return step;
    }
    if (ch == 0x1b || ch == 0x03) {
        step.done  = true;
        step.value = -1;
        return step;
    }
    if (ch >= '0' && ch <= '9') {
        buf.push_back(static_cast<char>(ch));
        return step;
    }
    if (ch == 8 || ch == 127) {
        if (!buf.empty()) {
            buf.pop_back();
        }
    }
    return step;
}

ConfirmStep confirm_feed(int ch) {
    ConfirmStep step;
    if (ch == 'y' || ch == 'Y' || ch == '\r' || ch == '\n') {
        step.decided = true;
        step.yes     = true;
    } else if (ch == 'n' || ch == 'N' || ch == 0x1b || ch == 0x03) {
        step.decided = true;
        step.yes     = false;
    }
    return step;
}

bool confirm(const std::string & question) {
    std::printf("%s [Y/n] ", question.c_str());
    for (;;) {
        const ConfirmStep step = confirm_feed(raw_getch());
        if (!step.decided) {
            continue;
        }
        std::printf(step.yes ? "yes\n" : "no\n");
        return step.yes;
    }
}

int ask_number(const std::string & prompt, int def, int max) {
    std::printf("%s [%d] ", prompt.c_str(), def);
    std::string buf;
    for (;;) {
        const int           ch   = raw_getch();
        const AskNumberStep step = ask_number_feed(ch, buf, def, max);
        if (step.done) {
            std::printf("\n");
            return step.value;
        }
        if (step.reprompt) {
            std::printf("\n%s [%d] ", prompt.c_str(), def);
            continue;
        }
        if (ch >= '0' && ch <= '9') {
            std::printf("%c", ch);
        } else if (ch == 8 || ch == 127) {
            std::printf("\b \b");
        }
    }
}

int pick_menu(const std::string & title, const std::vector<std::string> & rows, int cursor, const std::string & hint,
              const std::string & extra) {
    if (cursor < 0 || cursor >= static_cast<int>(rows.size())) {
        cursor = 0;
    }
    std::printf("%s\n", title.c_str());
    int        drawn = 0;
    const auto draw  = [&]() {
        for (int i = 0; i < drawn; i++) {
            std::printf("\x1b[A");
        }
        for (size_t i = 0; i < rows.size(); i++) {
            if (static_cast<int>(i) == cursor) {
                std::printf("\x1b[1G\x1b[K  %s\xe2\x9d\xaf %s%s\n", kBold, rows[i].c_str(), kReset);
            } else {
                std::printf("\x1b[1G\x1b[K    %s\n", rows[i].c_str());
            }
        }
        std::printf("\x1b[1G\x1b[K  %s%s%s\n", kDim, hint.c_str(), kReset);
        drawn = static_cast<int>(rows.size()) + 1;
    };
    draw();
    for (;;) {
        const int k = read_pick([]() { return raw_getch(); });
        if (k == kPickUp && cursor > 0) {
            cursor--;
            draw();
        } else if (k == kPickDown && cursor < static_cast<int>(rows.size()) - 1) {
            cursor++;
            draw();
        } else if (k == kPickEnter) {
            return cursor;
        } else if (k == kPickEsc) {
            return -1;
        } else if (k > 0 && k < 0x100 && extra.find(static_cast<char>(k)) != std::string::npos) {
            return -k;
        }
    }
}

bool clip_copy(const std::string & text) {
    const ProcessResult r = run_hidden({"clip"}, &text, false, true);
    return r.started && r.exit_code == 0;
}

} // namespace llmash
