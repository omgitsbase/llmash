#include "terminal.h"

#ifdef _WIN32
#include <windows.h>

#include <conio.h>
#include <io.h>

#include <cstdio>
#else
#include <sys/ioctl.h>
#include <termios.h>
#include <unistd.h>

#include <cstdio>
#endif

namespace llmash {

#ifdef _WIN32

int  raw_key() { return _getch(); }
bool stdin_is_terminal() { return _isatty(_fileno(stdin)) != 0; }
bool stdout_is_terminal() { return _isatty(_fileno(stdout)) != 0; }

int terminal_columns() {
    CONSOLE_SCREEN_BUFFER_INFO info;
    if (!GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &info)) {
        return 100;
    }
    const int w = info.srWindow.Right - info.srWindow.Left + 1;
    return w > 0 ? w : 100;
}

#else

namespace {

class RawMode {
public:
    RawMode() {
        if (tcgetattr(STDIN_FILENO, &saved_) != 0) {
            return;
        }
        ok_ = true;
        struct termios raw = saved_;
        raw.c_lflag &= ~static_cast<tcflag_t>(ICANON | ECHO);
        raw.c_cc[VMIN]  = 1;
        raw.c_cc[VTIME] = 0;
        tcsetattr(STDIN_FILENO, TCSANOW, &raw);
    }
    ~RawMode() {
        if (ok_) {
            tcsetattr(STDIN_FILENO, TCSANOW, &saved_);
        }
    }

private:
    struct termios saved_ {};
    bool           ok_ = false;
};

int read_byte() {
    unsigned char c = 0;
    return read(STDIN_FILENO, &c, 1) == 1 ? static_cast<int>(c) : -1;
}

} // namespace

int raw_key() {
    static int pending = 0;
    if (pending != 0) {
        const int k = pending;
        pending     = 0;
        return k;
    }

    const RawMode raw;
    const int     c = read_byte();
    if (c != 0x1b) {
        return c < 0 ? 0x03 : c;
    }

    struct termios t {};
    if (tcgetattr(STDIN_FILENO, &t) == 0) {
        struct termios peek = t;
        peek.c_cc[VMIN]     = 0;
        peek.c_cc[VTIME]    = 1;
        tcsetattr(STDIN_FILENO, TCSANOW, &peek);
        const int b1 = read_byte();
        const int b2 = b1 == '[' || b1 == 'O' ? read_byte() : -1;
        tcsetattr(STDIN_FILENO, TCSANOW, &t);
        if (b1 < 0) {
            return 0x1b;
        }
        switch (b2) {
            case 'A': pending = 0x48; return 0xE0;
            case 'B': pending = 0x50; return 0xE0;
            case 'C': pending = 0x4D; return 0xE0;
            case 'D': pending = 0x4B; return 0xE0;
            case 'H': pending = 0x47; return 0xE0;
            case 'F': pending = 0x4F; return 0xE0;
            case '3': read_byte(); pending = 0x53; return 0xE0;
            default: break;
        }
    }
    return 0x1b;
}

bool stdin_is_terminal() { return isatty(STDIN_FILENO) != 0; }
bool stdout_is_terminal() { return isatty(STDOUT_FILENO) != 0; }

int terminal_columns() {
    struct winsize ws {};
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0 && ws.ws_col > 0) {
        return ws.ws_col;
    }
    return 100;
}

#endif

} // namespace llmash
