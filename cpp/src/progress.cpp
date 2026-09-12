#include "progress.h"

#include "cli_format.h"
#include "terminal.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace llmash {
namespace {

constexpr int kTermWidth  = 80;
constexpr int kTermHeight = 24;

const char * const kSpinnerParts[] = {"⠋", "⠙", "⠹", "⠸", "⠼",
                                      "⠴", "⠦", "⠧", "⠇", "⠏"};
constexpr size_t kSpinnerCount = sizeof(kSpinnerParts) / sizeof(kSpinnerParts[0]);

void out(const char * s) { std::fputs(s, stderr); }

std::string pad_left(const std::string & s, int n) {
    const int k = n - static_cast<int>(s.size());
    return k > 0 ? std::string(static_cast<size_t>(k), ' ') + s : s;
}

std::string repeat(const char * unit, int n) {
    std::string s;
    for (int i = 0; i < n; i++) {
        s += unit;
    }
    return s;
}

// Two units at most: "1h20m", "45s".
std::string format_duration(double seconds) {
    if (seconds >= 100 * 3600) {
        return "99h+";
    }
    const long long s = static_cast<long long>(std::llround(seconds));
    if (s >= 3600) {
        return std::to_string(s / 3600) + "h" + std::to_string((s % 3600) / 60) + "m";
    }
    if (s >= 60) {
        return std::to_string(s / 60) + "m" + std::to_string(s % 60) + "s";
    }
    return std::to_string(s) + "s";
}

double secs_since(ProgClock::time_point t) {
    return std::chrono::duration<double>(ProgClock::now() - t).count();
}

} // namespace

// ------------------------------------------------------------------ spinner

ProgSpinner::ProgSpinner(std::string message)
    : message_(std::move(message)), last_(ProgClock::now()) {}

void ProgSpinner::set_message(std::string m) {
    std::lock_guard<std::mutex> lock(mu_);
    message_ = std::move(m);
}

void ProgSpinner::stop() {
    std::lock_guard<std::mutex> lock(mu_);
    stopped_ = true;
}

std::string ProgSpinner::str() {
    std::lock_guard<std::mutex> lock(mu_);
    if (!stopped_ && secs_since(last_) >= 0.1) {
        value_ = (value_ + 1) % kSpinnerCount;
        last_  = ProgClock::now();
    }
    std::string s = message_;
    if (!s.empty()) {
        s += " ";
    }
    if (!stopped_) {
        s += kSpinnerParts[value_];
        s += " ";
    }
    return s;
}

// ---------------------------------------------------------------------- bar

ProgBar::ProgBar(std::string message, int64_t max_value, int64_t initial, ProgUnit unit)
    : message_(std::move(message)), max_value_(max_value), initial_(initial),
      current_(initial), unit_(unit), started_(ProgClock::now()) {
    if (initial >= max_value && max_value > 0) {
        stopped_    = true;
        stopped_at_ = ProgClock::now();
    }
}

double ProgBar::percent_locked() const {
    return max_value_ > 0 ? static_cast<double>(current_) / static_cast<double>(max_value_) * 100.0 : 0.0;
}

double ProgBar::rate_locked() const {
    double num = 0, den = 0;
    if (stopped_) {
        num = static_cast<double>(current_ - initial_);
        den = std::chrono::duration<double>(stopped_at_ - started_).count();
    } else if (buckets_.size() > 1) {
        const Bucket & first = buckets_.front();
        const Bucket & last  = buckets_.back();
        num = static_cast<double>(last.value - first.value);
        den = std::chrono::duration<double>(last.updated - first.updated).count();
    }
    return den > 0 ? num / den : 0.0;
}

void ProgBar::set(int64_t value) {
    std::lock_guard<std::mutex> lock(mu_);
    if (max_value_ > 0 && value >= max_value_) {
        value = max_value_;
    }
    current_ = value;
    if (max_value_ > 0 && current_ >= max_value_ && !stopped_) {
        stopped_    = true;
        stopped_at_ = ProgClock::now();
    }
    if (buckets_.empty() || secs_since(buckets_.back().updated) > 1.0) {
        buckets_.push_back({ProgClock::now(), value});
        if (buckets_.size() > 10) {
            buckets_.erase(buckets_.begin());
        }
    }
}

std::string ProgBar::str() {
    int w = terminal_columns();
    if (w <= 0) {
        w = kTermWidth;
    }
    std::lock_guard<std::mutex> lock(mu_);

    std::string pre = message_;
    if (!pre.empty()) {
        pre += " ";
    }
    char pct[16];
    std::snprintf(pct, sizeof(pct), "%3.0f%%", percent_locked());
    pre += pct;

    const auto figure = [&](int64_t v) {
        return unit_ == ProgUnit::Bytes ? human_bytes(v) : std::to_string(v);
    };
    std::string suf;
    if (!stopped_) {
        suf += pad_left(figure(current_), 6) + "/" + pad_left(figure(max_value_), 6);
    } else {
        suf += pad_left(figure(max_value_), 6) + std::string(7, ' ');
    }
    const double rate = rate_locked();
    if (!stopped_ && rate > 0) {
        // A count of tensors per second tells no one anything; the time left does.
        suf += unit_ == ProgUnit::Bytes ? "  " + pad_left(human_bytes(static_cast<int64_t>(rate)), 6) + "/s"
                                        : std::string(9, ' ');
        suf += "  " + pad_left(format_duration(static_cast<double>(max_value_ - current_) / rate), 6);
    } else {
        suf += std::string(18, ' ');
    }

    // The bar fills whatever the message and the figures leave, minus the
    // four cells the brackets and their spaces take.
    int f = w - static_cast<int>(pre.size()) - static_cast<int>(suf.size()) - 5;
    if (f < 0) {
        f = 0;
    }
    const int n = static_cast<int>(static_cast<double>(f) * percent_locked() / 100.0);
    std::string mid = " ▕";
    mid += repeat("█", n);
    mid += std::string(static_cast<size_t>(f - n), ' ');
    mid += "▏ ";
    return pre + mid + suf;
}

// ----------------------------------------------------------------- progress

Progress::Progress() {
    thread_ = std::thread([this] { run(); });
}

Progress::~Progress() {
    done_ = true;
    if (thread_.joinable()) {
        thread_.join();
    }
}

void Progress::add(const std::shared_ptr<ProgState> & state) {
    std::lock_guard<std::mutex> lock(mu_);
    states_.push_back(state);
}

void Progress::run() {
    while (!done_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        if (!done_) {
            render();
        }
    }
}

void Progress::render() {
    std::lock_guard<std::mutex> lock(mu_);
    render_locked();
}

void Progress::render_locked() {
    if (!stderr_is_terminal()) {
        return;
    }
    // 2026 is synchronized output: the terminal shows the whole frame at once
    // instead of each line as it arrives, and 25l hides the caret, which is
    // what was flickering through every redraw.
    out("\x1b[?2026h");
    out("\x1b[?25l");
    for (int i = 0; i < pos_ - 1; i++) {
        out("\x1b[A");
    }
    out("\x1b[1G");
    const int max = std::min(static_cast<int>(states_.size()), kTermHeight);
    for (size_t i = states_.size() - static_cast<size_t>(max); i < states_.size(); i++) {
        const std::string line = states_[i]->str();
        std::fputs(line.c_str(), stderr);
        out("\x1b[K");
        if (i + 1 < states_.size()) {
            out("\n");
        }
    }
    pos_ = static_cast<int>(states_.size());
    out("\x1b[?25h");
    out("\x1b[?2026l");
    std::fflush(stderr);
}

void Progress::stop() {
    bool first = false;
    {
        std::lock_guard<std::mutex> lock(mu_);
        first   = !halted_;
        halted_ = true;
    }
    done_ = true;
    if (thread_.joinable()) {
        thread_.join();
    }
    if (!first) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(mu_);
        for (const auto & s : states_) {
            if (auto * sp = dynamic_cast<ProgSpinner *>(s.get())) {
                sp->stop();
            }
        }
        render_locked();
    }
    if (stderr_is_terminal()) {
        out("\n");
        std::fflush(stderr);
    }
}

void Progress::stop_and_clear() {
    int lines = 0;
    {
        std::lock_guard<std::mutex> lock(mu_);
        lines   = pos_;
        halted_ = true;
    }
    done_ = true;
    if (thread_.joinable()) {
        thread_.join();
    }
    if (!stderr_is_terminal()) {
        return;
    }
    out("\x1b[?25l");
    for (int i = 0; i < lines; i++) {
        if (i > 0) {
            out("\x1b[A");
        }
        out("\x1b[2K\x1b[1G");
    }
    out("\x1b[?25h");
    std::fflush(stderr);
    std::lock_guard<std::mutex> lock(mu_);
    pos_ = 0;
}

} // namespace llmash
