#pragma once

// Spinners and bars on stderr, redrawn in place ten times a second.

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace llmash {

using ProgClock = std::chrono::steady_clock;

class ProgState {
public:
    virtual ~ProgState() = default;
    virtual std::string str() = 0;
};

class ProgSpinner : public ProgState {
public:
    explicit ProgSpinner(std::string message);

    std::string str() override;
    void        set_message(std::string m);
    void        stop();

private:
    std::mutex          mu_;
    std::string         message_;
    size_t              value_ = 0;
    bool                stopped_ = false;
    ProgClock::time_point last_;
};

// What the two figures beside a bar count. Bytes get a rate; a count of
// tensors or files gets the plain numbers and the time left.
enum class ProgUnit { Bytes, Count };

class ProgBar : public ProgState {
public:
    ProgBar(std::string message, int64_t max_value, int64_t initial = 0, ProgUnit unit = ProgUnit::Bytes);

    std::string str() override;
    void        set(int64_t value);

private:
    struct Bucket {
        ProgClock::time_point updated;
        int64_t               value;
    };

    double percent_locked() const;
    double rate_locked() const;

    std::mutex            mu_;
    std::string           message_;
    int64_t               max_value_;
    int64_t               initial_;
    int64_t               current_;
    ProgUnit              unit_    = ProgUnit::Bytes;
    bool                  stopped_ = false;
    ProgClock::time_point started_;
    ProgClock::time_point stopped_at_;
    std::vector<Bucket>   buckets_;
};

class Progress {
public:
    Progress();
    ~Progress();

    void add(const std::shared_ptr<ProgState> & state);
    void stop(); // leaves the finished lines on screen
    void stop_and_clear();

private:
    void render();
    void render_locked();
    void run();

    std::mutex                              mu_;
    std::vector<std::shared_ptr<ProgState>> states_;
    int                                     pos_ = 0;
    std::atomic<bool>                       done_{false};
    bool                                    halted_ = false;
    std::thread                             thread_;
};

} // namespace llmash
