#include "tray_internal.h"

#include <subprocess.h>

#include <thread>

namespace llmash {

namespace {

// RAII over subprocess_s: subprocess_destroy closes the process/thread
// handles and the stdio pipes it created, on every return path.
class Subprocess {
public:
    Subprocess() = default;
    ~Subprocess() {
        if (created_) {
            subprocess_destroy(&proc_);
        }
    }
    Subprocess(const Subprocess &)             = delete;
    Subprocess & operator=(const Subprocess &) = delete;

    bool create(const char * const argv[], int options) {
        created_ = subprocess_create(argv, options, &proc_) == 0;
        return created_;
    }
    subprocess_s * get() { return &proc_; }

private:
    subprocess_s proc_{};
    bool         created_ = false;
};

} // namespace

PowerShellRun run_hidden_powershell(const std::string & script) {
    PowerShellRun result;

    const char * argv[] = {"powershell", "-NoProfile", "-NonInteractive", "-WindowStyle",
                           "Hidden",     "-Command",    script.c_str(),   nullptr};

    Subprocess sub;
    if (!sub.create(argv, subprocess_option_no_window | subprocess_option_search_user_path)) {
        return result;
    }
    result.started = true;

    // Drained concurrently with the wait below: a script that ever printed
    // more than one pipe buffer's worth would otherwise deadlock against
    // subprocess_join (child blocked writing, parent blocked joining).
    std::thread drain([&] {
        char     buf[4096];
        unsigned n;
        while ((n = subprocess_read_stdout(sub.get(), buf, sizeof(buf))) > 0) {
            result.stdout_text.append(buf, n);
        }
    });
    subprocess_join(sub.get(), &result.exit_code);
    drain.join();
    return result;
}

} // namespace llmash
