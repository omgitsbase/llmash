#include "cli_process.h"

#include <subprocess.h>

#include <cstdio>

namespace llmash {

namespace {

// Owns a subprocess_s: subprocess_destroy() always runs once create()
// succeeds, on every return path (RAII in place of Go's SysProcAttr flags +
// manual cmd.Process.Release()).
class ScopedProcess {
public:
    ~ScopedProcess() {
        if (created_) {
            subprocess_destroy(&proc_);
        }
    }

    bool create(const std::vector<std::string> & argv, int options) {
        std::vector<const char *> cargv;
        cargv.reserve(argv.size() + 1);
        for (const auto & a : argv) {
            cargv.push_back(a.c_str());
        }
        cargv.push_back(nullptr);
        created_ = subprocess_create(cargv.data(), options, &proc_) == 0;
        return created_;
    }

    subprocess_s * get() { return &proc_; }

private:
    subprocess_s proc_{};
    bool         created_ = false;
};

} // namespace

ProcessResult run_hidden(const std::vector<std::string> & argv, const std::string * stdin_data, bool capture_stdout,
                          bool wait) {
    ProcessResult  result;
    ScopedProcess  proc;
    const int      options = subprocess_option_no_window | subprocess_option_inherit_environment;
    if (!proc.create(argv, options)) {
        return result;
    }
    result.started = true;

    if (stdin_data != nullptr) {
        FILE * in = subprocess_stdin(proc.get());
        if (in != nullptr) {
            std::fwrite(stdin_data->data(), 1, stdin_data->size(), in);
            std::fflush(in);
        }
    }

    if (!wait) {
        return result;
    }

    if (capture_stdout) {
        FILE * out = subprocess_stdout(proc.get());
        if (out != nullptr) {
            char buf[4096];
            size_t n;
            while ((n = std::fread(buf, 1, sizeof(buf), out)) > 0) {
                result.out.append(buf, n);
            }
        }
    }

    subprocess_join(proc.get(), &result.exit_code);
    return result;
}

int run_streaming(const std::vector<std::string> & argv, const std::function<void(const std::string &)> & on_line) {
    ScopedProcess proc;
    const int     options = subprocess_option_no_window | subprocess_option_inherit_environment |
                        subprocess_option_combined_stdout_stderr;
    if (!proc.create(argv, options)) {
        return -1;
    }
    FILE * out = subprocess_stdout(proc.get());
    if (out != nullptr) {
        std::string line;
        char        buf[4096];
        size_t      n;
        while ((n = std::fread(buf, 1, sizeof(buf), out)) > 0) {
            for (size_t i = 0; i < n; i++) {
                if (buf[i] == '\n') {
                    while (!line.empty() && line.back() == '\r') {
                        line.pop_back();
                    }
                    if (on_line) {
                        on_line(line);
                    }
                    line.clear();
                } else {
                    line.push_back(buf[i]);
                }
            }
        }
        if (!line.empty() && on_line) {
            on_line(line);
        }
    }
    int code = -1;
    subprocess_join(proc.get(), &code);
    return code;
}

} // namespace llmash
