#include "log.h"

#include <cstdarg>
#include <ctime>
#include <mutex>

namespace llmash {
namespace {

std::mutex g_mu;
FILE *     g_out = nullptr;

} // namespace

bool set_log_file(const std::string & path) {
    FILE * f = std::fopen(path.c_str(), "ab");
    if (f == nullptr) {
        return false;
    }
    std::lock_guard<std::mutex> lock(g_mu);
    if (g_out != nullptr && g_out != stderr && g_out != stdout) {
        std::fclose(g_out);
    }
    g_out = f;
    return true;
}

void set_log_stream(FILE * f) {
    std::lock_guard<std::mutex> lock(g_mu);
    if (g_out != nullptr && g_out != stderr && g_out != stdout) {
        std::fclose(g_out);
    }
    g_out = f;
}

void log_line(const std::string & s) {
    std::lock_guard<std::mutex> lock(g_mu);
    FILE * out = g_out != nullptr ? g_out : stderr;
    char   stamp[32];
    const std::time_t t = std::time(nullptr);
    std::tm           tm{};
    localtime_s(&tm, &t);
    std::strftime(stamp, sizeof(stamp), "%Y/%m/%d %H:%M:%S", &tm);
    std::fprintf(out, "%s [llmash] %s\n", stamp, s.c_str());
    std::fflush(out);
}

void logf(const char * fmt, ...) {
    char    buf[4096];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    log_line(buf);
}

} // namespace llmash
