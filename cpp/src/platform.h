#pragma once

#include <cctype>
#include <cstdlib>
#include <ctime>
#include <string>

#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

namespace llmash {

inline void local_time(const std::time_t & t, std::tm & out) {
#ifdef _WIN32
    localtime_s(&out, &t);
#else
    localtime_r(&t, &out);
#endif
}

inline unsigned long current_pid() {
#ifdef _WIN32
    return static_cast<unsigned long>(_getpid());
#else
    return static_cast<unsigned long>(getpid());
#endif
}

inline void set_env(const char * name, const std::string & value) {
#ifdef _WIN32
    _putenv_s(name, value.c_str());
#else
    setenv(name, value.c_str(), 1);
#endif
}

inline bool starts_with_ci(const std::string & s, const std::string & prefix) {
    if (s.size() < prefix.size()) {
        return false;
    }
    for (size_t i = 0; i < prefix.size(); i++) {
        if (std::tolower(static_cast<unsigned char>(s[i])) != std::tolower(static_cast<unsigned char>(prefix[i]))) {
            return false;
        }
    }
    return true;
}

inline const char * llama_server_exe() {
#ifdef _WIN32
    return "llama-server.exe";
#else
    return "llama-server";
#endif
}

inline const char * llmash_exe() {
#ifdef _WIN32
    return "llmash.exe";
#else
    return "llmash";
#endif
}

inline const char * llmash_daemon_exe() {
#ifdef _WIN32
    return "llmashw.exe";
#else
    return "llmash";
#endif
}

} // namespace llmash
