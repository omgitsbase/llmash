#pragma once

#include <cstdio>
#include <string>

namespace llmash {

// The server's log. stderr until a sink is chosen.
void log_line(const std::string & s);
void logf(const char * fmt, ...);
bool set_log_file(const std::string & path);
void set_log_stream(FILE * f);

} // namespace llmash
