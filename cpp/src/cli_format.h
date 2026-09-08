#pragma once

// Port of the pure, non-interactive parts of cmd/llmash/table.go and
// cmd/llmash/render.go: the borderless table layout `list`/`ps`/`show`
// print, and ollama's byte/time/count formatters.

#include <cstdint>
#include <ctime>
#include <string>
#include <vector>

namespace llmash {

// Terminal cell width: 2 for the wide CJK/emoji blocks, 0 for combining marks.
int disp_width(const std::string & utf8);

// Ollama's tablewriter layout: no borders, no header rule, left-aligned,
// four-space gaps, and a wrap at wrap_at columns (0 = no wrap, table.go's
// default is 30).
class Table {
public:
    explicit Table(std::vector<std::string> header = {});

    void add(std::vector<std::string> cells);
    std::string str() const;

    int wrap_at = 30;

private:
    std::vector<std::string>              header_;
    std::vector<std::vector<std::string>> rows_;

    std::vector<std::string> split(const std::string & cell) const;
};

std::vector<std::string> wrap_string(const std::string & s, int lim);
std::string              tw_title(const std::string & name);

// Small string utilities shared with cli_commands.cpp (Go's strings.Split
// on "\n", strings.Fields, strings.Join and strings.TrimSpace).
std::vector<std::string> split_lines(const std::string & s);
std::vector<std::string> split_fields(const std::string & s);
std::string              join(const std::vector<std::string> & v, const std::string & sep);
std::string              trim(const std::string & s);

std::string human_bytes(int64_t b);
std::string human_number(uint64_t b);
std::string human_duration(double seconds);
// zero_value is returned for a zero time_t (Go's time.Time{}); is_zero lets a
// caller who has no valid timestamp at all say so without inventing a time_t.
std::string human_time(std::time_t t, bool is_zero, const std::string & zero_value, double now);
// Parses an RFC3339(-nano) timestamp (what the server's JSON carries); on a
// parse failure, or an empty string, returns zero_value.
std::string human_time_iso(const std::string & s, const std::string & zero_value, double now);

// cmds.go's own byte formatter (1024-based, used nowhere by the client
// commands themselves today but ported for fidelity: it is the same package-
// level `human` that api.go's create/copy handlers call).
std::string human(double n);

// A UTC "YYYY-MM-DDTHH:MM:SS[.fff][Z|+HH:MM]" timestamp as a Unix time, or
// -1 on a parse failure.
double parse_rfc3339(const std::string & s);

} // namespace llmash
