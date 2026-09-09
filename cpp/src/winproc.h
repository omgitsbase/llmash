#pragma once

#include <string>
#include <vector>

namespace llmash {

struct RunningProcess {
    unsigned long pid  = 0;
    unsigned long ppid = 0;
    std::string   name;  // lower case, with extension
    std::string   path;  // full image path, empty when it could not be read
};

std::vector<RunningProcess> running_processes();

// Every process whose image sits inside `dir`.
std::vector<RunningProcess> processes_under(const std::string & dir);

// Kills `pid` and any process named `child_name` whose parent it is.
int  kill_tree(unsigned long pid, const std::string & child_name);
bool kill_pid(unsigned long pid);
bool pid_alive(unsigned long pid, const std::string & expect_name = "");

// The pid the server records for itself, 0 when there is none.
unsigned long read_pid_file(const std::string & root);
void          write_pid_file(const std::string & root, unsigned long pid);
void          remove_pid_file(const std::string & root);

} // namespace llmash
