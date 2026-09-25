#pragma once

#include <string>
#include <vector>

namespace llmash {

inline constexpr const char * kRuntimeHostArg = "runtime";

std::vector<std::string> hosted_command(const std::vector<std::string> & argv);
int                      runtime_host_main();
bool                     is_runtime_host(unsigned long pid);
std::vector<std::string> runtime_process_names();

} // namespace llmash
