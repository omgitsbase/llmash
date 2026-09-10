#pragma once

#include <string>
#include <vector>

namespace llmash {

struct HttpReply {
    int         status = 0;
    std::string body;
    std::string error;
};

HttpReply http_get(const std::string & url, const std::vector<std::string> & headers, int timeout_s);

} // namespace llmash
