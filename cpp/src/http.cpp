#include "http.h"

#ifdef _WIN32
#include <windows.h>

#include <winhttp.h>
#pragma comment(lib, "winhttp.lib")
#else
#include <curl/curl.h>
#endif

namespace llmash {

#ifdef _WIN32

namespace {

std::wstring widen(const std::string & s) {
    if (s.empty()) {
        return L"";
    }
    const int    n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, nullptr, 0);
    std::wstring w(static_cast<size_t>(n), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, w.data(), n);
    w.resize(static_cast<size_t>(n - 1));
    return w;
}

struct Handle {
    HINTERNET h = nullptr;
    ~Handle() {
        if (h) {
            WinHttpCloseHandle(h);
        }
    }
};

} // namespace

HttpReply http_get(const std::string & url, const std::vector<std::string> & headers, int timeout_s) {
    HttpReply out;
    Handle    session{WinHttpOpen(L"llmash", WINHTTP_ACCESS_TYPE_DEFAULT_PROXY, WINHTTP_NO_PROXY_NAME,
                                  WINHTTP_NO_PROXY_BYPASS, 0)};
    if (!session.h) {
        out.error = "WinHttpOpen failed";
        return out;
    }
    const int ms = timeout_s > 0 ? timeout_s * 1000 : 30000;
    WinHttpSetTimeouts(session.h, ms, ms, ms, ms);

    URL_COMPONENTS uc{};
    uc.dwStructSize      = sizeof(uc);
    std::wstring wurl    = widen(url);
    std::wstring host(256, L'\0'), path(8192, L'\0');
    uc.lpszHostName      = host.data();
    uc.dwHostNameLength  = static_cast<DWORD>(host.size() - 1);
    uc.lpszUrlPath       = path.data();
    uc.dwUrlPathLength   = static_cast<DWORD>(path.size() - 1);
    if (!WinHttpCrackUrl(wurl.c_str(), static_cast<DWORD>(wurl.size()), 0, &uc)) {
        out.error = "could not parse " + url;
        return out;
    }

    Handle connect{WinHttpConnect(session.h, std::wstring(uc.lpszHostName, uc.dwHostNameLength).c_str(),
                                  uc.nPort, 0)};
    if (!connect.h) {
        out.error = "could not connect";
        return out;
    }
    Handle request{WinHttpOpenRequest(connect.h, L"GET", std::wstring(uc.lpszUrlPath, uc.dwUrlPathLength).c_str(),
                                      nullptr, WINHTTP_NO_REFERER, WINHTTP_DEFAULT_ACCEPT_TYPES,
                                      uc.nScheme == INTERNET_SCHEME_HTTPS ? WINHTTP_FLAG_SECURE : 0)};
    if (!request.h) {
        out.error = "could not build the request";
        return out;
    }

    std::wstring all;
    for (const std::string & h : headers) {
        all += widen(h) + L"\r\n";
    }
    if (!WinHttpSendRequest(request.h, all.empty() ? WINHTTP_NO_ADDITIONAL_HEADERS : all.c_str(),
                            all.empty() ? 0 : static_cast<DWORD>(-1), WINHTTP_NO_REQUEST_DATA, 0, 0, 0) ||
        !WinHttpReceiveResponse(request.h, nullptr)) {
        out.error = "no answer";
        return out;
    }

    DWORD status = 0, size = sizeof(status);
    WinHttpQueryHeaders(request.h, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER,
                        WINHTTP_HEADER_NAME_BY_INDEX, &status, &size, WINHTTP_NO_HEADER_INDEX);
    out.status = static_cast<int>(status);

    for (;;) {
        DWORD avail = 0;
        if (!WinHttpQueryDataAvailable(request.h, &avail) || avail == 0) {
            break;
        }
        std::string chunk(avail, '\0');
        DWORD       got = 0;
        if (!WinHttpReadData(request.h, chunk.data(), avail, &got)) {
            break;
        }
        out.body.append(chunk, 0, got);
        if (out.body.size() > (1u << 24)) {
            break;
        }
    }
    return out;
}

#else

namespace {

size_t collect(char * p, size_t size, size_t n, void * userdata) {
    static_cast<std::string *>(userdata)->append(p, size * n);
    return size * n;
}

} // namespace

HttpReply http_get(const std::string & url, const std::vector<std::string> & headers, int timeout_s) {
    HttpReply out;
    CURL *    curl = curl_easy_init();
    if (curl == nullptr) {
        out.error = "curl_easy_init failed";
        return out;
    }

    struct curl_slist * list = nullptr;
    for (const std::string & h : headers) {
        list = curl_slist_append(list, h.c_str());
    }

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "llmash");
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, static_cast<long>(timeout_s > 0 ? timeout_s : 30));
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, collect);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &out.body);
    if (list != nullptr) {
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, list);
    }

    const CURLcode rc = curl_easy_perform(curl);
    if (rc != CURLE_OK) {
        out.error = curl_easy_strerror(rc);
    } else {
        long code = 0;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &code);
        out.status = static_cast<int>(code);
    }

    if (list != nullptr) {
        curl_slist_free_all(list);
    }
    curl_easy_cleanup(curl);
    return out;
}

#endif

} // namespace llmash
