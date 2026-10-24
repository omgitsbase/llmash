#include "private_file.h"

#include <cstdio>
#include <system_error>

#ifdef _WIN32
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#    include <sddl.h>
#else
#    include <fcntl.h>
#    include <sys/stat.h>
#    include <unistd.h>
#endif

namespace fs = std::filesystem;

namespace llmash {

#ifdef _WIN32

namespace {

// the token's user, not the file's owner: an elevated admin creates files owned by Administrators
std::wstring current_user_sid() {
    HANDLE token = nullptr;
    if (!OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &token)) {
        return L"";
    }
    DWORD need = 0;
    GetTokenInformation(token, TokenUser, nullptr, 0, &need);
    std::wstring sid;
    if (need > 0) {
        std::string buf(need, '\0');
        if (GetTokenInformation(token, TokenUser, buf.data(), need, &need)) {
            LPWSTR s = nullptr;
            if (ConvertSidToStringSidW(reinterpret_cast<TOKEN_USER *>(buf.data())->User.Sid, &s)) {
                sid = s;
                LocalFree(s);
            }
        }
    }
    CloseHandle(token);
    return sid;
}

}  // namespace

void make_private(const fs::path & p) {
    std::error_code ec;
    if (!fs::exists(p, ec)) {
        return;
    }
    const std::wstring sid = current_user_sid();
    if (sid.empty()) {
        return;
    }
    const std::wstring   sddl = L"D:P(A;;FA;;;SY)(A;;FA;;;BA)(A;;FA;;;" + sid + L")";
    PSECURITY_DESCRIPTOR sd   = nullptr;
    if (ConvertStringSecurityDescriptorToSecurityDescriptorW(sddl.c_str(), SDDL_REVISION_1, &sd, nullptr)) {
        SetFileSecurityW(p.wstring().c_str(), DACL_SECURITY_INFORMATION | PROTECTED_DACL_SECURITY_INFORMATION, sd);
        LocalFree(sd);
    }
}

bool write_private_file(const fs::path & p, const std::string & content) {
    FILE * f = _wfopen(p.wstring().c_str(), L"wb");
    if (f == nullptr) {
        return false;
    }
    const bool ok = std::fwrite(content.data(), 1, content.size(), f) == content.size();
    std::fclose(f);
    make_private(p);
    return ok;
}

#else

void make_private(const fs::path & p) {
    ::chmod(p.c_str(), S_IRUSR | S_IWUSR);
}

bool write_private_file(const fs::path & p, const std::string & content) {
    const int fd = ::open(p.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, S_IRUSR | S_IWUSR);
    if (fd < 0) {
        return false;
    }
    // created earlier by a version that left it world-readable
    ::fchmod(fd, S_IRUSR | S_IWUSR);
    size_t done = 0;
    while (done < content.size()) {
        const ssize_t n = ::write(fd, content.data() + done, content.size() - done);
        if (n <= 0) {
            break;
        }
        done += static_cast<size_t>(n);
    }
    ::close(fd);
    return done == content.size();
}

#endif

}  // namespace llmash
