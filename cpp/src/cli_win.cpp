#include "cli_win.h"

#include "shortcut.h"
#include "cli_process.h"

#include <algorithm>
#include <cctype>
#include <cwchar>
#include <cwctype>
#include <filesystem>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

namespace llmash {

std::string ps_quote(const std::string & s) {
    std::string out;
    out.reserve(s.size());
    for (const char c : s) {
        out.push_back(c);
        if (c == '\'') {
            out.push_back('\'');
        }
    }
    return out;
}

std::string hidden_powershell(const std::string & script, bool wait, int * exit_code) {
    const ProcessResult r = run_hidden(
        {"powershell", "-NoProfile", "-NonInteractive", "-WindowStyle", "Hidden", "-Command", script}, nullptr, wait,
        wait);
    if (exit_code != nullptr) {
        *exit_code = r.exit_code;
    }
    return r.out;
}

#ifdef _WIN32

namespace {

std::wstring to_wide(const std::string & s) {
    if (s.empty()) {
        return L"";
    }
    const int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), nullptr, 0);
    std::wstring w(n, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), w.data(), n);
    return w;
}

std::string to_utf8(const std::wstring & w) {
    if (w.empty()) {
        return "";
    }
    const int n = WideCharToMultiByte(CP_UTF8, 0, w.data(), static_cast<int>(w.size()), nullptr, 0, nullptr, nullptr);
    std::string s(n, '\0');
    WideCharToMultiByte(CP_UTF8, 0, w.data(), static_cast<int>(w.size()), s.data(), n, nullptr, nullptr);
    return s;
}

// RegCloseKey on every return path, mirroring win.go's `defer procRegCloseKey.Call(h)`.
class ScopedHKey {
public:
    ~ScopedHKey() {
        if (open_) {
            RegCloseKey(key_);
        }
    }
    LSTATUS open(HKEY root, const std::wstring & subkey, REGSAM access) {
        const LSTATUS st = RegOpenKeyExW(root, subkey.c_str(), 0, access, &key_);
        open_             = st == ERROR_SUCCESS;
        return st;
    }
    HKEY get() const { return key_; }

private:
    HKEY key_  = nullptr;
    bool open_ = false;
};

} // namespace

bool reg_delete_hkcu_key(const std::string & subkey) {
    return RegDeleteKeyW(HKEY_CURRENT_USER, to_wide(subkey).c_str()) == ERROR_SUCCESS;
}

bool remove_from_user_path(const std::string & bin_dir) {
    ScopedHKey key;
    if (key.open(HKEY_CURRENT_USER, L"Environment", KEY_READ | KEY_WRITE) != ERROR_SUCCESS) {
        return false;
    }
    DWORD type = 0, size = 0;
    if (RegQueryValueExW(key.get(), L"Path", nullptr, &type, nullptr, &size) != ERROR_SUCCESS || size == 0) {
        return false;
    }
    std::wstring buf(size / 2 + 1, L'\0');
    if (RegQueryValueExW(key.get(), L"Path", nullptr, &type, reinterpret_cast<BYTE *>(buf.data()), &size) !=
        ERROR_SUCCESS) {
        return false;
    }
    buf.resize(wcsnlen(buf.c_str(), buf.size()));

    const auto clean_fold = [](std::wstring p) {
        while (!p.empty() && (p.back() == L'\\' || p.back() == L'/')) {
            p.pop_back();
        }
        std::transform(p.begin(), p.end(), p.begin(), [](wchar_t c) { return static_cast<wchar_t>(towlower(c)); });
        return p;
    };
    const std::wstring want = clean_fold(to_wide(bin_dir));

    std::vector<std::wstring> kept;
    bool                      changed = false;
    size_t                    pos     = 0;
    while (pos <= buf.size()) {
        const size_t      sep = buf.find(L';', pos);
        const std::wstring piece = buf.substr(pos, sep == std::wstring::npos ? std::wstring::npos : sep - pos);
        if (!piece.empty()) {
            if (clean_fold(piece) == want) {
                changed = true;
            } else {
                kept.push_back(piece);
            }
        }
        if (sep == std::wstring::npos) {
            break;
        }
        pos = sep + 1;
    }
    if (!changed) {
        return false;
    }
    std::wstring joined;
    for (size_t i = 0; i < kept.size(); i++) {
        if (i > 0) {
            joined += L';';
        }
        joined += kept[i];
    }
    joined.push_back(L'\0');
    const LSTATUS st = RegSetValueExW(key.get(), L"Path", 0, type, reinterpret_cast<const BYTE *>(joined.data()),
                                       static_cast<DWORD>(joined.size() * sizeof(wchar_t)));
    return st == ERROR_SUCCESS;
}

std::string which_exe(const std::string & exe) {
    wchar_t buf[MAX_PATH];
    const std::wstring w = to_wide(exe);
    const DWORD        n = SearchPathW(nullptr, w.c_str(), L".exe", MAX_PATH, buf, nullptr);
    if (n == 0 || n >= MAX_PATH) {
        return "";
    }
    // Go's LookPath lowercases the extension it appended; match it, so the
    // path reads the same way in `doctor`.
    std::string out = to_utf8(std::wstring(buf, n));
    const size_t dot = out.rfind('.');
    if (dot != std::string::npos && out.find_first_of("\\/", dot) == std::string::npos) {
        std::transform(out.begin() + dot, out.end(), out.begin() + dot,
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    }
    return out;
}

#else // !_WIN32

bool        reg_delete_hkcu_key(const std::string &) { return false; }
bool        remove_from_user_path(const std::string &) { return false; }
std::string which_exe(const std::string &) { return ""; }

#endif

std::string shortcut_target(const std::string & lnk_path) {
    return read_shortcut_target(lnk_path);
}

} // namespace llmash
