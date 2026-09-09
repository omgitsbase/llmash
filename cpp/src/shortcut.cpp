#include "shortcut.h"

#include <windows.h>

#include <objbase.h>
#include <shlobj.h>
#include <shobjidl.h>

#include <filesystem>

namespace llmash {
namespace {

std::wstring widen(const std::string & s) {
    if (s.empty()) {
        return L"";
    }
    const int    n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), static_cast<int>(s.size()), nullptr, 0);
    std::wstring w(static_cast<size_t>(n), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(), static_cast<int>(s.size()), w.data(), n);
    return w;
}

std::string narrow(const std::wstring & w) {
    if (w.empty()) {
        return "";
    }
    const int   n = WideCharToMultiByte(CP_UTF8, 0, w.c_str(), static_cast<int>(w.size()), nullptr, 0, nullptr, nullptr);
    std::string s(static_cast<size_t>(n), '\0');
    WideCharToMultiByte(CP_UTF8, 0, w.c_str(), static_cast<int>(w.size()), s.data(), n, nullptr, nullptr);
    return s;
}

// COM may already be up on this thread; either way it must be balanced.
class ComScope {
public:
    ComScope() { hr_ = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE); }
    ~ComScope() {
        if (hr_ == S_OK || hr_ == S_FALSE) {
            CoUninitialize();
        }
    }
    bool ok() const { return SUCCEEDED(hr_) || hr_ == RPC_E_CHANGED_MODE; }

private:
    HRESULT hr_ = E_FAIL;
};

} // namespace

std::string read_shortcut_target(const std::string & lnk_path) {
    std::error_code ec;
    if (!std::filesystem::is_regular_file(lnk_path, ec)) {
        return "";
    }
    const ComScope com;
    if (!com.ok()) {
        return "";
    }
    IShellLinkW * link = nullptr;
    if (FAILED(CoCreateInstance(CLSID_ShellLink, nullptr, CLSCTX_INPROC_SERVER, IID_IShellLinkW,
                                reinterpret_cast<void **>(&link)))) {
        return "";
    }
    std::string   out;
    IPersistFile * file = nullptr;
    if (SUCCEEDED(link->QueryInterface(IID_IPersistFile, reinterpret_cast<void **>(&file)))) {
        if (SUCCEEDED(file->Load(widen(lnk_path).c_str(), STGM_READ))) {
            wchar_t buf[MAX_PATH] = {};
            if (SUCCEEDED(link->GetPath(buf, MAX_PATH, nullptr, SLGP_RAWPATH))) {
                out = narrow(buf);
            }
        }
        file->Release();
    }
    link->Release();
    return out;
}

bool write_shortcut(const std::string & lnk_path, const std::string & target, const std::string & args,
                    const std::string & working_dir, const std::string & icon_path,
                    const std::string & description) {
    const ComScope com;
    if (!com.ok()) {
        return false;
    }
    IShellLinkW * link = nullptr;
    if (FAILED(CoCreateInstance(CLSID_ShellLink, nullptr, CLSCTX_INPROC_SERVER, IID_IShellLinkW,
                                reinterpret_cast<void **>(&link)))) {
        return false;
    }
    link->SetPath(widen(target).c_str());
    if (!args.empty()) {
        link->SetArguments(widen(args).c_str());
    }
    if (!working_dir.empty()) {
        link->SetWorkingDirectory(widen(working_dir).c_str());
    }
    if (!description.empty()) {
        link->SetDescription(widen(description).c_str());
    }
    if (!icon_path.empty()) {
        link->SetIconLocation(widen(icon_path).c_str(), 0);
    }
    link->SetShowCmd(SW_SHOWMINNOACTIVE);

    bool           ok   = false;
    IPersistFile * file = nullptr;
    if (SUCCEEDED(link->QueryInterface(IID_IPersistFile, reinterpret_cast<void **>(&file)))) {
        std::error_code ec;
        std::filesystem::create_directories(std::filesystem::path(lnk_path).parent_path(), ec);
        ok = SUCCEEDED(file->Save(widen(lnk_path).c_str(), TRUE));
        file->Release();
    }
    link->Release();
    return ok;
}

} // namespace llmash
