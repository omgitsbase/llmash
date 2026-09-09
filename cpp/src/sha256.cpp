#include "sha256.h"

#include <windows.h>

#include <bcrypt.h>

#include <vector>

namespace llmash {

std::string sha256_hex(const std::string & data) {
    BCRYPT_ALG_HANDLE alg = nullptr;
    if (BCryptOpenAlgorithmProvider(&alg, BCRYPT_SHA256_ALGORITHM, nullptr, 0) < 0) {
        return "";
    }
    DWORD hash_len = 0;
    DWORD got      = 0;
    BCryptGetProperty(alg, BCRYPT_HASH_LENGTH, reinterpret_cast<PUCHAR>(&hash_len), sizeof(hash_len), &got, 0);

    std::string        hex;
    std::vector<UCHAR> out(hash_len);
    BCRYPT_HASH_HANDLE h = nullptr;
    if (BCryptCreateHash(alg, &h, nullptr, 0, nullptr, 0, 0) >= 0) {
        BCryptHashData(h, reinterpret_cast<PUCHAR>(const_cast<char *>(data.data())),
                       static_cast<ULONG>(data.size()), 0);
        if (BCryptFinishHash(h, out.data(), hash_len, 0) >= 0) {
            static const char * digits = "0123456789abcdef";
            for (UCHAR b : out) {
                hex += digits[b >> 4];
                hex += digits[b & 0x0F];
            }
        }
        BCryptDestroyHash(h);
    }
    BCryptCloseAlgorithmProvider(alg, 0);
    return hex;
}

} // namespace llmash
