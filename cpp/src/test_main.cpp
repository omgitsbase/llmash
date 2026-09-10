#include "config.h"
#include "registry.h"

#include <cstdio>
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;
using namespace llmash;

namespace {

int failures = 0;

void check(bool ok, const char * what) {
    std::printf("%s  %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) {
        failures++;
    }
}

// A GGUF header with one metadata key and one tensor, enough to be read.
void write_gguf(const fs::path & p, const std::string & arch, const char * tensor) {
    std::ofstream out(p, std::ios::binary);
    const auto u32 = [&](uint32_t v) { out.write(reinterpret_cast<const char *>(&v), 4); };
    const auto u64 = [&](uint64_t v) { out.write(reinterpret_cast<const char *>(&v), 8); };
    const auto str = [&](const std::string & s) { u64(s.size()); out.write(s.data(), s.size()); };

    out.write("GGUF", 4);
    u32(3);
    u64(1);                       // tensors
    u64(1);                       // kv pairs
    str("general.architecture");
    u32(8);                       // string
    str(arch);

    str(tensor);
    u32(1);                       // dims
    u64(16);
    u32(0);                       // type
    u64(0);                       // offset
}

} // namespace

int main() {
    const fs::path dir = fs::temp_directory_path() / "llmash_cpp_test";
    fs::remove_all(dir);
    fs::create_directories(dir / "nested");

    write_gguf(dir / "Qwen3-8B-Q8_0.gguf", "qwen3", "blk.0.attn_norm.weight");
    write_gguf(dir / "nested" / "Nail-A3B.gguf", "qwen35moe", "blk.0.nextn.eh_proj.weight");
    write_gguf(dir / "Qwen3-8B-Q8_0.mtp.gguf", "qwen3", "blk.0.nextn.weight");
    { std::ofstream(dir / "notes.txt") << "not a model"; }

    const auto files = walk_gguf(dir.string());
    check(files.size() == 3, "walk finds every gguf, subfolders included");

    Config cfg;
    cfg.root        = dir.string();
    cfg.models_root = dir.string();
    Registry reg(cfg);

    const auto models = reg.all();
    check(models.size() == 2, "sidecars are not listed as models");

    const Model * nail = reg.find("nail-a3b");
    check(nail != nullptr, "a model in a subfolder is found by name");
    check(nail && nail->has_mtp, "an mtp head in the weights is seen");
    check(nail && nail->arch == "qwen35moe", "the architecture is read from the header");

    const Model * qwen = reg.find("qwen3-8b");
    check(qwen != nullptr, "a model at the top level is found");
    check(qwen && !qwen->has_mtp, "a model without an mtp head is not claimed to have one");
    check(qwen && !qwen->mtp_path.empty(), "a sidecar drafter beside it is picked up");

    check(reg.in_library((dir / "Qwen3-8B-Q8_0.gguf").string()), "a file in the folder is library");
#ifdef _WIN32
    check(!reg.in_library("C:/somewhere/else/x.gguf"), "a file outside it is not");
    check(same_dir("C:/A/B", "c:\\a\\b\\"), "paths compare the way Windows means it");
#else
    check(!reg.in_library("/somewhere/else/x.gguf"), "a file outside it is not");
    check(same_dir("/A/B", "/A/B/"), "a trailing separator is the same folder");
#endif

    fs::remove_all(dir);
    std::printf("\n%s\n", failures == 0 ? "all passed" : "FAILURES");
    return failures == 0 ? 0 : 1;
}
