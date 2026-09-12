#include "config.h"
#include "platform.h"
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

void eq(const std::string & got, const std::string & want, const char * what) {
    check(got == want, what);
    if (got != want) {
        std::printf("      got  [%s]\n      want [%s]\n", got.c_str(), want.c_str());
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

    const std::optional<Model> nail = reg.find("nail-a3b");
    check(nail.has_value(), "a model in a subfolder is found by name");
    check(nail && nail->has_mtp, "an mtp head in the weights is seen");
    check(nail && nail->arch == "qwen35moe", "the architecture is read from the header");

    // The Ollama library writes the quantisation into the tag, underscores
    // kept: llama3.2:1b-instruct-q8_0. Dropping it gave two builds one name.
    check(nail && nail->name == "nail-a3b:gguf", "no quantisation in the file name leaves the gguf tag");
    eq(loose_name("D:/m/Qwen3-8B-Q8_0.gguf"), "qwen3-8b:q8_0", "the quantisation is the tag");
    eq(loose_name("D:/m/Llama-3.2-1B-Instruct-RCO-3.9.gguf"), "llama-3.2-1b-instruct:rco-3.9",
       "a build assembled here is tagged by its width");
    eq(loose_name("D:/m/Qwen3-8B-Q4_K_M-00001-of-00002.gguf"), "qwen3-8b:q4_k_m", "shards resolve to one name");
    eq(loose_name("D:/m/Some_Model.gguf"), "some-model:gguf", "underscores in the name become dashes");

    const std::optional<Model> qwen = reg.find("qwen3-8b");
    check(qwen.has_value(), "a model at the top level is found");
    check(qwen && qwen->name == "qwen3-8b:q8_0", "and it is listed under its quantisation");
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

    // Run from the bin copy on PATH, the install is the folder above it.
    {
        const fs::path root = dir / "installed";
        const fs::path bin  = root / "bin";
        fs::create_directories(bin);
        { std::ofstream f(root / llmash_exe()); }
        { std::ofstream f(bin / llmash_exe()); }
        check(same_dir(install_root(bin.string()), root.string()), "the bin copy resolves to the install above it");
        check(same_dir(install_root(root.string()), root.string()), "the install itself resolves to itself");

        const fs::path lone = dir / "lonely" / "bin";
        fs::create_directories(lone);
        check(same_dir(install_root(lone.string()), lone.string()),
              "a bin folder with no program above it is left alone");
    }

    fs::remove_all(dir);
    std::printf("\n%s\n", failures == 0 ? "all passed" : "FAILURES");
    return failures == 0 ? 0 : 1;
}
