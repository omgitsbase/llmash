// Standalone test for pull.cpp and draft.cpp.

#include "pull.cpp"

#include "draft.h"

#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace llmash;

namespace {

int failures = 0;
int checks   = 0;

void check(bool ok, const std::string & what) {
    checks++;
    if (!ok) {
        failures++;
        std::cout << "  FAIL  " << what << "\n";
    }
}

template <typename A, typename B>
void check_eq(const A & got, const B & want, const std::string & what) {
    checks++;
    if (!(got == want)) {
        failures++;
        std::cout << "  FAIL  " << what << "\n         got  " << got << "\n         want " << want << "\n";
    }
}

void section(const char * name) { std::cout << "\n== " << name << "\n"; }

fs::path scratch() {
    static const fs::path dir = [] {
        std::error_code ec;
        const fs::path  d = fs::temp_directory_path(ec) / "llmash-pull-test";
        fs::remove_all(d, ec);
        fs::create_directories(d, ec);
        return d;
    }();
    return dir;
}

void write_file(const fs::path & p, const std::string & bytes) {
    std::ofstream out(p, std::ios::binary | std::ios::trunc);
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

std::string read_file(const fs::path & p) {
    std::ifstream in(p, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

std::string join_names(const std::vector<HfFile> & v) {
    std::string s;
    for (const auto & f : v) {
        if (!s.empty()) s += ",";
        s += f.name;
    }
    return s;
}

// ------------------------------------------------------------ GGUF builder

struct GgufBuilder {
    std::string kvs;
    uint64_t    n_kv = 0;
    std::string tensors;
    uint64_t    n_tensors = 0;

    static void put_u32(std::string & s, uint32_t v) { s.append(reinterpret_cast<const char *>(&v), 4); }
    static void put_u64(std::string & s, uint64_t v) { s.append(reinterpret_cast<const char *>(&v), 8); }
    static void put_str(std::string & s, const std::string & v) {
        put_u64(s, v.size());
        s += v;
    }

    void kv_string(const std::string & k, const std::string & v) {
        put_str(kvs, k);
        put_u32(kvs, GGUF_STRING);
        put_str(kvs, v);
        n_kv++;
    }
    void kv_u32(const std::string & k, uint32_t v) {
        put_str(kvs, k);
        put_u32(kvs, GGUF_U32);
        put_u32(kvs, v);
        n_kv++;
    }
    void kv_bool(const std::string & k, bool v) {
        put_str(kvs, k);
        put_u32(kvs, GGUF_BOOL);
        kvs += static_cast<char>(v ? 1 : 0);
        n_kv++;
    }
    void kv_u32_array(const std::string & k, const std::vector<uint32_t> & v) {
        put_str(kvs, k);
        put_u32(kvs, GGUF_ARRAY);
        put_u32(kvs, GGUF_U32);
        put_u64(kvs, v.size());
        for (const uint32_t x : v) put_u32(kvs, x);
        n_kv++;
    }
    void kv_f32_array(const std::string & k, size_t n) {
        put_str(kvs, k);
        put_u32(kvs, GGUF_ARRAY);
        put_u32(kvs, GGUF_F32);
        put_u64(kvs, n);
        for (size_t i = 0; i < n; i++) put_u32(kvs, 0x3f800000u);
        n_kv++;
    }
    void kv_tokens(const std::string & k, size_t n) {
        put_str(kvs, k);
        put_u32(kvs, GGUF_ARRAY);
        put_u32(kvs, GGUF_STRING);
        put_u64(kvs, n);
        for (size_t i = 0; i < n; i++) put_str(kvs, "t" + std::to_string(i));
        n_kv++;
    }
    void tensor(const std::string & name, const std::vector<uint64_t> & dims) {
        put_str(tensors, name);
        put_u32(tensors, static_cast<uint32_t>(dims.size()));
        for (const uint64_t d : dims) put_u64(tensors, d);
        put_u32(tensors, 0);  // ggml type
        put_u64(tensors, 0);  // offset
        n_tensors++;
    }

    std::string bytes(uint32_t version = 3) const {
        std::string s = "GGUF";
        put_u32(s, version);
        put_u64(s, n_tensors);
        put_u64(s, n_kv);
        s += kvs;
        s += tensors;
        return s;
    }
};

std::string target_gguf(int64_t embed, int64_t blocks, size_t vocab, bool size_label = true) {
    GgufBuilder b;
    b.kv_string("general.architecture", "qwen3");
    if (size_label) {
        b.kv_string("general.size_label", "8B");
    }
    b.kv_bool("qwen3.attention.causal", true);
    b.kv_u32("qwen3.embedding_length", static_cast<uint32_t>(embed));
    b.kv_u32("qwen3.block_count", static_cast<uint32_t>(blocks));
    b.kv_f32_array("qwen3.rope.scaling.factors", 6); // must be stepped over exactly
    b.kv_tokens("tokenizer.ggml.tokens", vocab);
    b.kv_string("tokenizer.chat_template", "{{ bos }}");
    b.tensor("token_embd.weight", {static_cast<uint64_t>(embed), vocab});
    return b.bytes();
}

std::string eagle3_gguf(const std::vector<uint32_t> & layers, int64_t fc_in, size_t vocab) {
    GgufBuilder b;
    b.kv_string("general.architecture", "eagle3");
    b.kv_u32("eagle3.embedding_length", 64);
    b.kv_u32("eagle3.block_count", 1);
    b.kv_u32_array("eagle3.target_layers", layers);
    b.kv_tokens("tokenizer.ggml.tokens", vocab);
    b.tensor("fc.weight", {static_cast<uint64_t>(fc_in), 64});
    return b.bytes();
}

GGUFSpec spec_of_bytes(const std::string & bytes, std::string & err) {
    std::istringstream in(bytes, std::ios::binary);
    GGUFSpec           s;
    err = scan_gguf(in, true, s);
    return s;
}

// ------------------------------------------------------- a recorded HF body

const char * const FAKE_HF_BODY = R"({
  "id": "unsloth/Qwen3.6-27B-GGUF",
  "siblings": [
    {"rfilename": "README.md", "size": 4096},
    {"rfilename": "Qwen3.6-27B-Q4_K_M.gguf", "size": 17000000000},
    {"rfilename": "Qwen3.6-27B-Q4_K_S.gguf", "size": 16000000000},
    {"rfilename": "Qwen3.6-27B-Q6_K.gguf", "size": 22000000000},
    {"rfilename": "Qwen3.6-27B-Q8_0.gguf", "size": 29000000000},
    {"rfilename": "Qwen3.6-27B-BF16-00001-of-00002.gguf", "size": 30000000000},
    {"rfilename": "Qwen3.6-27B-BF16-00002-of-00002.gguf", "size": 25000000000},
    {"rfilename": "mmproj-Qwen3.6-27B-f32.gguf", "size": 1400000000},
    {"rfilename": "mmproj-Qwen3.6-27B-f16.gguf", "size": 700000000},
    {"rfilename": "mtp-Qwen3.6-27B-Q8_0.gguf", "size": 900000000},
    {"rfilename": "config.json", "size": 900}
  ]
})";

// ============================================================ the tests

void test_quant_tag() {
    section("quant_tag");
    check_eq(quant_tag("Qwen3-8B-Q4_K_M-00001-of-00002.gguf"), std::string("Q4_K_M"), "a sharded Q4_K_M build");
    check_eq(quant_tag("Dirk-UD-Q4_K_XL.gguf"), std::string("UD-Q4_K_XL"), "unsloth's UD- prefix is kept");
    check_eq(quant_tag("gpt-oss-120b-MXFP4_MOE.gguf"), std::string("MXFP4_MOE"), "MXFP4_MOE");
    check_eq(quant_tag("model.BF16.gguf"), std::string("BF16"), "BF16 between dots");
    check_eq(quant_tag("model-IQ4_XS.gguf"), std::string("IQ4_XS"), "IQ4_XS");
    check_eq(quant_tag("mmproj-model-f16.gguf"), std::string("F16"), "F16, upper-cased");
    check_eq(quant_tag("Qwen3-8B.gguf"), std::string(""), "no quantisation in the name");
}

void test_kind_of() {
    section("kind_of");
    check(kind_of("mtp-Qwen3.6-27B-Q8_0.gguf") != nullptr && kind_of("mtp-Qwen3.6-27B-Q8_0.gguf")->name == "mtp",
          "an mtp- head");
    check(kind_of("model.mtp.gguf") != nullptr && kind_of("model.mtp.gguf")->name == "mtp", "a .mtp. sidecar");
    check(kind_of("org/Qwen3-EAGLE3-drafter") != nullptr && kind_of("org/Qwen3-EAGLE3-drafter")->name == "eagle3",
          "eagle3 wins over the later draft word");
    check(kind_of("org/qwen3-dspark") != nullptr && kind_of("org/qwen3-dspark")->name == "dspark", "dspark");
    check(kind_of("org/qwen3-speculator") != nullptr && kind_of("org/qwen3-speculator")->name == "draft",
          "speculator is a plain draft model");
    check(kind_of("Qwen3.6-27B-Q4_K_M.gguf") == nullptr, "a plain build is not a drafter");
    check_eq(draft_kinds()[0].spec_arg, std::string("draft-mtp"), "mtp's --spec-type");
}

void test_hf_parsing_and_pickers() {
    section("pick_gguf / quants_of / pick_mmproj against a recorded API body");

    std::vector<HfFile> files;
    check(parse_hf_siblings(FAKE_HF_BODY, files), "the recorded body parses");
    check_eq(files.size(), size_t(9), "only the .gguf siblings are kept");

    // exactly what was asked for
    check_eq(join_names(pick_gguf(files, "Q6_K")), std::string("Qwen3.6-27B-Q6_K.gguf"), "the quant asked for");
    check_eq(join_names(pick_gguf(files, "Q8_0")), std::string("Qwen3.6-27B-Q8_0.gguf"), "Q8_0 asked for");

    // sharded builds come back whole, in shard order
    check_eq(join_names(pick_gguf(files, "BF16")),
             std::string("Qwen3.6-27B-BF16-00001-of-00002.gguf,Qwen3.6-27B-BF16-00002-of-00002.gguf"),
             "every shard of the build, sorted");

    // a quantisation the repo does not carry falls back down the list
    check_eq(join_names(pick_gguf(files, "Q3_K_L")), std::string("Qwen3.6-27B-Q4_K_M.gguf"),
             "an absent quant falls back to Q4_K_M first");

    std::vector<HfFile> no_q4km;
    for (const auto & f : files) {
        if (f.name != "Qwen3.6-27B-Q4_K_M.gguf") no_q4km.push_back(f);
    }
    check_eq(join_names(pick_gguf(no_q4km, "Q3_K_L")), std::string("Qwen3.6-27B-Q4_K_S.gguf"),
             "then Q4_K_S, the next in the preference order");

    std::vector<HfFile> only_big;
    for (const auto & f : files) {
        const std::string q = quant_tag(f.name);
        if (q == "Q6_K" || q == "Q8_0" || q == "F16" || q == "BF16") only_big.push_back(f);
    }
    check_eq(join_names(pick_gguf(only_big, "Q3_K_L")), std::string("Qwen3.6-27B-Q6_K.gguf"),
             "Q6_K is preferred over Q8_0, and full precision is never the fallback");

    // drafters and projectors are not builds of the model
    for (const auto & f : pick_gguf(files, "Q4_K_M")) {
        check(!contains(lower(f.name), "mmproj") && kind_of(f.name) == nullptr,
              "pick_gguf never returns a projector or a draft head");
    }

    const std::vector<QuantInfo> quants = quants_of(files);
    std::string                  order;
    for (const auto & q : quants) {
        if (!order.empty()) order += ",";
        order += q.name;
    }
    check_eq(order, std::string("Q4_K_S,Q4_K_M,Q6_K,Q8_0,BF16"), "quants_of groups and orders by total size");
    for (const auto & q : quants) {
        if (q.name == "BF16") {
            check_eq(q.files, 2, "the sharded build counts two files");
            check_eq(q.size, int64_t(55000000000), "and their sizes add up");
        }
    }

    const std::optional<HfFile> proj = pick_mmproj(files);
    check(proj.has_value(), "a projector is found");
    check_eq(proj->name, std::string("mmproj-Qwen3.6-27B-f16.gguf"), "f16 wins over the larger f32 projector");
}

void test_already_have() {
    section("already_have");
    const fs::path dir = scratch() / "have";
    std::error_code ec;
    fs::create_directories(dir, ec);

    const fs::path missing = dir / "missing.gguf";
    const fs::path exact   = dir / "exact.gguf";
    const fs::path short_f = dir / "short.gguf";
    write_file(exact, std::string(1000, 'x'));
    write_file(short_f, std::string(400, 'x'));

    check(!already_have(missing.string(), 1000), "a file that is not there is not already had");
    check(already_have(exact.string(), 1000), "the right byte size counts as already had");
    check(!already_have(short_f.string(), 1000), "a half-written file does not");
    check(already_have(short_f.string(), 0), "an unknown size counts any existing file as already had");
    check(!already_have(missing.string(), 0), "but only if it exists");
    check(!already_have(dir.string(), 0), "a directory is not a file");
}

void test_block_map_resume() {
    section("resumable block map");

    check_eq(dl_block(), int64_t(1024), "LLMASH_DL_BLOCK is honoured");
    check_eq(block_count(10240), int64_t(10), "an exact multiple");
    check_eq(block_count(10241), int64_t(11), "the tail gets its own block");
    check_eq(block_count(0), int64_t(1), "an unknown size is still one block");

    const fs::path dir = scratch() / "resume";
    std::error_code ec;
    fs::create_directories(dir, ec);

    // A .part left by an older single-stream download: no map beside it.
    const int64_t  total = 10240;
    const fs::path part  = dir / "model.gguf.part";
    const fs::path idx   = dir / "model.gguf.part.idx";
    write_file(part, std::string(3500, 'a'));

    int64_t prefix = 0;
    check(seed_block_map(part.string(), total, prefix), "a .part with no map is seeded");
    check_eq(prefix, int64_t(3500), "the prefix already on disk is reported");

    const std::string map = read_file(idx);
    check_eq(map.size(), size_t(10), "the map has one byte per block");
    int marked = 0;
    for (const char c : map) marked += (c == 1);
    check_eq(marked, 3, "3500 bytes is three whole 1024-byte blocks, and the partial fourth is not trusted");
    check(map[0] == 1 && map[1] == 1 && map[2] == 1 && map[3] == 0, "and they are the first three");

    const std::vector<unsigned char> have = load_block_map(idx.string(), 10);
    const BlockPlan                  plan = plan_blocks(have, total);
    check_eq(plan.done, int64_t(3072), "the download resumes at block 3, byte 3072");
    check_eq(plan.todo.size(), size_t(7), "seven blocks are still outstanding");
    check(plan.todo.front() == 3 && plan.todo.back() == 9, "and they are blocks 3 through 9");

    // Seeding never clobbers a map that is already there.
    int64_t again = 0;
    check(!seed_block_map(part.string(), total, again), "a .part that already has a map is left alone");
    check_eq(read_file(idx), map, "the map on disk is untouched");

    // A map of the wrong length is from a different block size: start over.
    write_file(idx, std::string(4, '\1'));
    const BlockPlan fresh = plan_blocks(load_block_map(idx.string(), 10), total);
    check_eq(fresh.done, int64_t(0), "a map that does not match the block count is discarded");
    check_eq(fresh.todo.size(), size_t(10), "so every block is outstanding again");

    // done is clamped: a map claiming more bytes than the file has.
    const BlockPlan clamped = plan_blocks(std::vector<unsigned char>(10, 1), 9000);
    check_eq(clamped.done, int64_t(9000), "a full map never reports more than the total");
    check(clamped.todo.empty(), "and nothing is left to fetch");
}

void test_fetch_blocks_offline() {
    section("fetch_blocks without a server");

    const fs::path dir = scratch() / "fetch";
    std::error_code ec;
    fs::create_directories(dir, ec);
    const int64_t total = 4096; // four blocks

    // Every block already present: no request is made at all.
    {
        const fs::path part = dir / "done.part";
        const fs::path idx  = dir / "done.part.idx";
        write_file(part, std::string(4096, 'z'));
        write_file(idx, std::string(4, '\1'));

        int64_t     last = -1;
        std::string err;
        const bool  ok = fetch_blocks("http://127.0.0.1:1/never-asked", part.string(), total,
                                      [&](int64_t n) { last = n; }, err);
        check(ok, "a complete block map finishes without touching the network");
        check_eq(err, std::string(""), "and reports no error");
        check_eq(last, int64_t(-1), "progress is never called, because nothing was fetched");
        check(!fs::exists(idx, ec), "the block map is removed once the file is whole");
        check_eq(static_cast<int64_t>(fs::file_size(part, ec)), total, "the file is left at its full size");
    }

    // One block outstanding, and the endpoint is a closed port.
    {
        const fs::path part = dir / "partial.part";
        const fs::path idx  = dir / "partial.part.idx";
        write_file(part, std::string(2000, 'y')); // shorter than total on purpose
        std::string map(4, '\1');
        map[2] = 0;
        write_file(idx, map);

        std::string err;
        const bool  ok = fetch_blocks("http://127.0.0.1:1/refused", part.string(), total,
                                      [](int64_t) {}, err);
        check(!ok, "an unreachable endpoint fails the download");
        check(!err.empty(), "with a reason");
        check_eq(static_cast<int64_t>(fs::file_size(part, ec)), total,
                 "the destination is sized up front so every stream can seek into it");
        check(fs::exists(idx, ec), "the block map survives the failure, so the next run resumes");
        check_eq(read_file(idx), map, "with exactly the blocks that were already done");

        const BlockPlan plan = plan_blocks(load_block_map(idx.string(), 4), total);
        check_eq(plan.done, int64_t(3072), "the next run starts 3072 bytes in");
        check_eq(plan.todo.size(), size_t(1), "and has just the one block left");
        check_eq(plan.todo.front(), int64_t(2), "block 2");
    }
}

void test_refs_and_paths() {
    section("registry refs");

    std::string host, repo, tag;
    split_ref("llama3", host, repo, tag);
    check_eq(host, std::string("registry.ollama.ai"), "a bare name comes from the Ollama registry");
    check_eq(repo, std::string("library/llama3"), "under library/");
    check_eq(tag, std::string("latest"), "tagged latest");

    split_ref("unsloth/qwen3:8b", host, repo, tag);
    check_eq(repo, std::string("unsloth/qwen3"), "org/name keeps its namespace");
    check_eq(tag, std::string("8b"), "and its tag");

    split_ref("ghcr.io/some/deep/name:v2", host, repo, tag);
    check_eq(host, std::string("ghcr.io"), "three or more parts name a host");
    check_eq(repo, std::string("some/deep/name"), "and the rest is the repository");
    check_eq(tag, std::string("v2"), "with the tag");

    RegistryManifest m;
    split_ref("llama3", m.host, m.repo, m.tag);
    check_eq(m.name(), std::string("llama3:latest"), "library/ is trimmed off the Ollama host");
    check_eq(m.base(), std::string("https://registry.ollama.ai/v2/library/llama3"), "the v2 base");

    Config cfg;
    cfg.models_root = (scratch() / "store").string();
    const fs::path want =
        fs::path(cfg.models_root) / "manifests" / "registry.ollama.ai" / "library" / "llama3" / "latest";
    check_eq(m.manifest_path(cfg), want.string(), "the manifest sits under manifests/<host>/<repo>/<tag>");

    check_eq(short12("sha256:0123456789abcdef0123"), std::string("0123456789ab"), "short12 trims and cuts");
    check_eq(blob_path(cfg, "sha256:abc"), (fs::path(cfg.models_root) / "blobs" / "sha256-abc").string(),
             "a digest becomes a blob file name");
    check_eq(hf_download_url("unsloth/x", "a/b.gguf"),
             std::string("https://huggingface.co/unsloth/x/resolve/main/a/b.gguf"), "the resolve URL");

    cfg.gguf_dir.clear();
    check_eq(loose_dir(cfg), (fs::path(cfg.models_root) / "gguf").string(), "the loose folder defaults under root");
    cfg.gguf_dir = "D:\\models";
    check_eq(loose_dir(cfg), std::string("D:\\models"), "and is overridden when configured");
}

void test_aliases() {
    section("aliases.json");
    Config cfg;
    cfg.gguf_dir = (scratch() / "loose").string();

    set_alias(cfg, "Qwen3.6-27B-Q4_K_M.gguf", "qwen3.6:27b");
    set_alias(cfg, "other.gguf", "other:latest");

    const json j = json::parse(read_file(fs::path(cfg.gguf_dir) / "aliases.json"), nullptr, false);
    check(j.is_object(), "aliases.json is a JSON object");
    check_eq(j.value("Qwen3.6-27B-Q4_K_M.gguf", std::string()), std::string("qwen3.6:27b"), "the first name is kept");
    check_eq(j.value("other.gguf", std::string()), std::string("other:latest"), "and the second is added, not replaced");
}

void test_gguf_reader() {
    section("GGUFReader / scan_gguf");

    std::string err;
    GGUFSpec    target = spec_of_bytes(target_gguf(64, 8, 200), err);
    check_eq(err, std::string(""), "a synthesised target reads");
    check(!target.partial, "and reads completely");
    check_eq(target.arch, std::string("qwen3"), "its architecture");
    check_eq(target.embed, int64_t(64), "its embedding length");
    check_eq(target.blocks, int64_t(8), "its block count");
    check_eq(target.vocab, int64_t(200), "its vocabulary, counted from the token array's length");
    check_eq(target.tensors, int64_t(1), "its tensor count");

    GGUFSpec draft = spec_of_bytes(eagle3_gguf({1, 3, 5}, 192, 200), err);
    check_eq(err, std::string(""), "a synthesised EAGLE-3 drafter reads");
    check(draft.hidden(), "eagle3 reads hidden states");
    check_eq(draft.layers.size(), size_t(3), "its three target layers");
    check_eq(draft.enc, int64_t(192), "fc.weight's input width");

    std::string bad;
    spec_of_bytes("not a gguf at all", bad);
    check_eq(bad, std::string("not a GGUF file"), "a non-GGUF is rejected");

    GgufBuilder empty;
    empty.kv_string("general.architecture", "qwen3");
    spec_of_bytes(empty.bytes(), bad);
    check_eq(bad, std::string("no tensors in the file"), "a header with no tensors is rejected");

    GgufBuilder v9;
    v9.kv_string("general.architecture", "qwen3");
    v9.tensor("a", {1});
    spec_of_bytes(v9.bytes(9), bad);
    check_eq(bad, std::string("GGUF version 9, which this llama.cpp does not read"), "an unknown version is rejected");

    // A header cut short is a partial answer, not an error.
    const std::string whole = target_gguf(64, 8, 200);
    GGUFSpec          cut   = spec_of_bytes(whole.substr(0, 60), err);
    check_eq(err, std::string(""), "a truncated header is not an error");
    check(cut.partial, "it is marked partial");
}

void test_pairs() {
    section("pairs: does this drafter fit these weights");

    std::string err;
    const GGUFSpec target = spec_of_bytes(target_gguf(64, 8, 200), err);

    check_eq(pairs(target, spec_of_bytes(eagle3_gguf({1, 3, 5}, 192, 200), err)), std::string(""),
             "a matching EAGLE-3 drafter pairs");

    check_eq(pairs(target, spec_of_bytes(eagle3_gguf({1, 3, 5}, 192, 200 + 64), err)), std::string(""),
             "a vocabulary within 128 tokens is close enough");

    check_eq(pairs(target, spec_of_bytes(eagle3_gguf({1, 3, 5}, 192, 500), err)),
             std::string("its vocabulary has 500 tokens against this model's 200"), "a different vocabulary does not");

    check_eq(pairs(target, spec_of_bytes(eagle3_gguf({1, 3}, 128, 200), err)),
             std::string("EAGLE-3 reads exactly 3 layers, this one names 2"), "EAGLE-3 needs exactly three layers");

    check_eq(pairs(target, spec_of_bytes(eagle3_gguf({1, 3, 99}, 192, 200), err)),
             std::string("it reads layer 99, and this model has 8"), "a layer the target does not have");

    check_eq(pairs(target, spec_of_bytes(eagle3_gguf({1, 3, 5}, 128, 200), err)),
             std::string("its encoder takes 128 values where this model gives 192"),
             "an encoder shaped for a different hidden size");

    GGUFSpec no_layers = spec_of_bytes(eagle3_gguf({}, 192, 200), err);
    check_eq(pairs(target, no_layers), std::string("it is a eagle3 drafter but names no layers to read"),
             "a hidden-state drafter that names no layers");

    GGUFSpec plain;
    plain.arch  = "qwen3";
    plain.vocab = 200;
    check_eq(pairs(target, plain), std::string(""), "a plain draft model only has to match the vocabulary");

    check_eq(pairs(target, GGUFSpec{}), std::string("no architecture in its header"), "an unreadable drafter");
}

void test_draft_naming() {
    section("draft naming and rejection");

    check_eq(normalise("Qwen3.6-27B_it"), std::string("qwen3627bit"), "normalise keeps only letters and digits");

    check_eq(other_runtime("org/Qwen3-8B-MLX-4bit"), std::string("mlx"), "an MLX repack is not for llama.cpp");
    check_eq(other_runtime("org/Qwen3-8B-6.0bpw-exl2"), std::string("exl2"), "nor is exllama");
    check_eq(other_runtime("org/Qwen3-8B-int4"), std::string("int4"), "the leading dash is trimmed off");
    check_eq(other_runtime("unsloth/Qwen3-8B-GGUF"), std::string(""), "a GGUF repo is fine");

    const std::string want = normalise("Qwen3-8B");
    check_eq(foreign_base({"Qwen/Qwen3-8B"}, want), std::string(""), "the target itself is not a foreign base");
    check_eq(foreign_base({"Qwen/Qwen3-8B-Instruct-GGUF"}, want), std::string(""),
             "packaging and quantisation words do not make it foreign");
    check_eq(foreign_base({"huihui/Qwen3-8B-abliterated"}, want), std::string("huihui/Qwen3-8B-abliterated"),
             "an abliterated fine-tune is a different model");
    check_eq(foreign_base({"someone/Qwen3-8B-Nemotron"}, want), std::string("someone/Qwen3-8B-Nemotron"),
             "and so is anything with a name of its own left over");

    check(prefer_quant("model-Q4_K_M.gguf", "model-Q8_0.gguf"), "Q4_K_M is preferred to Q8_0");
    check(prefer_quant("model-IQ4_XS.gguf", "model-f16.gguf"), "IQ4_XS is preferred to f16");
    check(!prefer_quant("model-BF16.gguf", "model-Q4_K_M.gguf"), "and never the other way round");
    check(!prefer_quant("model.gguf", "model-Q8_0.gguf"), "an untagged file loses to a known quantisation");

    // draft_path puts the sidecar beside the weights, under the model's stem.
    const fs::path  dir = scratch() / "models";
    std::error_code ec;
    fs::create_directories(dir, ec);
    write_file(dir / "Qwen3-8B-Q4_K_M-00001-of-00002.gguf", "x");

    Model m;
    m.path = (dir / "Qwen3-8B-Q4_K_M-00001-of-00002.gguf").string();
    m.name = "qwen3:8b";
    Config cfg;
    cfg.gguf_dir = (scratch() / "loose").string();
    check_eq(draft_path(m, draft_kinds()[1], cfg), (dir / "Qwen3-8B-Q4_K_M.eagle3.gguf").string(),
             "the shard suffix is dropped and the kind becomes the extension");

    check_eq(installed_drafter(m), std::string(""), "no sidecar yet");
    write_file(dir / "Qwen3-8B-Q4_K_M.eagle3.gguf", "x");
    check_eq(installed_drafter(m), std::string("an EAGLE-3 drafter"), "and one is found once it is there");
    write_file(dir / "Qwen3-8B-Q4_K_M.mtp.gguf", "x");
    check_eq(installed_drafter(m), std::string("an MTP head"), "an MTP head outranks it");

    Model unwritable;
    unwritable.path = "Z:\\nowhere\\model.gguf";
    check_eq(draft_path(unwritable, draft_kinds()[0], cfg), (fs::path(cfg.gguf_dir) / "model.mtp.gguf").string(),
             "a folder that cannot be written to sends the sidecar to the loose folder");
}

void test_model_stem() {
    section("model_stem");

    const fs::path  dir = scratch() / "stem";
    std::error_code ec;
    fs::create_directories(dir, ec);

    // No usable repo_url in the header: the registry name and tag carry the size.
    const fs::path p = dir / "gemma-4-e4b.gguf";
    write_file(p, target_gguf(64, 8, 20));
    Model m;
    m.path = p.string();
    m.name = "gemma4:e4b";
    check_eq(model_stem(m), std::string("gemma4-e4b"), "a registry tag is folded into the stem");

    m.name = "qwen3:latest";
    check_eq(model_stem(m), std::string("qwen3-8B"), "a `latest` tag defers to general.size_label");

    m.name = "Qwen3-8B-Q4_K_M:gguf";
    check_eq(model_stem(m), std::string("Qwen3-8B-Q4_K_M-8B"),
             "a `gguf` tag also defers to the size label, before any suffix is stripped");

    // The same loose file with nothing in its header to name the size: now
    // the quantisation is the tail, and it is not part of the identity.
    const fs::path q = dir / "no-size-label.gguf";
    write_file(q, target_gguf(64, 8, 20, /*size_label=*/false));
    Model n;
    n.path = q.string();
    n.name = "Qwen3-8B-Q4_K_M:gguf";
    check_eq(model_stem(n), std::string("Qwen3-8B"), "the quantisation suffix is not part of the identity");

    n.name = "Qwen3-8B-Instruct-GGUF:gguf";
    check_eq(model_stem(n), std::string("Qwen3-8B"),
             "the packaging suffixes come off in order, -GGUF then -Instruct");
}

} // namespace

int main() {
    // Small blocks keep the block-map arithmetic legible; both must be set
    // before anything calls dl_block()/dl_streams(), which cache them.
    _putenv_s("LLMASH_DL_BLOCK", "1024");
    _putenv_s("LLMASH_DL_STREAMS", "4");

    std::cout << "pull.cpp / draft.cpp\n";
    std::cout << "scratch: " << scratch().string() << "\n";

    test_quant_tag();
    test_kind_of();
    test_hf_parsing_and_pickers();
    test_already_have();
    test_block_map_resume();
    test_refs_and_paths();
    test_aliases();
    test_gguf_reader();
    test_pairs();
    test_draft_naming();
    test_model_stem();
    test_fetch_blocks_offline(); // last: the retry backoff makes it the slow one

    std::cout << "\n" << (checks - failures) << "/" << checks << " checks passed\n";
    if (failures != 0) {
        std::cout << failures << " FAILED\n";
        return 1;
    }
    std::cout << "OK\n";
    return 0;
}
