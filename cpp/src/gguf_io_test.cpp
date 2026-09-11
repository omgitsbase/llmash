// Whitebox test for gguf_io.cpp: a header written here has to read back as
// the same tensors, and its size table has to agree with ggml's.
#include "gguf_io.cpp"

#include "rco.h"

#include <cstdio>
#include <filesystem>

using namespace llmash;
using namespace llmash::ggufio;

namespace {

int failures = 0;

void check(bool ok, const char * what) {
    if (!ok) {
        failures++;
        std::printf("FAIL %s\n", what);
    }
}

std::string put(const std::string & s) {
    std::string    out;
    const uint64_t n = s.size();
    out.append(reinterpret_cast<const char *>(&n), 8);
    return out + s;
}

// A GGUF with three metadata keys (one of them split bookkeeping) and three
// tensors, data included.
std::string tiny_model(int64_t & data_at) {
    const uint32_t t_str = 8, t_u16 = 2;
    std::string    kv;
    kv += put("general.architecture");
    kv.append(reinterpret_cast<const char *>(&t_str), 4);
    kv += put("llama");
    kv += put("general.name");
    kv.append(reinterpret_cast<const char *>(&t_str), 4);
    kv += put("tiny");
    kv += put("split.count");
    kv.append(reinterpret_cast<const char *>(&t_u16), 4);
    const uint16_t two = 2;
    kv.append(reinterpret_cast<const char *>(&two), 2);

    struct T {
        const char * name;
        int64_t      rows, cols;
    };
    const T ts[] = {{"token_embd.weight", 64, 8}, {"blk.0.attn_q.weight", 64, 8}, {"output_norm.weight", 64, 1}};

    std::string infos;
    int64_t     off = 0;
    for (const T & t : ts) {
        infos += put(t.name);
        const uint32_t nd = t.cols > 1 ? 2 : 1;
        infos.append(reinterpret_cast<const char *>(&nd), 4);
        infos.append(reinterpret_cast<const char *>(&t.rows), 8);
        if (nd == 2) {
            infos.append(reinterpret_cast<const char *>(&t.cols), 8);
        }
        const uint32_t type = 0; // F32
        infos.append(reinterpret_cast<const char *>(&type), 4);
        infos.append(reinterpret_cast<const char *>(&off), 8);
        off += (t.rows * t.cols * 4 + 31) / 32 * 32;
    }

    std::string    h   = "GGUF";
    const uint32_t ver = 3;
    h.append(reinterpret_cast<const char *>(&ver), 4);
    const uint64_t n_t = 3, n_kv = 3;
    h.append(reinterpret_cast<const char *>(&n_t), 8);
    h.append(reinterpret_cast<const char *>(&n_kv), 8);
    h += kv + infos;
    h.append((32 - h.size() % 32) % 32, '\0');
    data_at = static_cast<int64_t>(h.size());

    std::string data(static_cast<size_t>(off), '\0');
    for (size_t i = 0; i < data.size(); i++) {
        data[i] = static_cast<char>(i & 0x7f);
    }
    return h + data;
}

} // namespace

int main() {
    int64_t           data_at = 0;
    const std::string blob    = tiny_model(data_at);

    const Layout l = layout_from(blob);
    check(l.error.empty(), "a whole GGUF reads without error");
    check(l.version == 3, "the version comes through");
    check(l.kv.size() == 2, "the split bookkeeping is dropped and the rest kept");
    check(l.tensors.size() == 3, "all three tensors are listed");
    check(l.data_start.size() == 1 && l.data_start[0] == data_at, "the data offset matches the header length");
    check(l.tensors[0].bytes == 64 * 8 * 4, "an F32 tensor's size is rows x cols x 4");
    check(l.tensors[0].rows() == 8, "rows() is every dimension but the first");
    check(l.tensors[2].rows() == 1, "a 1-D tensor is one row");
    check(l.tensors[0].experts() == 1, "a 2-D tensor is one expert");
    check(l.file_offset(l.tensors[1]) == data_at + l.tensors[1].offset, "and the file offset adds the data start");

    // A second shard contributes tensors, not metadata.
    {
        Layout two = layout_from(blob);
        check(append_part(two, blob), "a second part appends");
        check(two.kv.size() == 2, "without repeating the metadata");
        check(two.tensors.size() == 6, "and its tensors are added");
        check(two.tensors[3].part == 1, "tagged with the part they came from");
        check(two.data_start.size() == 2, "each part keeping its own data offset");
    }

    // Written back out, then read: the same tensors at the same sizes.
    {
        const std::string path = "gguf_io_test.gguf";
        std::string       err;
        const int64_t     head = write_header(path, l, err);
        check(head > 0, "a header is written");
        {
            std::fstream out(path, std::ios::binary | std::ios::in | std::ios::out);
            out.seekp(0, std::ios::end);
            for (const TensorEntry & t : l.tensors) {
                out.write(blob.data() + l.file_offset(t), t.bytes);
                out.write(std::string(static_cast<size_t>((32 - t.bytes % 32) % 32), '\0').data(),
                          (32 - t.bytes % 32) % 32);
            }
        }
        const Layout back = read_layout(path);
        check(back.error.empty(), "and reads back as a GGUF");
        check(back.tensors.size() == l.tensors.size(), "with every tensor");
        check(back.tensors[1].name == l.tensors[1].name, "named as they were");
        check(back.data_start[0] == head, "its data starting where the header ended");

        // patch_entry finds the right field for each tensor.
        {
            std::fstream out(path, std::ios::binary | std::ios::in | std::ios::out);
            check(patch_entry(out, l, 1, 12, 4096), "an entry is patched");
        }
        const Layout patched = read_layout(path);
        check(patched.tensors[1].type == 12 && patched.tensors[1].offset == 4096, "with the type and offset given");
        check(patched.tensors[0].type == 0 && patched.tensors[2].type == 0, "and its neighbours untouched");

        std::error_code ec;
        std::filesystem::remove(path, ec);
    }

    // The size table has to agree with the ggml the runtime ships, since a
    // wrong row size reads the next tensor's bytes as this one's.
    Config      cfg;
    cfg.llama_bin = env_str("LLAMA_BIN");
    rco::Ggml   g;
    std::string err;
    if (!cfg.llama_bin.empty() && g.load(cfg, err)) {
        bool agree = true;
        for (const int t : {0, 1, 8, 10, 11, 12, 13, 14, 16, 17, 18, 20, 21, 22, 23, 29, 30}) {
            TensorEntry e;
            e.dims = {256, 4};
            e.type = static_cast<uint32_t>(t);
            const int64_t ours   = tensor_bytes(e);
            const int64_t theirs = static_cast<int64_t>(g.row_size(t, 256)) * 4;
            if (ours != theirs) {
                agree = false;
                std::printf("  %s: ours %lld, ggml %lld\n", g.name(t), (long long) ours, (long long) theirs);
            }
        }
        check(agree, "every type's size matches what ggml reports");
    } else {
        std::printf("  (ggml not loaded: %s)\n", err.empty() ? "LLAMA_BIN unset" : err.c_str());
    }

    std::printf("%s (%d failure%s)\n", failures == 0 ? "all passed" : "FAILURES", failures,
                failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
