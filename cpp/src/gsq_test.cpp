// Whitebox test for gsq.cpp: the chunk header it writes has to be a GGUF
// llama-quantize accepts, and assembling the pieces back has to give one
// file with every tensor in it.
#include "gsq.cpp"

#include <cstdio>

using namespace llmash;
using namespace llmash::gsq;

namespace {

int failures = 0;

void check(bool ok, const char * what) {
    if (!ok) {
        failures++;
        std::printf("FAIL %s\n", what);
    }
}

std::string put(const std::string & s) {
    std::string out;
    const uint64_t n = s.size();
    out.append(reinterpret_cast<const char *>(&n), 8);
    return out + s;
}

// A GGUF with two string keys and three tensors, data included.
std::string tiny_model(int64_t & data_at) {
    std::string kv;
    kv += put("general.architecture");
    const uint32_t t_str = 8;
    kv.append(reinterpret_cast<const char *>(&t_str), 4);
    kv += put("llama");
    kv += put("general.name");
    kv.append(reinterpret_cast<const char *>(&t_str), 4);
    kv += put("tiny");

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

    std::string h = "GGUF";
    const uint32_t ver = 3;
    h.append(reinterpret_cast<const char *>(&ver), 4);
    const uint64_t n_t = 3, n_kv = 2;
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
    check(l.kv.size() == 2, "both metadata entries are kept");
    check(l.tensors.size() == 3, "all three tensors are listed");
    check(l.data_start.size() == 1 && l.data_start[0] == data_at, "the data offset matches the header length");
    check(l.tensors[0].bytes == 64 * 8 * 4, "an F32 tensor's size is rows x cols x 4");

    const std::vector<Chunk> one = plan_chunks(l, 1 << 30);
    check(one.size() == 1, "a budget larger than the model is one chunk");
    check(one[0].first == 0 && one[0].last == 2, "and it holds every tensor");

    const std::vector<Chunk> many = plan_chunks(l, 2048);
    check(many.size() > 1, "a small budget splits it");
    check(many.front().first == 0, "the first chunk starts at the first tensor");
    check(many.back().last == l.tensors.size() - 1, "the last one ends at the last tensor");
    bool consecutive = true;
    for (size_t i = 1; i < many.size(); i++) {
        consecutive = consecutive && many[i - 1].last + 1 == many[i].first;
    }
    check(consecutive, "and every tensor is in exactly one of them, in order");

    // A chunk's header is a GGUF of its own: same metadata, its own tensors,
    // offsets rebased to it.
    const std::string path = "gsq_test_chunk.gguf";
    std::string       err;
    const int64_t     head = write_chunk_header(path, l, many[0], err);
    check(head > 0, "the chunk header is written");

    const std::vector<Piece> pieces = chunk_pieces(l, many[0], head);
    check(pieces.size() == many[0].last - many[0].first + 1, "one piece per tensor in the chunk");
    check(pieces[0].into == head, "the first tensor sits right after the header");
    check(pieces[0].from == data_at + l.tensors[0].offset, "and comes from where the source keeps it");

    // Fill it the way a fetch would, then read it back as a model.
    {
        std::ofstream out(path, std::ios::binary | std::ios::app);
        for (const Piece & p : pieces) {
            out.write(blob.data() + p.from, p.bytes);
            out.write(std::string(static_cast<size_t>((32 - p.bytes % 32) % 32), '\0').data(),
                      (32 - p.bytes % 32) % 32);
        }
    }
    const Layout back = read_layout(path);
    check(back.error.empty(), "the chunk reads back as a GGUF");
    check(back.kv.size() == 2, "carrying the whole metadata");
    check(back.tensors.size() == many[0].last - many[0].first + 1, "and only its own tensors");
    check(back.tensors[0].name == l.tensors[0].name, "named as they were");

    std::error_code ec;
    std::filesystem::remove(path, ec);

    check(bpw_of_quant("GSQ-3.5") == 3.5, "GSQ-3.5 asks for 3.5 bits");
    check(bpw_of_quant("gsq") == 3.5, "a bare GSQ means the widest one");
    check(bpw_of_quant("Q4_K_M") == 0, "an ordinary quant is not a GSQ quality");
    check(quant_name(3.5) == "GSQ-3.5", "and the name round-trips");
    check(quant_name(3.0) == "GSQ-3", "a whole number loses the decimal");
    check(quant_name(2.75) == "GSQ-2.75", "two decimals survive");
    check(bpw_of_quant(quant_name(2.75)) == 2.75, "and parse back");

    const auto alloc = parse_allocation("blk.0.attn_q.weight: IQ3_S\n# a comment\nbad line\n");
    check(alloc.size() == 1, "only well-formed allocation lines count");
    check(alloc.at("blk.0.attn_q.weight") == "iq3_s", "and the type is lower-cased");

    const Plan p = plan_types(l, alloc);
    check(p.total == 2, "only 2-D .weight tensors can be quantized");
    check(p.covered == 1, "one of them is named by the allocation");
    check(p.lines.size() == 1 && p.lines[0] == "blk.0.attn_q.weight=iq3_s", "which is the line that goes to the file");

    std::printf("%s (%d failure%s)\n", failures == 0 ? "all passed" : "FAILURES", failures,
                failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
