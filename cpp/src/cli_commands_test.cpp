// Exercises the pure logic cli_commands.cpp owns: flag parsing, the quant
// name readers pull's build picker runs on, the two encoders `link` mints
// its key with, path containment, and the row filter / table renderers
// list and ps drive, against a fake /api/tags-shaped document.

#include "cli_commands.h"
#include "cli_format.h"

#include <cstdio>
#include <string>
#include <vector>

using namespace llmash;
using json = nlohmann::json;

namespace {

int g_failed = 0;
int g_ran    = 0;

void check(bool ok, const std::string & what) {
    g_ran++;
    if (!ok) {
        g_failed++;
        std::printf("FAIL  %s\n", what.c_str());
    }
}

void eq_str(const std::string & got, const std::string & want, const std::string & what) {
    g_ran++;
    if (got != want) {
        g_failed++;
        std::printf("FAIL  %s\n      got  [%s]\n      want [%s]\n", what.c_str(), got.c_str(), want.c_str());
    }
}

void eq_num(double got, double want, const std::string & what) {
    g_ran++;
    if (got != want) {
        g_failed++;
        std::printf("FAIL  %s: got %g, want %g\n", what.c_str(), got, want);
    }
}

std::vector<std::string> A(std::initializer_list<const char *> xs) {
    std::vector<std::string> v;
    for (const char * x : xs) {
        v.emplace_back(x);
    }
    return v;
}

// ------------------------------------------------------------ parse_simple

void test_parse_simple() {
    const std::vector<std::string> bools{"--insecure", "--draft", "--no-draft"};
    const std::vector<std::string> vals{"--quant", "-q"};

    {
        const ParsedArgs o = parse_simple(A({"qwen3:8b"}), bools, vals);
        check(o.pos.size() == 1 && o.pos[0] == "qwen3:8b", "bare positional");
        check(!o.has_flag("--no-draft"), "unset bool flag is false");
        eq_str(o.val("--quant"), "", "unset value flag is empty");
    }
    {
        // --x=v, --x v, a flag after the positional, and a repeat that wins
        const ParsedArgs o = parse_simple(A({"--quant=Q4_K_M", "hf:owner/repo", "--no-draft", "-q", "Q8_0"}), bools,
                                          vals);
        check(o.pos == A({"hf:owner/repo"}), "flags in any position leave one positional");
        check(o.has_flag("--no-draft"), "--no-draft set");
        check(!o.has_flag("--draft"), "--draft not set");
        eq_str(o.val("--quant"), "Q4_K_M", "--quant=v form");
        eq_str(o.val("-q"), "Q8_0", "-q v form");
        check(o.has_val("-q") && !o.has_val("--insecure"), "has_val only for value flags");
    }
    {
        // a lone dash is a positional, not a flag
        const ParsedArgs o = parse_simple(A({"-", "a"}), {}, {});
        check(o.pos == A({"-", "a"}), "\"-\" is positional");
    }
    {
        // the same value flag twice: the last one wins, as a Go map would
        const ParsedArgs o = parse_simple(A({"-q", "Q4_K_M", "-q", "Q6_K", "m"}), {}, vals);
        eq_str(o.val("-q"), "Q6_K", "repeated value flag keeps the last");
        check(o.vals.size() == 1, "repeated value flag stores one entry");
    }
    {
        // the same bool flag twice stays a single set entry
        const ParsedArgs o = parse_simple(A({"--draft", "--draft"}), bools, vals);
        check(o.bool_flags_set.size() == 1, "repeated bool flag stores one entry");
    }
    {
        bool threw = false;
        try {
            parse_simple(A({"--nope"}), bools, vals);
        } catch (const CliExit & e) {
            threw = e.code == 1 && std::string(e.what()) == "Error: unknown flag: '--nope'";
        }
        check(threw, "unknown flag dies with cobra's wording");
    }
    {
        bool threw = false;
        try {
            parse_simple(A({"model", "--quant"}), bools, vals);
        } catch (const CliExit & e) {
            threw = std::string(e.what()) == "Error: flag needs an argument: --quant";
        }
        check(threw, "value flag with no argument dies");
    }
    {
        // --x= is an empty value, not a missing one
        const ParsedArgs o = parse_simple(A({"--quant=", "m"}), bools, vals);
        check(o.has_val("--quant") && o.val("--quant").empty(), "--quant= is an empty value");
    }
}

// -------------------------------------------------------- quant name logic

void test_bits_of() {
    eq_num(bits_of("Q8_0"), 8, "bits_of Q8_0");
    eq_num(bits_of("Q4_K_M"), 4.5, "bits_of Q4_K_M");
    eq_num(bits_of("Q5_K_S"), 5.5, "bits_of Q5_K_S");
    eq_num(bits_of("Q4_0"), 4, "bits_of Q4_0");
    eq_num(bits_of("IQ3_XXS"), 3, "bits_of IQ3_XXS");
    eq_num(bits_of("IQ4_XS"), 4, "bits_of IQ4_XS");
    eq_num(bits_of("UD-Q4_K_XL"), 4.5, "bits_of UD-Q4_K_XL adds half a bit once");
    eq_num(bits_of("TQ1_0"), 1, "bits_of TQ1_0");
    eq_num(bits_of("F16"), 16, "bits_of F16");
    eq_num(bits_of("BF16"), 16, "bits_of BF16 reads as F16");
    eq_num(bits_of("F32"), 32, "bits_of F32");
    eq_num(bits_of("MXFP4"), 0, "bits_of MXFP4 is unknown");
    eq_num(bits_of("q6_k"), 6.5, "bits_of is case-insensitive");
}

QuantInfo Q(const char * name, int64_t size) {
    QuantInfo q;
    q.name  = name;
    q.size  = size;
    q.files = 1;
    return q;
}

void test_tiers_of() {
    {
        // the named builds win outright, whatever their order or size
        const std::vector<QuantInfo> qs{Q("IQ3_XXS", 3), Q("Q4_K_M", 5), Q("Q6_K", 7), Q("Q8_0", 9)};
        const Tiers                  t = tiers_of(qs);
        eq_str(qs[t.tiny].name, "IQ3_XXS", "tiers tiny");
        eq_str(qs[t.medium].name, "Q4_K_M", "tiers medium");
        eq_str(qs[t.large].name, "Q8_0", "tiers large prefers Q8_0 over Q6_K");
    }
    {
        // nothing named: the bit-range fallbacks pick largest / largest /
        // smallest inside their windows
        const std::vector<QuantInfo> qs{Q("IQ2_M", 2), Q("IQ3_S", 3), Q("IQ3_M", 4), Q("Q5_1", 6), Q("Q7_9", 8)};
        const Tiers                  t = tiers_of(qs);
        eq_str(qs[t.tiny].name, "IQ3_M", "tiers tiny falls back to a named 3-bit build");
        eq_str(qs[t.medium].name, "Q5_1", "tiers medium falls back to the biggest 4-5.5 bit build");
        eq_str(qs[t.large].name, "Q7_9", "tiers large falls back to the biggest 5-8.5 bit build");
    }
    {
        // no candidate fits any window: the positional fallbacks
        const std::vector<QuantInfo> qs{Q("F16", 1), Q("F32", 2), Q("MXFP4", 3)};
        const Tiers                  t = tiers_of(qs);
        eq_num(t.tiny, 0, "tiers tiny falls back to the first build");
        eq_num(t.medium, 1, "tiers medium falls back to the middle build");
        eq_num(t.large, 2, "tiers large falls back to the last build");
    }
    {
        // a custom build has a row of its own above the three sizes, so the
        // picker hands tiers_of the ordinary quants only
        const std::vector<QuantInfo> qs{Q("Q3_K_M", 4), Q("Q4_K_M", 6), Q("Q8_0", 9)};
        const Tiers                  t = tiers_of(qs);
        eq_str(qs[t.tiny].name, "Q3_K_M", "tiers small is the named 3-bit build");
        eq_str(qs[t.medium].name, "Q4_K_M", "medium is the 4-bit one");
        eq_str(qs[t.large].name, "Q8_0", "large is the 8-bit one");
    }
}

void test_quant_tag() {
    eq_str(quant_tag("Qwen3-8B-Q4_K_M-00001-of-00002.gguf"), "Q4_K_M", "quant_tag of a sharded file");
    eq_str(quant_tag("model.BF16.gguf"), "BF16", "quant_tag BF16");
    eq_str(quant_tag("foo-UD-Q4_K_XL.gguf"), "UD-Q4_K_XL", "quant_tag keeps the UD- prefix");
    eq_str(quant_tag("gpt-oss-20b-MXFP4_MOE.gguf"), "MXFP4_MOE", "quant_tag MXFP4_MOE");
    eq_str(quant_tag("Q8_0.gguf"), "Q8_0", "quant_tag at the start of the name");
    eq_str(quant_tag("llama-3-8b-instruct.gguf"), "", "quant_tag of a name with no quant");
    eq_str(quant_tag("qwen3-8b-q6_k.gguf"), "Q6_K", "quant_tag upper-cases what it finds");
    eq_str(quant_tag("Model-IQ3_XXS.mtp.gguf"), "IQ3_XXS", "quant_tag of an MTP sidecar");
}

void test_is_hf_ref() {
    check(is_hf_ref("hf:owner/repo"), "hf: is a hub ref");
    check(is_hf_ref("hf.co/owner/repo"), "hf.co/ is a hub ref");
    check(is_hf_ref("huggingface.co/owner/repo"), "huggingface.co/ is a hub ref");
    check(!is_hf_ref("qwen3:8b"), "a registry name is not a hub ref");
    check(!is_hf_ref("HF:owner/repo"), "the prefix match is case-sensitive");
}

// -------------------------------------------------------------- encoders

void test_url_query_escape() {
    eq_str(url_query_escape("hello world"), "hello+world", "space becomes +");
    eq_str(url_query_escape("unsloth/Qwen3-8B-GGUF"), "unsloth%2FQwen3-8B-GGUF", "slash is escaped");
    eq_str(url_query_escape("qwen3:8b"), "qwen3%3A8b", "colon is escaped");
    eq_str(url_query_escape("a-b_c.d~e"), "a-b_c.d~e", "unreserved characters pass through");
    eq_str(url_query_escape("hf:owner/repo@Q4_K_M"), "hf%3Aowner%2Frepo%40Q4_K_M", "a pinned hub ref");
    eq_str(url_query_escape("+"), "%2B", "a literal plus is escaped");
}

std::string b64(const std::string & s) {
    return base64_url_encode(reinterpret_cast<const unsigned char *>(s.data()), s.size());
}

void test_base64_url_encode() {
    // RFC 4648 section 10's vectors, unpadded (Go's RawURLEncoding)
    eq_str(b64(""), "", "base64 of empty");
    eq_str(b64("f"), "Zg", "base64 of f");
    eq_str(b64("fo"), "Zm8", "base64 of fo");
    eq_str(b64("foo"), "Zm9v", "base64 of foo");
    eq_str(b64("foob"), "Zm9vYg", "base64 of foob");
    eq_str(b64("fooba"), "Zm9vYmE", "base64 of fooba");
    eq_str(b64("foobar"), "Zm9vYmFy", "base64 of foobar");

    const unsigned char urlsafe[3] = {0xFF, 0xEF, 0xBE};
    eq_str(base64_url_encode(urlsafe, 3), "_---", "62/63 use - and _ , not + and /");

    const unsigned char zeros[3] = {0x00, 0x00, 0x00};
    eq_str(base64_url_encode(zeros, 3), "AAAA", "base64 of three zero bytes");

    // what linkKey actually mints: 24 raw bytes -> 32 unpadded characters
    unsigned char raw[24];
    for (size_t i = 0; i < sizeof(raw); i++) {
        raw[i] = static_cast<unsigned char>(i * 7);
    }
    const std::string key = base64_url_encode(raw, sizeof(raw));
    check(key.size() == 32, "a 24-byte key encodes to 32 characters with no padding");
    check(key.find('=') == std::string::npos, "no padding in a raw encoding");
}

// -------------------------------------------------------------- path_under

void test_path_under() {
    check(path_under("C:\\llmash\\bin\\llmash.exe", "C:\\llmash"), "a file inside the root is under it");
    check(path_under("C:\\llmash", "C:\\llmash"), "the root is under itself");
    check(path_under("C:\\llmash\\a\\..\\b\\x.txt", "C:\\llmash"), "a path is normalised before comparing");
    check(!path_under("C:\\other\\llmash.exe", "C:\\llmash"), "a sibling directory is not under the root");
    check(!path_under("C:\\llmashx\\a", "C:\\llmash"), "a name that merely shares a prefix is not under it");
    check(!path_under("D:\\llmash\\a", "C:\\llmash"), "another volume is not under the root");
    check(!path_under("", "C:\\llmash"), "an empty path is not under anything");
}

// ------------------------------------------------------ filter_rows / tables

json fake_tags() {
    return json::parse(R"({
      "models": [
        {"name":"qwen3:8b",       "digest":"sha256:aaaaaaaaaaaabbbbbbbb", "size":4900000000,
         "modified_at":"2026-09-08T00:00:00Z", "size_vram":4900000000, "context_length":8192,
         "expires_at":"2126-09-08T00:00:00Z"},
        {"name":"Qwen3:27b",      "digest":"sha256:ccccccccccccdddddddd", "size":17000000000,
         "modified_at":"2026-09-01T00:00:00Z", "size_vram":8500000000, "context_length":16384,
         "expires_at":"2126-09-08T00:00:00Z"},
        {"name":"nomic-embed",    "digest":"274d1",                       "size":274000000,
         "modified_at":"", "size_vram":0, "context_length":2048, "expires_at":""},
        "not-an-object"
      ]
    })");
}

void test_filter_rows() {
    const json doc = fake_tags();

    const auto all = filter_rows(doc, "", true);
    check(all.size() == 3, "an empty prefix keeps every object row and drops the non-object");

    const auto folded = filter_rows(doc, "qwen", true);
    check(folded.size() == 2, "list folds case, so qwen matches both");

    const auto exact = filter_rows(doc, "qwen", false);
    check(exact.size() == 1 && exact[0]["name"] == "qwen3:8b", "ps does not fold case");

    check(filter_rows(doc, "Qwen", false).size() == 1, "ps matches the capitalised name only");
    check(filter_rows(doc, "zzz", true).empty(), "a prefix nothing starts with matches nothing");
    check(filter_rows(json::object(), "", true).empty(), "a document with no models array is empty");
    check(filter_rows(json::parse(R"({"models":"nope"})"), "", true).empty(), "a non-array models field is empty");
}

void test_render_tables() {
    const auto rows = filter_rows(fake_tags(), "", true);

    const std::string list = render_list(rows);
    check(list.rfind("NAME", 0) == 0, "render_list starts with the NAME header");
    check(list.find("aaaaaaaaaaaa ") != std::string::npos, "render_list shortens the digest to 12 characters");
    check(list.find("aaaaaaaaaaaab") == std::string::npos, "render_list does not print the 13th digest character");
    check(list.find("274d1") != std::string::npos, "a short digest is printed whole");
    check(list.find("Never") != std::string::npos, "an unparsable modified_at renders as Never");
    check(list.find("4.9 GB") != std::string::npos, "render_list humanises the size");

    const std::string ps = render_ps(rows);
    check(ps.rfind("NAME", 0) == 0, "render_ps starts with the NAME header");
    check(ps.find("PROCESSOR") != std::string::npos, "render_ps has a PROCESSOR column");
    check(ps.find("100% GPU") != std::string::npos, "size_vram == size is 100% GPU");
    check(ps.find("100% CPU") != std::string::npos, "size_vram == 0 is 100% CPU");
    check(ps.find("50%/50% CPU/GPU") != std::string::npos, "half in VRAM is a split");
    check(ps.find("8192") != std::string::npos, "render_ps prints the context length");
}

// -------------------------------------------------------- elide / show_info

void test_elide() {
    eq_str(elide({"a", "b"}, 10), "[a b]", "a short list is printed whole");
    eq_str(elide({}, 10), "[]", "an empty list");
    // widths accumulate as 1 + len(first) + 2 + len(next) + ...; at 10 only
    // "llama" (5) fits, at 14 "qwen" (4, plus the 2-column separator) joins it
    eq_str(elide({"llama", "qwen", "gemma", "phi"}, 10), "[llama ...+3 more]", "a long list is cut");
    eq_str(elide({"llama", "qwen", "gemma", "phi"}, 14), "[llama qwen ...+2 more]", "a wider column fits one more");
    eq_str(elide({"averyverylongsinglevalue"}, 10), "[averyverylongsinglevalue]",
           "the first value is always kept, however wide");
    eq_str(elide({"averyverylongsinglevalue", "x"}, 10), "[averyverylongsinglevalue ...+1 more]",
           "the second value is dropped when the first already overflows");
}

void test_show_info() {
    const json resp = json::parse(R"({
      "model_info": {
        "general.architecture": "qwen3",
        "general.parameter_count": 8190735360,
        "qwen3.context_length": 40960,
        "qwen3.embedding_length": 4096,
        "qwen3.attention.head_count": 32,
        "tokenizer.ggml.tokens": ["a","b","c","d","e","f","g","h"],
        "general.file_type": true
      },
      "details": {"family":"qwen3","parameter_size":"8.2B","quantization_level":"Q4_K_M"},
      "capabilities": ["completion","tools"],
      "parameters": "stop \"<|im_end|>\"\ntemperature 0.6\n",
      "system": "line one\nline two\nline three\n",
      "license": "MIT"
    })");

    std::string out;
    show_info(resp, false, out);
    check(out.rfind("  Model\n", 0) == 0, "the Model block comes first, indented two spaces");
    check(out.find("architecture") != std::string::npos, "architecture row");
    check(out.find("qwen3") != std::string::npos, "architecture value");
    check(out.find("8.2B") != std::string::npos, "details.parameter_size wins over the raw count");
    check(out.find("40960") != std::string::npos, "context length is printed without a decimal point");
    check(out.find("4096") != std::string::npos, "embedding length");
    check(out.find("Q4_K_M") != std::string::npos, "quantization row");
    check(out.find("  Capabilities\n") != std::string::npos, "the Capabilities block");
    check(out.find("  Parameters\n") != std::string::npos, "the Parameters block");
    check(out.find("temperature") != std::string::npos, "a parameter name");
    check(out.find("  System\n") != std::string::npos, "the System block");
    check(out.find("line three") == std::string::npos, "only the first two System lines are shown");
    check(out.find("...") != std::string::npos, "the elision marker for a longer System block");
    check(out.find("  License\n") != std::string::npos, "the License block");
    check(out.find("  Metadata\n") == std::string::npos, "no Metadata block without -v");

    std::string verbose;
    show_info(resp, true, verbose);
    check(verbose.find("  Metadata\n") != std::string::npos, "-v adds the Metadata block");
    check(verbose.find("qwen3.attention.head_count") != std::string::npos, "a metadata key");
    check(verbose.find("true") != std::string::npos, "a boolean metadata value");
    check(verbose.find("...+") != std::string::npos, "a long array metadata value is elided");
    const size_t arch = verbose.find("general.architecture");
    const size_t file = verbose.find("general.file_type");
    check(arch != std::string::npos && file != std::string::npos && arch < file, "metadata keys are sorted");

    // no model_info at all: the details-only shape
    std::string bare;
    show_info(json::parse(R"({"details":{"family":"llama","parameter_size":"7B","quantization_level":"Q8_0"}})"), false,
              bare);
    check(bare.find("llama") != std::string::npos, "details.family stands in for the architecture");
    check(bare.find("7B") != std::string::npos, "details.parameter_size stands in for the count");
    check(bare.find("Q8_0") != std::string::npos, "quantization is always printed");
}

// ------------------------------------------------------------------- prog

// Display columns: every UTF-8 lead byte is one cell in these rows.
size_t cols(const std::string & s) {
    size_t n = 0;
    for (const unsigned char c : s) {
        if ((c & 0xC0) != 0x80) {
            n++;
        }
    }
    return n;
}

void test_tradeoff() {
    QuantInfo custom = Q("RCO-3.9", 996 * 1000 * 1000);
    custom.fetch     = 2200LL * 1000 * 1000;
    const std::vector<QuantInfo> plain{Q("IQ3_XS", 923 * 1000 * 1000), Q("Q4_K_M", 1100LL * 1000 * 1000),
                                       Q("Q8_0", 2200LL * 1000 * 1000)};

    const QuantInfo * ref = nearest_by_size(plain, custom.size);
    check(ref != nullptr && ref->name == "IQ3_XS", "the nearest published build by size is IQ3_XS");

    const std::vector<std::string> rows = tradeoff_rows(custom, ref);
    check(rows.size() == 3, "a heading and two builds");
    check(rows[0].find("download") != std::string::npos && rows[0].find("on disk") != std::string::npos,
          "the heading names both figures");
    check(rows[1].rfind("RCO-3.9", 0) == 0, "the custom build comes first");
    check(rows[2].rfind("IQ3_XS", 0) == 0, "the published build comes second");
    check(rows[1].find("2.2 GB") != std::string::npos && rows[1].find("996 MB") != std::string::npos,
          "the custom row shows the download and what is kept");
    check(rows[2].find("923 MB") != std::string::npos, "the published row shows one figure twice");
    check(rows[1].find("\xe2\x96\x88") != std::string::npos && rows[2].find("\xe2\x96\x88") != std::string::npos,
          "both rows carry a bar");
    check(cols(rows[0]) == cols(rows[1]) && cols(rows[1]) == cols(rows[2]), "the rows line up");

    const std::vector<std::string> note = tradeoff_note(custom, ref);
    check(!note.empty() && note[0].rfind("2.4x", 0) == 0, "the note leads with the download multiple");
    check(tradeoff_note(custom, nullptr).empty(), "no comparison, no note");

    QuantInfo published = Q("Q4_K_M", 1100LL * 1000 * 1000);
    check(tradeoff_note(published, ref).empty(), "a build that downloads what it keeps has nothing to trade");
}

void test_prog() {
    const std::string was = prog();
    eq_str(prog(), "llmash", "prog defaults to llmash");
    set_prog("ollama");
    eq_str(prog(), "ollama", "set_prog takes");
    set_prog("");
    eq_str(prog(), "llmash", "an empty prog falls back to llmash");
    set_prog(was);
}

} // namespace

int main() {
    test_parse_simple();
    test_bits_of();
    test_tiers_of();
    test_quant_tag();
    test_is_hf_ref();
    test_url_query_escape();
    test_base64_url_encode();
    test_path_under();
    test_filter_rows();
    test_render_tables();
    test_elide();
    test_show_info();
    test_tradeoff();
    test_prog();

    std::printf("%s: %d checks, %d failed\n", g_failed == 0 ? "PASS" : "FAIL", g_ran, g_failed);
    return g_failed == 0 ? 0 : 1;
}
