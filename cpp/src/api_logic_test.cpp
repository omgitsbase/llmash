#include "api_logic.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using json   = nlohmann::json;
using namespace llmash;

namespace {

int failures = 0;

void check(bool ok, const std::string & what) {
    std::printf("%s  %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) {
        failures++;
    }
}

void eq(const std::string & got, const std::string & want, const std::string & what) {
    check(got == want, what + (got == want ? "" : "  [got \"" + got + "\", want \"" + want + "\"]"));
}

void eq_num(double got, double want, const std::string & what) {
    const bool ok = std::abs(got - want) < 1e-9;
    check(ok, what + (ok ? "" : "  [got " + std::to_string(got) + "]"));
}

void set_env(const char * name, const std::string & value) {
#ifdef _WIN32
    _putenv_s(name, value.c_str());
#else
    if (value.empty()) {
        unsetenv(name);
    } else {
        setenv(name, value.c_str(), 1);
    }
#endif
}

void write_file(const fs::path & p, const std::string & body) {
    std::ofstream out(p, std::ios::binary | std::ios::trunc);
    out << body;
}

std::string read_file(const fs::path & p) {
    std::ifstream in(p, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

Model fake_model(const fs::path & dir) {
    Model m;
    m.name         = "qwen3:8b";
    m.path         = (dir / "Qwen3-8B-Q4_K_M.gguf").string();
    m.quant        = "Q4_K_M";
    m.arch         = "qwen3";
    m.size         = 4661000000ULL;
    m.digest       = "sha256:abc123";
    m.family       = "qwen3";
    m.param_size   = "8B";
    m.tmpl         = "{{ .Prompt }}";
    m.modified     = 1700000000;
    m.ctx_train    = 262144;
    m.experts      = 128;
    m.experts_used = 8;
    m.caps         = {"completion", "tools"};
    return m;
}

const double inf = std::numeric_limits<double>::infinity();

} // namespace

int main() {
    const fs::path dir = fs::temp_directory_path() / "llmash_api_logic_test";
    fs::remove_all(dir);
    fs::create_directories(dir);
    set_env("LLMASH_GGUF", "");
    set_env("LLMASH_LINK_KEY", "");

    // ------------------------------------------------------------ iso / bytes
    eq(iso(0), "1970-01-01T00:00:00+00:00", "iso: the epoch, no fractional part");
    eq(iso(1700000000), "2023-11-14T22:13:20+00:00", "iso: a whole second");
    eq(iso(1700000000.5), "2023-11-14T22:13:20.5+00:00", "iso: trailing zeros trimmed");
    eq(iso(1700000000.000125), "2023-11-14T22:13:20.000125+00:00", "iso: microseconds kept");
    check(now_unix() > 1600000000, "now_unix: a plausible epoch time");

    eq(human_bytes(500), "500 B", "human_bytes: bytes print whole");
    eq(human_bytes(2048), "2 KB", "human_bytes: KB print whole");
    eq(human_bytes(5 * 1024 * 1024), "5.0 MB", "human_bytes: MB keep one decimal");
    eq(human_bytes(1610612736), "1.5 GB", "human_bytes: GB keep one decimal");
    eq(human_bytes(1024.0 * 1024 * 1024 * 1024 * 3), "3.0 TB", "human_bytes: TB");
    eq(human_bytes(1024.0 * 1024 * 1024 * 1024 * 1024 * 2), "2.0 PB", "human_bytes: past TB");

    // --------------------------------------------------------- safe_model_name
    eq(safe_model_name("Qwen3 8B:latest"), "Qwen3-8B", "safe_model_name: tag dropped, space folded");
    eq(safe_model_name("hf.co/user/repo"), "hf.co-user-repo", "safe_model_name: slashes folded");
    eq(safe_model_name("a_b-c.d"), "a_b-c.d", "safe_model_name: legal characters kept");
    eq(safe_model_name(":only-tag"), "", "safe_model_name: nothing before the colon");

    // -------------------------------------------------------------- keep_alive
    eq_num(parse_keep_alive(json(nullptr), "15m"), 900, "keep_alive: null falls back to the default");
    eq_num(parse_keep_alive(json(nullptr), ""), 300, "keep_alive: no default at all is 300");
    eq_num(parse_keep_alive(json("5m"), "15m"), 300, "keep_alive: 5m");
    eq_num(parse_keep_alive(json("1h"), "15m"), 3600, "keep_alive: 1h");
    eq_num(parse_keep_alive(json("45s"), "15m"), 45, "keep_alive: 45s");
    eq_num(parse_keep_alive(json("500ms"), "15m"), 0.5, "keep_alive: milliseconds");
    eq_num(parse_keep_alive(json("120"), "15m"), 120, "keep_alive: a bare number string");
    eq_num(parse_keep_alive(json(30), "15m"), 30, "keep_alive: a JSON number");
    eq_num(parse_keep_alive(json(0), "15m"), 0, "keep_alive: zero means unload now");
    check(std::isinf(parse_keep_alive(json(-1), "15m")), "keep_alive: -1 pins forever");
    check(std::isinf(parse_keep_alive(json("-1"), "15m")), "keep_alive: \"-1\" pins forever");
    check(std::isinf(parse_keep_alive(json("-1s"), "15m")), "keep_alive: \"-1s\" pins forever");
    check(std::isinf(parse_keep_alive(json(-90), "15m")), "keep_alive: any negative pins forever");
    eq_num(parse_keep_alive(json("banana"), "15m"), 300, "keep_alive: junk falls back to 300");
    eq_num(parse_keep_alive(json("12x"), "15m"), 300, "keep_alive: an unknown unit is junk");
    eq_num(parse_keep_alive(json(true), "15m"), 300, "keep_alive: a bool is junk");
    eq_num(keep_alive_out(inf), -1, "keep_alive_out: forever reports as -1");
    eq_num(keep_alive_out(300), 300, "keep_alive_out: a finite value passes through");

    // -------------------------------------------------------------------- auth
    check(check_api_key("", "secret", "secret"), "auth: X-API-Key accepted");
    check(check_api_key("Bearer secret", "", "secret"), "auth: bearer accepted");
    check(check_api_key("BEARER secret", "", "secret"), "auth: bearer scheme is case-insensitive");
    check(check_api_key("Bearer  secret ", "", "secret"), "auth: bearer token is trimmed");
    check(check_api_key("Bearer secret", "wrong", "secret"), "auth: bearer wins over X-API-Key");
    check(!check_api_key("Bearer wrong", "secret", "secret"), "auth: a bad bearer is not rescued by X-API-Key");
    check(!check_api_key("", "", "secret"), "auth: no credentials rejected");
    check(!check_api_key("", "secret", ""), "auth: an empty expected key always rejects");
    check(!check_api_key("Bearer secret", "", ""), "auth: empty expected key rejects bearers too");
    check(!check_api_key("", "secre", "secret"), "auth: a prefix is not a match");
    check(!check_api_key("Bearer", "", "Bearer"), "auth: a short header is not a bearer scheme");

    // ------------------------------------------------------------- loadable
    Config cfg;
    cfg.root        = dir.string();
    cfg.models_root = (dir / "models").string();
    cfg.ctx         = 8192;
    cfg.public_port = 11435;

    Model m = fake_model(dir);
    check(loadable(m), "loadable: an ordinary arch loads");
    {
        Model bad = m;
        bad.family = "GPTOSS";
        check(!loadable(bad), "loadable: gptoss is refused, case-folded");
        bad.family = "glm4moelite";
        check(!loadable(bad), "loadable: glm4moelite is refused");
    }
    check(advertised_ctx(cfg) == 8192, "advertised_ctx: the configured default");
    { Config c2; c2.ctx = 0; check(advertised_ctx(c2) == 8192, "advertised_ctx: zero falls back"); }

    // ------------------------------------------------------------- tag entry
    const json e = tag_entry_json(m, cfg);
    eq(e.value("name", ""), "qwen3:8b", "tagEntry: name");
    eq(e.value("model", ""), "qwen3:8b", "tagEntry: model repeats the name");
    eq(e.value("modified_at", ""), "2023-11-14T22:13:20+00:00", "tagEntry: modified_at is an iso stamp");
    check(e["size"] == 4661000000ULL, "tagEntry: size in bytes");
    eq(e.value("digest", ""), "abc123", "tagEntry: the sha256: prefix is stripped");
    check(e["loadable"] == true, "tagEntry: loadable flag");
    check(e["capabilities"] == json::array({"completion", "tools"}), "tagEntry: capabilities");
    const json d = e["details"];
    eq(d.value("parent_model", "x"), "", "tagEntry.details: parent_model empty");
    eq(d.value("format", ""), "gguf", "tagEntry.details: format");
    eq(d.value("family", ""), "qwen3", "tagEntry.details: family");
    check(d["families"] == json::array({"qwen3"}), "tagEntry.details: families holds the family");
    eq(d.value("parameter_size", ""), "8B", "tagEntry.details: parameter_size");
    eq(d.value("quantization_level", ""), "Q4_K_M", "tagEntry.details: quantization_level");
    check(d["context_length"] == 8192, "tagEntry.details: context_length");
    check(d["expert_count"] == 128, "tagEntry.details: expert_count");
    check(d["expert_used_count"] == 8, "tagEntry.details: expert_used_count");
    check(d.size() == 9, "tagEntry.details: no extra fields");
    check(e.size() == 8, "tagEntry: no extra fields");
    {
        Model plain = m;
        plain.caps.clear();
        plain.family.clear();
        const json pe = tag_entry_json(plain, cfg);
        check(pe["capabilities"] == json::array({"completion"}), "tagEntry: capabilities default to completion");
        check(pe["details"]["families"] == json::array(), "tagEntry: no family means an empty families array");
    }
    {
        Model nodigest = m;
        nodigest.digest.clear();
        const std::string a = tag_entry_json(nodigest, cfg).value("digest", "");
        check(a.size() == 64, "digest_of: stands in with 64 hex characters");
        check(a == digest_of(nodigest), "digest_of: the entry uses it verbatim");
        check(a == digest_of(nodigest), "digest_of: stable across calls");
        Model other = nodigest;
        other.size += 1;
        check(digest_of(other) != a, "digest_of: a different size is a different digest");
        check(a.find_first_not_of("0123456789abcdef") == std::string::npos, "digest_of: hex only");
    }

    // ---------------------------------------------------------------- listings
    write_file(m.path, "GGUF-not-really");
    Model missing = m;
    missing.name  = "ghost:latest";
    missing.path  = (dir / "gone.gguf").string();
    Model unloadable = m;
    unloadable.name   = "gpt-oss:120b";
    unloadable.family = "gptoss";
    unloadable.path   = (dir / "oss.gguf").string();
    write_file(unloadable.path, "GGUF");

    const json tags = tags_json({m, missing, unloadable}, cfg);
    check(tags["models"].is_array() && tags["models"].size() == 1, "tags: only the present, loadable model");
    eq(tags["models"][0].value("name", ""), "qwen3:8b", "tags: the surviving row");
    check(tags.size() == 1, "tags: one key, models");

    const json v1e = v1_entry_json(m, cfg);
    eq(v1e.value("id", ""), "qwen3:8b", "v1Entry: id");
    eq(v1e.value("object", ""), "model", "v1Entry: object");
    check(v1e["created"] == 1700000000, "v1Entry: created is a unix second");
    eq(v1e.value("owned_by", ""), "llmash", "v1Entry: owned_by");
    check(v1e["context_length"] == 8192 && v1e["max_model_len"] == 8192 &&
              v1e["max_context_length"] == 8192 && v1e["context_window"] == 8192,
          "v1Entry: all four context aliases");
    check(v1e.size() == 8, "v1Entry: no extra fields");

    const json v1m = v1_models_json({m, missing, unloadable}, cfg);
    eq(v1m.value("object", ""), "list", "v1Models: object is list");
    check(v1m["data"].size() == 1, "v1Models: same filtering as tags");

    const json sh = show_json(m, cfg);
    eq(sh.value("license", "x"), "", "show: license empty");
    eq(sh.value("modelfile", ""), "FROM " + m.path, "show: modelfile is a FROM line");
    eq(sh.value("template", ""), "{{ .Prompt }}", "show: template");
    eq(sh.value("system", "x"), "", "show: system");
    eq(sh.value("parameters", "x"), "", "show: parameters");
    check(sh["details"] == d, "show: details are the tagEntry details");
    eq(sh["model_info"].value("general.architecture", ""), "qwen3", "show: model_info architecture");
    check(sh["model_info"]["qwen3.context_length"] == 8192, "show: model_info arch-scoped context_length");
    check(sh["model_info"]["context_length"] == 8192, "show: model_info context_length");
    eq(sh["model_info"].value("general.parameter_count", ""), "8B", "show: model_info parameter_count");
    check(sh["capabilities"] == json::array({"completion", "tools"}), "show: capabilities");
    {
        Model nofam = m;
        nofam.family.clear();
        const json s2 = show_json(nofam, cfg);
        eq(s2["model_info"].value("general.architecture", ""), "llama", "show: a missing family reads as llama");
        check(s2["model_info"].contains("llama.context_length"), "show: the arch key follows the fallback");
    }

    const json oe = openai_error_json("model 'x' not found", "invalid_request_error", "model_not_found");
    eq(oe["error"].value("message", ""), "model 'x' not found", "openai error: message");
    eq(oe["error"].value("type", ""), "invalid_request_error", "openai error: type");
    eq(oe["error"].value("code", ""), "model_not_found", "openai error: code");
    check(oe.size() == 1 && oe["error"].size() == 3, "openai error: nested under one error key");

    // ---------------------------------------------------------------------- ps
    InstanceView v;
    v.model      = m;
    v.vram_gb    = 2;
    v.on_gpu     = true;
    v.ctx        = 16384;
    v.expires_at = 1700000600;
    const json pe = ps_entry_json(v, cfg);
    check(pe["size"] == 2147483648LL, "ps: size is the resident VRAM, not the file size");
    check(pe["size_vram"] == 2147483648LL, "ps: size_vram matches when the model is on the card");
    eq(pe.value("expires_at", ""), "2023-11-14T22:23:20+00:00", "ps: expires_at");
    check(pe["context_length"] == 16384, "ps: context_length is the instance's, not the default");
    eq(pe.value("name", ""), "qwen3:8b", "ps: the tagEntry fields are still there");
    check(pe["details"]["context_length"] == 8192, "ps: details keep the advertised context");
    {
        InstanceView cpu = v;
        cpu.on_gpu       = false;
        check(ps_entry_json(cpu, cfg)["size_vram"] == 0, "ps: nothing resident when it is not on the GPU");
        InstanceView pinned = v;
        pinned.expires_at   = inf;
        eq(ps_entry_json(pinned, cfg).value("expires_at", ""), "9999-12-31T23:59:59Z",
           "ps: a pinned model gets the sentinel stamp");
        pinned.expires_at = 1e16;
        eq(ps_entry_json(pinned, cfg).value("expires_at", ""), "9999-12-31T23:59:59Z",
           "ps: an absurd expiry is the sentinel too");
    }
    {
        InstanceView loading = v;
        loading.ready        = false;
        const json list      = ps_json({v, loading}, cfg);
        check(list["models"].size() == 1, "ps: a model still loading is not listed");
        check(list.size() == 1 && list.contains("models"), "ps: one key, models");
    }

    // ------------------------------------------------------------- loose_dir
    eq(loose_dir(cfg), (fs::path(cfg.models_root) / "gguf").string(), "loose_dir: <models_root>/gguf");
    {
        Config c2   = cfg;
        c2.gguf_dir = (dir / "loose").string();
        eq(loose_dir(c2), c2.gguf_dir, "loose_dir: gguf_dir wins over models_root");
        set_env("LLMASH_GGUF", (dir / "env").string());
        eq(loose_dir(c2), (dir / "env").string(), "loose_dir: LLMASH_GGUF wins over both");
        set_env("LLMASH_GGUF", "");
    }

    // -------------------------------------------------------- find_projector
    {
        const fs::path pdir = dir / "proj";
        fs::create_directories(pdir);
        write_file(pdir / "Foo-Q4_K_M.gguf", "x");
        check(find_projector((pdir / "Foo-Q4_K_M.gguf").string()).empty(), "find_projector: none is empty");
        write_file(pdir / "Foo-Q8_0-mmproj.gguf", "x");
        eq(find_projector((pdir / "Foo-Q4_K_M.gguf").string()), (pdir / "Foo-Q8_0-mmproj.gguf").string(),
           "find_projector: matched across quantisations");
        write_file(pdir / "Foo-Q4_K_M.mmproj.gguf", "x");
        eq(find_projector((pdir / "Foo-Q4_K_M.gguf").string()), (pdir / "Foo-Q4_K_M.mmproj.gguf").string(),
           "find_projector: the exact sidecar name wins");
        write_file(pdir / "Bar-00001-of-00002.gguf", "x");
        write_file(pdir / "Bar.mmproj.gguf", "x");
        eq(find_projector((pdir / "Bar-00001-of-00002.gguf").string()), (pdir / "Bar.mmproj.gguf").string(),
           "find_projector: the shard suffix is stripped first");
    }

    // -------------------------------------------------------------- copy file
    {
        const fs::path src = dir / "src.bin";
        const fs::path dst = dir / "dst.bin";
        write_file(src, "hello projector");
        copy_file_preserving_mtime(src.string(), dst.string());
        eq(read_file(dst), "hello projector", "copy_file: contents");
        check(fs::last_write_time(dst) == fs::last_write_time(src), "copy_file: mtime preserved");
        bool threw = false;
        try {
            copy_file_preserving_mtime((dir / "nope.bin").string(), dst.string());
        } catch (const std::exception &) {
            threw = true;
        }
        check(threw, "copy_file: a missing source throws");
    }

    // ------------------------------------------------------------- quantize
    {
        Config qcfg     = cfg;
        qcfg.llama_bin  = (dir / "bin" / "llama-server.exe").string();
        std::string msg;
        try {
            quantize_gguf(qcfg, m.path, (dir / "out.gguf").string(), "q4_k_m");
        } catch (const std::exception & ex) {
            msg = ex.what();
        }
        eq(msg, "llama-quantize.exe not found next to llama-server", "quantize: reports a missing tool");
    }

    // ---------------------------------------------------------------- create
    {
        Config ccfg     = cfg;
        ccfg.gguf_dir   = (dir / "loose").string();
        ccfg.llama_bin  = (dir / "bin" / "llama-server.exe").string();

        std::vector<json> ev;
        const Emit emit = [&](const json & j) { ev.push_back(j); };

        run_create(ccfg, "notes", (dir / "notes.txt").string(), "", "", emit);
        check(ev.size() == 1 && ev[0].contains("error"), "create: a non-gguf source is refused");
        eq(ev[0].value("error", ""),
           "'" + (dir / "notes.txt").string() + "' isn't a local .gguf, so there's nothing to import.",
           "create: the refusal names the file");

        ev.clear();
        run_create(ccfg, "My Model:latest", m.path, "", "", emit);
        const fs::path dest = fs::path(ccfg.gguf_dir) / "My-Model.gguf";
        check(fs::exists(dest), "create: the import landed under a safe name");
        eq(read_file(dest), "GGUF-not-really", "create: the bytes were copied");
        check(ev.size() == 3, "create: three ndjson lines");
        eq(ev[0].value("status", ""),
           "importing Qwen3-8B-Q4_K_M.gguf -> " + dest.string() + " (15 B) ...", "create: the importing line");
        eq(ev[1].value("status", ""), "created 'My Model:latest'  (llmash serves it as my-model)",
           "create: the created line");
        eq(ev[2].value("status", ""), "success", "create: the final line");

        ev.clear();
        run_create(ccfg, "My Model:latest", m.path, "", "", emit);
        check(ev.size() == 1 && ev[0].contains("error"), "create: an existing destination is refused");
        eq(ev[0].value("error", ""),
           "'My-Model.gguf' already exists in " + ccfg.gguf_dir + ". Remove it first (rm) or pick another name.",
           "create: the refusal names the folder");

        ev.clear();
        run_create(ccfg, "quantme", m.path, "q4_k_m", "", emit);
        check(ev.size() == 2, "create: the quantize path stops at the failure");
        eq(ev[0].value("status", ""), "quantizing Qwen3-8B-Q4_K_M.gguf -> quantme.gguf (Q4_K_M) ...",
           "create: the quantizing line upper-cases the level");
        eq(ev[1].value("error", ""), "llama-quantize.exe not found next to llama-server",
           "create: the quantizer's failure is streamed, not thrown");

        // a source with a projector beside it
        ev.clear();
        const fs::path vsrc = dir / "Vision-Q4_K_M.gguf";
        write_file(vsrc, "V");
        write_file(dir / "Vision-Q4_K_M.mmproj.gguf", "MM");
        run_create(ccfg, "vision", vsrc.string(), "", "", emit);
        check(fs::exists(fs::path(ccfg.gguf_dir) / "vision.mmproj.gguf"), "create: the projector came along");
        check(ev.size() == 4 && ev[1].value("status", "") == "also imported its projector (Vision-Q4_K_M.mmproj.gguf)",
              "create: the projector line");
    }

    // ------------------------------------------------------------------ copy
    {
        Config ccfg   = cfg;
        ccfg.gguf_dir = (dir / "loose2").string();
        std::vector<json> ev;
        const Emit emit = [&](const json & j) { ev.push_back(j); };

        run_copy(ccfg, m, "Copy Of:tag", emit);
        const fs::path dest = fs::path(ccfg.gguf_dir) / "Copy-Of.gguf";
        check(fs::exists(dest), "copy: the duplicate landed");
        check(ev.size() == 3, "copy: three ndjson lines");
        eq(ev[0].value("status", ""), "copying Qwen3-8B-Q4_K_M.gguf -> Copy-Of.gguf (15 B) ...",
           "copy: the copying line");
        eq(ev[1].value("status", ""), "copied 'qwen3:8b' to 'Copy Of:tag'", "copy: the copied line");
        eq(ev[2].value("status", ""), "success", "copy: the final line");

        ev.clear();
        run_copy(ccfg, m, "Copy Of:tag", emit);
        check(ev.size() == 1 && ev[0].contains("error"), "copy: an existing destination is refused");

        ev.clear();
        Model bad = m;
        bad.path  = (dir / "notes.txt").string();
        write_file(bad.path, "text");
        run_copy(ccfg, bad, "whatever", emit);
        eq(ev[0].value("error", ""), "can't copy 'qwen3:8b': its backing file isn't a GGUF llmash can duplicate.",
           "copy: a non-GGUF source is refused");
    }

    // ---------------------------------------------------------------- delete
    {
        const fs::path ddir = dir / "del";
        fs::create_directories(ddir);
        write_file(ddir / "Shardy-00001-of-00002.gguf", "a");
        write_file(ddir / "Shardy-00002-of-00002.gguf", "b");
        write_file(ddir / "Shardy.mmproj.gguf", "c");
        write_file(ddir / "Other.gguf", "d");

        Model sharded  = m;
        sharded.name   = "shardy:gguf";
        sharded.path   = (ddir / "Shardy-00001-of-00002.gguf").string();
        const DeleteOutcome ok = run_delete(sharded);
        check(ok.status == 200, "delete: a loose model is removed");
        eq(ok.body.value("status", ""), "success", "delete: status success");
        check(ok.body["removed"].size() == 3, "delete: every shard and the projector went with it");
        check(!fs::exists(ddir / "Shardy-00002-of-00002.gguf"), "delete: the second shard is gone");
        check(fs::exists(ddir / "Other.gguf"), "delete: an unrelated model is untouched");

        const DeleteOutcome again = run_delete(sharded);
        check(again.status == 409, "delete: nothing left to delete is a 409");
        eq(again.body.value("error", ""),
           "found shardy:gguf but nothing to delete. Its file is at " + sharded.path +
               ", outside the model directory",
           "delete: the nothing-to-delete message");

        Model lib      = m;
        lib.name       = "readonly:gguf";
        lib.in_library = true;
        const DeleteOutcome ro = run_delete(lib);
        check(ro.status == 409, "delete: a library model is refused");
        eq(ro.body.value("error", ""),
           "readonly:gguf is read from " + lib.path + ", a folder llmash only reads; delete the file yourself",
           "delete: the library message");
        check(fs::exists(lib.path), "delete: the library file is still there");
    }

    // ------------------------------------------------------------- cli cache
    {
        CliTextCache cache;
        int          builds = 0;
        const auto   build  = [&]() { builds++; return "list #" + std::to_string(builds); };
        eq(cache.get("list", 1.0, build), "list #1", "cli cache: the first read builds");
        eq(cache.get("list", 1.0, build), "list #1", "cli cache: a fresh entry is served from memory");
        check(builds == 1, "cli cache: only one build so far");
        eq(cache.get("ps", 1.0, build), "list #2", "cli cache: each kind is cached apart");
        eq(cache.get("list", 0.0, build), "list #3", "cli cache: a zero ttl always rebuilds");
        cache.invalidate();
        eq(cache.get("list", 1.0, build), "list #4", "cli cache: invalidate forces a rebuild");
    }

    // --------------------------------------------------------------- link key
    {
        Config lcfg = cfg;
        lcfg.root   = (dir / "install").string();
        fs::create_directories(lcfg.root);
        const std::string k1 = link_key(lcfg);
        check(k1.rfind("sk-llmash-", 0) == 0, "link key: minted with the sk-llmash- prefix");
        check(k1.size() == 10 + 32, "link key: 24 random bytes as unpadded base64url");
        check(k1.find('=') == std::string::npos, "link key: no base64 padding");
        check(fs::exists(fs::path(lcfg.root) / "link.json"), "link key: persisted to link.json");
        eq(link_key(lcfg), k1, "link key: the second read returns the same key");
        const json saved = json::parse(read_file(fs::path(lcfg.root) / "link.json"));
        eq(saved.value("key", ""), k1, "link key: link.json holds the key");
        check(saved["public_port"] == 11435, "link key: link.json records the public port");
        set_env("LLMASH_LINK_KEY", "sk-env-override");
        eq(link_key(lcfg), "sk-env-override", "link key: the environment wins");
        set_env("LLMASH_LINK_KEY", "");
    }

    // ------------------------------------------------------------- subprocess
    {
        Subprocess proc;
        check(!proc.start({}), "subprocess: an empty argv does not start");
        check(proc.join() == -1, "subprocess: joining an unstarted process is -1");
        Subprocess run;
        if (run.start({"cmd.exe", "/c", "exit 7"})) {
            check(run.join() == 7, "subprocess: the child's exit code comes back");
        } else {
            check(false, "subprocess: cmd.exe started");
        }
    }

    fs::remove_all(dir);
    std::printf("\n%s (%d failure%s)\n", failures == 0 ? "PASS" : "FAIL", failures,
                failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
