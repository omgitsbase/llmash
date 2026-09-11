// Standalone whitebox test for cli_run.cpp: #includes the .cpp directly (not
// just the header) so it can exercise the anonymous-namespace internals
// (InterruptGuard, Spinner, the wire JSON helpers) directly, the same way
// test_main.cpp checks registry.cpp/gguf.cpp through their public headers.
#include "cli_run.cpp"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>

using namespace llmash;

namespace {

int failures = 0;

void check(bool ok, const char * what) {
    std::printf("%s  %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) failures++;
}

} // namespace

int main() {
    // ------------------------------------------------------- parse_run_args

    check(parse_run_args({"mymodel"}).model == "mymodel", "parse_run_args: bare model, no flags");
    check(parse_run_args({"mymodel"}).prompt.empty(), "parse_run_args: no prompt when none given");

    {
        const RunArgs r = parse_run_args({"--verbose", "m", "hello", "world"});
        check(r.verbose, "parse_run_args: --verbose sets verbose");
        check(r.model == "m", "parse_run_args: model is the first positional");
        check(r.prompt == "hello world", "parse_run_args: remaining positionals join with spaces");
    }
    check(parse_run_args({"--nowordwrap", "m"}).nowordwrap, "parse_run_args: --nowordwrap");
    check(parse_run_args({"--hidethinking", "m"}).hidethinking, "parse_run_args: --hidethinking");
    check(parse_run_args({"--format", "json", "m"}).format == "json", "parse_run_args: --format value form");
    check(parse_run_args({"--format=json", "m"}).format == "json", "parse_run_args: --format=value form");
    check(parse_run_args({"--keepalive", "10m", "m"}).keepalive == "10m", "parse_run_args: --keepalive value form");
    check(parse_run_args({"--keepalive=0", "m"}).keepalive == "0", "parse_run_args: --keepalive=value form");
    check(parse_run_args({"--ctx", "8192", "m"}).ctx == 8192, "parse_run_args: --ctx value form");
    check(parse_run_args({"--ctx=4096", "m"}).ctx == 4096, "parse_run_args: --ctx=value form");
    check(parse_run_args({"--ctx", "notanumber", "m"}).ctx == 0,
          "parse_run_args: an invalid --ctx silently stays 0 (fmt.Sscan semantics), no throw");

    check(!parse_run_args({"m"}).think_set, "parse_run_args: --think unset by default");
    {
        const RunArgs r = parse_run_args({"--think", "m"});
        check(r.think_set && r.think == "true", "parse_run_args: bare --think is true");
    }
    {
        const RunArgs r = parse_run_args({"--think=high", "m"});
        check(r.think_set && r.think == "high", "parse_run_args: --think=level");
    }
    {
        bool threw = false;
        try {
            parse_run_args({"--think=bogus", "m"});
        } catch (const CliUsageError &) {
            threw = true;
        }
        check(threw, "parse_run_args: an invalid --think level throws");
    }

    check(!parse_run_args({"m"}).truncate.has_value(), "parse_run_args: --truncate unset by default");
    {
        const RunArgs r = parse_run_args({"--truncate", "m"});
        check(r.truncate.has_value() && *r.truncate == true, "parse_run_args: bare --truncate is true");
    }
    {
        const RunArgs r = parse_run_args({"--truncate=false", "m"});
        check(r.truncate.has_value() && *r.truncate == false, "parse_run_args: --truncate=false");
    }
    {
        bool threw = false;
        try {
            parse_run_args({"--truncate=maybe", "m"});
        } catch (const CliUsageError &) {
            threw = true;
        }
        check(threw, "parse_run_args: an invalid --truncate value throws");
    }

    {
        const RunArgs r = parse_run_args({"--temperature", "0.7", "m"});
        check(r.temperature.has_value() && std::fabs(*r.temperature - 0.7) < 1e-9, "parse_run_args: --temperature");
    }
    check(!parse_run_args({"--temperature", "nope", "m"}).temperature.has_value(),
          "parse_run_args: an invalid --temperature is silently left unset, no throw");
    check(parse_run_args({"--dimensions", "512", "m"}).dimensions == 512, "parse_run_args: --dimensions");

    {
        bool threw = false;
        try {
            parse_run_args({"--bogus", "m"});
        } catch (const CliUsageError & e) {
            threw = true;
            check(std::string(e.what()).find("unknown flag") != std::string::npos,
                  "parse_run_args: the unknown-flag message names the flag");
        }
        check(threw, "parse_run_args: an unknown flag throws");
    }
    {
        bool threw = false;
        try {
            parse_run_args({});
        } catch (const CliUsageError &) {
            threw = true;
        }
        check(threw, "parse_run_args: no model argument throws");
    }
    {
        bool threw = false;
        try {
            parse_run_args({"--format"});
        } catch (const CliUsageError & e) {
            threw = true;
            check(std::string(e.what()).find("flag needs an argument") != std::string::npos,
                  "parse_run_args: a value flag with nothing after it names the flag");
        }
        check(threw, "parse_run_args: --format with no value throws");
    }
    {
        const RunArgs r = parse_run_args({"--insecure", "--width", "512", "m"});
        check(r.model == "m", "parse_run_args: --insecure/--width are accepted but do nothing for run");
    }
    {
        const RunArgs r = parse_run_args({"--verbose", "--think", "--ctx=100", "m", "say", "hi"});
        check(r.verbose && r.think_set && r.think == "true" && r.ctx == 100 && r.model == "m" && r.prompt == "say hi",
              "parse_run_args: several flags combined with a multi-word prompt");
    }

    // ----------------------------------------------------------- format_value

    check(format_value("").is_null(), "format_value: empty string becomes null");
    check(format_value("json") == "json", "format_value: the literal \"json\" passes through");
    check(format_value(R"({"a":1})")["a"] == 1, "format_value: a valid JSON object is parsed");
    check(format_value("not json") == "not json", "format_value: non-JSON falls back to the raw string");

    // ----------------------------------------------------------- format_param

    {
        const json v = format_param("stop", {"a", "b"});
        check(v.is_array() && v.size() == 2 && v[0] == "a" && v[1] == "b", "format_param: stop becomes a string array");
    }
    check(format_param("num_ctx", {"4096"}).get<long long>() == 4096, "format_param: an int param parses");
    {
        bool threw = false;
        try {
            format_param("num_ctx", {"abc"});
        } catch (const CliUsageError &) {
            threw = true;
        }
        check(threw, "format_param: a bad int param throws");
    }
    check(format_param("use_mmap", {"true"}).get<bool>() == true, "format_param: a bool param parses");
    {
        bool threw = false;
        try {
            format_param("use_mmap", {"nope"});
        } catch (const CliUsageError &) {
            threw = true;
        }
        check(threw, "format_param: a bad bool param throws");
    }
    check(std::fabs(format_param("temperature", {"0.8"}).get<double>() - 0.8) < 1e-9,
          "format_param: an unlisted key parses as a float");
    {
        bool threw = false;
        try {
            format_param("temperature", {"nope"});
        } catch (const CliUsageError &) {
            threw = true;
        }
        check(threw, "format_param: a bad float param throws");
    }

    // ------------------------------------------------------ normalize_file_path

    check(normalize_file_path("C:\\Users\\me\\my\\ pic.png") == "C:\\Users\\me\\my pic.png",
          "normalize_file_path: unescapes a backslash-escaped space");
    check(normalize_file_path("a\\(1\\).jpg") == "a(1).jpg", "normalize_file_path: unescapes parens");
    check(normalize_file_path("plain\\path.png") == "plain\\path.png",
          "normalize_file_path: an ordinary backslash that isn't escaping anything is left alone");

    // ----------------------------------------------------------- encode_base64

    check(encode_base64("hello") == "aGVsbG8=", "encode_base64: known vector");
    check(encode_base64("").empty(), "encode_base64: empty input");
    check(encode_base64("Ab") == "QWI=", "encode_base64: two-byte input pads with one '='");

    // ------------------------------------------------- extract_file_data (real files)

    {
        char tmp_dir[MAX_PATH];
        GetTempPathA(MAX_PATH, tmp_dir);
        const std::string path = std::string(tmp_dir) + "cli_run_test_img.png";
        {
            std::ofstream out(path, std::ios::binary);
            out << "fakepngbytes";
        }
        const std::string      prompt = "describe " + path + " please";
        const FileExtraction fe     = extract_file_data(prompt);
        check(fe.ok, "extract_file_data: reads a file that really exists on disk");
        check(fe.images_b64.size() == 1 && fe.images_b64[0] == encode_base64("fakepngbytes"),
              "extract_file_data: base64-encodes the file's real bytes");
        check(fe.text.find(path) == std::string::npos, "extract_file_data: the path is stripped out of the text");
        // Go erases the path and trims the ends only, so the gap it leaves behind stays.
        check(fe.text == "describe  please", "extract_file_data: the rest of the prompt survives, ends trimmed");
        DeleteFileA(path.c_str());
    }
    {
        const FileExtraction fe = extract_file_data("just text, no files");
        check(fe.ok && fe.images_b64.empty() && fe.text == "just text, no files",
              "extract_file_data: plain text with no file reference is untouched");
    }
    {
        const FileExtraction fe = extract_file_data("look at C:/does/not/exist_at_all.png now");
        check(!fe.ok, "extract_file_data: a referenced file that can't be read is reported, not silently dropped");
    }

    // ------------------------------------------------------------- disp_width

    check(disp_width("hello") == 5, "disp_width: plain ASCII is one column per byte");
    check(disp_width("") == 0, "disp_width: empty string");
    check(disp_width("\xe4\xbd\xa0") == 2, "disp_width: a CJK character (U+4F60) is two columns");
    check(disp_width("\xcc\x81") == 0, "disp_width: a combining accent (U+0301) is zero columns");

    // ---------------------------------------------- display_response bookkeeping

    {
        DisplayState st;
        display_response("go", true, st);
        check(st.line_length == 2, "display_response: line_length tracks printed width");
        check(st.word_buffer == "go", "display_response: an in-progress word accumulates in the buffer");
        display_response(" ", true, st);
        check(st.word_buffer.empty(), "display_response: a space flushes the word buffer");
        display_response("x\n", true, st);
        check(st.line_length == 0, "display_response: a newline resets line_length");
    }

    // -------------------------------------------------------- RunOptions

    {
        RunOptions o;
        o.model = "m";
        o.messages.push_back(Message{{"role", "user"}, {"content", "hi"}});
        o.options["seed"] = 1;
        RunOptions c = o.copy();
        c.messages.push_back(Message{{"role", "user"}, {"content", "second"}});
        c.options["seed"] = 2;
        check(o.messages.size() == 1, "RunOptions::copy: mutating the copy's messages leaves the original alone");
        check(o.options["seed"] == 1, "RunOptions::copy: mutating the copy's options leaves the original alone");
    }
    {
        RunOptions o;
        o.model      = "m";
        const json b = o.body(json::object());
        check(b["model"] == "m" && !b.contains("options") && !b.contains("format") && !b.contains("think") &&
                  !b.contains("keep_alive"),
              "RunOptions::body: unset options/format/think/keep_alive are all left off the wire");
        o.options["seed"] = 1;
        o.format          = "json";
        o.think           = true;
        o.keep_alive      = "10m";
        const json b2     = o.body({{"stream", true}});
        check(b2["options"]["seed"] == 1 && b2["format"] == "json" && b2["think"] == true &&
                  b2["keep_alive"] == "10m" && b2["stream"] == true,
              "RunOptions::body: set fields and the extra overlay all land on the wire");
    }

    // ------------------------------------------------------ has_cap / infer_thinking

    {
        const json info = {{"capabilities", json::array({"thinking", "vision"})}};
        check(has_cap(info, "thinking"), "has_cap: finds a listed capability");
        check(!has_cap(info, "embedding"), "has_cap: an absent capability is false");

        RunOptions o;
        infer_thinking(info, o, false);
        check(o.think == true, "infer_thinking: defaults on for a thinking-capable model");

        RunOptions o2;
        infer_thinking(json::object(), o2, false);
        check(o2.think.is_null(), "infer_thinking: stays off for a model without the capability");

        RunOptions o3;
        o3.think = false;
        infer_thinking(info, o3, true);
        check(o3.think == false, "infer_thinking: an explicit flag is never overridden by the model's capabilities");
    }

    // ----------------------------------------------------- RAII: InterruptGuard

    {
        InterruptGuard g1;
        check(g1.is_installed(), "InterruptGuard: construction acquires the console ctrl handler");
        check(!g1.interrupted(), "InterruptGuard: starts uninterrupted");
        console_ctrl_handler(CTRL_C_EVENT);
        check(g1.interrupted(), "InterruptGuard: a Ctrl+C event sets the flag");
    } // g1's destructor releases the handler here
    {
        InterruptGuard g2;
        check(!g2.interrupted(),
              "InterruptGuard: a freshly-acquired guard starts clean (the earlier guard's release really ran)");
    }

    // ----------------------------------------------------------- spinner

    {
        ProgSpinner       s("");
        const std::string first = s.str();
        check(first == "\xe2\xa0\x8b ", "ProgSpinner: an empty message draws just the glyph and a space");
        std::this_thread::sleep_for(std::chrono::milliseconds(120));
        check(s.str() != first, "ProgSpinner: the glyph advances every 100 ms");
        s.stop();
        check(s.str().empty(), "ProgSpinner: stopped with an empty message, it draws nothing");
    }
    {
        WaitSpinner sp;
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
        sp.stop_and_clear();
        sp.stop_and_clear();
        check(true, "WaitSpinner: stop_and_clear twice returns instead of deadlocking");
    }

    std::printf("\n%s (%d failure%s)\n", failures == 0 ? "all passed" : "FAILURES", failures, failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
