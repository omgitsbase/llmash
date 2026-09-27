// The classify module's pure parts: questions as lines and as JSON, the options as the models spell them, the
// prompts of the letter and verdict readouts, the head's sequence, the recipe on disk and the answers as text.
#include "classify.h"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>

using namespace llmash;
using json  = nlohmann::json;
using ojson = nlohmann::ordered_json;

namespace {

int failures = 0;

void check(bool ok, const char * what) {
    if (!ok) {
        failures++;
        std::printf("FAIL %s\n", what);
    }
}

}  // namespace

int main() {
    // ---- questions as lines
    Question q;
    check(question_line("Which team? [billing, technical, sales]", q) && q.type == "choice" && q.text == "Which team?" &&
              q.criteria.size() == 3 && q.criteria[1].first == "technical",
          "a choice line");
    check(question_line("score: How urgent? [not at all | slightly | very]", q) && q.type == "score" && q.criteria.size() == 3 &&
              q.criteria[0].first == "not at all",
          "a score line, | between options");
    check(question_line("Is it spam?", q) && q.type == "noul" && q.criteria.size() == 2, "a bare question is yes or no");
    check(question_line("Which? [billing: payments and refunds | sales: new purchases]", q) && q.criteria[0].first == "billing" &&
              q.criteria[0].second == "payments and refunds",
          "name: description");
    check(!question_line("The customer wrote in on Monday.", q), "a statement is not a question");
    check(!question_line("Pick one [only]", q), "one option is not a choice");
    check(!question_line("/questions", q), "a command is not a question");

    ParsedText pt = parse_text("I was charged twice this month.\n\nWhich team? [billing, technical, sales]\nIs the customer angry?\n");
    check(pt.state == "I was charged twice this month." && pt.questions.size() == 2 && pt.questions[0].first == "q1" &&
              pt.questions[1].second.type == "noul",
          "questions after the text");
    pt = parse_text("Which team? [a, b]\n\nThe charge was doubled.\nI want it back.");
    check(pt.state == "The charge was doubled.\nI want it back." && pt.questions.size() == 1, "questions before the text");
    pt = parse_text("Just a text, with a line that asks?\n\nIs it spam?");
    check(pt.state == "Just a text, with a line that asks?" && pt.questions.size() == 1, "a blank line ends the block");
    pt = parse_text("Just a text.");
    check(pt.questions.empty() && pt.state == "Just a text.", "no questions");

    // ---- questions as JSON
    std::string err;
    check(parse_question(json::parse(R"({"type":"choice","instructions":"Which team?","criteria":{"billing":"payments","sales":null}})"), q, err) &&
              q.criteria.size() == 2 && q.criteria[0].second == "payments" && q.criteria[1].second.empty(),
          "a choice from JSON");
    check(parse_question(json::parse(R"({"type":"score","instructions":"How urgent?","criteria":["low","mid","high"]})"), q, err) &&
              q.type == "score" && q.criteria.size() == 3 && q.criteria[2].first == "high",
          "a score from JSON");
    check(parse_question(json::parse(R"({"type":"noul","criteria":{"true":"it is spam","false":"it is not"}})"), q, err) &&
              q.text == "Which answer fits the context?" && q.criteria[1].second == "it is spam",
          "a yes/no without instructions takes the fixed question");
    check(!parse_question(json::parse(R"({"type":"choice","instructions":"x","criteria":["only"]})"), q, err), "one option is refused");
    check(!parse_question(json::parse(R"({"type":"rank","instructions":"x"})"), q, err), "an unknown type is refused");

    Question sc;
    sc.type     = "score";
    sc.text     = "How urgent?";
    sc.criteria = {{"low", ""}, {"high", ""}};
    check(render_options(sc, "typed")[0] == "level 0: low" && render_options(sc, "plain")[1] == "1: high", "score options in both spellings");
    Question no;
    no.type     = "noul";
    no.text     = "Is it spam?";
    no.criteria = {{"false", ""}, {"true", ""}};
    check(render_options(no, "typed")[0] == "false: no, the statement does not hold" && render_options(no, "plain")[1] == "yes",
          "yes/no options in both spellings");
    check(option_names(sc) == std::vector<std::string>{"0", "1"} && option_names(no)[1] == "true", "what the answers report");

    // ---- the letter readout
    Question ch;
    ch.type     = "choice";
    ch.text     = "Which team?";
    ch.criteria = {{"billing", ""}, {"sales", "new purchases"}};
    std::vector<std::string> pieces;
    std::string              p = letters_prompt(letters_recipe("decider"), "The invoice is wrong.", ch, pieces);
    check(p == "Context:\nThe invoice is wrong.\n\nQuestion: Which team?\nOptions:\n(A) billing\n(B) sales: new purchases\n\nAnswer: (" &&
              pieces == std::vector<std::string>{"A", "B"},
          "the context-first layout, read at the bare letter");
    p = letters_prompt(letters_recipe("jev"), "s", ch, pieces);
    check(p.find("[Options]\nA. billing\nB. sales: new purchases\n\nAnswer:") != std::string::npos && pieces[0] == " A",
          "the decision-function layout, read at the spaced letter");

    // ---- the verdict readout
    Recipe v;
    v.kind                                = "verdict";
    const std::vector<std::string> slots  = verdict_prompts(v, "The film was excellent.", ch);
    check(slots.size() == 2 &&
              slots[0] == "State:\nThe film was excellent.\n\nQuestion [choice]: Which team?\nOptions:\n- billing\n- sales: new purchases\n"
                          "Judge each option:\nbilling ->" &&
              slots[1] == slots[0] + "\nsales: new purchases ->",
          "one prompt per option, each ending at its slot");

    const json comp = json::parse(R"({"completion_probabilities":[{"token":"A","logprob":-0.1,
        "top_logprobs":[{"token":"A","logprob":-0.2},{"token":"B","logprob":-1.8},{"token":"\n","logprob":-5.0}]}]})");
    const std::vector<double> lp = piece_logprobs(comp, {"A", "B", "C"});
    check(lp[0] == -0.2 && lp[1] == -1.8 && lp[2] <= -1e8, "letters read out of the top list, a missing one at nothing");
    const std::vector<float> pr = softmax_scaled({lp[0], lp[1]}, 1.0);
    check(std::fabs(pr[0] - 0.832) < 0.01 && std::fabs(pr[0] + pr[1] - 1.0) < 1e-5, "renormalised over the letters");
    const std::vector<float> flat = softmax_scaled({lp[0], lp[1]}, 1000.0);
    check(std::fabs(flat[0] - 0.5) < 0.01, "a high temperature flattens");

    // ---- the head's sequence: [CLS] question [SEP] ([MASK] option)* [SEP] state [SEP]
    const auto words = [](const std::string & text) {
        std::vector<int> ids;
        std::string      cur;
        for (const char c : text + " ") {
            if (c == ' ') {
                if (!cur.empty()) {
                    ids.push_back(1000 + static_cast<int>(cur.size()));
                }
                cur.clear();
            } else {
                cur += c;
            }
        }
        return ids;
    };
    const HeadSeq seq = head_sequence(words, "The invoice is wrong.", ch, 1024, 256, 1, 2, 3);
    // 1 | choice question: Which team? (4) | 2 | 3 billing (1) | 3 sales: new purchases (3) | 2 | 4 words | 2
    check(seq.ids.size() == 1 + 4 + 1 + 2 + 4 + 1 + 4 + 1 && seq.ids[0] == 1 && seq.ids[5] == 2 && seq.markers == std::vector<int>{6, 8} &&
              seq.ids[6] == 3 && seq.ids[8] == 3 && seq.ids[12] == 2 && seq.ids.back() == 2 && seq.qtype == 0,
          "the sequence around the markers");
    const HeadSeq tight = head_sequence(words, "one two three four five six seven eight nine ten", ch, 16, 256, 1, 2, 3);
    check(tight.ids.size() == 16 && tight.ids.back() == 2 && tight.markers.size() == 2, "a long state is cut to the length, the end kept");
    check(temp_bucket("choice", 3) == "choice:3-5" && temp_bucket("noul", 2) == "noul:2" && temp_bucket("choice", 40) == "choice:11+",
          "the calibration buckets");

    // ---- the recipe on disk
    const std::filesystem::path tmp = std::filesystem::temp_directory_path() / "llmash-classify-test.classifier.json";
    {
        Recipe d           = letters_recipe("decider");
        d.source           = "someone/decider-GGUF";
        d.pipeline         = "text-classification";
        d.temperature[""]  = 1.145;
        d.temperature["noul"] = 1.624;
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        out << recipe_json(d).dump(2);
    }
    const Recipe back = read_recipe(tmp.string(), err);
    check(err.empty() && back.kind == "letters" && back.tmpl == letters_recipe("decider").tmpl && back.isolated_levels &&
              back.source == "someone/decider-GGUF" && std::fabs(back.temperature.at("noul") - 1.624) < 1e-9 && std::fabs(back.T - 1.145) < 1e-9,
          "a letters recipe reads back");
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        out << R"({"kind":"verdict","source":"x","pipeline_tag":"text-classification","yes":" yes","no":" no","slot":" ->","temperature":0.88,"clamp":[0.3,5.0]})";
    }
    const Recipe vb = read_recipe(tmp.string(), err);
    check(err.empty() && vb.kind == "verdict" && std::fabs(vb.T - 0.88) < 1e-9 && vb.slot == " ->" && vb.T_hi == 5.0, "a verdict recipe reads back");
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        out << R"({"kind":"pointer"})";
    }
    read_recipe(tmp.string(), err);
    check(!err.empty(), "an unknown kind is refused");
    std::filesystem::remove(tmp);

    // ---- answers
    const json a = answer_json(ch, {0.32f, 0.68f});
    check(a["choice"] == "sales" && std::fabs(a["probabilities"]["billing"].get<double>() - 0.32) < 1e-6 && a["confidence"].get<double>() > 0.05,
          "a choice answer");
    const json s = answer_json(sc, {0.25f, 0.75f});
    check(std::fabs(s["score"].get<double>() - 0.75) < 1e-6 && s["legend"]["1"] == "high", "a score answer is the expected level");
    const json n = answer_json(no, {0.97f, 0.03f});
    check(std::fabs(n["noul"].get<double>() - 0.03) < 1e-6, "a yes/no answer is the probability of yes");

    const Questions qs = {{"q1", ch}, {"q2", no}, {"urgency", sc}};
    const json      ans = json::parse(R"({"q1":{"type":"choice","choice":"billing","probabilities":{"billing":0.68,"sales":0.32}},
        "q2":{"type":"noul","noul":0.03},"urgency":{"type":"score","score":0.75,"probabilities":{"0":0.25,"1":0.75}}})");
    check(answers_text(qs, ans) == "Which team?  billing 68%  ·  sales 32%\nIs it spam?  no 97%\nurgency      0.8 of 0-1  ·  low 25%  ·  high 75%\n",
          "the answers as lines");

    check(py_dumps(ojson::parse(R"({"b": 1, "a": [true, null, "x\ny"]})")) == "{\"b\": 1, \"a\": [true, null, \"x\\ny\"]}",
          "a state object is spelt the way Python spells it");
    const ojson qj = questions_json(qs);
    check(qj["q1"]["criteria"]["sales"] == "new purchases" && qj["q1"]["criteria"]["billing"].is_null() && qj["urgency"]["criteria"][1] == "high" &&
              qj["q2"]["type"] == "noul",
          "questions back to JSON");

    if (failures == 0) {
        std::printf("classify_test: all passed\n");
    }
    return failures == 0 ? 0 : 1;
}
