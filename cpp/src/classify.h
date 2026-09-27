#pragma once

// Classifiers: models that answer typed questions about a text from one forward pass and generate nothing.
// The hub tags them text-classification, and what sits beside the model says how its answers are read:
//
//   head     a decision head over an encoder's per-token states, in <stem>.classifier.gguf
//   verdict  a decoder read at one slot per option, yes against no
//   letters  a decoder read at an answer slot, one letter per option
//   labels   a scorer inside the GGUF (a cls head), read through the runtime's rerank route
//
// The recipe is <stem>.classifier.json, written at pull from what the repository carries, and the request and
// answer are the shape every decision model speaks:
//
//   {"state": text | object, "questions": {id: {"type": choice|score|noul, "instructions": ..., "criteria": ...}}}
//   {"answers": {id: {"type", "choice" | "score" | "noul", "probabilities", "confidence"}}, "usage": {...}}

#include "config.h"
#include "gguf.h"
#include "manager.h"
#include "registry.h"

#include <nlohmann/json.hpp>

#include <functional>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace httplib {
struct Request;
struct Response;
}  // namespace httplib

namespace llmash {

// ------------------------------------------------------------------ what a model is

// "" for a model that is not one.
std::string classifier_kind(const std::string & gguf, const GGUFInfo & g);
std::string classifier_recipe_path(const std::string & gguf);  // <stem>.classifier.json, when there
std::string classifier_head_path(const std::string & gguf);    // <stem>.classifier.gguf, when there
// The runtime flags a kind needs: an encoder serves its states, a scorer its rank.
std::vector<std::string> classifier_launch_flags(const Model & m);

// A model pulled before its kind was known: its source is found (the record beside it, the header, or the file
// on the hub) and looked up once, and a classifier gets its pieces. True when the registry has to be re-read.
bool ensure_classifier(const Model & m, const Config & cfg, Registry & reg);

// At pull: the pipeline tag recorded, and the head or readout config taken from the repository that
// published the GGUF (`repo`) or the model it was converted from (`from`).
void install_classifier(const std::string & repo, const std::string & from, const std::string & gguf,
                        const std::string & pipeline, const std::function<void(const nlohmann::json &)> & emit);

// ------------------------------------------------------------------ recipes

struct Recipe {
    std::string kind;      // head, verdict, letters, labels
    std::string source;    // the repository
    std::string pipeline;  // the hub's tag
    // letters: the prompt, one option line and the token read, with {state} {question} {options} {letter} {option}
    std::string tmpl, option, letter;
    std::string option_style = "plain";  // how options are spelt: plain (name: description) or typed (level i: ..., true: ...)
    bool        isolated_levels = false; // a score question judged one level at a time, yes or no
    std::map<std::string, double> temperature;  // by question type; "" for every type
    // verdict: the three tokens and the calibration
    std::string yes = " yes", no = " no", slot = " ->";
    double      T = 1.0, T_lo = 0.3, T_hi = 5.0;
    std::map<std::string, double> groups;  // "family|type|bucket" -> T
};
Recipe letters_recipe(const std::string & flavour);  // "decider" or "jev": the two published layouts
Recipe read_recipe(const std::string & path, std::string & err);
nlohmann::json recipe_json(const Recipe & r);

// ------------------------------------------------------------------ questions

struct Question {
    std::string                                      type;      // choice, score or noul
    std::string                                      text;      // the instructions
    std::vector<std::pair<std::string, std::string>> criteria;  // choice: name, description; score: level, ""; noul: false/true, description
};
using Questions = std::vector<std::pair<std::string, Question>>;

bool                     parse_question(const nlohmann::json & q, Question & out, std::string & err);
std::vector<std::string> render_options(const Question & q, const std::string & style);
std::vector<std::string> option_names(const Question & q);  // what the answer reports

// Questions as lines, for a chat message or the console:
//   Which team? [billing, technical, sales]         a choice; "name: description" describes one
//   score: How urgent? [not at all, slightly, very]  ordered levels, lowest first
//   Is it spam?                                     yes or no
// A block of such lines at the start or the end of the text; the rest is the state.
struct ParsedText {
    std::string state;
    Questions   questions;
};
ParsedText parse_text(const std::string & text);
nlohmann::ordered_json questions_json(const Questions & qs);  // the request's shape, ids kept in order
bool       question_line(const std::string & line, Question & out);

// ------------------------------------------------------------------ answers

nlohmann::json answer_json(const Question & q, const std::vector<float> & probs);
std::string    answers_text(const Questions & qs, const nlohmann::json & answers);
std::string    py_dumps(const nlohmann::ordered_json & v);  // Python's json.dumps, the spelling the models were trained on

// ------------------------------------------------------------------ readouts, the pure parts

// letters: the prompt, and the token piece that stands for each option
std::string letters_prompt(const Recipe & r, const std::string & state, const Question & q, std::vector<std::string> & pieces);
// verdict: one prompt per option, each ending at that option's slot
std::vector<std::string> verdict_prompts(const Recipe & r, const std::string & state, const Question & q);
// The log probability of each piece in a runtime completion's top list; -1e9 when it is not there.
std::vector<double> piece_logprobs(const nlohmann::json & completion, const std::vector<std::string> & pieces);
std::vector<float>  softmax_scaled(const std::vector<double> & scores, double T);

// head: the sequence over the encoder
struct HeadSeq {
    std::vector<int> ids;
    std::vector<int> markers;  // where each option's mask token sits
    int              qtype = 0;
};
HeadSeq head_sequence(const std::function<std::vector<int>(const std::string &)> & tok, const std::string & state,
                      const Question & q, int max_len, int head_max_len, int cls, int sep, int mask);
std::string temp_bucket(const std::string & type, int k);

// ------------------------------------------------------------------ the run

// {"model", "state", "questions"} -> {"model", "answers", "usage"}, or {"error"} with the status set.
nlohmann::json classify(const nlohmann::ordered_json & request, Config & cfg, Manager & mgr, Registry & reg, int & status);
// POST /api/classify and /v1/systemone
void handle_classify(const httplib::Request & req, httplib::Response & res, Config & cfg, Manager & mgr, Registry & reg);
// A chat request to a classifier: the last user message is the state, the questions are lines in it or in the
// system message, and the answer comes back as text. `shape` is chat, generate or openai. False when the model
// is not a classifier, and the caller carries on.
bool classify_chat(const nlohmann::json & body, httplib::Response & res, Config & cfg, Manager & mgr, Registry & reg,
                   const std::string & shape);

}  // namespace llmash
