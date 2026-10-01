#pragma once

// Finding a drafter published for a model already on disk, proving from its
// GGUF header alone that it pairs with those exact weights, and installing
// it as a sidecar beside them.

#include "config.h"
#include "pull.h"
#include "registry.h"

#include <functional>
#include <istream>
#include <string>
#include <vector>

namespace llmash {

// A line of explanation for the user, already formatted.
using Say = std::function<void(const std::string &)>;

// ------------------------------------------------------------- candidates

struct DraftCand {
    std::string              repo;
    std::string              file;
    const DraftKind *        kind = nullptr; // one of pull.h's draft_kinds()
    int64_t                  size = 0;
    std::vector<std::string> bases;
    int                      score = 0;
    std::string              note;
    bool                     embedded = false; // an MTP head inside a full build: only the head is fetched
    std::string              via;             // the model it was trained on, when that is not this model
};

// What a file is, by its own header: the model these weights are (Qwen3.5-4B), and the model it was tuned from
// when the header names one (Qwen3.5-9B for a MiMo distill of it).
struct ModelIdent {
    std::string name;
    std::string tuned_from;
};
ModelIdent identify(const Model & m);

// Anything larger is a whole model that happens to mention a drafter.
constexpr int64_t DRAFT_SIZE_CEILING = 6ll << 30;

// Lower-cased, everything but [a-z0-9] dropped.
std::string normalise(const std::string & s);
// normalise() with the words a name adds around a model's (Instruct, it, chat, base, GGUF) taken out, so
// LLaMA3.1-Instruct-8B and Llama-3.1-8B-Instruct-GGUF both read as llama318b
std::string model_key(const std::string & s);

std::string model_stem(const Model & m);  // identify(m).name

bool hub_info_reason(const std::string & repo, HubModel & out, std::string & err);

// A base that is a derivative of the target (an abliterated fine-tune, say),
// or "" when every base named is the target itself.
std::string foreign_base(const std::vector<std::string> & bases, const std::string & want);

// Repacks for other runtimes match on name but will never load here; the
// runtime's own name, or "".
std::string other_runtime(const std::string & repo);

// Is `a` a more useful quantisation to take than `b`.
bool prefer_quant(const std::string & a, const std::string & b);

// A repo named as an assistant drafter (Gemma 4's MTP heads): <model>-assistant.
bool assistant_named(const std::string & repo);

// One search hit judged: false, with `say` told why, for anything that is not
// a drafter for this exact model.
bool consider_repo(const HubModel & hit, const std::string & want, const std::string & stem, const Say & say,
                   DraftCand & out);

// Every drafter published for this model, best first. `say` is called only
// when verbose, matching findDrafters' own flag.
std::vector<DraftCand> find_drafters(const Model & m, bool verbose, const Say & say = Say(), double budget_s = 0);

struct GGUFSpec {
    std::string          arch;
    int64_t              embed   = 0; // <arch>.embedding_length
    int64_t              blocks  = 0; // <arch>.block_count
    int64_t              vocab   = 0; // number of tokens in the vocabulary
    std::vector<int64_t> layers;      // <arch>.target_layers, what a drafter reads from
    int64_t              enc     = 0; // fc.weight's input width = layers.size() * target embed
    int64_t              nextn_layer = -1; // the block an MTP head sits at, when the file carries one
    int64_t              embed_out = 0; // <arch>.embedding_length_out: the width an assistant drafter hands back
    int64_t              tensors = 0;
    bool                 partial = false; // the read ended early: a zero field means "not reached"

    // A drafter that reads the target's hidden states, not just its tokens.
    bool hidden() const;
};

// "" on success. Tensor shapes sit after the vocabulary, so want_tensors
// costs a few megabytes.
std::string scan_gguf(std::istream & in, bool want_tensors, GGUFSpec & out);
std::string spec_of_file(const std::string & path, GGUFSpec & out);

// `unreadable` marks a problem reaching the hub as opposed to a verdict on
// the file, which is what draft_verify.go's errUnreadable sentinel carries.
std::string spec_of_url(const std::string & url, GGUFSpec & out, bool & unreadable);

// Why a drafter cannot serve a target, or "" if it can.
std::string pairs(const GGUFSpec & target, const GGUFSpec & draft);

// pairs() for a candidate not yet downloaded: its header is range-fetched.
// Reads the candidate's header; a file that turns out to hold an MTP head on its own is re-labelled as one.
std::string fits_target(const Model & m, DraftCand & c);

// ------------------------------------------------------------- installing

// Beside the weights when that folder can be written to, in the loose-GGUF
// folder otherwise.
std::string draft_path(const Model & m, const DraftKind & kind, const Config & cfg);
bool        writable(const std::string & dir);

// The sidecar `<stem><suffix>` beside the weights, or "".
std::string sidecar_path(const std::string & gguf, const char * suffix);
// Where that sidecar goes, whether or not it exists yet: the path sidecar_path looks for.
std::string sidecar_target(const std::string & gguf, const char * suffix);
// Where a model's ablation vector is written and looked for: beside the weights, or in llmash's own folder
// for a model kept in an Ollama store's content-addressed blobs.
std::string ablation_path(const std::string & gguf, const Config & cfg);
// A DSpark drafter beside the weights, matched by name with the
// quantisation ignored, or "".
std::string dspark_path(const std::string & gguf);
// A DFlash drafter beside the weights, as <stem>.dflash.gguf or trained for the base model the
// target's header names, or "".
std::string dflash_path(const std::string & gguf);
// <stem>.mtp.gguf, or the MTP drafter beside another build of the same model.
std::string mtp_path(const std::string & gguf);

// "an MTP head" / "a DSpark drafter" / "a draft model" / "an EAGLE-3
// drafter", or "" when none is installed.
std::string installed_drafter(const Model & m);

// installed_drafter, plus "an MTP head of its own" for weights that carry
// multi-token-prediction heads already.
std::string has_own_drafter(const Model & m);

// The self-speculation a model keeps when no drafter fits (LLMASH_SPEC_FALLBACK).
std::string spec_fallback();

std::string verify_draft(const Model & m, const std::string & path);

// Downloads, verifies and renames into place; the installed path, or "" with
// `err` set. An already-installed sidecar is returned untouched.
std::string install_draft(const Model & m, const DraftCand & c, const Config & cfg, const ProgressFn & progress,
                          std::string & err);


// A drafter named rather than searched for: REPO, REPO:FILE or REPO@QUANT, with or without hf.co/. Its best build
// or the file asked for, typed by its name or else by the architecture in its header.
bool drafter_from(const std::string & ref, DraftCand & out, std::string & err);

} // namespace llmash
