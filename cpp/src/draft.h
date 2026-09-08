#pragma once

// Port of cmd/llmash/draft.go, draft_verify.go and draft_install.go: finding
// a drafter published for a model already on disk, proving from its GGUF
// header alone that it pairs with those exact weights, and installing it as
// a sidecar beside them.

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
};

// Anything larger is a whole model that happens to mention a drafter.
constexpr int64_t DRAFT_SIZE_CEILING = 6ll << 30;

// Lower-cased, everything but [a-z0-9] dropped.
std::string normalise(const std::string & s);

// The name a drafter would have been published under: the model's own name,
// or its base repository's, with the size that tells one member of a family
// from another kept and the packaging/quantisation words dropped.
std::string model_stem(const Model & m);

// draft.go's hubGet reports why it failed, and consider_repo() prints that
// reason: "rate limited" reads very differently from "no such repository".
// pull.h's hub_info() answers only yes/no, so this carries the text.
bool hub_info_reason(const std::string & repo, HubModel & out, std::string & err);

// A base that is a derivative of the target (an abliterated fine-tune, say),
// or "" when every base named is the target itself.
std::string foreign_base(const std::vector<std::string> & bases, const std::string & want);

// Repacks for other runtimes match on name but will never load here; the
// runtime's own name, or "".
std::string other_runtime(const std::string & repo);

// Is `a` a more useful quantisation to take than `b`.
bool prefer_quant(const std::string & a, const std::string & b);

// One search hit judged: false, with `say` told why, for anything that is not
// a drafter for this exact model.
bool consider_repo(const HubModel & hit, const std::string & want, const std::string & stem, const Say & say,
                   DraftCand & out);

// Every drafter published for this model, best first. `say` is called only
// when verbose, matching findDrafters' own flag.
std::vector<DraftCand> find_drafters(const Model & m, bool verbose, const Say & say = Say());

// ----------------------------------------------------------- verification
//
// llama.cpp accepts a drafter only if the vocabularies match and its encoder
// is shaped for the target's hidden size. Both are in the GGUF header, so the
// answer costs a few kilobytes instead of a load.

struct GGUFSpec {
    std::string          arch;
    int64_t              embed   = 0; // <arch>.embedding_length
    int64_t              blocks  = 0; // <arch>.block_count
    int64_t              vocab   = 0; // number of tokens in the vocabulary
    std::vector<int64_t> layers;      // <arch>.target_layers, what a drafter reads from
    int64_t              enc     = 0; // fc.weight's input width = layers.size() * target embed
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
std::string fits_target(const Model & m, const DraftCand & c);

// ------------------------------------------------------------- installing

// Beside the weights when that folder can be written to, in the loose-GGUF
// folder otherwise.
std::string draft_path(const Model & m, const DraftKind & kind, const Config & cfg);
bool        writable(const std::string & dir);

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

// The first candidate whose header says it fits, checked before downloading.
bool pick_drafter(const Model & m, const std::vector<DraftCand> & cands, const Say & say, DraftCand & out);

} // namespace llmash
