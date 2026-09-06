package main

import (
	"encoding/json"
	"testing"
)

// The matching rules, against the repository names and metadata the hub really
// returns. Picking a drafter trained on a different fine-tune is the failure
// that matters: it loads, drafts badly, and leaves the model slower than no
// drafter at all, so most of these cases are rejections.

func TestKindOf(t *testing.T) {
	cases := map[string]string{
		"RedHatAI/gemma-4-26B-A4B-it-speculator.eagle3":      "eagle3",
		"williamliao/Qwen3.6-35B-A3B-DSPARK-GGUF":            "dspark",
		"undeadindustries/gemma-4-26b-a4b-it-dflash-drafter": "dflash",
		"someone/Qwen3-4B-draft-GGUF":                        "draft",
		"unsloth/gemma-4-26B-A4B-it-GGUF":                    "",
		"google/gemma-4-26B-A4B-it":                          "",
	}
	for repo, want := range cases {
		got := ""
		if k := kindOf(repo); k != nil {
			got = k.name
		}
		if got != want {
			t.Errorf("kindOf(%q) = %q, want %q", repo, got, want)
		}
	}
}

func TestKindRanking(t *testing.T) {
	if kindOf("x-eagle3").rank <= kindOf("x-dspark").rank {
		t.Error("eagle3 should outrank dspark")
	}
	if kindOf("x-dspark").rank <= kindOf("x-draft").rank {
		t.Error("dspark should outrank a plain draft model")
	}
}

func TestNormalise(t *testing.T) {
	same := [][2]string{
		{"Qwen3.6-35B-A3B", "qwen3_6_35b_a3b"},
		{"gemma-4-26B-A4B-it", "Gemma.4.26b.A4B.IT"},
	}
	for _, p := range same {
		if normalise(p[0]) != normalise(p[1]) {
			t.Errorf("%q and %q should normalise the same: %q vs %q",
				p[0], p[1], normalise(p[0]), normalise(p[1]))
		}
	}
	if normalise("Qwen3.6-35B-A3B") == normalise("Qwen3.8-27B") {
		t.Error("different models must not normalise the same")
	}
}

// foreignBase is what keeps a drafter for an abliterated or captioning
// fine-tune away from the plain model.
func TestForeignBase(t *testing.T) {
	want := normalise("gemma-4-26B-A4B-it")
	reject := [][]string{
		{"huihui-ai/Huihui-gemma-4-26B-A4B-it-abliterated", "google/gemma-4-26B-A4B-it"},
		{"someone/gemma-4-26B-A4B-it-Caption-v2"},
		{"x/gemma-4-26B-A4B-it-Uncensored-Merge"},
	}
	for _, bases := range reject {
		if foreignBase(bases, want) == "" {
			t.Errorf("should have rejected %v", bases)
		}
	}
	accept := [][]string{
		{"google/gemma-4-26B-A4B-it"},
		{"RedHatAI/gemma-4-26B-A4B-it-speculator.eagle3", "unsloth/gemma-4-26B-A4B-it-GGUF"},
	}
	for _, bases := range accept {
		if b := foreignBase(bases, want); b != "" {
			t.Errorf("should have accepted %v, rejected because of %q", bases, b)
		}
	}
}

func TestPreferQuant(t *testing.T) {
	if !preferQuant("draft-Q4_K_M.gguf", "draft-F16.gguf") {
		t.Error("Q4_K_M should be preferred over F16")
	}
	if preferQuant("draft-F16.gguf", "draft-Q4_K_M.gguf") {
		t.Error("preference must not be symmetric")
	}
	if !preferQuant("draft-IQ4_XS.gguf", "draft-Q8_0.gguf") {
		t.Error("IQ4_XS should be preferred over Q8_0")
	}
}

// considerRepo decides on one repository. These are the real shapes the hub
// returns, so a change in the rules shows up here rather than at run time.
func TestConsiderRepo(t *testing.T) {
	mk := func(id string, tags []string, base any) hubModel {
		var h hubModel
		h.ID, h.Tags = id, tags
		h.CardData.BaseModel = base
		return h
	}
	want := normalise("gemma-4-26B-A4B-it")
	quiet := func(string, ...any) {}

	// no drafter word anywhere: not a candidate at all
	if _, ok := considerRepo(mk("unsloth/gemma-4-26B-A4B-it-GGUF",
		[]string{"gguf"}, "google/gemma-4-26B-A4B-it"), want, "gemma-4-26B-A4B-it", quiet); ok {
		t.Error("a plain GGUF repack is not a drafter")
	}
	// right words, wrong target
	if _, ok := considerRepo(mk("someone/Qwen3.6-35B-A3B-eagle3",
		[]string{"eagle3"}, "Qwen/Qwen3.6-35B-A3B"), want, "gemma-4-26B-A4B-it", quiet); ok {
		t.Error("a drafter for another model must be rejected")
	}
	// right words, a derived fine-tune
	if _, ok := considerRepo(mk("coolthor/Huihui-gemma-4-26B-A4B-it-abliterated-eagle3-draft",
		[]string{"eagle3"},
		[]any{"huihui-ai/Huihui-gemma-4-26B-A4B-it-abliterated", "google/gemma-4-26B-A4B-it"}),
		want, "gemma-4-26B-A4B-it", quiet); ok {
		t.Error("a drafter for an abliterated fine-tune must be rejected")
	}
}

func TestHubModelBases(t *testing.T) {
	var one, many, none hubModel
	if err := json.Unmarshal([]byte(`{"cardData":{"base_model":"a/b"}}`), &one); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal([]byte(`{"cardData":{"base_model":["a/b","c/d"]}}`), &many); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal([]byte(`{"cardData":{}}`), &none); err != nil {
		t.Fatal(err)
	}
	if got := one.bases(); len(got) != 1 || got[0] != "a/b" {
		t.Errorf("a single base_model should read back as one entry, got %v", got)
	}
	if got := many.bases(); len(got) != 2 {
		t.Errorf("a list of base models should read back whole, got %v", got)
	}
	if got := none.bases(); got != nil {
		t.Errorf("a missing base_model should be empty, got %v", got)
	}
}

// A repack of a fine-tune's drafter must be rejected even though its own base
// is a drafter: the fine-tune's name is still in the lineage. This was found
// by searching the live hub, where a captioning fine-tune's EAGLE-3 drafter
// ranked first for the plain model.
func TestRejectsRepackOfAFineTuneDrafter(t *testing.T) {
	want := normalise("Qwen3.6-35B-A3B")
	bases := []string{"PatchyTisa/Qwen3.6-35B-A3B-Caption-Eagle3DraftModel"}
	if got := foreignBase(bases, want); got == "" {
		t.Error("a repack of a captioning fine-tune's drafter must be rejected")
	}
	// the repository name alone is enough evidence
	if got := foreignBase([]string{"EntityDeletr/Qwen3.6-35B-A3B-Caption-Eagle3DraftModel-GGUF"},
		want); got == "" {
		t.Error("the repository name should be read for fine-tune markers too")
	}
	// a plain repack of the plain drafter still passes
	ok := []string{"RedHatAI/Qwen3.6-35B-A3B-speculator.dspark", "Qwen/Qwen3.6-35B-A3B"}
	if got := foreignBase(ok, want); got != "" {
		t.Errorf("a plain drafter repack must be accepted, rejected for %q", got)
	}
}

// Repacks for other runtimes match on name but cannot load in llama.cpp.
func TestRejectsOtherRuntimes(t *testing.T) {
	for _, repo := range []string{
		"funnygeeker/Qwen3.6-35B-A3B-DFlash-MLX-6bit",
		"UnstableLlama/Qwen3.6-35B-A3B-DFlash-exl3-2.50bpw",
		"circulus/Qwen3.6-35B-A3B-eagle3-ov-int4",
		"someone/model-AWQ",
	} {
		if otherRuntime(repo) == "" {
			t.Errorf("%s is for another runtime and should be rejected", repo)
		}
	}
	// A GGUF is loadable whatever the weights were quantised from, so the
	// source quantisation alone is not a reason to skip it. Whether it drafts
	// well is settled by loading it, not by reading its name.
	for _, repo := range []string{
		"Koopah/Qwen3.6-35B-A3B-NVFP4-DSPARK-v2-GGUF",
		"williamliao/Qwen3.6-35B-A3B-DSPARK-GGUF",
		"adriabama06/Qwen3.6-35B-A3B-speculator.dspark-GGUF",
	} {
		if r := otherRuntime(repo); r != "" {
			t.Errorf("%s is a GGUF and should be a candidate, rejected as %q", repo, r)
		}
	}
}

// A repository that declares no base model is judged on its name alone, and a
// name carrying its own project identity is not a drafter for the plain model.
func TestNameOnlyEvidence(t *testing.T) {
	want := normalise("Qwen3.6-35B-A3B")
	if foreignBase([]string{"Dogacel/specdrift-qwen3.6-35b-a3b-eagle3"}, want) == "" {
		t.Error("specdrift is a project of its own and should be rejected")
	}
	if got := foreignBase([]string{"williamliao/Qwen3.6-35B-A3B-DSPARK-GGUF"}, want); got != "" {
		t.Errorf("a plain repack should pass on its name, rejected as %q", got)
	}
}
