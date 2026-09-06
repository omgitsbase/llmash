package main

import (
	"encoding/binary"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// Which drafter a model launches with. The measured order is what this pins:
// a model's own head beats a downloaded one, because the built-in head is
// trained on the same weights and accepts roughly twice as many of its own
// guesses.

// writeStubGGUF makes the smallest file hasMTP and the launcher will read: a
// header and one tensor name. mtp says whether that name marks an MTP head.
func writeStubGGUF(t *testing.T, path string, mtp bool) {
	t.Helper()
	name := "blk.0.ffn_down.weight"
	if mtp {
		name = "blk.0.nextn.embed_tokens.weight"
	}
	var b []byte
	u32 := func(v uint32) { b = binary.LittleEndian.AppendUint32(b, v) }
	u64 := func(v uint64) { b = binary.LittleEndian.AppendUint64(b, v) }
	b = append(b, "GGUF"...)
	u32(3)                 // version
	u64(1)                 // one tensor
	u64(0)                 // no key-values
	u64(uint64(len(name))) // the tensor's name
	b = append(b, name...)
	u32(1) // one dimension
	u64(8)
	u32(0) // ggml type
	u64(0) // offset
	if err := os.WriteFile(path, b, 0o644); err != nil {
		t.Fatal(err)
	}
	delete(mtpCache, path)
}

func specTypeIn(args []string) string {
	for i, a := range args {
		if a == "--spec-type" && i+1 < len(args) {
			return args[i+1]
		}
	}
	return ""
}

func TestDrafterPrecedence(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("LLMASH_TUNE_OFF", "1")
	specFallback = "ngram-mod"

	withMTP := filepath.Join(dir, "with-mtp.gguf")
	noMTP := filepath.Join(dir, "no-mtp.gguf")
	writeStubGGUF(t, withMTP, true)
	writeStubGGUF(t, noMTP, false)
	eagle := filepath.Join(dir, "with-mtp.eagle3.gguf")
	if err := os.WriteFile(eagle, []byte("GGUF"), 0o644); err != nil {
		t.Fatal(err)
	}
	plainEagle := filepath.Join(dir, "no-mtp.eagle3.gguf")
	if err := os.WriteFile(plainEagle, []byte("GGUF"), 0o644); err != nil {
		t.Fatal(err)
	}

	spec := func(gguf, eagle3 string) string {
		in := &Instance{Model: &Model{Name: "test", GGUF: gguf, Eagle3: eagle3}, Ctx: 4096}
		return specTypeIn(in.args())
	}

	if got := spec(withMTP, ""); got != "draft-mtp" {
		t.Errorf("a model with its own head should use it, got %q", got)
	}
	if got := spec(withMTP, eagle); got != "draft-mtp" {
		t.Errorf("a downloaded head must not displace the model's own, got %q", got)
	}
	if got := spec(noMTP, plainEagle); got != "draft-eagle3" {
		t.Errorf("a model with no head of its own should use the downloaded one, got %q", got)
	}
	if got := spec(noMTP, ""); got != "ngram-mod" {
		t.Errorf("a model with no drafter at all should fall back, got %q", got)
	}

	// asked for, in writing, by pulldraft --force
	marker := preferMarker(withMTP)
	if err := os.WriteFile(marker, nil, 0o644); err != nil {
		t.Fatal(err)
	}
	if got := spec(withMTP, eagle); got != "draft-eagle3" {
		t.Errorf("a marked model should use the drafter that was forced on it, got %q", got)
	}
	// the marker outliving the drafter must not cost the model its own head
	if got := spec(withMTP, ""); got != "draft-mtp" {
		t.Errorf("a marker with no drafter installed should fall back to the head, got %q", got)
	}
	os.Remove(marker)
	if got := spec(withMTP, eagle); got != "draft-mtp" {
		t.Errorf("deleting the marker should undo the choice, got %q", got)
	}
}

// The marker belongs to one model, not to every model in the folder.
func TestPreferMarkerIsPerModel(t *testing.T) {
	dir := t.TempDir()
	a := filepath.Join(dir, "one.gguf")
	b := filepath.Join(dir, "two.gguf")
	if err := os.WriteFile(preferMarker(a), nil, 0o644); err != nil {
		t.Fatal(err)
	}
	if !preferDraft(a) {
		t.Error("the model it was written for should read it")
	}
	if preferDraft(b) {
		t.Error("another model in the same folder must not")
	}
	if !strings.HasSuffix(preferMarker(a), ".prefer-draft") {
		t.Errorf("the marker should say what it is, got %s", preferMarker(a))
	}
}
