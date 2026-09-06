package main

import (
	"bytes"
	"io"
	"os"
	"strings"
	"testing"
)

// The pairing rules are llama.cpp's own arithmetic, so they can be checked
// exactly. These are the numbers real files carry: a Qwen3.6 target with a
// 248320-token vocabulary and 41 blocks of width 2048, an EAGLE-3 drafter that
// reads 3 layers through a 6144-wide encoder, and a DFlash drafter that reads
// 8 through a 16384-wide one.

var (
	qwenTarget  = ggufSpec{Arch: "qwen35moe", Embed: 2048, Blocks: 41, Vocab: 248320}
	gemmaTarget = ggufSpec{Arch: "gemma4", Embed: 2816, Blocks: 30, Vocab: 262144}
	qwenEagle   = ggufSpec{Arch: "eagle3", Embed: 2048, Blocks: 1, Vocab: 248320,
		Layers: []int64{2, 20, 37}, Enc: 6144}
	qwenDflash = ggufSpec{Arch: "dflash", Embed: 2048, Blocks: 6, Vocab: 248320,
		Layers: []int64{2, 7, 12, 17, 23, 28, 33, 38}, Enc: 16384}
)

func TestPairsAcceptsRealDrafters(t *testing.T) {
	if err := pairs(qwenTarget, qwenEagle); err != nil {
		t.Errorf("the EAGLE-3 drafter should pair with its target: %v", err)
	}
	if err := pairs(qwenTarget, qwenDflash); err != nil {
		t.Errorf("the DFlash drafter should pair with its target: %v", err)
	}
}

// The vocabulary is the first thing llama.cpp compares, and it is what keeps a
// Qwen drafter away from a Gemma model however well the names match.
func TestPairsRejectsAnotherFamily(t *testing.T) {
	err := pairs(gemmaTarget, qwenEagle)
	if err == nil {
		t.Fatal("a Qwen drafter must not pair with a Gemma model")
	}
	if !strings.Contains(err.Error(), "vocabulary") {
		t.Errorf("the vocabulary should be the reason, got %v", err)
	}
}

// Same family, different size: the vocabulary matches and the layer indices are
// plausible, so only the encoder width catches it. This is the case that would
// otherwise download and fail at load.
func TestPairsRejectsTheWrongSizeInTheSameFamily(t *testing.T) {
	small := qwenTarget
	small.Embed = 1024 // a smaller sibling with the same tokenizer
	err := pairs(small, qwenEagle)
	if err == nil {
		t.Fatal("an encoder built for a 2048-wide model must not pair with a 1024-wide one")
	}
	if !strings.Contains(err.Error(), "encoder") {
		t.Errorf("the encoder width should be the reason, got %v", err)
	}
}

func TestPairsRejectsUnreachableLayers(t *testing.T) {
	shallow := qwenTarget
	shallow.Blocks = 20 // the drafter reads layer 37
	if err := pairs(shallow, qwenEagle); err == nil {
		t.Error("a drafter reading layer 37 must not pair with a 20-layer model")
	}
}

func TestPairsRejectsAMalformedEagle(t *testing.T) {
	twoLayers := qwenEagle
	twoLayers.Layers = []int64{2, 20}
	twoLayers.Enc = 4096
	if err := pairs(qwenTarget, twoLayers); err == nil {
		t.Error("EAGLE-3 reads exactly three layers; two is not an EAGLE-3")
	}
	none := qwenEagle
	none.Layers = nil
	if err := pairs(qwenTarget, none); err == nil {
		t.Error("a hidden-state drafter naming no layers must be rejected")
	}
}

// A plain small model used as a draft-simple drafter reads no hidden states, so
// only the vocabulary applies to it.
func TestPairsAllowsAPlainDrafter(t *testing.T) {
	plain := ggufSpec{Arch: "qwen3", Embed: 1024, Blocks: 28, Vocab: 248320}
	if err := pairs(qwenTarget, plain); err != nil {
		t.Errorf("a plain drafter with the right vocabulary should pair: %v", err)
	}
	plain.Vocab = 32000
	if err := pairs(qwenTarget, plain); err == nil {
		t.Error("a plain drafter with the wrong vocabulary should not pair")
	}
}

// llama.cpp tolerates a small difference, and so must this.
func TestPairsToleratesASlightlyDifferentVocabulary(t *testing.T) {
	trimmed := qwenEagle
	trimmed.Vocab = qwenTarget.Vocab - 64
	if err := pairs(qwenTarget, trimmed); err != nil {
		t.Errorf("64 tokens is within what llama.cpp accepts: %v", err)
	}
	trimmed.Vocab = qwenTarget.Vocab - 200
	if err := pairs(qwenTarget, trimmed); err == nil {
		t.Error("200 tokens is beyond what llama.cpp accepts")
	}
}

// What arrives over the wire is not always a model.
func TestScanGGUFRejectsRubbish(t *testing.T) {
	cases := map[string][]byte{
		"an HTML error page": []byte("<!DOCTYPE html><html><body>Not Found</body></html>"),
		"an empty file":      {},
		"a truncated header": append([]byte("GGUF"), 3, 0, 0),
	}
	for what, body := range cases {
		if _, err := scanGGUF(bytes.NewReader(body), true); err == nil {
			t.Errorf("%s should not read as a GGUF file", what)
		}
	}
	// the right magic, a version nothing here can load
	head := append([]byte("GGUF"), 9, 0, 0, 0)
	head = append(head, make([]byte, 16)...)
	if _, err := scanGGUF(bytes.NewReader(head), true); err == nil {
		t.Error("an unreadable GGUF version should be reported")
	}
}

// A file whose header is cut short must come back marked partial rather than
// with zeroes that would read as "no vocabulary, so nothing to disagree with".
func TestScanGGUFReportsAShortRead(t *testing.T) {
	target := os.Getenv("LLMASH_TEST_GGUF")
	if target == "" || !fileExists(target) {
		t.Skip("set LLMASH_TEST_GGUF to a local model to run this")
	}
	f, err := os.Open(target)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	s, err := scanGGUF(&limitedReader{f, 4096}, true)
	if err != nil {
		t.Fatalf("a short read should not be an error, got %v", err)
	}
	if !s.partial {
		t.Error("a 4 KB window cannot hold a whole header; it should read as partial")
	}
}

type limitedReader struct {
	r io.Reader
	n int64
}

func (l *limitedReader) Read(p []byte) (int, error) {
	if l.n <= 0 {
		return 0, io.EOF
	}
	if int64(len(p)) > l.n {
		p = p[:l.n]
	}
	n, err := l.r.Read(p)
	l.n -= int64(n)
	return n, err
}

// Against a real model on this machine, when one is named.
func TestScanGGUFLocal(t *testing.T) {
	path := os.Getenv("LLMASH_TEST_GGUF")
	if path == "" || !fileExists(path) {
		t.Skip("set LLMASH_TEST_GGUF to a local model to run this")
	}
	s, err := specOfFile(path)
	if err != nil {
		t.Fatalf("%s: %v", path, err)
	}
	if s.partial {
		t.Error("a local file should always read to the end of its header")
	}
	if s.Arch == "" || s.Vocab == 0 || s.Embed == 0 {
		t.Errorf("%s read as arch=%q vocab=%d embed=%d", path, s.Arch, s.Vocab, s.Embed)
	}
	t.Logf("%s: %s, %d blocks of %d, %d tokens, %d tensors",
		path, s.Arch, s.Blocks, s.Embed, s.Vocab, s.Tensors)
}
