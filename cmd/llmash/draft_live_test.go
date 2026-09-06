package main

import (
	"errors"
	"os"
	"strings"
	"testing"
)

// Against the real hub. Skipped unless LLMASH_LIVE=1, because it needs the
// network; it needs no account and no token, which is the point being tested.
func TestFindDraftersLive(t *testing.T) {
	if os.Getenv("LLMASH_LIVE") != "1" {
		t.Skip("set LLMASH_LIVE=1 to search Hugging Face")
	}
	cases := []struct {
		name     string
		stem     string
		wantKind string // "" means nothing published, which is a valid answer
	}{
		{"gemma4:26b", "gemma-4-26B-A4B-it", "eagle3"},
		{"nail", "Qwen3.6-35B-A3B", "dspark"},
		{"llama3.2", "Llama-3.2-3B-Instruct", ""},
	}
	for _, c := range cases {
		m := &Model{Name: c.stem}
		cands := findDrafters(m, testing.Verbose())
		t.Logf("%s: %d candidate(s)", c.stem, len(cands))
		for _, cand := range cands {
			t.Logf("    %-58s %s", cand.Repo, cand.Note)
			// whatever comes back must at least name the target
			if !strings.Contains(normalise(cand.Repo)+normalise(strings.Join(cand.Bases, " ")),
				normalise(c.stem)) {
				t.Errorf("%s: candidate %s does not reference the target anywhere",
					c.stem, cand.Repo)
			}
			if cand.File == "" || !strings.HasSuffix(strings.ToLower(cand.File), ".gguf") {
				t.Errorf("%s: candidate %s has no GGUF to download", c.stem, cand.Repo)
			}
		}
		if c.wantKind != "" && len(cands) > 0 && cands[0].Kind.name != c.wantKind {
			t.Logf("%s: best is %s, expected %s (not fatal, the hub changes)",
				c.stem, cands[0].Kind.name, c.wantKind)
		}
	}
}

// The hub is somebody else's server: every call has to fail softly.
func TestHubFailsSoftly(t *testing.T) {
	if got := hubSearch("::::nonsense::::", 5); len(got) != 0 {
		t.Errorf("a nonsense search should come back empty, got %d", len(got))
	}
	if _, err := hubInfo("definitely/not-a-real-repo-xyzzy"); err == nil {
		t.Error("a missing repository should report an error")
	}
	if got := hubFiles("definitely/not-a-real-repo-xyzzy"); got != nil {
		t.Errorf("a missing repository should list no files, got %v", got)
	}
	// an empty stem must not send a wildcard search to the hub
	if got := findDrafters(&Model{Name: ""}, false); got != nil {
		t.Errorf("a nameless model should find nothing, got %d", len(got))
	}
}

// The remote header read, against files that are really on the hub. The point
// being tested is that a candidate can be accepted or rejected before any of
// its weights are downloaded, with no account and no token.
func TestSpecOfURLLive(t *testing.T) {
	if os.Getenv("LLMASH_LIVE") != "1" {
		t.Skip("set LLMASH_LIVE=1 to read from Hugging Face")
	}
	cases := []struct{ repo, file, arch string }{
		{"Koopah/Qwen3.6-35B-A3B-NVFP4-DSPARK-v2-GGUF",
			"Qwen3.6-35B-A3B-NVFP4-DSPARK-v2-Q8_0.gguf", "dflash"},
		{"williamliao/gemma-4-26B-A4B-it-speculator.eagle3-F16-GGUF", "", "eagle3"},
	}
	for i := range cases {
		if cases[i].file != "" {
			continue
		}
		for _, f := range hubFiles(cases[i].repo) {
			if strings.HasSuffix(strings.ToLower(f.Path), ".gguf") {
				cases[i].file = f.Path
				break
			}
		}
	}
	for _, c := range cases {
		if c.file == "" {
			t.Errorf("%s: no GGUF listed", c.repo)
			continue
		}
		s, err := specOfURL(hubDownloadURL(c.repo, c.file))
		if err != nil {
			t.Errorf("%s: %v", c.repo, err)
			continue
		}
		t.Logf("%s: arch=%s embed=%d blocks=%d vocab=%d layers=%v enc=%d tensors=%d",
			c.repo, s.Arch, s.Embed, s.Blocks, s.Vocab, s.Layers, s.Enc, s.Tensors)
		if s.partial {
			t.Errorf("%s: the header did not fit the window", c.repo)
		}
		if c.arch != "" && s.Arch != c.arch {
			t.Errorf("%s: read as %q, expected %q", c.repo, s.Arch, c.arch)
		}
		if s.hidden() && (len(s.Layers) == 0 || s.Enc == 0) {
			t.Errorf("%s: a hidden-state drafter must name its layers and its encoder", c.repo)
		}
	}
}

// The whole pipeline for the models on this machine: search, filter, and check
// the winner's header remotely. Nothing is downloaded and nothing is loaded.
func TestPickDrafterLive(t *testing.T) {
	if os.Getenv("LLMASH_LIVE") != "1" {
		t.Skip("set LLMASH_LIVE=1 to search Hugging Face")
	}
	for _, path := range []string{
		os.Getenv("LLMASH_TEST_GGUF"), os.Getenv("LLMASH_TEST_GGUF2"),
	} {
		if path == "" || !fileExists(path) {
			continue
		}
		m := &Model{Name: stemOf(path), GGUF: path}
		target, err := specOfFile(path)
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("%s: %s, %d blocks of %d, %d tokens",
			m.Name, target.Arch, target.Blocks, target.Embed, target.Vocab)
		cands := findDrafters(m, false)
		for _, c := range cands {
			t.Logf("  candidate %-58s %s", c.Repo, c.Note)
		}
		best, ok := pickDrafter(m, cands, func(f string, a ...any) { t.Logf(f, a...) })
		if !ok {
			t.Logf("  -> nothing that fits")
			continue
		}
		t.Logf("  -> %s (%s)", best.Repo, best.Note)
	}
}

// A file that is not a model must be told apart from a hub that is merely
// unavailable: one is a verdict, the other is not.
func TestSpecOfURLTellsAMissingFileFromABadOne(t *testing.T) {
	if os.Getenv("LLMASH_LIVE") != "1" {
		t.Skip("set LLMASH_LIVE=1 to read from Hugging Face")
	}
	_, err := specOfURL(hubDownloadURL("Koopah/Qwen3.6-35B-A3B-NVFP4-DSPARK-v2-GGUF", "no-such-file.gguf"))
	if !errors.Is(err, errUnreadable) {
		t.Errorf("a missing file is the hub's answer, not the file's: %v", err)
	}
	// a real page of HTML, served with a 200
	_, err = specOfURL("https://huggingface.co/api/models/Koopah/Qwen3.6-35B-A3B-NVFP4-DSPARK-v2-GGUF")
	if err == nil || errors.Is(err, errUnreadable) {
		t.Errorf("something that is not a GGUF should be rejected outright: %v", err)
	}
}
