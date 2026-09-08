package main

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

// A folder of GGUFs the way llama.cpp users keep one: nested, several
// quantisations of one model side by side, a projector named the HuggingFace
// way, and a shard set.
func fakeLibrary(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	old := time.Now().Add(-time.Hour)
	for _, rel := range []string{
		"qwen/Qwen3-8B-Q4_K_M.gguf",
		"qwen/Qwen3-8B-Q8_0.gguf",
		"gemma/gemma-4-26b-it-Q4_K_M.gguf",
		"gemma/mmproj-gemma-4-26b-it-F16.gguf",
		"big/Big-Model-00001-of-00002.gguf",
		"big/Big-Model-00002-of-00002.gguf",
		".cache/hidden-model.gguf",
		"notes.txt",
	} {
		p := filepath.Join(dir, filepath.FromSlash(rel))
		os.MkdirAll(filepath.Dir(p), 0o755)
		if err := os.WriteFile(p, []byte("x"), 0o644); err != nil {
			t.Fatal(err)
		}
		os.Chtimes(p, old, old) // not "still being written"
	}
	return dir
}

func TestWalkGGUFIsRecursiveAndSkipsHiddenFolders(t *testing.T) {
	dir := fakeLibrary(t)
	got := walkGGUF(dir)
	if len(got) != 6 {
		t.Fatalf("want 6 GGUFs, got %d: %v", len(got), got)
	}
	for _, p := range got {
		if filepath.Base(filepath.Dir(p)) == ".cache" {
			t.Errorf("hidden folder was walked: %s", p)
		}
	}
}

func TestLibraryFolderIsListedByName(t *testing.T) {
	dir := fakeLibrary(t)
	r := &Registry{Root: t.TempDir(), Library: []string{dir}, cache: map[string]*Model{}, libCache: map[string]libScan{}}
	r.Blobs, r.Manifests = filepath.Join(r.Root, "blobs"), filepath.Join(r.Root, "manifests")

	names := map[string]string{}
	for _, e := range r.looseNamed() {
		names[e.name] = filepath.Base(e.path)
	}
	want := map[string]string{
		"qwen3-8b:gguf":       "Qwen3-8B-Q4_K_M.gguf", // first keeps the plain name
		"qwen3-8b:q8-0":       "Qwen3-8B-Q8_0.gguf",   // the other is told apart by its quant
		"gemma-4-26b-it:gguf": "gemma-4-26b-it-Q4_K_M.gguf",
		"big-model:gguf":      "Big-Model-00001-of-00002.gguf", // one entry for the shard set
	}
	for name, file := range want {
		if names[name] != file {
			t.Errorf("%s: want %s, got %q", name, file, names[name])
		}
	}
	if len(names) != len(want) {
		t.Errorf("want %d names, got %d: %v", len(want), len(names), names)
	}
	for name := range names {
		if name == "mmproj-gemma-4-26b-it:gguf" || name == "mmproj-gemma-4-26b-it-f16:gguf" {
			t.Errorf("a projector was listed as a model: %s", name)
		}
	}
}

func TestBareNameFindsTheLooseFile(t *testing.T) {
	dir := fakeLibrary(t)
	r := &Registry{Root: t.TempDir(), Library: []string{dir}, cache: map[string]*Model{}, libCache: map[string]libScan{}}
	r.Blobs, r.Manifests = filepath.Join(r.Root, "blobs"), filepath.Join(r.Root, "manifests")

	m := r.get("qwen3-8b", false)
	if m == nil {
		t.Fatal("run qwen3-8b should find qwen3-8b:gguf")
	}
	if m.Name != "qwen3-8b:gguf" || filepath.Base(m.GGUF) != "Qwen3-8B-Q4_K_M.gguf" {
		t.Errorf("got %s from %s", m.Name, m.GGUF)
	}
	if r.get("qwen3-8b:q8-0", false) == nil {
		t.Error("the second quantisation should be reachable under its own tag")
	}
	if r.get("qwen3-8b:latest", false) != nil {
		t.Error("an explicit :latest is not a loose file")
	}
	if r.get("nothing-here", false) != nil {
		t.Error("a name that matches nothing should stay nil")
	}
}

func TestInLibraryOnlyCoversTheLibrary(t *testing.T) {
	dir := fakeLibrary(t)
	r := &Registry{Root: t.TempDir(), Library: []string{dir}}
	if !r.InLibrary(filepath.Join(dir, "qwen", "Qwen3-8B-Q8_0.gguf")) {
		t.Error("a file under the library folder should be in it")
	}
	if r.InLibrary(filepath.Join(r.Root, "gguf", "x.gguf")) {
		t.Error("the store is not the library")
	}
	if r.InLibrary(dir + "-other") {
		t.Error("a sibling folder sharing the prefix is not inside it")
	}
}

// One setting, OLLAMA_MODELS, read for what it points at: a store is the
// store, a folder of GGUFs is read in place and the store stays where it was.
func TestModelsSettingIsReadForWhatItIs(t *testing.T) {
	folder := fakeLibrary(t)
	store := t.TempDir()
	os.MkdirAll(filepath.Join(store, "manifests"), 0o755)
	saved := localModelsRoot
	defer func() { localModelsRoot = saved }()
	t.Setenv("OLLAMA_MODELS", "")
	t.Setenv("LLMASH_MODELS", "")

	localModelsRoot = folder
	if got := libraryDirs(); len(got) != 1 || got[0] != folder {
		t.Errorf("a GGUF folder in local.json should be read in place, got %v", got)
	}
	if defaultModelsRoot() == folder {
		t.Error("a GGUF folder must not become the store")
	}

	localModelsRoot = store
	if libraryDirs() != nil {
		t.Error("a store is not a folder to read in place")
	}
	if defaultModelsRoot() != store {
		t.Errorf("a store in local.json should be the store, got %s", defaultModelsRoot())
	}

	t.Setenv("LLMASH_MODELS", folder)
	if got := libraryDirs(); len(got) != 1 || got[0] != folder {
		t.Errorf("LLMASH_MODELS should be read like OLLAMA_MODELS and win over local.json, got %v", got)
	}
	t.Setenv("OLLAMA_MODELS", store)
	if defaultModelsRoot() != store || libraryDirs() != nil {
		t.Error("OLLAMA_MODELS wins over LLMASH_MODELS")
	}
}

func TestSidecarsAreNotModels(t *testing.T) {
	for _, stem := range []string{"Qwen3-8B.mmproj", "mmproj-Qwen3-8B-F16", "Qwen3-8B-mmproj", "x.draft", "x.dspark", "x.eagle3", "x.mtp"} {
		if !isSidecar(stem) {
			t.Errorf("%s is a sidecar", stem)
		}
	}
	for _, stem := range []string{"Qwen3-8B-Q4_K_M", "gemma-4-26b-it", "Big-Model-00001-of-00002"} {
		if isSidecar(stem) {
			t.Errorf("%s is a model", stem)
		}
	}
}
