package main

import (
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// `llmash models`: where models are read from, and `llmash models set DIR`
// to point llmash somewhere else. It sets the same thing OLLAMA_MODELS does,
// which llmash honours as well; this is the command Ollama never had.
//
// A folder is read for what it is. An Ollama store, with a manifests folder
// inside, becomes the store, where pulls go. Any other folder is taken as a
// folder of GGUFs, the kind llama.cpp keeps, and is read where it is,
// subfolders included: nothing in it is copied, moved or deleted.

func cmdModels(args []string) {
	o := parseSimple(args, nil, nil)
	log.SetOutput(io.Discard)
	loadConfig()
	log.SetOutput(os.Stdout)

	if len(o.pos) == 0 {
		showModelDirs()
		return
	}
	dir := o.pos[len(o.pos)-1]
	if o.pos[0] == "set" && len(o.pos) < 2 {
		die("Error: %s models set needs a folder", prog)
	}
	if o.pos[0] != "set" && len(o.pos) > 1 {
		die("Error: unknown command \"%s\" for \"%s models\"", o.pos[0], prog)
	}
	abs, err := filepath.Abs(dir)
	if err != nil {
		die("Error: %v", err)
	}
	if !dirExists(abs) {
		die("Error: %s is not a folder", abs)
	}
	setModelDir(abs)
}

func sameDir(a, b string) bool { return strings.EqualFold(filepath.Clean(a), filepath.Clean(b)) }

func hasDir(xs []string, dir string) bool {
	for _, x := range xs {
		if sameDir(x, dir) {
			return true
		}
	}
	return false
}

func strList(xs []string) []any {
	out := make([]any, 0, len(xs))
	for _, x := range xs {
		out = append(out, x)
	}
	return out
}

func setModelDir(dir string) {
	cfg, err := readLocal()
	if err != nil {
		die("Error: %v", err)
	}
	// the variable wins over local.json, so a different one would make this a no-op
	for _, name := range []string{"OLLAMA_MODELS", "LLMASH_MODELS"} {
		if v := strings.TrimSpace(os.Getenv(name)); v != "" && !sameDir(v, dir) {
			die("%s is set to %s in your environment, and that wins.\nChange it there, or unset it:  setx %s \"\"", name, v, name)
		}
	}

	if isOllamaStore(dir) {
		old := defaultModelsRoot()
		if sameDir(dir, old) {
			fmt.Printf("%s is already the model store (%d models)\n", dir, countManifests(dir))
			return
		}
		// what was pulled into the old store is still worth reading, as the installer does
		extras := localStringsOf(cfg, "extra_roots")
		if isOllamaStore(old) && !hasDir(extras, old) {
			extras = append(extras, old)
		}
		cfg["models_root"] = dir
		cfg["extra_roots"] = strList(extras)
		if err := writeLocal(cfg); err != nil {
			die("Error: %v", err)
		}
		os.Setenv("OLLAMA_MODELS", dir)
		fmt.Printf("model store is now %s (%d models; pulls go here)\n", dir, countManifests(dir))
		restartForPaths()
		return
	}

	if !isGGUFFolder(dir) {
		die("%s has no GGUF in it and is not an Ollama store, so there is nothing to read", dir)
	}
	cfg["models_root"] = dir
	if err := writeLocal(cfg); err != nil {
		die("Error: %v", err)
	}
	clean, skipped := countLibrary(dir)
	line := fmt.Sprintf("reading %s (%d models", dir, clean)
	if skipped > 0 {
		line += fmt.Sprintf(", %d files skipped", skipped)
	}
	fmt.Println(line + ")")
	reloadPaths()
}

// countLibrary is how many models a folder yields, and how many of its GGUFs
// are left out: sidecars, later shards, and files whose header will not read.
func countLibrary(dir string) (clean, skipped int) {
	for _, p := range walkGGUF(dir) {
		stem := stemOf(p)
		if isSidecar(stem) || (shardSuffix.MatchString(stem) && !firstShard.MatchString(stem)) {
			continue
		}
		if metaStr(readGGUFMeta(p), "general.architecture") == "" {
			skipped++
			continue
		}
		clean++
	}
	return clean, skipped
}

func countManifests(store string) int {
	n := 0
	filepath.Walk(filepath.Join(store, "manifests"), func(p string, info os.FileInfo, err error) error {
		if err == nil && !info.IsDir() {
			n++
		}
		return nil
	})
	return n
}

func localStringsOf(cfg map[string]any, key string) []string {
	var out []string
	for _, v := range list(cfg, key) {
		if s, _ := v.(string); strings.TrimSpace(s) != "" {
			out = append(out, strings.TrimSpace(s))
		}
	}
	return out
}

func serverUp() bool {
	_, err := call("GET", "/api/version", nil, 2*time.Second)
	return err == nil
}

// reloadPaths tells a running server to re-read local.json. Without one the
// change simply applies at the next start.
func reloadPaths() {
	if !serverUp() {
		return
	}
	if _, _, err := callJSON("POST", "/api/paths", map[string]any{}, 30*time.Second); err != nil {
		fmt.Printf("the running server could not reload (%v); restart it from the tray\n", err)
	}
}

// restartForPaths: the store root is fixed for the life of a server, since
// pulls are writing into it, so a new one means a restart.
func restartForPaths() {
	if !serverUp() {
		return
	}
	fmt.Print("restarting the server ... ")
	stopServerProcess()
	if startServerProcess() {
		fmt.Println("done")
	} else {
		fmt.Printf("\nit did not come back; start it with:  %s tray\n", prog)
	}
}

func showModelDirs() {
	reg := newRegistry()
	type row struct{ kind, dir, note string }
	rows := []row{{"store", reg.Root, fmt.Sprintf("%d models; pulls go here", countManifests(reg.Root))}}
	if loose := reg.LooseDir(); dirExists(loose) {
		n, _ := countLibrary(loose)
		rows = append(rows, row{"loose", loose, fmt.Sprintf("%d models; HuggingFace pulls go here", n)})
	}
	for _, e := range reg.extras() {
		rows = append(rows, row{"also", e, fmt.Sprintf("%d models; an older store, still read", countManifests(e))})
	}
	for _, l := range libraryDirs() {
		if !dirExists(l) {
			rows = append(rows, row{"folder", l, "missing"})
			continue
		}
		clean, skipped := countLibrary(l)
		note := fmt.Sprintf("%d models, read in place", clean)
		if skipped > 0 {
			note += fmt.Sprintf(", %d files skipped", skipped)
		}
		rows = append(rows, row{"folder", l, note})
	}
	width := 0
	for _, r := range rows {
		width = max(width, len(r.dir))
	}
	for _, r := range rows {
		fmt.Printf("  %-7s %-*s  %s\n", r.kind, width, r.dir, r.note)
	}
	for _, g := range ollamaStoreGuesses() {
		if isOllamaStore(g) && !sameDir(g, reg.Root) && !hasDir(reg.extras(), g) {
			fmt.Printf("\nOllama's models at %s are not being read:  %s models set %s\n", g, prog, g)
		}
	}
}

func ollamaStoreGuesses() []string {
	var out []string
	if v := modelsDir(); v != "" {
		out = append(out, v)
	}
	home, _ := os.UserHomeDir()
	out = append(out, filepath.Join(home, ".ollama", "models"))
	if local := os.Getenv("LOCALAPPDATA"); local != "" {
		out = append(out, filepath.Join(local, "Ollama", "models"))
	}
	return out
}
