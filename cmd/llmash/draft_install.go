package main

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// Installing a drafter: download it beside the model, under the suffix llmash
// already scans for, and it is in use on the next load. Nothing is run on the
// GPU to decide: the file is checked as a file, and a model that already has a
// draft head of its own keeps it.

func draftPath(m *Model, kind *draftKind) string {
	stem := shardSuffix.ReplaceAllString(stemOf(m.GGUF), "")
	dir := filepath.Dir(m.GGUF)
	if !dirExists(dir) || !writable(dir) {
		dir = reg.LooseDir()
	}
	return filepath.Join(dir, stem+"."+kind.name+".gguf")
}

func writable(dir string) bool {
	f, err := os.CreateTemp(dir, ".llmash-*")
	if err != nil {
		return false
	}
	name := f.Name()
	f.Close()
	os.Remove(name)
	return true
}

// A model that ships its own multi-token-prediction head, or already has a
// drafter beside it, is left alone: its own head is trained with the weights
// and a downloaded one would replace it, not add to it.
func hasOwnDrafter(m *Model) string {
	if m.Dspark != "" && fileExists(m.Dspark) {
		return "a DSpark drafter"
	}
	if m.Draft != "" && fileExists(m.Draft) {
		return "a draft model"
	}
	if m.Eagle3 != "" && fileExists(m.Eagle3) {
		return "an EAGLE-3 drafter"
	}
	if hasMTP(m.GGUF) {
		return "an MTP head of its own"
	}
	return ""
}

// A model's own head wins by default, so asking for a downloaded one instead
// has to be recorded. The marker sits beside the model and deleting it undoes
// the choice.
func preferMarker(gguf string) string {
	return shardSuffix.ReplaceAllString(stemOf(gguf), "") + ".prefer-draft"
}

func preferDraft(gguf string) bool { return fileExists(preferMarker(gguf)) }

// verifyDraft checks a downloaded file against the model it will draft for.
// A truncated download, an HTML error page saved under a .gguf name and a
// drafter built for different weights are all caught here, from the header.
func verifyDraft(m *Model, path string) error {
	draft, err := specOfFile(path)
	if err != nil {
		return err
	}
	target, err := specOfFile(m.GGUF)
	if err != nil {
		return nil
	}
	return pairs(target, draft)
}

// installDraft downloads a candidate and puts it where llmash will find it.
func installDraft(m *Model, c draftCand) (string, error) {
	dest := draftPath(m, c.Kind)
	if fileExists(dest) {
		return dest, nil
	}
	tmp := dest + ".part"
	os.Remove(tmp)
	bar := newProgress(os.Stderr)
	pb := newBar("pulling "+filepath.Base(c.File)+":", c.Size, 0)
	bar.add(pb)
	err := fetchBlocks(context.Background(), hubDownloadURL(c.Repo, c.File), tmp, c.Size,
		func(done int64) { pb.set(done) })
	bar.stop()
	if err != nil {
		os.Remove(tmp)
		os.Remove(tmp + ".idx")
		return "", err
	}
	if err := verifyDraft(m, tmp); err != nil {
		os.Remove(tmp)
		os.Remove(tmp + ".idx")
		return "", err
	}
	if err := os.Rename(tmp, dest); err != nil {
		os.Remove(tmp)
		return "", err
	}
	os.Remove(tmp + ".idx")
	return dest, nil
}

// pickDrafter takes the candidates in order and returns the first whose header
// says it fits. Reading the head of a remote file costs a moment; downloading
// gigabytes that cannot load costs rather more.
func pickDrafter(m *Model, cands []draftCand, say func(string, ...any)) (draftCand, bool) {
	for _, c := range cands {
		if err := fitsTarget(m, c); err != nil {
			say("    %s: %v", c.Repo, err)
			continue
		}
		return c, true
	}
	return draftCand{}, false
}

// ------------------------------------------------------------- the command

func cmdPullDraft(name string, yes, force bool) {
	loadConfig()
	needServer()
	m := modelFor(name)
	if m == nil {
		die("Error: %s is not a local GGUF, so there is nothing to pair a drafter with", name)
	}
	if own := hasOwnDrafter(m); own != "" && !force {
		fmt.Printf("%s already has %s, which is trained for these exact weights.\n", name, own)
		fmt.Printf("Nothing to add. `%s pulldraft %s --force` replaces it anyway.\n", prog, name)
		return
	}

	fmt.Printf("looking for a draft model for %s\n", name)
	cands := findDrafters(m, true)
	if len(cands) == 0 {
		fmt.Println("\nnothing published for this model. A drafter has to be trained against")
		fmt.Println("these exact weights, and either none exists or the ones that do ship")
		fmt.Printf("only safetensors. %s keeps its self-speculation (%s).\n", name, specFallback)
		return
	}

	fmt.Println("\nfound:")
	for i, c := range cands {
		fmt.Printf("  %d. %-58s %s\n", i+1, c.Repo, c.Note)
	}
	best := cands[0]
	fmt.Printf("\nbest match: %s (%s)\n\n", best.Repo, best.Note)
	if !yes && !confirm("Install it?") {
		return
	}
	path, err := installDraft(m, best)
	if err != nil {
		die("could not install it: %v", err)
	}
	fmt.Printf("\ninstalled as %s\n", filepath.Base(path))
	fmt.Printf("%s now loads with --spec-type %s.\n", name, best.Kind.specArg)
}

// modelFor resolves a name to the local GGUF behind it.
func modelFor(name string) *Model {
	info, code := showModel(name)
	if code != 200 {
		die("Error: %s", first(str(info, "error"), "model not found"))
	}
	gguf := strings.TrimPrefix(str(info, "modelfile"), "FROM ")
	if !fileExists(gguf) {
		return nil
	}
	m := &Model{Name: name, GGUF: gguf}
	m.Dspark = findDspark(gguf)
	m.Draft = findDraft(gguf)
	m.Eagle3 = findEagle3(gguf)
	return m
}

// offerDraft runs at the end of a pull. It asks once, and says nothing at all
// when the model already has a head or nothing is published for it.
func offerDraft(name string) {
	if !isConsole(os.Stdin) || !isConsole(os.Stdout) {
		return
	}
	loadConfig()
	m := modelFor(name)
	if m == nil || hasOwnDrafter(m) != "" {
		return
	}
	fmt.Printf("\n%slooking for a draft model...%s ", dim, reset)
	cands := findDrafters(m, false)
	if len(cands) == 0 {
		fmt.Printf("%snone published%s\n", dim, reset)
		return
	}
	best := cands[0]
	fmt.Printf("\n  %s (%s) drafts for this model and usually makes it\n", best.Repo, best.Note)
	fmt.Println("  meaningfully faster.")
	if !confirm("  Install it?") {
		fmt.Printf("  skipped. `%s pulldraft %s` does it later.\n", prog, name)
		return
	}
	path, err := installDraft(m, best)
	if err != nil {
		fmt.Printf("  could not install it: %v\n", err)
		return
	}
	fmt.Printf("  installed as %s\n", filepath.Base(path))
}

// confirm asks a yes/no question, defaulting to yes on Enter.
func confirm(question string) bool {
	fmt.Printf("%s [Y/n] ", question)
	for {
		switch ch := getch(); ch {
		case 'y', 'Y', '\r', '\n':
			fmt.Println("yes")
			return true
		case 'n', 'N', 0x1b, 0x03:
			fmt.Println("no")
			return false
		}
	}
}
