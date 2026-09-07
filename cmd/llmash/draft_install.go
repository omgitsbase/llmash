package main

import (
	"context"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"strings"
)

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

func hasOwnDrafter(m *Model) string {
	if d := installedDrafter(m); d != "" {
		return d
	}
	if hasMTP(m.GGUF) {
		return "an MTP head of its own"
	}
	return ""
}

func installedDrafter(m *Model) string {
	switch {
	case m.Mtp != "" && fileExists(m.Mtp):
		return "an MTP head"
	case m.Dspark != "" && fileExists(m.Dspark):
		return "a DSpark drafter"
	case m.Draft != "" && fileExists(m.Draft):
		return "a draft model"
	case m.Eagle3 != "" && fileExists(m.Eagle3):
		return "an EAGLE-3 drafter"
	}
	return ""
}

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

// The first candidate whose header says it fits, checked before downloading.
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

func cmdPullDraft(name string, yes, force bool) {
	log.SetOutput(io.Discard)
	loadConfig()
	needServer()
	m := modelFor(name)
	if m == nil {
		die("Error: %s is not a local GGUF, so there is nothing to pair a drafter with", name)
	}
	if hasMTP(m.GGUF) {
		fmt.Printf("%s has an MTP head of its own, trained with these exact weights.\n", name)
		fmt.Println("There is nothing to look for.")
		return
	}
	if installed := installedDrafter(m); installed != "" && !force {
		fmt.Printf("%s already has %s installed.\n", name, installed)
		fmt.Printf("`%s pulldraft %s --force` fetches it again.\n", prog, name)
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
	fmt.Println("\nchecking which of them fits these weights")
	best, ok := pickDrafter(m, cands, func(f string, a ...any) { fmt.Printf(f+"\n", a...) })
	if !ok {
		fmt.Printf("\nnone of them pairs with %s. It keeps its self-speculation (%s).\n",
			name, specFallback)
		return
	}
	fmt.Printf("\nbest match: %s (%s)\n\n", best.Repo, best.Note)
	if !yes && !confirm("Install it?") {
		return
	}
	if force {
		os.Remove(draftPath(m, best.Kind))
	}
	path, err := installDraft(m, best)
	if err != nil {
		die("could not install it: %v", err)
	}
	fmt.Printf("\ninstalled as %s\n", filepath.Base(path))
	fmt.Printf("%s now loads with --spec-type %s.\n", name, best.Kind.specArg)
}

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
	m.Mtp = findMtp(gguf)
	return m
}

// Runs at the end of a pull, and says nothing when there is nothing to offer.
func offerDraft(name string) {
	if !isConsole(os.Stdin) || !isConsole(os.Stdout) {
		return
	}
	log.SetOutput(io.Discard)
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
	var fit []draftCand
	for _, c := range cands {
		if fitsTarget(m, c) == nil {
			fit = append(fit, c)
		}
		if len(fit) == 4 {
			break
		}
	}
	if len(fit) == 0 {
		fmt.Printf("%snone that fit these weights%s\n", dim, reset)
		return
	}
	fmt.Printf("\n  a drafter usually makes this model meaningfully faster. These fit:\n")
	for i, c := range fit {
		fmt.Printf("    %2d. %-56s %s\n", i+1, c.Repo, c.Note)
	}
	n := askNumber("  Install one? (0 for none)", 1, len(fit))
	if n <= 0 {
		fmt.Printf("  skipped. `%s pulldraft %s` does it later.\n", prog, name)
		return
	}
	path, err := installDraft(m, fit[n-1])
	if err != nil {
		fmt.Printf("  could not install it: %v\n", err)
		return
	}
	fmt.Printf("  installed as %s\n", filepath.Base(path))
}

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
