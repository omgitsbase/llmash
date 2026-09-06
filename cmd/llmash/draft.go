package main

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"sort"
	"strings"
	"time"
)

// A drafter is trained against one specific target, and the wrong one is worse
// than none: it loads, drafts badly, and costs speed. So this rejects rather
// than guesses. The Hugging Face API used here needs no account and no token.

type draftKind struct {
	name    string // what llmash calls it, and the file suffix it is saved under
	specArg string // llama.cpp's --spec-type
	rank    int    // higher wins when several exist
	words   []string
}

var draftKinds = []draftKind{
	{"eagle3", "draft-eagle3", 40, []string{"eagle3", "eagle-3", "eagle_3"}},
	{"dspark", "draft-dspark", 30, []string{"dspark", "d-spark"}},
	{"dflash", "draft-dflash", 20, []string{"dflash", "d-flash"}},
	{"draft", "draft-simple", 10, []string{"draft", "speculator", "speculative"}},
}

func kindOf(text string) *draftKind {
	low := strings.ToLower(text)
	for i := range draftKinds {
		for _, w := range draftKinds[i].words {
			if strings.Contains(low, w) {
				return &draftKinds[i]
			}
		}
	}
	return nil
}

type draftCand struct {
	Repo  string
	File  string
	Kind  *draftKind
	Size  int64
	Bases []string
	Score int
	Note  string
}

const hubAPI = "https://huggingface.co/api"

func hubGet(path string, out any) error {
	req, err := http.NewRequest("GET", path, nil)
	if err != nil {
		return err
	}
	req.Header.Set("User-Agent", "llmash")
	req.Header.Set("Accept", "application/json")
	resp, err := (&http.Client{Timeout: 45 * time.Second}).Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode == 429 {
		return fmt.Errorf("Hugging Face is rate limiting this address; try again in a minute")
	}
	if resp.StatusCode != 200 {
		return fmt.Errorf("hub answered %d", resp.StatusCode)
	}
	return json.NewDecoder(io.LimitReader(resp.Body, 8<<20)).Decode(out)
}

type hubModel struct {
	ID        string   `json:"id"`
	Tags      []string `json:"tags"`
	Downloads int      `json:"downloads"`
	CardData  struct {
		BaseModel any `json:"base_model"`
	} `json:"cardData"`
}

func (h hubModel) bases() []string {
	switch v := h.CardData.BaseModel.(type) {
	case string:
		return []string{v}
	case []any:
		var out []string
		for _, x := range v {
			out = append(out, fmt.Sprint(x))
		}
		return out
	}
	return nil
}

type hubFile struct {
	Path string `json:"path"`
	Type string `json:"type"`
	Size int64  `json:"size"`
}

func hubSearch(query string, limit int) []hubModel {
	var out []hubModel
	u := fmt.Sprintf("%s/models?search=%s&limit=%d&sort=downloads&direction=-1",
		hubAPI, url.QueryEscape(query), limit)
	if err := hubGet(u, &out); err != nil {
		return nil
	}
	return out
}

func hubInfo(repo string) (hubModel, error) {
	var m hubModel
	err := hubGet(hubAPI+"/models/"+repo, &m)
	return m, err
}

func hubFiles(repo string) []hubFile {
	var out []hubFile
	if err := hubGet(hubAPI+"/models/"+repo+"/tree/main?recursive=true", &out); err != nil {
		return nil
	}
	return out
}

func hubDownloadURL(repo, file string) string {
	return "https://huggingface.co/" + repo + "/resolve/main/" + file
}

func modelStem(m *Model) string {
	name := m.Name
	if repo := hfRepoOf(readGGUFMeta(m.GGUF)); repo != "" {
		if i := strings.Index(repo, "/"); i >= 0 {
			name = repo[i+1:]
		} else {
			name = repo
		}
	}
	name = strings.SplitN(name, ":", 2)[0]
	for _, junk := range []string{"-GGUF", "-gguf", "-it-GGUF", "-UD", "-Instruct"} {
		name = strings.TrimSuffix(name, junk)
	}
	name = quantSuffix.ReplaceAllString(name, "")
	return strings.Trim(name, "-_. ")
}

func normalise(s string) string {
	var b strings.Builder
	for _, r := range strings.ToLower(s) {
		if (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') {
			b.WriteRune(r)
		}
	}
	return b.String()
}

func findDrafters(m *Model, verbose bool) []draftCand {
	stem := modelStem(m)
	if stem == "" {
		return nil
	}
	want := normalise(stem)
	seen := map[string]bool{}
	var cands []draftCand

	say := func(format string, a ...any) {
		if verbose {
			fmt.Printf(format+"\n", a...)
		}
	}
	say("  looking for a drafter trained on %s", stem)

	queries := []string{stem + " eagle3", stem + " dspark", stem + " speculator",
		stem + " draft GGUF", stem + " dflash"}
	for _, q := range queries {
		for _, hit := range hubSearch(q, 25) {
			if seen[hit.ID] {
				continue
			}
			seen[hit.ID] = true
			if c, ok := considerRepo(hit, want, stem, say); ok {
				cands = append(cands, c)
			}
		}
	}
	sort.SliceStable(cands, func(i, j int) bool { return cands[i].Score > cands[j].Score })
	return cands
}

// Every rejection is reported: a silent one looks like "none exists".
func considerRepo(hit hubModel, want, stem string, say func(string, ...any)) (draftCand, bool) {
	var c draftCand
	kind := kindOf(hit.ID)
	if kind == nil {
		for _, t := range hit.Tags {
			if kind = kindOf(t); kind != nil {
				break
			}
		}
	}
	if kind == nil {
		return c, false
	}

	info, err := hubInfo(hit.ID)
	if err != nil {
		say("    %s: %v", hit.ID, err)
		return c, false
	}
	bases := info.bases()

	matched := false
	for _, b := range bases {
		if strings.Contains(normalise(b), want) {
			matched = true
		}
	}
	if !matched && !strings.Contains(normalise(hit.ID), want) {
		say("    %s: does not name %s as a base", hit.ID, stem)
		return c, false
	}
	if derived := foreignBase(bases, want); derived != "" {
		say("    %s: built for %s, a different model", hit.ID, derived)
		return c, false
	}
	if runtime := otherRuntime(hit.ID); runtime != "" {
		say("    %s: a %s build, which llama.cpp does not load", hit.ID, runtime)
		return c, false
	}
	if len(bases) == 0 {
		if derived := foreignBase([]string{hit.ID}, want); derived != "" {
			say("    %s: the name says it is a project of its own, not a drafter for %s",
				hit.ID, stem)
			return c, false
		}
	}

	files := hubFiles(hit.ID)
	var best hubFile
	for _, f := range files {
		if f.Type != "file" || !strings.HasSuffix(strings.ToLower(f.Path), ".gguf") {
			continue
		}
		if best.Path == "" || preferQuant(f.Path, best.Path) {
			best = f
		}
	}
	if best.Path == "" {
		say("    %s: %s, but only safetensors (needs converting)", hit.ID, kind.name)
		return c, false
	}

	if best.Size > draftSizeCeiling {
		say("    %s: %s is too big for a drafter; that is a whole model",
			hit.ID, humanBytes(best.Size))
		return c, false
	}
	c = draftCand{Repo: hit.ID, File: best.Path, Kind: kind, Size: best.Size,
		Bases: bases, Score: kind.rank}
	if info.Downloads > 100 {
		c.Score += 5
	}
	if best.Size > 0 && best.Size < 1<<30 {
		c.Score += 3
	}
	c.Note = fmt.Sprintf("%s, %s", kind.name, humanBytes(best.Size))
	return c, true
}

// A base that is a derivative of the target, e.g. an abliterated fine-tune.
func foreignBase(bases []string, want string) string {
	markers := []string{"abliterat", "uncensored", "caption", "distill", "merge",
		"roleplay", "magic", "agentic", "heretic", "aggressive"}
	for _, b := range bases {
		low := strings.ToLower(b)
		for _, m := range markers {
			if strings.Contains(low, m) {
				return b
			}
		}
		// Take the model name without the organisation, remove the target and
		// the words that mean nothing (packaging, quantisation, drafter
		// kinds). Whatever is left is a fine-tune's own name.
		name := b
		if i := strings.LastIndex(name, "/"); i >= 0 {
			name = name[i+1:]
		}
		rest := strings.Replace(normalise(name), want, "", 1)
		for _, innocuous := range []string{
			"instruct", "it", "chat", "base", "gguf", "hf", "llamacpp", "llama",
			"nvfp", "fp", "unsloth", "quant",
			"speculator", "eagle3", "eagle", "dspark", "dflash", "draft", "model",
			"f16", "bf16", "fp16", "q4km", "q4", "q8", "iq4xs", "test", "preview",
			"v1", "v2", "v3", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
		} {
			rest = strings.ReplaceAll(rest, innocuous, "")
		}
		if len(rest) > 3 {
			return b
		}
	}
	return ""
}

// Repacks for other runtimes match on name but will never load here.
func otherRuntime(repo string) string {
	for _, r := range []string{"mlx", "exl3", "exl2", "bpw", "ov-int", "openvino",
		"awq", "gptq", "-int4", "-int8", "rocm", "trt", "tensorrt"} {
		if strings.Contains(strings.ToLower(repo), r) {
			return strings.TrimPrefix(r, "-")
		}
	}
	return ""
}

// Anything larger is a whole model that happens to mention a drafter.
const draftSizeCeiling = 6 << 30

var quantOrder = []string{"q4_k_m", "q4_k", "iq4_xs", "q5_k_m", "q8_0", "f16", "bf16"}

func preferQuant(a, b string) bool {
	rank := func(s string) int {
		low := strings.ToLower(s)
		for i, q := range quantOrder {
			if strings.Contains(low, q) {
				return i
			}
		}
		return len(quantOrder)
	}
	return rank(a) < rank(b)
}
