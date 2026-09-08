package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"sync"
	"time"
)

// Ollama's on-disk model store, read directly: OCI-style manifests plus blobs,
// and beside it a loose-GGUF directory for models fetched from HuggingFace.

// The quantisation on the end of a filename is not part of the model's
// identity: `Dirk-Q6_K` and `Dirk-UD-Q4_K_XL` are one model at two precisions.
var quantSuffix = regexp.MustCompile(`(?i)-(?:i?q\d+(?:_[a-z0-9]+)*|f16|bf16|f32|mxfp4)$`)
var shardSuffix = regexp.MustCompile(`-\d+-of-\d+$`)
var firstShard = regexp.MustCompile(`-0*1-of-\d+$`)

func baseName(p string) string { return filepath.Base(p) }

func stemOf(p string) string {
	b := filepath.Base(p)
	return strings.TrimSuffix(b, filepath.Ext(b))
}

func pairStem(stem string) string {
	prev := ""
	for prev != stem {
		prev = stem
		stem = quantSuffix.ReplaceAllString(stem, "")
	}
	return strings.ToLower(regexp.MustCompile(`(?i)-ud$`).ReplaceAllString(stem, ""))
}

var mmprojTail = regexp.MustCompile(`^[.\-_]*mmproj[.\-_a-z0-9]*\.gguf$`)

// findProjector: the multimodal projector belonging to this model, whatever
// it is called. Name tests first, then quant-stripped equality, then the
// header (same base repo AND matching projection width, unique match only).
func findProjector(gguf string, meta map[string]any) string {
	stem := shardSuffix.ReplaceAllString(stemOf(gguf), "")
	dir := filepath.Dir(gguf)
	for _, name := range []string{stem + ".mmproj.gguf", stem + "-mmproj.gguf"} {
		if fileExists(filepath.Join(dir, name)) {
			return filepath.Join(dir, name)
		}
	}
	cands, _ := filepath.Glob(filepath.Join(dir, "*mmproj*.gguf"))
	sort.Strings(cands)
	lowStem := strings.ToLower(stem)
	for _, c := range cands {
		low := strings.ToLower(filepath.Base(c))
		if strings.HasPrefix(low, lowStem) && mmprojTail.MatchString(low[len(lowStem):]) {
			return c
		}
	}
	want := pairStem(stem)
	for _, c := range cands {
		base := regexp.MustCompile(`(?i)[.-]mmproj$`).ReplaceAllString(stemOf(c), "")
		base = regexp.MustCompile(`(?i)^mmproj[.\-_]`).ReplaceAllString(base, "")
		if pairStem(base) == want {
			return c
		}
	}
	return projectorByArch(gguf, cands, meta)
}

func projectorByArch(gguf string, cands []string, meta map[string]any) string {
	if len(cands) == 0 {
		return ""
	}
	if meta == nil {
		meta = readGGUFMeta(gguf)
	}
	arch := metaStr(meta, "general.architecture")
	emb, ok := metaInt(meta, arch+".embedding_length")
	if !ok {
		return ""
	}
	repo := hfRepoOf(meta)
	if repo == "" {
		return ""
	}
	var hits []string
	for _, c := range cands {
		mm := readGGUFMeta(c)
		if dim, ok := metaInt(mm, "clip.vision.projection_dim"); !ok || dim != emb {
			continue
		}
		if hfRepoOf(mm) != repo {
			continue
		}
		hits = append(hits, c)
	}
	if len(hits) == 1 {
		return hits[0]
	}
	return ""
}

func findDspark(gguf string) string {
	stem := shardSuffix.ReplaceAllString(stemOf(gguf), "")
	dir := filepath.Dir(gguf)
	if c := filepath.Join(dir, stem+".dspark.gguf"); fileExists(c) {
		return c
	}
	want := pairStem(stem)
	cands, _ := filepath.Glob(filepath.Join(dir, "*.dspark.gguf"))
	sort.Strings(cands)
	for _, c := range cands {
		if pairStem(strings.TrimSuffix(stemOf(c), ".dspark")) == want {
			return c
		}
	}
	return ""
}

func findEagle3(gguf string) string {
	stem := shardSuffix.ReplaceAllString(stemOf(gguf), "")
	if c := filepath.Join(filepath.Dir(gguf), stem+".eagle3.gguf"); fileExists(c) {
		return c
	}
	return ""
}

func findMtp(gguf string) string {
	stem := shardSuffix.ReplaceAllString(stemOf(gguf), "")
	c := filepath.Join(filepath.Dir(gguf), stem+".mtp.gguf")
	if fileExists(c) {
		return c
	}
	return ""
}

func findDraft(gguf string) string {
	stem := shardSuffix.ReplaceAllString(stemOf(gguf), "")
	c := filepath.Join(filepath.Dir(gguf), stem+".draft.gguf")
	if fileExists(c) {
		return c
	}
	return ""
}

// ----------------------------------------------------------- capabilities

var pipelineCaps = map[string][]string{
	"any-to-any":                   {"vision", "audio", "video"},
	"image-text-to-text":           {"vision", "video"},
	"video-text-to-text":           {"vision", "video"},
	"visual-question-answering":    {"vision"},
	"image-to-text":                {"vision"},
	"audio-text-to-text":           {"audio"},
	"automatic-speech-recognition": {"audio"},
	"audio-classification":         {"audio"},
	"text-generation":              {},
	"text2text-generation":         {},
}

var hfRepoRe = regexp.MustCompile(`huggingface\.co/([^/\s]+/[^/\s]+)`)

func hfRepoOf(meta map[string]any) string {
	for _, key := range []string{"general.base_model.0.repo_url", "general.repo_url"} {
		if m := hfRepoRe.FindStringSubmatch(metaStr(meta, key)); m != nil {
			return strings.TrimRight(m[1], "/")
		}
	}
	org := metaStr(meta, "general.base_model.0.organization")
	name := metaStr(meta, "general.base_model.0.name")
	if org != "" && name != "" {
		return org + "/" + strings.ReplaceAll(name, " ", "-")
	}
	return ""
}

var (
	hfCapsMu  sync.Mutex
	hfCapsMem = map[string]*[]string{} // nil pointer value = unknown
)

// hfCaps: modalities HF lists for the repo, cached on disk beside the models
// so the network is touched once per repo ever and never when offline.
func hfCaps(repo, root string) *[]string {
	if repo == "" {
		return nil
	}
	hfCapsMu.Lock()
	defer hfCapsMu.Unlock()
	if v, ok := hfCapsMem[repo]; ok {
		return v
	}
	cf := filepath.Join(root, "hf_caps.json")
	disk := map[string]*[]string{}
	if b, err := os.ReadFile(cf); err == nil {
		json.Unmarshal(b, &disk)
	}
	if v, ok := disk[repo]; ok {
		hfCapsMem[repo] = v
		return v
	}
	var caps *[]string
	settled := false
	// Short: this runs inside the first `show` of a newly pulled model, and a
	// slow hub should cost a beat, not six seconds.
	client := &http.Client{Timeout: 2 * time.Second}
	resp, err := client.Get("https://huggingface.co/api/models/" + repo)
	if err == nil {
		func() {
			defer resp.Body.Close()
			if resp.StatusCode == 200 {
				var d struct {
					Tags        []string `json:"tags"`
					PipelineTag string   `json:"pipeline_tag"`
				}
				if json.NewDecoder(resp.Body).Decode(&d) == nil {
					tags := map[string]bool{}
					for _, t := range d.Tags {
						tags[strings.ToLower(t)] = true
					}
					pipeline := strings.ToLower(d.PipelineTag)
					set := map[string]bool{}
					for key, val := range pipelineCaps {
						if pipeline == key || tags[key] {
							for _, v := range val {
								set[v] = true
							}
						}
					}
					_, known := pipelineCaps[pipeline]
					if len(set) == 0 && !known {
						caps = nil
					} else {
						out := []string{}
						for k := range set {
							out = append(out, k)
						}
						sort.Strings(out)
						caps = &out
					}
					settled = true
				}
			} else if resp.StatusCode == 401 || resp.StatusCode == 403 || resp.StatusCode == 404 || resp.StatusCode == 410 {
				settled = true
			}
		}()
	}
	hfCapsMem[repo] = caps
	if settled {
		disk[repo] = caps
		if b, err := json.MarshalIndent(disk, "", " "); err == nil {
			os.WriteFile(cf, append(b, '\n'), 0o644)
		}
	}
	return caps
}

func projectorCaps(mm string) []string {
	meta := readGGUFMeta(mm)
	if len(meta) == 0 {
		return []string{"vision"}
	}
	on := func(k string) bool {
		v := strings.ToLower(metaStr(meta, k))
		return v == "true" || v == "1"
	}
	var caps []string
	if on("clip.has_vision_encoder") {
		caps = append(caps, "vision")
	}
	if on("clip.has_audio_encoder") {
		caps = append(caps, "audio")
	}
	if len(caps) == 0 {
		return []string{"vision"}
	}
	return caps
}

var embedArch = map[string]bool{"bert": true, "nomic-bert": true, "nomic-bert-moe": true,
	"jina-bert-v2": true, "xlm-roberta": true, "mpnet": true, "gte": true, "t5encoder": true}

// An embedding model has no chat side: ollama reports the one capability.
func isEmbedding(meta map[string]any, arch string) bool {
	if embedArch[strings.ToLower(arch)] {
		return true
	}
	_, pooled := metaInt(meta, arch+".pooling_type")
	return pooled
}

func capsFor(projector, template, arch string, published *[]string, embed bool) []string {
	if embed {
		return []string{"embedding"}
	}
	caps := []string{"completion"}
	installed := map[string]bool{}
	if projector != "" {
		for _, c := range projectorCaps(projector) {
			installed[c] = true
		}
	}
	var extra []string
	if published == nil {
		for c := range installed {
			extra = append(extra, c)
		}
	} else {
		for _, c := range *published {
			if installed[c] {
				extra = append(extra, c)
			}
		}
		if contains(*published, "video") && installed["vision"] && !contains(extra, "video") {
			extra = append(extra, "video")
		}
	}
	sort.Strings(extra)
	caps = append(caps, extra...)
	tl := strings.ToLower(template)
	if strings.Contains(tl, "tool") || strings.Contains(tl, "function") || arch == "gpt-oss" {
		caps = append(caps, "tools")
	}
	if strings.Contains(tl, "think") || strings.Contains(tl, "reason") || arch == "gpt-oss" {
		caps = append(caps, "thinking")
	}
	return caps
}

func missingModalities(published *[]string, projector string) []string {
	if published == nil || len(*published) == 0 {
		return nil
	}
	if projector != "" {
		for _, c := range projectorCaps(projector) {
			if c == "vision" || c == "audio" {
				return nil
			}
		}
	}
	out := append([]string{}, *published...)
	sort.Strings(out)
	return out
}

func contains(xs []string, s string) bool {
	for _, x := range xs {
		if x == s {
			return true
		}
	}
	return false
}

// ------------------------------------------------------------------ store

// modelsDir is the one models setting: OLLAMA_MODELS as Ollama takes it,
// LLMASH_MODELS as the same thing under this program's name, or local.json's
// models_root, which `llmash models set` writes. It may name an Ollama store
// or a plain folder of GGUFs; the folder is read for what it is.
func modelsDir() string {
	if v := strings.TrimSpace(os.Getenv("OLLAMA_MODELS")); v != "" {
		return v
	}
	if v := strings.TrimSpace(env("LLMASH_MODELS")); v != "" {
		return v
	}
	return localModelsRoot
}

func isOllamaStore(dir string) bool { return dirExists(filepath.Join(dir, "manifests")) }

// isGGUFFolder: a folder that is not an Ollama store and has GGUFs in it
// somewhere. Ollama would make a store of it; llmash reads it in place.
func isGGUFFolder(dir string) bool {
	return dirExists(dir) && !isOllamaStore(dir) && len(walkGGUF(dir)) > 0
}

func defaultModelsRoot() string {
	if v := modelsDir(); v != "" && !isGGUFFolder(v) {
		return v
	}
	home, _ := os.UserHomeDir()
	guesses := []string{filepath.Join(home, ".ollama", "models")}
	if local := os.Getenv("LOCALAPPDATA"); local != "" {
		guesses = append(guesses, filepath.Join(local, "Ollama", "models"))
	}
	guesses = append(guesses, `/usr/share/ollama/.ollama/models`, `/var/lib/ollama/.ollama/models`)
	for _, g := range guesses {
		if dirExists(filepath.Join(g, "manifests")) {
			return g
		}
	}
	return filepath.Join(home, ".ollama", "models")
}

var media = map[string]string{
	"model":     "application/vnd.ollama.image.model",
	"projector": "application/vnd.ollama.image.projector",
	"template":  "application/vnd.ollama.image.template",
	"params":    "application/vnd.ollama.image.params",
	"system":    "application/vnd.ollama.image.system",
	"license":   "application/vnd.ollama.image.license",
}

type Model struct {
	Name        string
	GGUF        string
	Size        int64
	Digest      string
	Projector   string
	Draft       string
	Eagle3      string
	Dspark      string
	Mtp         string
	Template    string
	System      string
	Params      map[string]any
	Family      string
	ParamSize   string
	Quant       string
	Ctx         int
	Modified    float64
	Caps        []string
	Missing     []string
	Experts     int
	ExpertsUsed int
}

type Registry struct {
	Root      string
	Blobs     string
	Manifests string
	Extras    []string // other stores read alongside the root: their manifests, blobs and gguf folders
	Library   []string // folders of GGUFs read in place, every subfolder included; never written to

	cfgMu    sync.RWMutex // Extras and Library, which Reload replaces while requests run
	libMu    sync.Mutex
	libCache map[string]libScan
	mu       sync.Mutex
	cache    map[string]*Model
	looseFP  string
	fpAt     time.Time
	fpVal    string
	allFP    string
	allDeep  bool
	allVal   []*Model
	aliasAt  time.Time
	aliasVal map[string]string
	aliasKey string
	doneMu   sync.Mutex
	done     map[string]bool
}

const fpTTL = 2 * time.Second

func newRegistry() *Registry {
	root := defaultModelsRoot()
	r := &Registry{Root: root, Blobs: filepath.Join(root, "blobs"), Manifests: filepath.Join(root, "manifests"),
		cache: map[string]*Model{}, libCache: map[string]libScan{}}
	r.Extras, r.Library = r.configuredDirs()
	return r
}

// configuredDirs reads the extra stores and library folders as configured
// right now, dropping the root itself and anything that is not a folder.
func (r *Registry) configuredDirs() (extras, library []string) {
	for _, e := range extraRoots() {
		if filepath.Clean(e) != filepath.Clean(r.Root) && dirExists(e) {
			extras = append(extras, e)
		}
	}
	for _, l := range libraryDirs() {
		if dirExists(l) {
			library = append(library, l)
		}
	}
	return extras, library
}

// Reload re-reads local.json's extra_roots and library, so `llmash models`
// takes effect without a restart. The store root itself stays: pulls are
// writing into it.
func (r *Registry) Reload() {
	loadLocal()
	extras, library := r.configuredDirs()
	r.cfgMu.Lock()
	r.Extras, r.Library = extras, library
	r.cfgMu.Unlock()
	r.libMu.Lock()
	r.libCache = map[string]libScan{}
	r.libMu.Unlock()
	r.Invalidate()
}

func (r *Registry) extras() []string {
	r.cfgMu.RLock()
	defer r.cfgMu.RUnlock()
	return r.Extras
}

func (r *Registry) library() []string {
	r.cfgMu.RLock()
	defer r.cfgMu.RUnlock()
	return r.Library
}

// InLibrary says whether a file lives under one of the library folders,
// which llmash reads but never deletes from.
func (r *Registry) InLibrary(p string) bool {
	abs, err := filepath.Abs(p)
	if err != nil {
		return false
	}
	for _, l := range r.library() {
		if la, err := filepath.Abs(l); err == nil {
			if rel, err := filepath.Rel(la, abs); err == nil && rel != ".." && !strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
				return true
			}
		}
	}
	return false
}

// splitDirs takes a ';'-separated list of folders, the way LLMASH_EXTRA_ROOTS
// is written.
func splitDirs(s string) []string {
	var out []string
	for _, e := range strings.Split(s, ";") {
		if e = strings.TrimSpace(e); e != "" {
			out = append(out, e)
		}
	}
	return out
}

// extraRoots lists the other model stores to read: local.json's extra_roots,
// and LLMASH_EXTRA_ROOTS with ';' between them.
func extraRoots() []string { return mergeDirs(localExtraRoots, env("LLMASH_EXTRA_ROOTS")) }

// libraryDirs is the folder of GGUFs read in place, when the models setting
// names one rather than an Ollama store.
func libraryDirs() []string {
	if v := modelsDir(); v != "" && isGGUFFolder(v) {
		return []string{v}
	}
	return nil
}

// mergeDirs is the configured folders followed by the variable's, each once.
func mergeDirs(configured []string, variable string) []string {
	var out []string
	seen := map[string]bool{}
	for _, d := range append(append([]string{}, configured...), splitDirs(variable)...) {
		key := strings.ToLower(filepath.Clean(d))
		if !seen[key] {
			seen[key] = true
			out = append(out, d)
		}
	}
	return out
}

// manifestDirs is every manifests folder read, the root's first.
func (r *Registry) manifestDirs() []string {
	out := []string{r.Manifests}
	for _, e := range r.extras() {
		out = append(out, filepath.Join(e, "manifests"))
	}
	return out
}

// looseDirs is every folder of loose GGUFs read: the configured one, then
// the gguf folder of the root and of each extra store.
func (r *Registry) looseDirs() []string {
	seen := map[string]bool{}
	var out []string
	add := func(d string) {
		c := filepath.Clean(d)
		if d != "" && !seen[c] {
			seen[c] = true
			out = append(out, d)
		}
	}
	add(r.LooseDir())
	add(filepath.Join(r.Root, "gguf"))
	for _, e := range r.extras() {
		add(filepath.Join(e, "gguf"))
	}
	return out
}

type libScan struct {
	at    time.Time
	files []string
}

const libTTL = 10 * time.Second

// libraryFiles is every GGUF under the library folders. A walk of a big tree
// is not free, and the fingerprint asks every couple of seconds, so each
// folder's listing is kept for a little while.
func (r *Registry) libraryFiles() []string {
	var out []string
	now := time.Now()
	r.libMu.Lock()
	defer r.libMu.Unlock()
	for _, d := range r.library() {
		key := filepath.Clean(d)
		if c, ok := r.libCache[key]; ok && now.Sub(c.at) < libTTL {
			out = append(out, c.files...)
			continue
		}
		files := walkGGUF(d)
		r.libCache[key] = libScan{at: now, files: files}
		out = append(out, files...)
	}
	return out
}

// walkGGUF lists the GGUFs under dir, subfolders included, sorted by path.
// Hidden folders are skipped, and the walk does not follow links.
func walkGGUF(dir string) []string {
	var out []string
	filepath.WalkDir(dir, func(p string, d os.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		if d.IsDir() {
			if p != dir && strings.HasPrefix(d.Name(), ".") {
				return filepath.SkipDir
			}
			return nil
		}
		if d.Type().IsRegular() && strings.EqualFold(filepath.Ext(p), ".gguf") {
			out = append(out, p)
		}
		return nil
	})
	sort.Strings(out)
	return out
}

func (r *Registry) Invalidate() {
	r.mu.Lock()
	r.cache = map[string]*Model{}
	r.fpAt = time.Time{}
	r.allVal = nil
	r.mu.Unlock()
}

func (r *Registry) nameOf(mf string) string {
	rel := mf
	for _, d := range r.manifestDirs() {
		if x, err := filepath.Rel(d, mf); err == nil && !strings.HasPrefix(x, "..") {
			rel = x
			break
		}
	}
	parts := strings.Split(filepath.ToSlash(rel), "/")
	if len(parts) < 2 {
		return rel
	}
	tag := parts[len(parts)-1]
	name := parts[len(parts)-2]
	namespace := "library"
	if len(parts) >= 3 {
		namespace = parts[len(parts)-3]
	}
	registry := parts[0]
	var base string
	switch {
	case registry == "registry.ollama.ai" && namespace == "library":
		base = name
	case registry == "registry.ollama.ai":
		base = namespace + "/" + name
	default:
		base = registry + "/" + namespace + "/" + name
	}
	return base + ":" + tag
}

func (r *Registry) blob(digest string) string {
	name := strings.ReplaceAll(digest, ":", "-")
	p := filepath.Join(r.Blobs, name)
	if fileExists(p) {
		return p
	}
	for _, e := range r.extras() {
		if q := filepath.Join(e, "blobs", name); fileExists(q) {
			return q
		}
	}
	return p
}

func (r *Registry) manifestFiles() []string {
	var out []string
	seen := map[string]bool{}
	for _, d := range r.manifestDirs() {
		if !dirExists(d) {
			continue
		}
		filepath.Walk(d, func(p string, info os.FileInfo, err error) error {
			if err == nil && !info.IsDir() {
				// the root's copy of a name wins over an extra store's
				if n := r.nameOf(p); !seen[n] {
					seen[n] = true
					out = append(out, p)
				}
			}
			return nil
		})
	}
	return out
}

func (r *Registry) LooseDir() string {
	if env := env("LLMASH_GGUF"); env != "" {
		return env
	}
	return filepath.Join(r.Root, "gguf")
}

func (r *Registry) aliases() map[string]string {
	f := filepath.Join(r.LooseDir(), "aliases.json")
	st, err := os.Stat(f)
	if err != nil {
		return map[string]string{}
	}
	key := fmt.Sprintf("%d:%d", st.ModTime().UnixNano(), st.Size())
	if r.aliasVal != nil && r.aliasKey == key {
		return r.aliasVal
	}
	data := map[string]string{}
	if b, err := os.ReadFile(f); err == nil {
		json.Unmarshal(b, &data)
	}
	r.aliasVal, r.aliasKey = data, key
	return data
}

func (r *Registry) setAlias(file, name string) error {
	data := map[string]string{}
	for k, v := range r.aliases() {
		data[k] = v
	}
	data[file] = name
	b, _ := json.MarshalIndent(data, "", "  ")
	if err := os.WriteFile(filepath.Join(r.LooseDir(), "aliases.json"), b, 0o644); err != nil {
		return err
	}
	r.aliasVal = nil
	r.Invalidate()
	return nil
}

// removeManifestModel drops a registry model: its manifest, and every blob
// no other manifest still names.
func (r *Registry) removeManifestModel(mfPath string) {
	type manifestRefs struct {
		Layers []struct {
			Digest string `json:"digest"`
		} `json:"layers"`
		Config *struct {
			Digest string `json:"digest"`
		} `json:"config"`
	}
	read := func(p string) (manifestRefs, bool) {
		var m manifestRefs
		b, err := os.ReadFile(p)
		if err != nil || json.Unmarshal(b, &m) != nil {
			return m, false
		}
		return m, true
	}
	gone, ok := read(mfPath)
	if !ok {
		return
	}
	used := map[string]bool{}
	for _, other := range r.manifestFiles() {
		if filepath.Clean(other) == filepath.Clean(mfPath) {
			continue
		}
		m, ok := read(other)
		if !ok {
			continue
		}
		for _, l := range m.Layers {
			used[l.Digest] = true
		}
		if m.Config != nil {
			used[m.Config.Digest] = true
		}
	}
	os.Remove(mfPath)
	for _, l := range gone.Layers {
		if !used[l.Digest] {
			os.Remove(r.blob(l.Digest))
		}
	}
	if gone.Config != nil && !used[gone.Config.Digest] {
		os.Remove(r.blob(gone.Config.Digest))
	}
	r.Invalidate()
}

// markComplete records a file this server finished writing, so the scan
// need not wait to see whether it is still growing.
func (r *Registry) markComplete(p string) {
	r.doneMu.Lock()
	if r.done == nil {
		r.done = map[string]bool{}
	}
	r.done[filepath.Clean(p)] = true
	r.doneMu.Unlock()
}

func (r *Registry) isComplete(p string) bool {
	r.doneMu.Lock()
	defer r.doneMu.Unlock()
	return r.done[filepath.Clean(p)]
}

func (r *Registry) looseName(p string) string {
	if alias := r.aliases()[filepath.Base(p)]; alias != "" {
		return alias
	}
	stem := shardSuffix.ReplaceAllString(stemOf(p), "")
	stem = quantSuffix.ReplaceAllString(stem, "")
	return strings.ReplaceAll(strings.ToLower(stem), "_", "-") + ":gguf"
}

func (r *Registry) looseFiles() []string {
	var files []string
	for _, d := range r.looseDirs() {
		if !dirExists(d) {
			continue
		}
		fs, _ := filepath.Glob(filepath.Join(d, "*.gguf"))
		sort.Strings(fs)
		files = append(files, fs...)
	}
	files = append(files, r.libraryFiles()...)
	var out []string
	now := time.Now()
	seen := map[string]bool{}
	for _, p := range files {
		if seen[filepath.Clean(p)] {
			continue
		}
		seen[filepath.Clean(p)] = true
		stem := stemOf(p)
		if shardSuffix.MatchString(stem) && !firstShard.MatchString(stem) {
			continue
		}
		if isSidecar(stem) {
			continue
		}
		st, err := os.Stat(p)
		if err != nil || (!r.isComplete(p) && now.Sub(st.ModTime()) < 20*time.Second) {
			continue // still being written
		}
		out = append(out, p)
	}
	return out
}

// isSidecar: a projector, drafter or MTP head that belongs to a model, not a
// model in its own right.
func isSidecar(stem string) bool {
	low := strings.ToLower(stem)
	return strings.Contains(low, "mmproj") ||
		strings.HasSuffix(low, ".draft") || strings.HasSuffix(low, ".dspark") ||
		strings.HasSuffix(low, ".eagle3") || strings.HasSuffix(low, ".mtp")
}

type looseEntry struct{ path, name string }

// looseNamed is every loose GGUF with the name it is served under. Two files
// that strip to the same name, the same model at two quantisations say, would
// otherwise be one name twice: the first keeps it and the next is told apart
// by its quantisation (`qwen3-8b:q8-0`), or by a counter when it has none.
func (r *Registry) looseNamed() []looseEntry {
	var out []looseEntry
	taken := map[string]bool{}
	for _, p := range r.looseFiles() {
		name := r.looseName(p)
		if taken[name] {
			stem := shardSuffix.ReplaceAllString(stemOf(p), "")
			base := strings.TrimSuffix(name, ":gguf")
			if q := quantSuffix.FindString(stem); q != "" && strings.HasSuffix(name, ":gguf") {
				name = base + ":" + strings.ReplaceAll(strings.ToLower(strings.TrimPrefix(q, "-")), "_", "-")
			}
			for n := 2; taken[name]; n++ {
				name = fmt.Sprintf("%s:gguf-%d", base, n)
			}
		}
		taken[name] = true
		out = append(out, looseEntry{p, name})
	}
	return out
}

func shardBytes(p string) int64 {
	st, err := os.Stat(p)
	if err != nil {
		return 0
	}
	m := regexp.MustCompile(`-(\d+)-of-(\d+)$`).FindStringSubmatch(stemOf(p))
	if m == nil {
		return st.Size()
	}
	stem := stemOf(p)
	head := stem[:len(stem)-len(m[0])]
	var total int
	fmt.Sscan(m[2], &total)
	width := len(m[1])
	var size int64
	for i := 1; i <= total; i++ {
		part := filepath.Join(filepath.Dir(p), fmt.Sprintf("%s-%0*d-of-%s%s", head, width, i, m[2], filepath.Ext(p)))
		if pst, err := os.Stat(part); err == nil {
			size += pst.Size()
		}
	}
	if size == 0 {
		return st.Size()
	}
	return size
}

func fileDigest(p string) string {
	st, err := os.Stat(p)
	if err != nil {
		return ""
	}
	sum := sha256.Sum256([]byte(fmt.Sprintf("%s:%d:%d", filepath.Base(p), st.Size(), st.ModTime().Unix())))
	return hex.EncodeToString(sum[:])
}

var sizeInName = regexp.MustCompile(`[-_ ][A-Za-z]?\d+(?:\.\d+)?[bBmM]\b`)

func (r *Registry) loadLoose(p, name string, deep bool) *Model {
	st, err := os.Stat(p)
	if err != nil {
		return nil
	}
	m := &Model{Name: name, GGUF: p, Size: shardBytes(p), Digest: "sha256:" + fileDigest(p),
		Modified: float64(st.ModTime().UnixNano()) / 1e9, Params: map[string]any{}}
	if !deep {
		return m
	}
	meta := readGGUFMeta(p)
	arch := metaStr(meta, "general.architecture")
	base := strings.TrimSpace(metaStr(meta, "general.basename"))
	if loc := sizeInName.FindStringIndex(base); loc != nil {
		base = base[:loc[0]]
	}
	base = strings.Trim(base, "-_ ")
	if base == "" {
		base = arch
	}
	m.Family = base
	if v, ok := metaInt(meta, arch+".context_length"); ok {
		m.Ctx = int(v)
	}
	if n, ok := metaInt(meta, "general.parameter_count"); ok && n > 0 {
		m.ParamSize = prettyParams(n)
	} else if s := metaStr(meta, "general.size_label"); s != "" {
		m.ParamSize = s
	}
	if v, ok := metaInt(meta, arch+".expert_count"); ok {
		m.Experts = int(v)
	}
	if v, ok := metaInt(meta, arch+".expert_used_count"); ok {
		m.ExpertsUsed = int(v)
	}
	if ft, ok := metaInt(meta, "general.file_type"); ok {
		if q, ok := quantNames[int(ft)]; ok {
			m.Quant = q
		} else {
			m.Quant = fmt.Sprint(ft)
		}
	}
	m.Template = metaStr(meta, "tokenizer.chat_template")
	m.Projector = findProjector(p, meta)
	m.Draft = findDraft(p)
	m.Eagle3 = findEagle3(p)
	m.Dspark = findDspark(p)
	m.Mtp = findMtp(p)
	pub := hfCaps(hfRepoOf(meta), filepath.Dir(p))
	m.Caps = capsFor(m.Projector, m.Template, arch, pub, isEmbedding(meta, arch))
	m.Missing = missingModalities(pub, m.Projector)
	return m
}

func (r *Registry) loadManifest(mf, name string, deep bool) *Model {
	b, err := os.ReadFile(mf)
	if err != nil {
		return nil
	}
	var data struct {
		Layers []struct {
			MediaType string `json:"mediaType"`
			Digest    string `json:"digest"`
			Size      int64  `json:"size"`
		} `json:"layers"`
		Config struct {
			Digest string `json:"digest"`
		} `json:"config"`
	}
	if json.Unmarshal(b, &data) != nil {
		return nil
	}
	m := &Model{Name: name, Params: map[string]any{}}
	for _, l := range data.Layers {
		blob := r.blob(l.Digest)
		switch l.MediaType {
		case media["model"]:
			m.GGUF, m.Size, m.Digest = blob, l.Size, l.Digest
		case media["projector"]:
			m.Projector = blob
		case media["template"]:
			if t, err := os.ReadFile(blob); err == nil {
				m.Template = string(t)
			}
		case media["system"]:
			if t, err := os.ReadFile(blob); err == nil {
				m.System = string(t)
			}
		case media["params"]:
			if t, err := os.ReadFile(blob); err == nil {
				json.Unmarshal(t, &m.Params)
			}
		}
	}
	if m.GGUF == "" {
		return nil
	}
	if st, err := os.Stat(mf); err == nil {
		m.Modified = float64(st.ModTime().UnixNano()) / 1e9
	}
	m.Draft = findDraft(m.GGUF)
	m.Eagle3 = findEagle3(m.GGUF)
	m.Dspark = findDspark(m.GGUF)
	m.Mtp = findMtp(m.GGUF)
	if data.Config.Digest != "" {
		if t, err := os.ReadFile(r.blob(data.Config.Digest)); err == nil {
			var cfg struct {
				ModelFamily   string   `json:"model_family"`
				ModelFamilies []string `json:"model_families"`
				ModelType     string   `json:"model_type"`
				FileType      string   `json:"file_type"`
			}
			if json.Unmarshal(t, &cfg) == nil {
				m.Family = cfg.ModelFamily
				if m.Family == "" && len(cfg.ModelFamilies) > 0 {
					m.Family = cfg.ModelFamilies[0]
				}
				m.ParamSize = cfg.ModelType
				m.Quant = cfg.FileType
			}
		}
	}
	if deep && fileExists(m.GGUF) {
		meta := readGGUFMeta(m.GGUF)
		arch := metaStr(meta, "general.architecture")
		if m.Family == "" {
			m.Family = arch
		}
		if v, ok := metaInt(meta, arch+".context_length"); ok {
			m.Ctx = int(v)
		}
		if m.ParamSize == "" {
			if n, ok := metaInt(meta, "general.parameter_count"); ok && n > 0 {
				m.ParamSize = prettyParams(n)
			}
		}
		if ft, ok := metaInt(meta, "general.file_type"); ok && m.Quant == "" {
			if q, ok := quantNames[int(ft)]; ok {
				m.Quant = q
			} else {
				m.Quant = fmt.Sprint(ft)
			}
		}
		if m.Template == "" {
			m.Template = metaStr(meta, "tokenizer.chat_template")
		}
		pub := hfCaps(hfRepoOf(meta), r.LooseDir())
		archFor := arch
		if m.Template == "" && strings.Contains(m.Name, "gpt-oss") {
			archFor = "gpt-oss"
		}
		m.Caps = capsFor(m.Projector, m.Template, archFor, pub, isEmbedding(meta, arch))
		m.Missing = missingModalities(pub, m.Projector)
	}
	return m
}

// fingerprint: a cheap stat of everything the listing depends on.
func (r *Registry) fingerprint() string {
	var sb strings.Builder
	for _, p := range r.looseFiles() {
		if st, err := os.Stat(p); err == nil {
			fmt.Fprintf(&sb, "%s|%d|%d;", p, st.ModTime().UnixNano(), st.Size())
		}
	}
	for _, d := range r.looseDirs() {
		if !dirExists(d) {
			continue
		}
		mm, _ := filepath.Glob(filepath.Join(d, "*mmproj*.gguf"))
		sort.Strings(mm)
		for _, p := range mm {
			if st, err := os.Stat(p); err == nil {
				fmt.Fprintf(&sb, "%s|%d|%d;", p, st.ModTime().UnixNano(), st.Size())
			}
		}
	}
	for _, p := range r.manifestFiles() {
		if st, err := os.Stat(p); err == nil {
			fmt.Fprintf(&sb, "%s|%d;", p, st.ModTime().UnixNano())
		}
	}
	return sb.String()
}

func (r *Registry) fingerprintCached() string {
	if !r.fpAt.IsZero() && time.Since(r.fpAt) < fpTTL {
		return r.fpVal
	}
	r.fpVal = r.fingerprint()
	r.fpAt = time.Now()
	return r.fpVal
}

// Get resolves a name; loose GGUFs win over manifests on a clash.
func (r *Registry) Get(name string) *Model {
	return r.get(name, true)
}

func (r *Registry) get(name string, deep bool) *Model {
	bare := !strings.Contains(name, ":")
	if bare {
		name += ":latest"
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if m := r.getExact(name, deep); m != nil {
		return m
	}
	// A loose file has no tag of its own and is listed as `name:gguf`, so a
	// bare name that nothing was pulled under still finds the file.
	if bare {
		return r.getExact(strings.TrimSuffix(name, ":latest")+":gguf", deep)
	}
	return nil
}

// getExact resolves one full name; the caller holds r.mu.
func (r *Registry) getExact(name string, deep bool) *Model {
	for _, e := range r.looseNamed() {
		if e.name != name {
			continue
		}
		fp := r.fingerprintCached()
		if r.looseFP != fp {
			for k := range r.cache {
				if strings.HasPrefix(k, "loose|") {
					delete(r.cache, k)
				}
			}
			r.looseFP = fp
		}
		key := fmt.Sprintf("loose|%s|%v", name, deep)
		if m, ok := r.cache[key]; ok {
			return m
		}
		m := r.loadLoose(e.path, e.name, deep)
		if m != nil {
			r.cache[key] = m
		}
		return m
	}
	key := fmt.Sprintf("%s|%v", name, deep)
	if m, ok := r.cache[key]; ok {
		return m
	}
	for _, mf := range r.manifestFiles() {
		if r.nameOf(mf) == name {
			m := r.loadManifest(mf, name, deep)
			if m != nil {
				r.cache[key] = m
			}
			return m
		}
	}
	return nil
}

// All lists every model, newest first; cached on the store's fingerprint.
func (r *Registry) All(deep bool) []*Model {
	r.mu.Lock()
	defer r.mu.Unlock()
	fp := r.fingerprintCached()
	if r.allVal != nil && r.allFP == fp && r.allDeep == deep {
		return append([]*Model{}, r.allVal...)
	}
	var out []*Model
	seen := map[string]bool{}
	for _, e := range r.looseNamed() {
		if m := r.loadLoose(e.path, e.name, deep); m != nil {
			out = append(out, m)
			seen[m.Name] = true
		}
	}
	for _, mf := range r.manifestFiles() {
		name := r.nameOf(mf)
		if seen[name] {
			continue
		}
		if m := r.loadManifest(mf, name, deep); m != nil {
			out = append(out, m)
		}
	}
	sort.SliceStable(out, func(i, j int) bool { return out[i].Modified > out[j].Modified })
	r.allVal, r.allFP, r.allDeep = out, fp, deep
	return append([]*Model{}, out...)
}
