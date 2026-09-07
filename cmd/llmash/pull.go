package main

import (
	"bufio"
	"bytes"
	"math"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

// /api/pull: straight from the Ollama registry into the same blob store, or
// from HuggingFace into the loose-GGUF directory (hf:owner/repo[@QUANT], and
// the names whose Ollama build upstream llama.cpp cannot read).

const (
	ollamaRegistry = "registry.ollama.ai"
	hfBase         = "https://huggingface.co"
)

var hfReplacements = map[string][2]string{
	"gemma4:e4b":           {"unsloth/gemma-4-E4B-it-GGUF", "Q4_K_M"},
	"gemma4:26b":           {"unsloth/gemma-4-26B-A4B-it-GGUF", "Q4_K_M"},
	"gemma4:31b":           {"unsloth/gemma-4-31B-it-GGUF", "Q4_K_M"},
	"Qwen3.6:27B":          {"unsloth/Qwen3.6-27B-GGUF", "Q4_K_M"},
	"qwen3.5:35b":          {"unsloth/Qwen3.5-35B-A3B-GGUF", "Q4_K_M"},
	"glm-4.7-flash:latest": {"lmstudio-community/GLM-4.7-Flash-GGUF", "Q4_K_M"},
	"gpt-oss:120b":         {"lmstudio-community/gpt-oss-120b-GGUF", "MXFP4"},
}

var pullClient = &http.Client{}

func splitRef(ref string) (host, repo, tag string) {
	host = ollamaRegistry
	name, tag, _ := strings.Cut(ref, ":")
	if tag == "" {
		tag = "latest"
	}
	parts := strings.Split(name, "/")
	switch len(parts) {
	case 1:
		repo = "library/" + parts[0]
	case 2:
		repo = name
	default:
		host, repo = parts[0], strings.Join(parts[1:], "/")
	}
	return
}

type hfFile struct {
	Name string `json:"rfilename"`
	Size int64  `json:"size"`
}

func hfFiles(ctx context.Context, repo string) ([]hfFile, error) {
	req, _ := http.NewRequestWithContext(ctx, "GET", hfBase+"/api/models/"+repo+"?blobs=true", nil)
	resp, err := pullClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return nil, nil
	}
	var d struct {
		Siblings []hfFile `json:"siblings"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&d); err != nil {
		return nil, err
	}
	var out []hfFile
	for _, f := range d.Siblings {
		if strings.HasSuffix(strings.ToLower(f.Name), ".gguf") {
			out = append(out, f)
		}
	}
	return out, nil
}

func pickGGUF(files []hfFile, quant string) []hfFile {
	q := strings.ToLower(quant)
	var cand []hfFile
	for _, f := range files {
		if strings.Contains(strings.ToLower(f.Name), q) {
			cand = append(cand, f)
		}
	}
	if len(cand) == 0 {
		for _, f := range files {
			if strings.Contains(strings.ToLower(f.Name), "q4_k_m") {
				cand = append(cand, f)
			}
		}
		if len(cand) == 0 {
			cand = files
		}
	}
	var main []hfFile
	for _, f := range cand {
		if !strings.Contains(strings.ToLower(f.Name), "mmproj") {
			main = append(main, f)
		}
	}
	if len(main) > 0 {
		cand = main
	}
	var shards []hfFile
	for _, f := range cand {
		if strings.Contains(f.Name, "-of-") {
			shards = append(shards, f)
		}
	}
	if len(shards) > 0 {
		sort.Slice(shards, func(i, j int) bool { return shards[i].Name < shards[j].Name })
		return shards
	}
	if len(cand) == 0 {
		return nil
	}
	sort.Slice(cand, func(i, j int) bool { return cand[i].Size < cand[j].Size })
	return []hfFile{cand[len(cand)-1]}
}

func pickMmproj(files []hfFile) *hfFile {
	var mm []hfFile
	for _, f := range files {
		if strings.Contains(strings.ToLower(f.Name), "mmproj") {
			mm = append(mm, f)
		}
	}
	if len(mm) == 0 {
		return nil
	}
	smallest := func(xs []hfFile) *hfFile {
		sort.Slice(xs, func(i, j int) bool { return xs[i].Size < xs[j].Size })
		return &xs[0]
	}
	for _, want := range []string{"-f16", "f16", "-bf16", "bf16"} {
		var hit []hfFile
		for _, f := range mm {
			if strings.Contains(strings.ToLower(f.Name), want) {
				hit = append(hit, f)
			}
		}
		if len(hit) > 0 {
			return smallest(hit)
		}
	}
	return smallest(mm)
}

var (
	dlStreams = envInt("LLMASH_DL_STREAMS", 8)
	dlBlock   = int64(envInt("LLMASH_DL_BLOCK", 64*1024*1024))
)

// fetchBlocks downloads url into tmp with several ranged connections at once,
// keeping a block map beside it so an interrupted download resumes.
// fetchBlob pulls a file by ranged blocks over several connections when the
// server honours ranges, and over one connection when it does not. The
// registry answers a ranged GET with 206 from its CDN; anything that answers
// 200 gets the whole file on the stream it opened.
func fetchBlob(ctx context.Context, url, tmp string, total int64, progress func(int64)) error {
	if total > 2*dlBlock {
		req, _ := http.NewRequestWithContext(ctx, "GET", url, nil)
		req.Header.Set("Range", "bytes=0-0")
		resp, err := pullClient.Do(req)
		if err == nil {
			resp.Body.Close()
			if resp.StatusCode == 206 {
				return fetchBlocks(ctx, url, tmp, total, progress)
			}
		}
	}
	req, _ := http.NewRequestWithContext(ctx, "GET", url, nil)
	resp, err := pullClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return fmt.Errorf("HTTP %d", resp.StatusCode)
	}
	fh, err := os.Create(tmp)
	if err != nil {
		return err
	}
	defer fh.Close()
	var done int64
	last := time.Time{}
	buf := make([]byte, 1<<20)
	for {
		n, err := resp.Body.Read(buf)
		if n > 0 {
			if _, werr := fh.Write(buf[:n]); werr != nil {
				return werr
			}
			done += int64(n)
			if time.Since(last) > 200*time.Millisecond {
				last = time.Now()
				progress(done)
			}
		}
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return err
		}
	}
}

func fetchBlocks(ctx context.Context, url, tmp string, total int64, progress func(int64)) error {
	nblocks := (total + dlBlock - 1) / dlBlock
	if nblocks < 1 {
		nblocks = 1
	}
	idxPath := tmp + ".idx"
	have := make([]byte, nblocks)
	if b, err := os.ReadFile(idxPath); err == nil && int64(len(b)) == nblocks {
		copy(have, b)
	}
	f, err := os.OpenFile(tmp, os.O_CREATE|os.O_RDWR, 0o644)
	if err != nil {
		return err
	}
	if err := f.Truncate(total); err != nil {
		f.Close()
		return err
	}
	f.Close()

	var todo []int64
	var done int64
	for i := int64(0); i < nblocks; i++ {
		if have[i] == 1 {
			done += dlBlock
		} else {
			todo = append(todo, i)
		}
	}
	if done > total {
		done = total
	}
	var mu sync.Mutex
	var firstErr error
	next := 0
	takeBlock := func() (int64, bool) {
		mu.Lock()
		defer mu.Unlock()
		if next >= len(todo) || firstErr != nil {
			return 0, false
		}
		i := todo[next]
		next++
		return i, true
	}
	workers := dlStreams
	if int64(workers) > nblocks {
		workers = int(nblocks)
	}
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			fh, err := os.OpenFile(tmp, os.O_RDWR, 0o644)
			if err != nil {
				mu.Lock()
				firstErr = err
				mu.Unlock()
				return
			}
			defer fh.Close()
			for {
				i, ok := takeBlock()
				if !ok {
					return
				}
				start := i * dlBlock
				end := start + dlBlock
				if end > total {
					end = total
				}
				var lastErr error
				for attempt := 0; attempt < 4; attempt++ {
					lastErr = func() error {
						req, _ := http.NewRequestWithContext(ctx, "GET", url, nil)
						req.Header.Set("Range", fmt.Sprintf("bytes=%d-%d", start, end-1))
						resp, err := pullClient.Do(req)
						if err != nil {
							return err
						}
						defer resp.Body.Close()
						if resp.StatusCode != 200 && resp.StatusCode != 206 {
							return fmt.Errorf("HTTP %d", resp.StatusCode)
						}
						off := start
						buf := make([]byte, 1<<20)
						for {
							n, err := resp.Body.Read(buf)
							if n > 0 {
								if _, werr := fh.WriteAt(buf[:n], off); werr != nil {
									return werr
								}
								off += int64(n)
								mu.Lock()
								done += int64(n)
								progress(done)
								mu.Unlock()
							}
							if err == io.EOF {
								return nil
							}
							if err != nil {
								return err
							}
						}
					}()
					if lastErr == nil {
						break
					}
					time.Sleep(time.Duration(1500*(attempt+1)) * time.Millisecond)
				}
				if lastErr != nil {
					mu.Lock()
					firstErr = lastErr
					mu.Unlock()
					return
				}
				mu.Lock()
				have[i] = 1
				os.WriteFile(idxPath, have, 0o644)
				mu.Unlock()
			}
		}()
	}
	wg.Wait()
	if firstErr != nil {
		return firstErr
	}
	os.Remove(idxPath)
	return nil
}

func hfPull(ctx context.Context, repo, quant string, emit func(map[string]any)) (string, bool) {
	destDir := reg.LooseDir()
	os.MkdirAll(destDir, 0o755)
	emit(map[string]any{"status": fmt.Sprintf("looking up %s on Hugging Face", repo)})
	files, err := hfFiles(ctx, repo)
	if err != nil {
		emit(errorObj(err.Error()))
		return "", false
	}
	if len(files) == 0 {
		emit(errorObj("no GGUF files in " + repo))
		return "", false
	}
	want := pickGGUF(files, quant)
	if len(want) == 0 {
		emit(errorObj(fmt.Sprintf("no %s build in %s", quant, repo)))
		return "", false
	}
	type job struct {
		f      hfFile
		saveAs string
	}
	var jobs []job
	var wantBytes int64
	for _, f := range want {
		jobs = append(jobs, job{f, filepath.Base(f.Name)})
		wantBytes += f.Size
	}
	if len(want) > 1 {
		emit(map[string]any{"status": fmt.Sprintf("taking the %s build, %s split over %d files",
			quant, humanBytes(wantBytes), len(want))})
	} else {
		emit(map[string]any{"status": fmt.Sprintf("taking the %s build, %s", quant, humanBytes(wantBytes))})
	}
	if proj := pickMmproj(files); proj != nil {
		stem := shardSuffix.ReplaceAllString(stemOf(want[0].Name), "")
		jobs = append(jobs, job{*proj, stem + ".mmproj.gguf"})
		emit(map[string]any{"status": "this model can read images, so its vision projector comes too"})
	}
	for _, j := range jobs {
		total := j.f.Size
		dest := filepath.Join(destDir, j.saveAs)
		if st, err := os.Stat(dest); err == nil && (total == 0 || st.Size() == total) {
			emit(map[string]any{"status": "already have " + j.saveAs, "total": total, "completed": total})
			continue
		}
		tmp := dest + ".part"
		url := hfBase + "/" + repo + "/resolve/main/" + j.f.Name
		if total == 0 {
			req, _ := http.NewRequestWithContext(ctx, "HEAD", url, nil)
			if resp, err := pullClient.Do(req); err == nil {
				resp.Body.Close()
				total, _ = strconv.ParseInt(resp.Header.Get("Content-Length"), 10, 64)
			}
		}
		if total == 0 {
			emit(errorObj("could not determine the size of " + j.f.Name))
			return "", false
		}
		idxPath := tmp + ".idx"
		if st, err := os.Stat(tmp); err == nil && !fileExists(idxPath) {
			prefix := st.Size()
			nblocks := (total + dlBlock - 1) / dlBlock
			if nblocks < 1 {
				nblocks = 1
			}
			have := make([]byte, nblocks)
			for i := int64(0); i < prefix/dlBlock; i++ {
				have[i] = 1
			}
			os.WriteFile(idxPath, have, 0o644)
			emit(map[string]any{"status": fmt.Sprintf("resuming %s from %.1f GB", j.saveAs, float64(prefix)/(1<<30))})
		}
		var mu sync.Mutex
		var seen int64
		doneCh := make(chan error, 1)
		go func() {
			doneCh <- fetchBlocks(ctx, url, tmp, total, func(n int64) {
				mu.Lock()
				seen = n
				mu.Unlock()
			})
		}()
	wait:
		for {
			select {
			case err := <-doneCh:
				if err != nil {
					emit(errorObj(fmt.Sprintf("%T: %v", err, err)))
					return "", false
				}
				break wait
			case <-time.After(250 * time.Millisecond):
				mu.Lock()
				n := seen
				mu.Unlock()
				if n > total {
					n = total
				}
				emit(map[string]any{"status": "pulling " + j.saveAs, "total": total, "completed": n})
			}
		}
		os.Rename(tmp, dest)
		emit(map[string]any{"status": "pulling " + j.saveAs, "total": total, "completed": total})
	}
	reg.Invalidate()
	cliInvalidate()
	first := filepath.Join(destDir, filepath.Base(want[0].Name))
	emit(map[string]any{"status": "checking the download reads as a model"})
	meta := readGGUFMeta(first)
	arch := metaStr(meta, "general.architecture")
	if arch == "" {
		arch = "?"
	}
	pulled := int64(0)
	for _, j := range jobs {
		if st, err := os.Stat(filepath.Join(destDir, j.saveAs)); err == nil {
			pulled += st.Size()
		}
	}
	emit(map[string]any{"status": fmt.Sprintf("%s is ready: %s of %s weights in %s",
		reg.looseName(first), humanBytes(pulled), arch, destDir)})
	return first, true
}

// probeGGUFHeader reads the first few MB of a remote GGUF: enough for every
// key that precedes the tokenizer, which is what tells one build from another.
func probeGGUFHeader(ctx context.Context, url string) map[string]any {
	req, _ := http.NewRequestWithContext(ctx, "GET", url, nil)
	req.Header.Set("Range", "bytes=0-4194303")
	resp, err := pullClient.Do(req)
	if err != nil {
		return map[string]any{}
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 && resp.StatusCode != 206 {
		return map[string]any{}
	}
	head, _ := io.ReadAll(io.LimitReader(resp.Body, 4<<20))
	return ggufMetaFrom(bufio.NewReader(bytes.NewReader(head)))
}

// unloadableBuild names what makes a registry build unreadable for llama.cpp,
// or returns "" for a plain text-model GGUF. Ollama packs the vision and audio
// encoders into the language model's file; llama.cpp keeps them in a separate
// projector and rejects the combined file for its tensor count.
func unloadableBuild(meta map[string]any) string {
	arch := metaStr(meta, "general.architecture")
	if arch == "" {
		return ""
	}
	if arch == "mllama" {
		return "is packaged for Ollama's own runtime"
	}
	vision, audio := false, false
	for k := range meta {
		if strings.HasPrefix(k, arch+".vision.") {
			vision = true
		}
		if strings.HasPrefix(k, arch+".audio.") {
			audio = true
		}
	}
	switch {
	case vision && audio:
		return "bundles its vision and audio encoders into the one file"
	case vision:
		return "bundles its vision encoder into the one file"
	case audio:
		return "bundles its audio encoder into the one file"
	}
	return ""
}

var ggufFileTypes = map[int64]string{
	0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1", 10: "Q2_K",
	11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S", 15: "Q4_K_M", 16: "Q5_K_S",
	17: "Q5_K_M", 18: "Q6_K", 19: "IQ2_XXS", 20: "IQ2_XS", 21: "Q2_K_S", 22: "IQ3_XS",
	23: "IQ3_XXS", 24: "IQ1_S", 25: "IQ4_NL", 26: "IQ3_S", 27: "IQ3_M", 28: "IQ2_S",
	29: "IQ2_M", 30: "IQ4_XS", 31: "IQ1_M", 32: "BF16", 38: "MXFP4",
}

func quantOfFileType(meta map[string]any) string {
	if ft, ok := metaInt(meta, "general.file_type"); ok {
		if q, ok := ggufFileTypes[ft]; ok {
			return q
		}
	}
	return "Q4_K_M"
}

var sizeTagJunk = regexp.MustCompile(`(?i)^(i?q\d|f16|bf16|f32|fp16|mxfp4|instruct|it|chat|text|latest)`)

var familyDigits = regexp.MustCompile(`([a-zA-Z])(\d)`)

// hfEquivalent finds the Hugging Face GGUF repository that carries the same
// model as a registry reference: same family, same size, from a quantiser
// whose files are known to load.
func hfEquivalent(repo, tag string, meta map[string]any) (string, bool) {
	family := repo
	if i := strings.LastIndex(family, "/"); i >= 0 {
		family = family[i+1:]
	}
	var sizeParts []string
	for _, part := range strings.Split(tag, "-") {
		if part == "" || sizeTagJunk.MatchString(part) {
			continue
		}
		sizeParts = append(sizeParts, part)
	}
	size := normalise(strings.Join(sizeParts, ""))
	if size == "" {
		size = normalise(metaStr(meta, "general.size_label"))
	}
	if size == "" {
		// a bare family name matches every size the family comes in
		return "", false
	}
	spaced := familyDigits.ReplaceAllString(family, "$1-$2")
	sizeText := strings.Join(sizeParts, "-")
	queries := []string{family + " " + sizeText, spaced + " " + sizeText, spaced}
	orgRank := map[string]int{"ggml-org": 5, "unsloth": 4, "bartowski": 3, "lmstudio-community": 2}
	markers := []string{"abliterat", "uncensored", "heretic", "distill", "merge", "roleplay", "mobile", "caption"}
	seen := map[string]bool{}
	best, bestScore := "", -1
	for _, q := range queries {
		for _, hit := range hubSearch(q, 40) {
			if seen[hit.ID] || !strings.Contains(hit.ID, "/") {
				continue
			}
			seen[hit.ID] = true
			low := strings.ToLower(hit.ID)
			n := normalise(hit.ID)
			if !strings.Contains(low, "gguf") || !strings.Contains(n, normalise(family)) || !strings.Contains(n, size) {
				continue
			}
			foreign := false
			for _, m := range markers {
				if strings.Contains(low, m) {
					foreign = true
				}
			}
			if foreign {
				continue
			}
			score := orgRank[low[:strings.Index(low, "/")]] * 100
			if strings.Contains(low, "-it") || strings.Contains(low, "instruct") {
				score += 30
			}
			if strings.Contains(low, "qat") {
				score -= 20
			}
			score += int(math.Log10(float64(hit.Downloads)+1)) * 3
			if score > bestScore {
				best, bestScore = hit.ID, score
			}
		}
	}
	return best, best != ""
}

func registryPull(ctx context.Context, ref string, emit func(map[string]any)) {
	host, repo, tag := splitRef(ref)
	base := "https://" + host + "/v2/" + repo
	emit(map[string]any{"status": fmt.Sprintf("looking up %s on %s", repo, host)})
	req, _ := http.NewRequestWithContext(ctx, "GET", base+"/manifests/"+tag, nil)
	req.Header.Set("Accept", "application/vnd.docker.distribution.manifest.v2+json")
	resp, err := pullClient.Do(req)
	if err != nil {
		emit(errorObj(fmt.Sprintf("%T: %v", err, err)))
		return
	}
	raw, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if resp.StatusCode != 200 {
		emit(errorObj(fmt.Sprintf("manifest %d for %s", resp.StatusCode, ref)))
		return
	}
	var manifest struct {
		Layers []struct {
			MediaType string `json:"mediaType"`
			Digest    string `json:"digest"`
			Size      int64  `json:"size"`
		} `json:"layers"`
		Config *struct {
			MediaType string `json:"mediaType"`
			Digest    string `json:"digest"`
			Size      int64  `json:"size"`
		} `json:"config"`
	}
	if json.Unmarshal(raw, &manifest) != nil {
		emit(errorObj("bad manifest for " + ref))
		return
	}
	name := repo + ":" + tag
	if host == ollamaRegistry {
		name = strings.TrimPrefix(name, "library/")
	}
	mfPath := filepath.Join(reg.Manifests, host, filepath.FromSlash(repo), tag)
	for _, layer := range manifest.Layers {
		if !strings.HasSuffix(layer.MediaType, ".model") {
			continue
		}
		meta := probeGGUFHeader(ctx, base+"/blobs/"+layer.Digest)
		why := unloadableBuild(meta)
		if why == "" {
			break
		}
		emit(map[string]any{"status": "the registry build of " + name + " " + why + ", which llama.cpp does not load"})
		hfRepo, ok := hfEquivalent(repo, tag, meta)
		if !ok {
			emit(errorObj("no HuggingFace build of " + name + " was found to take instead; pull one directly with `llmash pull hf:<org>/<repo>`"))
			return
		}
		quant := quantOfFileType(meta)
		emit(map[string]any{"status": fmt.Sprintf("taking %s from Hugging Face instead, as %s", hfRepo, name)})
		first, ok := hfPull(ctx, hfRepo, quant, emit)
		if !ok {
			return
		}
		if err := reg.setAlias(filepath.Base(first), name); err != nil {
			emit(errorObj("could not record the name: " + err.Error()))
			return
		}
		os.Remove(mfPath)
		reg.Invalidate()
		cliInvalidate()
		emit(map[string]any{"status": name + " now serves the HuggingFace build"})
		return
	}
	layers := manifest.Layers
	if manifest.Config != nil {
		layers = append(layers, *manifest.Config)
	}
	for _, layer := range layers {
		total := layer.Size
		dest := reg.blob(layer.Digest)
		short := short12(layer.Digest)
		if st, err := os.Stat(dest); err == nil && st.Size() == total {
			emit(map[string]any{"status": "pulling " + short, "digest": layer.Digest, "total": total, "completed": total})
			continue
		}
		os.MkdirAll(filepath.Dir(dest), 0o755)
		tmp := dest + ".partial"
		url := base + "/blobs/" + layer.Digest
		if err := fetchBlob(ctx, url, tmp, total, func(done int64) {
			emit(map[string]any{"status": "pulling " + short, "digest": layer.Digest, "total": total, "completed": done})
		}); err != nil {
			emit(errorObj(fmt.Sprintf("%s: %v", short, err)))
			return
		}
		os.Rename(tmp, dest)
		emit(map[string]any{"status": "pulling " + short, "digest": layer.Digest, "total": total, "completed": total})
	}
	os.MkdirAll(filepath.Dir(mfPath), 0o755)
	os.WriteFile(mfPath, raw, 0o644)
	reg.Invalidate()
	cliInvalidate()
	var pulled int64
	for _, l := range manifest.Layers {
		pulled += l.Size
	}
	emit(map[string]any{"status": fmt.Sprintf("%s is ready, %s in %s", ref, humanBytes(pulled), reg.Blobs)})
}

var hfPrefixes = []string{"hf:", "hf.co/", "huggingface.co/"}

func apiPull(w http.ResponseWriter, r *http.Request) {
	body, _ := readBody(r)
	ref := first(str(body, "model"), str(body, "name"))
	n := newNDJSON(w)
	emit := func(ev map[string]any) { n.send(ev) }
	quant := str(body, "quant")
	for _, p := range hfPrefixes {
		if strings.HasPrefix(ref, p) {
			spec := ref[len(p):]
			repo, q, _ := strings.Cut(spec, "@")
			if quant != "" {
				q = quant
			}
			if q == "" {
				q = "Q4_K_M"
			}
			hfPull(r.Context(), repo, q, emit)
			return
		}
	}
	fromOllama, _ := body["from_ollama"].(bool)
	if rep, ok := hfReplacements[ref]; ok && !fromOllama {
		hfPull(r.Context(), rep[0], first(quant, rep[1]), emit)
		return
	}
	registryPull(r.Context(), ref, emit)
}

// apiQuants lists the GGUF builds a Hugging Face repository offers, by
// quantisation, with the size of each, so a pull can choose one.
func apiQuants(w http.ResponseWriter, r *http.Request) {
	repo := strings.TrimSpace(r.URL.Query().Get("repo"))
	for _, p := range hfPrefixes {
		repo = strings.TrimPrefix(repo, p)
	}
	repo, _, _ = strings.Cut(repo, "@")
	files, err := hfFiles(r.Context(), repo)
	if err != nil {
		writeJSON(w, 502, errorObj(err.Error()))
		return
	}
	writeJSON(w, 200, map[string]any{"repo": repo, "quants": quantsOf(files)})
}

type quantInfo struct {
	Name  string `json:"name"`
	Size  int64  `json:"size"`
	Files int    `json:"files"`
}

// quantsOf groups a repository's GGUF files by quantisation. Shards count
// towards one build; projectors are left out.
func quantsOf(files []hfFile) []quantInfo {
	byQuant := map[string]*quantInfo{}
	var order []string
	for _, f := range files {
		low := strings.ToLower(f.Name)
		if !strings.HasSuffix(low, ".gguf") || strings.Contains(low, "mmproj") || kindOf(f.Name) != nil {
			continue // a projector or a draft head is not a build of the model
		}
		q := quantTag(f.Name)
		if q == "" {
			continue
		}
		qi, ok := byQuant[q]
		if !ok {
			qi = &quantInfo{Name: q}
			byQuant[q] = qi
			order = append(order, q)
		}
		qi.Size += f.Size
		qi.Files++
	}
	out := make([]quantInfo, 0, len(order))
	for _, q := range order {
		out = append(out, *byQuant[q])
	}
	sort.SliceStable(out, func(i, j int) bool { return out[i].Size < out[j].Size })
	return out
}

var quantTagRe = regexp.MustCompile(`(?i)(?:^|[-_.])((?:UD-)?(?:IQ|Q|TQ)[1-8](?:_[0-9A-Z]+)*|BF16|F16|F32|MXFP4(?:_MOE)?|NVFP4)(?:[-_.]|$)`)

// quantTag pulls the quantisation out of a GGUF file name.
func quantTag(name string) string {
	m := quantTagRe.FindStringSubmatch(name)
	if m == nil {
		return ""
	}
	return strings.ToUpper(m[1])
}
