package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
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

func hfPull(ctx context.Context, repo, quant string, emit func(map[string]any)) {
	destDir := reg.LooseDir()
	os.MkdirAll(destDir, 0o755)
	emit(map[string]any{"status": fmt.Sprintf("looking up %s on Hugging Face", repo)})
	files, err := hfFiles(ctx, repo)
	if err != nil {
		emit(errorObj(err.Error()))
		return
	}
	if len(files) == 0 {
		emit(errorObj("no GGUF files in " + repo))
		return
	}
	want := pickGGUF(files, quant)
	if len(want) == 0 {
		emit(errorObj(fmt.Sprintf("no %s build in %s", quant, repo)))
		return
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
			return
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
					return
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
			Digest string `json:"digest"`
			Size   int64  `json:"size"`
		} `json:"layers"`
		Config *struct {
			Digest string `json:"digest"`
			Size   int64  `json:"size"`
		} `json:"config"`
	}
	if json.Unmarshal(raw, &manifest) != nil {
		emit(errorObj("bad manifest for " + ref))
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
		req, _ := http.NewRequestWithContext(ctx, "GET", base+"/blobs/"+layer.Digest, nil)
		resp, err := pullClient.Do(req)
		if err != nil {
			emit(errorObj(fmt.Sprintf("%T: %v", err, err)))
			return
		}
		if resp.StatusCode != 200 {
			resp.Body.Close()
			emit(errorObj(fmt.Sprintf("blob %d for %s", resp.StatusCode, short)))
			return
		}
		fh, err := os.Create(tmp)
		if err != nil {
			resp.Body.Close()
			emit(errorObj(err.Error()))
			return
		}
		var done int64
		last := time.Time{}
		buf := make([]byte, 1<<20)
		for {
			n, err := resp.Body.Read(buf)
			if n > 0 {
				fh.Write(buf[:n])
				done += int64(n)
				if time.Since(last) > 200*time.Millisecond {
					last = time.Now()
					emit(map[string]any{"status": "pulling " + short, "digest": layer.Digest, "total": total, "completed": done})
				}
			}
			if err != nil {
				break
			}
		}
		fh.Close()
		resp.Body.Close()
		os.Rename(tmp, dest)
		emit(map[string]any{"status": "pulling " + short, "digest": layer.Digest, "total": total, "completed": total})
	}
	mfPath := filepath.Join(reg.Manifests, host, filepath.FromSlash(repo), tag)
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
	for _, p := range hfPrefixes {
		if strings.HasPrefix(ref, p) {
			spec := ref[len(p):]
			repo, q, _ := strings.Cut(spec, "@")
			if q == "" {
				q = "Q4_K_M"
			}
			hfPull(r.Context(), repo, q, emit)
			return
		}
	}
	fromOllama, _ := body["from_ollama"].(bool)
	if rep, ok := hfReplacements[ref]; ok && !fromOllama {
		hfPull(r.Context(), rep[0], rep[1], emit)
		return
	}
	registryPull(r.Context(), ref, emit)
}
