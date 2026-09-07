package main

import (
	"context"
	"crypto/subtle"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"time"
)

var reg *Registry

type ctxKey int

const publicPortKey ctxKey = 1

func isPublic(r *http.Request) bool {
	v, _ := r.Context().Value(publicPortKey).(bool)
	return v
}

// ------------------------------------------------------------ helpers

func writeJSON(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	json.NewEncoder(w).Encode(v)
}

func writeText(w http.ResponseWriter, code int, s string) {
	w.Header().Set("Content-Type", "text/plain; charset=utf-8")
	w.WriteHeader(code)
	io.WriteString(w, s)
}

func readBody(r *http.Request) (map[string]any, error) {
	b, err := io.ReadAll(io.LimitReader(r.Body, 512<<20))
	if err != nil {
		return nil, err
	}
	d := map[string]any{}
	if len(strings.TrimSpace(string(b))) == 0 {
		return d, nil
	}
	return d, json.Unmarshal(b, &d)
}

// ndjson streams one JSON object per line and flushes each.
type ndjson struct {
	w http.ResponseWriter
	f http.Flusher
}

func newNDJSON(w http.ResponseWriter) *ndjson {
	w.Header().Set("Content-Type", "application/x-ndjson")
	w.WriteHeader(200)
	f, _ := w.(http.Flusher)
	return &ndjson{w, f}
}

func (n *ndjson) send(v any) {
	b, _ := json.Marshal(v)
	n.w.Write(append(b, '\n'))
	if n.f != nil {
		n.f.Flush()
	}
}

func errorObj(msg string) map[string]any { return map[string]any{"error": msg} }

var linkKeyVal string

func serverLinkKey() string {
	if k := env("LLMASH_LINK_KEY"); k != "" {
		return k
	}
	return linkKey()
}

func guarded(public bool, next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Ollama ships permissive CORS on its local API and browser clients rely on it.
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "*")
		w.Header().Set("Access-Control-Allow-Headers", "*")
		if r.Method == "OPTIONS" {
			w.WriteHeader(200)
			return
		}
		r = r.WithContext(context.WithValue(r.Context(), publicPortKey, public))
		if public {
			auth := r.Header.Get("Authorization")
			key := r.Header.Get("X-API-Key")
			if len(auth) > 7 && strings.EqualFold(auth[:7], "bearer ") {
				key = strings.TrimSpace(auth[7:])
			}
			if linkKeyVal == "" || subtle.ConstantTimeCompare([]byte(key), []byte(linkKeyVal)) != 1 {
				writeJSON(w, 401, errorObj("unauthorized: missing or invalid API key"))
				return
			}
		}
		next.ServeHTTP(w, r)
	})
}

// -------------------------------------------------------------- routes

func buildMux() *http.ServeMux {
	mux := http.NewServeMux()
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/" {
			writeJSON(w, 404, errorObj("not found"))
			return
		}
		writeText(w, 200, "Ollama is running")
	})
	mux.HandleFunc("/api/version", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, 200, map[string]any{"version": serverVersion, "build": serverBuild})
	})
	mux.HandleFunc("/cli/list", func(w http.ResponseWriter, r *http.Request) { writeText(w, 200, cliCached("list")) })
	mux.HandleFunc("/cli/ps", func(w http.ResponseWriter, r *http.Request) { writeText(w, 200, cliCached("ps")) })
	mux.HandleFunc("/api/paths", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, 200, map[string]any{"root": root, "loose_dir": reg.LooseDir(), "models": reg.Root})
	})
	mux.HandleFunc("/api/create", apiCreate)
	mux.HandleFunc("/api/copy", apiCopy)
	mux.HandleFunc("/api/keep_alive", apiKeepAlive)
	mux.HandleFunc("/api/tags", func(w http.ResponseWriter, r *http.Request) { writeJSON(w, 200, tags()) })
	mux.HandleFunc("/v1/models", v1Models)
	mux.HandleFunc("/v1/models/", v1Model)
	mux.HandleFunc("/v1/chat/completions", func(w http.ResponseWriter, r *http.Request) { v1Proxy("/v1/chat/completions", w, r) })
	mux.HandleFunc("/v1/completions", func(w http.ResponseWriter, r *http.Request) { v1Proxy("/v1/completions", w, r) })
	mux.HandleFunc("/api/ps", func(w http.ResponseWriter, r *http.Request) { writeJSON(w, 200, ps()) })
	mux.HandleFunc("/api/show", apiShow)
	mux.HandleFunc("/api/delete", apiDelete)
	mux.HandleFunc("/api/pull", apiPull)
	mux.HandleFunc("/api/quants", apiQuants)
	mux.HandleFunc("/api/resolve", apiResolve)
	mux.HandleFunc("/api/chat", apiChat)
	mux.HandleFunc("/api/generate", apiGenerate)
	mux.HandleFunc("/api/embed", apiEmbed)
	mux.HandleFunc("/api/embeddings", apiEmbed)
	mux.HandleFunc("/api/loading", apiLoading)
	return mux
}

// ------------------------------------------------------------ listing

func tagEntry(m *Model) map[string]any {
	fams := []string{}
	if m.Family != "" {
		fams = []string{m.Family}
	}
	caps := m.Caps
	if len(caps) == 0 {
		caps = []string{"completion"}
	}
	return map[string]any{
		"name": m.Name, "model": m.Name,
		"modified_at": iso(m.Modified),
		"size":        m.Size,
		"digest":      strings.TrimPrefix(m.Digest, "sha256:"),
		"details": map[string]any{
			"parent_model": "", "format": "gguf",
			"family": m.Family, "families": fams,
			"parameter_size": m.ParamSize, "quantization_level": m.Quant,
			"context_length": advertisedCtx(m),
			"expert_count":   m.Experts, "expert_used_count": m.ExpertsUsed,
		},
		"capabilities": caps,
		"loadable":     loadable(m),
	}
}

func tags() map[string]any {
	rows := []map[string]any{}
	for _, m := range reg.All(true) {
		if fileExists(m.GGUF) && loadable(m) {
			rows = append(rows, tagEntry(m))
		}
	}
	return map[string]any{"models": rows}
}

func v1Entry(m *Model) map[string]any {
	ctx := advertisedCtx(m)
	return map[string]any{"id": m.Name, "object": "model", "created": int64(m.Modified),
		"owned_by": "llmash", "context_length": ctx, "max_model_len": ctx,
		"max_context_length": ctx, "context_window": ctx}
}

func v1Models(w http.ResponseWriter, r *http.Request) {
	data := []map[string]any{}
	for _, m := range reg.All(true) {
		if fileExists(m.GGUF) && loadable(m) {
			data = append(data, v1Entry(m))
		}
	}
	writeJSON(w, 200, map[string]any{"object": "list", "data": data})
}

func v1Model(w http.ResponseWriter, r *http.Request) {
	name := strings.TrimPrefix(r.URL.Path, "/v1/models/")
	m := reg.Get(name)
	if m == nil || !fileExists(m.GGUF) {
		writeJSON(w, 404, map[string]any{"error": map[string]any{"message": fmt.Sprintf("model '%s' not found", name),
			"type": "invalid_request_error", "code": "model_not_found"}})
		return
	}
	writeJSON(w, 200, v1Entry(m))
}

func ps() map[string]any {
	out := []map[string]any{}
	seen := map[string]bool{}
	for _, in := range mgr.Loaded() {
		e := tagEntry(in.Model)
		e["size"] = int64(in.VRAMGB() * (1 << 30))
		// only what actually went to the card counts as resident there
		if in.OnGPU() {
			e["size_vram"] = e["size"]
		} else {
			e["size_vram"] = int64(0)
		}
		_, exp, _, _ := in.Snapshot()
		if exp == exp+1 || exp > 1e15 { // +Inf
			e["expires_at"] = "9999-12-31T23:59:59Z"
		} else {
			e["expires_at"] = iso(exp)
		}
		e["context_length"] = in.Ctx
		seen[in.Model.Name] = true
		out = append(out, e)
	}
	for _, route := range fastRoutes {
		container := routeKey(route)
		if !routeUp(route) {
			continue
		}
		match := strings.ToLower(str(route, "match"))
		for _, m := range reg.All(false) {
			if !strings.Contains(strings.ToLower(m.Name), match) || seen[m.Name] {
				continue
			}
			e := tagEntry(m)
			e["size_vram"] = m.Size
			if exp := remoteExpiry(container); exp > 1e15 {
				e["expires_at"] = "9999-12-31T23:59:59Z"
			} else {
				e["expires_at"] = iso(exp)
			}
			mc := int(num(route, "max_context"))
			if mc == 0 {
				mc = remoteMaxCtx
			}
			e["context_length"] = mc
			seen[m.Name] = true
			out = append(out, e)
		}
	}
	return map[string]any{"models": out}
}

func apiShow(w http.ResponseWriter, r *http.Request) {
	body, _ := readBody(r)
	name := str(body, "model")
	if name == "" {
		name = str(body, "name")
	}
	m := reg.Get(name)
	if m == nil {
		writeJSON(w, 404, errorObj("model not found"))
		return
	}
	arch := m.Family
	if arch == "" {
		arch = "llama"
	}
	var params []string
	for k, v := range m.Params {
		params = append(params, fmt.Sprintf("%s %v", k, v))
	}
	caps := m.Caps
	if len(caps) == 0 {
		caps = []string{"completion"}
	}
	writeJSON(w, 200, map[string]any{
		"license": "", "modelfile": "FROM " + m.GGUF,
		"parameters": strings.Join(params, "\n"),
		"template":   m.Template, "system": m.System,
		"details": tagEntry(m)["details"],
		"model_info": map[string]any{
			"general.architecture":    arch,
			arch + ".context_length":  advertisedCtx(m),
			"context_length":          advertisedCtx(m),
			"general.parameter_count": m.ParamSize,
		},
		"capabilities": caps,
	})
}

func apiDelete(w http.ResponseWriter, r *http.Request) {
	body, _ := readBody(r)
	name := str(body, "model")
	if name == "" {
		name = str(body, "name")
	}
	m := reg.Get(name)
	if m == nil {
		writeJSON(w, 404, errorObj("model not found"))
		return
	}
	mgr.Unload(m.Name)
	var removed []string
	for _, mf := range reg.manifestFiles() {
		if reg.nameOf(mf) == m.Name {
			os.Remove(mf)
			removed = append(removed, filepath.Base(mf))
		}
	}
	loose, _ := filepath.Abs(reg.LooseDir())
	gguf, _ := filepath.Abs(m.GGUF)
	if fileExists(gguf) && strings.EqualFold(filepath.Dir(gguf), loose) {
		stem := shardSuffix.ReplaceAllString(stemOf(gguf), "")
		parts, _ := filepath.Glob(filepath.Join(loose, stem+"*.gguf"))
		for _, part := range parts {
			if err := os.Remove(part); err != nil {
				writeJSON(w, 500, errorObj(fmt.Sprintf("could not delete %s: %v", filepath.Base(part), err)))
				return
			}
			removed = append(removed, filepath.Base(part))
		}
	}
	reg.Invalidate()
	cliInvalidate()
	if len(removed) == 0 {
		writeJSON(w, 409, errorObj(fmt.Sprintf("found %s but nothing to delete. Its file is at %s, outside the model directory", m.Name, m.GGUF)))
		return
	}
	writeJSON(w, 200, map[string]any{"status": "success", "removed": removed})
}

func apiKeepAlive(w http.ResponseWriter, r *http.Request) {
	body, _ := readBody(r)
	name := str(body, "model")
	kaRaw, ok := body["keep_alive"]
	if !ok {
		kaRaw = defaultKeep
	}
	ka := parseKeepAlive(kaRaw)

	in := mgr.Find(name)
	if in == nil {
		// A model on a fast backend is a whole server, not an instance:
		// the same timers act on that backend's idle clock.
		if route := fastRoute(name); route != nil && routeUp(route) {
			container := routeKey(route)
			if ka == 0 {
				stopRemoteFor(name)
				writeJSON(w, 200, map[string]any{"model": name, "done_reason": "unload"})
				return
			}
			setRemoteKeep(container, ka)
			var expOut any
			if exp := remoteExpiry(container); exp < 1e15 {
				expOut = iso(exp)
			}
			writeJSON(w, 200, map[string]any{"model": name, "keep_alive": kaOut(ka), "expires_at": expOut})
			return
		}
		writeJSON(w, 404, errorObj(fmt.Sprintf("model '%s' is not loaded", name)))
		return
	}
	if ka == 0 {
		mgr.Unload(in.Model.Name)
		writeJSON(w, 200, map[string]any{"model": in.Model.Name, "done_reason": "unload"})
		return
	}
	in.SetKeepAlive(ka)
	_, exp, _, _ := in.Snapshot()
	var expOut any
	if exp < 1e15 {
		expOut = iso(exp)
	}
	writeJSON(w, 200, map[string]any{"model": in.Model.Name, "keep_alive": kaOut(ka), "expires_at": expOut})
}

// JSON has no infinity: a pinned model reports ollama's -1.
func kaOut(ka float64) float64 {
	if math.IsInf(ka, 1) {
		return -1
	}
	return ka
}

func apiLoading(w http.ResponseWriter, r *http.Request) {
	out := []map[string]any{}
	for name, in := range mgr.Live() {
		if in.Ready() {
			continue
		}
		last, _, _, _ := in.Snapshot()
		out = append(out, map[string]any{"name": name, "pct": float64(int(in.Progress()*1000)) / 10,
			"size": in.Model.Size, "ctx": in.Ctx, "elapsed": float64(int((nowF()-last)*10)) / 10,
			"backend": "llama.cpp"})
	}
	for _, container := range loadingContainers() {
		pct, size, elapsed, ok := remoteLoadPct(container)
		if !ok {
			continue
		}
		for _, name := range remoteLoadNames(container) {
			out = append(out, map[string]any{"name": name, "pct": pct, "size": size, "ctx": remoteMaxCtx,
				"elapsed": elapsed, "backend": "remote"})
		}
	}
	writeJSON(w, 200, map[string]any{"loading": out})
}

// ------------------------------------------------------- create / copy

var unsafeModelName = regexp.MustCompile(`[^A-Za-z0-9._-]`)

func safeModelName(name string) string {
	return unsafeModelName.ReplaceAllString(strings.SplitN(name, ":", 2)[0], "-")
}

func quantizeGGUF(src, dest, level string) error {
	qexe := filepath.Join(filepath.Dir(llamaBin), "llama-quantize.exe")
	if !fileExists(qexe) {
		return fmt.Errorf("llama-quantize.exe not found next to llama-server")
	}
	cmd := quiet(exec.Command(qexe, "--allow-requantize", src, dest, strings.ToUpper(level)))
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("llama-quantize failed (%v)", err)
	}
	return nil
}

func copyFile(src, dst string) error {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()
	out, err := os.Create(dst)
	if err != nil {
		return err
	}
	if _, err := io.Copy(out, in); err != nil {
		out.Close()
		return err
	}
	if err := out.Close(); err != nil {
		return err
	}
	if st, err := os.Stat(src); err == nil {
		os.Chtimes(dst, st.ModTime(), st.ModTime())
	}
	return nil
}

func apiCreate(w http.ResponseWriter, r *http.Request) {
	body, _ := readBody(r)
	name := first(str(body, "model"), str(body, "name"))
	from := str(body, "from")
	quant := str(body, "quantize")
	draft := str(body, "draft_quantize")
	n := newNDJSON(w)
	src := from
	if !strings.EqualFold(filepath.Ext(src), ".gguf") || !fileExists(src) {
		n.send(errorObj(fmt.Sprintf("'%s' isn't a local .gguf, so there's nothing to import.", from)))
		return
	}
	destDir := reg.LooseDir()
	os.MkdirAll(destDir, 0o755)
	safe := safeModelName(name)
	dest := filepath.Join(destDir, safe+".gguf")
	if fileExists(dest) {
		n.send(errorObj(fmt.Sprintf("'%s' already exists in %s. Remove it first (rm) or pick another name.", filepath.Base(dest), destDir)))
		return
	}
	st, _ := os.Stat(src)
	if quant != "" {
		n.send(map[string]any{"status": fmt.Sprintf("quantizing %s -> %s (%s) ...", filepath.Base(src), filepath.Base(dest), strings.ToUpper(quant))})
		if err := quantizeGGUF(src, dest, quant); err != nil {
			n.send(errorObj(err.Error()))
			return
		}
	} else {
		n.send(map[string]any{"status": fmt.Sprintf("importing %s -> %s (%s) ...", filepath.Base(src), dest, human(float64(st.Size())))})
		if err := copyFile(src, dest); err != nil {
			n.send(errorObj(err.Error()))
			return
		}
	}
	if draft != "" {
		d := filepath.Join(destDir, safe+".draft.gguf")
		n.send(map[string]any{"status": fmt.Sprintf("quantizing %s -> %s (%s) ...", filepath.Base(src), filepath.Base(d), strings.ToUpper(draft))})
		if err := quantizeGGUF(src, d, draft); err != nil {
			n.send(errorObj(err.Error()))
			return
		}
	}
	if mm := findProjector(src, nil); mm != "" {
		copyFile(mm, filepath.Join(destDir, safe+".mmproj.gguf"))
		n.send(map[string]any{"status": fmt.Sprintf("also imported its projector (%s)", filepath.Base(mm))})
	}
	reg.Invalidate()
	cliInvalidate()
	n.send(map[string]any{"status": fmt.Sprintf("created '%s'  (llmash serves it as %s)", name, reg.looseName(dest))})
	n.send(map[string]any{"status": "success"})
}

func apiCopy(w http.ResponseWriter, r *http.Request) {
	body, _ := readBody(r)
	source := str(body, "source")
	destination := str(body, "destination")
	n := newNDJSON(w)
	m := reg.Get(source)
	if m == nil {
		n.send(errorObj(source + ": model not found"))
		return
	}
	if !fileExists(m.GGUF) || !strings.EqualFold(filepath.Ext(m.GGUF), ".gguf") {
		n.send(errorObj(fmt.Sprintf("can't copy '%s': its backing file isn't a GGUF llmash can duplicate.", source)))
		return
	}
	destDir := reg.LooseDir()
	os.MkdirAll(destDir, 0o755)
	dest := filepath.Join(destDir, safeModelName(destination)+".gguf")
	if fileExists(dest) {
		n.send(errorObj(fmt.Sprintf("'%s' already exists in %s. Remove it first (rm) or pick another name.", filepath.Base(dest), destDir)))
		return
	}
	st, _ := os.Stat(m.GGUF)
	n.send(map[string]any{"status": fmt.Sprintf("copying %s -> %s (%s) ...", filepath.Base(m.GGUF), filepath.Base(dest), human(float64(st.Size())))})
	if err := copyFile(m.GGUF, dest); err != nil {
		n.send(errorObj(err.Error()))
		return
	}
	reg.Invalidate()
	cliInvalidate()
	n.send(map[string]any{"status": fmt.Sprintf("copied '%s' to '%s'", source, destination)})
	n.send(map[string]any{"status": "success"})
}

// ---------------------------------------------------------- cli cache

var (
	cliMu   sync.Mutex
	cliText = map[string]struct {
		at   time.Time
		text string
	}{}
)

func cliInvalidate() {
	cliMu.Lock()
	cliText = map[string]struct {
		at   time.Time
		text string
	}{}
	cliMu.Unlock()
}

func cliCacheWrite(name, text string) {
	dir := filepath.Join(root, "cache")
	os.MkdirAll(dir, 0o755)
	tmp := filepath.Join(dir, name+".tmp")
	if os.WriteFile(tmp, []byte(text), 0o644) == nil {
		os.Rename(tmp, filepath.Join(dir, name))
	}
}

func cliBuild(kind string) string {
	var text string
	if kind == "list" {
		text = renderList(rowsOf(tags()))
	} else {
		text = renderPs(rowsOf(ps()))
	}
	cliMu.Lock()
	cliText[kind] = struct {
		at   time.Time
		text string
	}{time.Now(), text}
	cliMu.Unlock()
	cliCacheWrite(kind+".txt", text)
	return text
}

func rowsOf(d map[string]any) []map[string]any {
	var out []map[string]any
	for _, m := range d["models"].([]map[string]any) {
		out = append(out, m)
	}
	return out
}

func cliCached(kind string) string {
	cliMu.Lock()
	hit, ok := cliText[kind]
	cliMu.Unlock()
	if ok && time.Since(hit.at) < time.Second {
		return hit.text
	}
	return cliBuild(kind)
}

func dirStamp() string {
	var sb strings.Builder
	for _, d := range []string{reg.Manifests, reg.Blobs, reg.LooseDir()} {
		if st, err := os.Stat(d); err == nil {
			fmt.Fprintf(&sb, "%d;", st.ModTime().UnixNano())
		} else {
			sb.WriteString("0;")
		}
	}
	return sb.String()
}

func cliCacheTick(ctx context.Context) {
	lastStamp := ""
	for {
		func() {
			defer func() {
				if e := recover(); e != nil {
					logf("cli cache: %v", e)
				}
			}()
			dir := filepath.Join(root, "cache")
			os.MkdirAll(dir, 0o755)
			alive := filepath.Join(dir, "alive")
			if f, err := os.OpenFile(alive, os.O_CREATE|os.O_WRONLY, 0o644); err == nil {
				f.Close()
			}
			now := time.Now()
			os.Chtimes(alive, now, now)
			stamp := dirStamp()
			cliMu.Lock()
			_, haveList := cliText["list"]
			cliMu.Unlock()
			if stamp != lastStamp || !haveList {
				cliBuild("list")
				lastStamp = stamp
			}
			cliBuild("ps")
		}()
		select {
		case <-ctx.Done():
			return
		case <-time.After(2 * time.Second):
		}
	}
}
