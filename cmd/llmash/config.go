package main

import (
	"encoding/json"
	"fmt"
	"log"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"
)

// Everything the server reads from the environment and from local.json, the
// file that holds one machine's settings and is never shipped. A fresh
// install has an empty LOCAL and behaves like plain Ollama on llama.cpp.

const (
	serverVersion = "0.3.4"
	serverBuild   = "llmash"
)

var local map[string]any

func loadLocal() {
	local = map[string]any{}
	b, err := os.ReadFile(filepath.Join(root, "local.json"))
	if err != nil {
		return
	}
	if err := json.Unmarshal(b, &local); err != nil {
		fmt.Fprintf(os.Stderr, "[llmash] bad local.json (%v); ignoring it\n", err)
		local = map[string]any{}
	}
	if v := str(local, "models_root"); v != "" && os.Getenv("OLLAMA_MODELS") == "" {
		os.Setenv("OLLAMA_MODELS", v)
	}
	if v := str(local, "gguf_dir"); v != "" && env("LLMASH_GGUF") == "" {
		os.Setenv("LLMASH_GGUF", v)
	}
	if env("LLMASH_EXTRA_ROOTS") == "" {
		var extras []string
		for _, e := range list(local, "extra_roots") {
			if s, _ := e.(string); strings.TrimSpace(s) != "" {
				extras = append(extras, strings.TrimSpace(s))
			}
		}
		if len(extras) > 0 {
			os.Setenv("LLMASH_EXTRA_ROOTS", strings.Join(extras, ";"))
		}
	}
}

func localStrings(key string) []string {
	var out []string
	for _, v := range list(local, key) {
		out = append(out, fmt.Sprint(v))
	}
	return out
}

func localMap(key string) map[string]any { return sub(local, key) }

// The project used to be called llamash, so LLAMASH_* is still read.
func env(name string) string {
	if v := os.Getenv(name); v != "" {
		return v
	}
	if legacy, ok := strings.CutPrefix(name, "LLMASH_"); ok {
		return os.Getenv("LLAMASH_" + legacy)
	}
	return ""
}

func envStr(name, def string) string {
	if v := env(name); v != "" {
		return v
	}
	return def
}

func envFloat(name string, def float64) float64 {
	if v := env(name); v != "" {
		if f, err := strconv.ParseFloat(v, 64); err == nil {
			return f
		}
	}
	return def
}

func csv(s string) []string {
	var out []string
	for _, x := range strings.Split(s, ",") {
		if x = strings.TrimSpace(x); x != "" {
			out = append(out, strings.ToLower(x))
		}
	}
	return out
}

// ---------------------------------------------------------------- knobs

var (
	llamaBin        string
	logDir          string
	vramBudgetGB    float64
	kvType          string
	loadMode        string
	defaultCtx      int
	defaultKeep     string
	pinned          map[string]bool
	ramFloorGB      float64
	vramHeadroomGB  float64
	busyGrace       float64
	nParallel       int
	mtpDraft        int
	dsparkDraft     int
	dsparkDraftMin  int
	lowlatPredict   int
	thinkOffDefault []string
	noMmproj        []string
	mmprojOnDemand  bool
	specFallback    string
	v1Ctx           int
	injectOn        bool
	thinkBudget     int
	nudgeAfterS     float64
	ctxOverride     map[string]int
	ctxMax          map[string]int
	launchExtras    map[string][]string
	modelSampling   map[string]map[string]any
	noInject        []string
	remoteContainer string
	remoteMaxCtx    int
	remoteIdle      float64
	remoteBootWait  float64
	remoteLoadGuess float64
	tokenizerJSON   string
	genCountEvery   float64
)

const ctxTrainNative = 262144

var badArch = map[string]bool{"gptoss": true, "glm4moelite": true}

var (
	brokenMu sync.Mutex
	broken   = map[string]bool{"gpt-oss:120b": true, "glm-4.7-flash:latest": true}
)

var samplingKeys = []string{"temperature", "temp", "top_k", "top_p", "min_p", "typical_p",
	"top_a", "presence_penalty", "frequency_penalty", "repeat_penalty",
	"repeat_last_n", "tfs_z", "mirostat", "mirostat_tau", "mirostat_eta"}

func loadConfig() {
	loadLocal()
	llamaBin = findLlamaBin()
	logDir = filepath.Join(root, "logs")
	vramBudgetGB = envFloat("LLMASH_VRAM_GB", 80)
	kvType = envStr("LLMASH_KV", "f16")
	loadMode = envStr("LLMASH_LOAD_MODE", "dio")
	defaultCtx = envInt("LLMASH_CTX", 8192)
	defaultKeep = envStr("OLLAMA_KEEP_ALIVE", "15m")
	pinned = map[string]bool{}
	pinSrc := env("LLMASH_PIN")
	if pinSrc == "" {
		pinSrc = strings.Join(localStrings("pin"), ",")
	}
	for _, p := range strings.Split(pinSrc, ",") {
		if p = strings.TrimSpace(p); p != "" {
			pinned[p] = true
		}
	}
	ramFloorGB = envFloat("LLMASH_RAM_FLOOR", 12)
	vramHeadroomGB = envFloat("LLMASH_VRAM_HEADROOM", 6)
	busyGrace = envFloat("LLMASH_BUSY_GRACE", 120)
	// Four slots split the KV cache four ways and cost about 6% of decode
	// speed even when only one is in use, measured on Qwen3.6-35B-A3B at
	// 318 tok/s on one slot against 300 on four. One personal server rarely
	// serves four conversations at once, so one is the default.
	nParallel = envInt("LLMASH_PARALLEL", 1)
	mtpDraft = envInt("LLMASH_MTP_DRAFT", 4)
	dsparkDraft = envInt("LLMASH_DSPARK_DRAFT", 6)
	dsparkDraftMin = envInt("LLMASH_DSPARK_DRAFT_MIN", 6)
	lowlatPredict = envInt("LLMASH_LOWLAT_PREDICT", 8)
	thinkOffDefault = csv(envStr("LLMASH_THINK_OFF", "gemma4,gemma-4"))
	if v, ok := os.LookupEnv("LLMASH_NO_MMPROJ"); ok {
		noMmproj = csv(v)
	} else {
		noMmproj = csv(strings.Join(localStrings("no_mmproj"), ","))
	}
	mmprojOnDemand = envStr("LLMASH_MMPROJ_ON_DEMAND", "1") == "1"
	// Upstream's own default speculative config; it drafts from the text
	// already in the context, so every model gets it with no draft model.
	specFallback = envStr("LLMASH_SPEC_FALLBACK", "ngram-mod")
	v1Ctx = envInt("LLMASH_V1_CTX", 32768)
	injectOn = env("LLMASH_INJECT") == "1"
	thinkBudget = envInt("LLMASH_THINK_BUDGET", 32000)
	nudgeAfterS = envFloat("LLMASH_NUDGE_S", 15)
	publicPort = envInt("LLMASH_PUBLIC_PORT", 11435)
	ctxOverride = map[string]int{}
	for k, v := range localMap("ctx_override") {
		ctxOverride[strings.ToLower(k)] = int(toFloat(v))
	}
	ctxMax = map[string]int{}
	for k, v := range localMap("ctx_max") {
		ctxMax[strings.ToLower(k)] = int(toFloat(v))
	}
	launchExtras = map[string][]string{}
	for k, v := range localMap("launch_extra") {
		var xs []string
		if arr, ok := v.([]any); ok {
			for _, x := range arr {
				xs = append(xs, fmt.Sprint(x))
			}
		}
		launchExtras[strings.ToLower(k)] = xs
	}
	modelSampling = map[string]map[string]any{}
	for k, v := range localMap("sampling") {
		if mm, ok := v.(map[string]any); ok {
			modelSampling[strings.ToLower(k)] = mm
		}
	}
	noInject = localStrings("no_inject")
	for i := range noInject {
		noInject[i] = strings.ToLower(noInject[i])
	}
	remoteContainer = env("LLMASH_REMOTE_CONTAINER")
	remoteMaxCtx = envInt("LLMASH_REMOTE_MAX_CONTEXT", 131072)
	remoteIdle = envFloat("LLMASH_REMOTE_IDLE", 900)
	remoteBootWait = envFloat("LLMASH_REMOTE_BOOT_WAIT", 45)
	remoteLoadGuess = envFloat("LLMASH_REMOTE_LOAD_GUESS", 20)
	tokenizerJSON = envStr("LLMASH_TOKENIZER_JSON", str(local, "tokenizer_json"))
	genCountEvery = envFloat("LLMASH_GEN_COUNT_INTERVAL", 0.07)
}

func toFloat(v any) float64 {
	switch t := v.(type) {
	case float64:
		return t
	case int:
		return float64(t)
	case string:
		f, _ := strconv.ParseFloat(t, 64)
		return f
	}
	return 0
}

// findLlamaBin: LLAMA_BIN, local.json, the install's runtime folder, PATH.
func findLlamaBin() string {
	if env := os.Getenv("LLAMA_BIN"); env != "" {
		return env
	}
	if v := str(local, "llama_bin"); v != "" {
		return v
	}
	exe := "llama-server.exe"
	cands := []string{}
	for _, v := range []string{"ProgramData", "LOCALAPPDATA"} {
		if base := os.Getenv(v); base != "" {
			cands = append(cands, filepath.Join(base, "llmash", "runtime", exe))
		}
	}
	cands = append(cands, filepath.Join(root, "runtime", exe), filepath.Join(root, "llama.cpp", exe))
	for _, c := range cands {
		if fileExists(c) {
			return c
		}
	}
	if p, err := exec.LookPath(exe); err == nil {
		return p
	}
	return filepath.Join(root, "runtime", exe)
}

func matchKey[T any](table map[string]T, name string) (T, bool) {
	low := strings.ToLower(name)
	for k, v := range table {
		if strings.Contains(low, k) {
			return v, true
		}
	}
	var zero T
	return zero, false
}

func samplingOverride(name string) map[string]any {
	v, _ := matchKey(modelSampling, name)
	return v
}

func skipInject(name string) bool {
	low := strings.ToLower(name)
	for _, k := range noInject {
		if strings.Contains(low, k) {
			return true
		}
	}
	return false
}

func ctxTarget(name string) int {
	v, _ := matchKey(ctxOverride, name)
	return v
}

func ctxCeiling(name string, native int) int {
	if v, ok := matchKey(ctxMax, name); ok && v > native {
		return v
	}
	return native
}

func launchExtra(name string) []string {
	low := strings.ToLower(name)
	var out []string
	for k, v := range launchExtras {
		if strings.Contains(low, k) {
			out = append(out, v...)
		}
	}
	return out
}

func advertisedCtx(m *Model) int {
	native := m.Ctx
	if native == 0 {
		native = defaultCtx
	}
	ctx := ctxTarget(m.Name)
	if ctx == 0 {
		ctx = ctxCeiling(m.Name, native)
	}
	if route := fastRoute(m.Name); route != nil {
		mc := int(num(route, "max_context"))
		if mc == 0 {
			mc = remoteMaxCtx
		}
		if mc < ctx {
			ctx = mc
		}
	}
	return ctx
}

func isPinned(name string) bool { return pinned[name] }

func loadable(m *Model) bool {
	brokenMu.Lock()
	b := broken[m.Name]
	brokenMu.Unlock()
	if b {
		return false
	}
	return !badArch[strings.ToLower(m.Family)]
}

func markBroken(name string) {
	brokenMu.Lock()
	broken[name] = true
	brokenMu.Unlock()
}

// ------------------------------------------------------------------ logging

var logMu sync.Mutex

func logf(format string, a ...any) {
	logMu.Lock()
	defer logMu.Unlock()
	log.Printf("[llmash] "+format, a...)
}

func iso(ts float64) string {
	us := int64(math.Round(ts * 1e6)) // Python rounds to the microsecond
	return time.Unix(0, us*1000).UTC().Format("2006-01-02T15:04:05.999999+00:00")
}

func nowF() float64 { return float64(time.Now().UnixNano()) / 1e9 }
