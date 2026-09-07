package main

import (
	"context"
	"errors"
	"fmt"
	"math"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"
	"unsafe"
)

// One llama-server process per loaded model.

var (
	procGlobalMemoryStatusEx = kernel32.NewProc("GlobalMemoryStatusEx")
	procOpenProcess          = kernel32.NewProc("OpenProcess")
	procCloseHandle          = kernel32.NewProc("CloseHandle")
	procGetProcessIoCounters = kernel32.NewProc("GetProcessIoCounters")
	psapi                    = syscall.NewLazyDLL("psapi.dll")
	procGetProcessMemoryInfo = psapi.NewProc("GetProcessMemoryInfo")
)

const createNoWindow = 0x08000000

func quiet(cmd *exec.Cmd) *exec.Cmd {
	cmd.SysProcAttr = &syscall.SysProcAttr{HideWindow: true, CreationFlags: createNoWindow}
	return cmd
}

var vramMu sync.Mutex
var vramAt time.Time
// how long a load may make no progress at all before it is called stuck
const loadStallTimeout = 10 * time.Minute

var vramFree float64
var vramKnown bool // whether vramFree came from a real card

// freeVRAMGB as the driver reports it, cached for two seconds.
// Free VRAM, and whether that number means anything. A machine with no
// NVIDIA card answers nothing, and answering "the budget" there told every
// caller it had an 80 GB card: the context was never trimmed, the batch was
// widened for a big card, and the weights went to system RAM anyway. Unknown
// is its own answer now, and callers treat it as no GPU.
func freeVRAM() (float64, bool) {
	vramMu.Lock()
	defer vramMu.Unlock()
	if time.Since(vramAt) < 2*time.Second {
		return vramFree, vramKnown
	}
	free, known := 0.0, false
	if v := envFloat("LLMASH_VRAM_GB", 0); v > 0 {
		free, known = v, true // set by hand, believed
	} else {
		cmd := quiet(exec.Command("nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"))
		done := make(chan struct{})
		var out []byte
		var err error
		go func() { out, err = cmd.Output(); close(done) }()
		select {
		case <-done:
			if err == nil {
				if line := strings.TrimSpace(strings.SplitN(string(out), "\n", 2)[0]); line != "" {
					if mb, e := strconv.ParseFloat(line, 64); e == nil {
						free, known = mb/1024, true
					}
				}
			}
		case <-time.After(8 * time.Second):
			cmd.Process.Kill()
		}
	}
	vramAt, vramFree, vramKnown = time.Now(), free, known
	return free, known
}

// what the old callers wanted: a number, with unknown reading as no GPU
func freeVRAMGB() float64 {
	free, _ := freeVRAM()
	return free
}

func freeRAMGB() float64 {
	var ms struct {
		Length               uint32
		MemoryLoad           uint32
		TotalPhys            uint64
		AvailPhys            uint64
		TotalPageFile        uint64
		AvailPageFile        uint64
		TotalVirtual         uint64
		AvailVirtual         uint64
		AvailExtendedVirtual uint64
	}
	ms.Length = uint32(unsafe.Sizeof(ms))
	if r, _, _ := procGlobalMemoryStatusEx.Call(uintptr(unsafe.Pointer(&ms))); r == 0 {
		return 999
	}
	return float64(ms.AvailPhys) / (1 << 30)
}

func freePort() int {
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return 0
	}
	defer l.Close()
	return l.Addr().(*net.TCPAddr).Port
}

type Instance struct {
	Model     *Model
	Ctx       int
	Vision    bool
	Port      int
	cmd       *exec.Cmd
	exited    chan struct{}
	exitCode  int
	ready     bool
	onGPU     bool // the weights reached the card, per llama.cpp's own log
	plainArgs bool // this runtime rejected our optional flags
	lastUsed  float64
	expiresAt float64
	keepAlive float64
	loaded    chan struct{}
	err       string
	loadMode  string
	tuneNote  string
	specNote  string
	logfile   string
	logfh     *os.File
	mu        sync.Mutex
}

func newInstance(m *Model, ctx int, vision bool) *Instance {
	return &Instance{Model: m, Ctx: ctx, Vision: vision, Port: freePort(), lastUsed: nowF(),
		expiresAt: math.Inf(1), keepAlive: math.Inf(1), loaded: make(chan struct{}), loadMode: loadMode}
}

func (in *Instance) URL() string { return fmt.Sprintf("http://127.0.0.1:%d", in.Port) }

// Touch marks the instance in use now and pushes its expiry out; inf stays inf.
func (in *Instance) Touch() {
	in.mu.Lock()
	in.lastUsed = nowF()
	if math.IsInf(in.keepAlive, 1) {
		in.expiresAt = math.Inf(1)
	} else {
		in.expiresAt = in.lastUsed + in.keepAlive
	}
	in.mu.Unlock()
}

func (in *Instance) SetKeepAlive(ka float64) {
	in.mu.Lock()
	in.keepAlive = ka
	in.mu.Unlock()
	in.Touch()
}

func (in *Instance) Snapshot() (lastUsed, expiresAt, keepAlive float64, ready bool) {
	in.mu.Lock()
	defer in.mu.Unlock()
	return in.lastUsed, in.expiresAt, in.keepAlive, in.ready
}

func (in *Instance) Ready() bool {
	in.mu.Lock()
	defer in.mu.Unlock()
	return in.ready
}

func (in *Instance) VRAMGB() float64 { return float64(in.Model.Size) / (1 << 30) * 1.05 }

func (in *Instance) alive() bool {
	if in.cmd == nil {
		return false
	}
	select {
	case <-in.exited:
		return false
	default:
		return true
	}
}

func (in *Instance) tailLog(n int) string {
	if in.logfile == "" {
		return ""
	}
	b, err := os.ReadFile(in.logfile)
	if err != nil {
		return ""
	}
	if len(b) > n {
		b = b[len(b)-n:]
	}
	return string(b)
}

// Progress 0..1: bytes read (dio) or working set (mmap) against the file size.
func (in *Instance) Progress() float64 {
	if in.Ready() {
		return 1
	}
	if !in.alive() {
		return 0
	}
	h, _, _ := procOpenProcess.Call(0x0410, 0, uintptr(in.cmd.Process.Pid))
	if h == 0 {
		return 0
	}
	defer procCloseHandle.Call(h)
	var pmc struct {
		Cb                         uint32
		PageFaultCount             uint32
		PeakWorkingSetSize         uintptr
		WorkingSetSize             uintptr
		QuotaPeakPagedPoolUsage    uintptr
		QuotaPagedPoolUsage        uintptr
		QuotaPeakNonPagedPoolUsage uintptr
		QuotaNonPagedPoolUsage     uintptr
		PagefileUsage              uintptr
		PeakPagefileUsage          uintptr
	}
	pmc.Cb = uint32(unsafe.Sizeof(pmc))
	if r, _, _ := procGetProcessMemoryInfo.Call(h, uintptr(unsafe.Pointer(&pmc)), uintptr(pmc.Cb)); r == 0 {
		return 0
	}
	var io struct {
		ReadOperationCount, WriteOperationCount, OtherOperationCount uint64
		ReadTransferCount, WriteTransferCount, OtherTransferCount    uint64
	}
	var read uint64
	if r, _, _ := procGetProcessIoCounters.Call(h, uintptr(unsafe.Pointer(&io))); r != 0 {
		read = io.ReadTransferCount
	}
	total := float64(in.Model.Size)
	if total < 1 {
		total = 1
	}
	best := float64(read)
	if ws := float64(pmc.WorkingSetSize); ws > best {
		best = ws
	}
	return math.Max(0, math.Min(0.99, best/total))
}

func (in *Instance) args() []string {
	m := in.Model
	a := []string{llamaBin, "-m", m.GGUF, "--host", "127.0.0.1", "--port", strconv.Itoa(in.Port)}
	if _, gpu := freeVRAM(); gpu {
		a = append(a, "-ngl", "999")
	}
	// llama.cpp adjusts what it was not given (--fit, on by default), so on a
	// machine with no GPU it places the layers itself rather than being told
	// to offload 999 of them to nothing
	a = append(a, []string{"-c", strconv.Itoa(in.Ctx * nParallel),
		"--jinja", "--no-webui", "-fa", "on", "--cache-type-k", kvType, "--cache-type-v", kvType,
		"--parallel", strconv.Itoa(nParallel)}...)
	if !in.plainArgs {
		// only a build of ours is known to take these
		a = append(a, "--load-mode", in.loadMode, "-bs")
	}
	native := m.Ctx
	if native == 0 {
		native = defaultCtx
	}
	if in.Ctx > native {
		a = append(a, "--rope-scaling", "yarn", "--rope-scale", fmt.Sprintf("%.4f", float64(in.Ctx)/float64(native)),
			"--yarn-orig-ctx", strconv.Itoa(native))
	}
	extra := launchExtra(m.Name)
	if tuned, why := in.autoTune(); len(tuned) > 0 {
		tuned = dropOverridden(tuned, extra)
		a = append(a, tuned...)
		in.tuneNote = why
	}
	a = append(a, extra...)
	low := strings.ToLower(m.Name)
	blocked := false
	for _, k := range noMmproj {
		if strings.Contains(low, k) {
			blocked = true
		}
	}
	if m.Projector != "" && fileExists(m.Projector) && (in.Vision || !(blocked || mmprojOnDemand)) {
		a = append(a, "--mmproj", m.Projector)
	}
	// A model's own head is trained with its weights and wins: on
	// Qwen3.6-35B-A3B it ran 308 tok/s at 60% acceptance against a downloaded
	// DSpark head's 265 at 33%. Where a model has none, a downloaded head is
	// worth a great deal: gemma-4 26B-A4B ran 342 tok/s on a fetched EAGLE-3
	// against 244 on ngram self-speculation and 218 on none.
	switch {
	case hasMTP(m.GGUF):
		a = append(a, "--spec-type", "draft-mtp", "--spec-draft-n-max", strconv.Itoa(mtpDraft))
	case m.Mtp != "" && fileExists(m.Mtp):
		a = append(a, "--spec-type", "draft-mtp", "--model-draft", m.Mtp, "-ngld", "999",
			"--spec-draft-n-max", strconv.Itoa(mtpDraft))
		in.specNote = "mtp"
	case m.Eagle3 != "" && fileExists(m.Eagle3):
		a = append(a, "--spec-type", "draft-eagle3", "--model-draft", m.Eagle3, "-ngld", "999",
			"--spec-draft-n-max", strconv.Itoa(mtpDraft))
		in.specNote = "eagle3"
	case m.Dspark != "" && fileExists(m.Dspark):
		a = append(a, "--spec-type", "draft-dspark", "--model-draft", m.Dspark, "-ngld", "999",
			"--spec-draft-n-max", strconv.Itoa(dsparkDraft), "--spec-draft-n-min", strconv.Itoa(dsparkDraftMin))
	case m.Draft != "" && fileExists(m.Draft):
		a = append(a, "--spec-type", "draft-simple", "--model-draft", m.Draft, "-ngld", "999",
			"--spec-draft-n-max", strconv.Itoa(mtpDraft))
	case specFallback != "" && specFallback != "none":
		// No drafter of its own: llama.cpp can still speculate from the
		// text itself, which costs no model and no VRAM.
		a = append(a, "--spec-type", specFallback)
		if !strings.HasPrefix(specFallback, "ngram") {
			a = append(a, "--spec-draft-n-max", strconv.Itoa(mtpDraft))
		}
		in.specNote = specFallback
	}
	return a
}

func explainLoadFailure(raw string) string {
	low := strings.ToLower(raw)
	switch {
	case strings.Contains(low, "unknown model architecture") && !ownRuntime():
		return "This llama.cpp build does not know this model's architecture, which usually means the runtime is older " +
			"than the model. Run `llmash update -Runtime cuda` (or vulkan, or cpu) to replace the runtime in " +
			runtimeDir() + "."
	case strings.Contains(low, "wrong number of tensors") || strings.Contains(low, "check_tensor_dims") ||
		strings.Contains(low, "unknown model architecture"):
		return "llama.cpp cannot load this Ollama-packaged build: it does not carry the tensors llama.cpp expects for " +
			"this architecture, which happens when a model is packaged for Ollama's own fork. Run `llmash pull` for " +
			"this model again to take the Hugging Face build instead."
	case strings.Contains(low, "unable to allocate") || strings.Contains(low, "out of memory") || strings.Contains(low, "cudamalloc"):
		return "Not enough VRAM to load this model at the requested context size. Lower the context window, or unload whatever else is resident."
	case strings.Contains(low, "failed to fit") || strings.Contains(low, "common_fit_params"):
		return "llama.cpp couldn't fit this model in the available memory. Lower the context window and try again."
	case strings.Contains(low, "no such file") || strings.Contains(low, "failed to open"):
		return "The model file is missing from disk."
	}
	lines := strings.Split(strings.TrimSpace(raw), "\n")
	for i := len(lines) - 1; i >= 0; i-- {
		if l := strings.TrimSpace(lines[i]); l != "" {
			if len(l) > 300 {
				l = l[:300]
			}
			return l
		}
	}
	return "llama-server failed to start."
}

var unsafeName = regexp.MustCompile(`[^A-Za-z0-9._-]`)

func (in *Instance) start() error {
	logf("loading %s ctx=%d port=%d mode=%s", in.Model.Name, in.Ctx, in.Port, in.loadMode)
	os.MkdirAll(logDir, 0o755)
	in.logfile = filepath.Join(logDir, unsafeName.ReplaceAllString(in.Model.Name, "_")+".log")
	fh, err := os.Create(in.logfile)
	if err != nil {
		return err
	}
	in.logfh = fh
	argv := in.args()
	if in.tuneNote != "" || in.specNote != "" {
		logf("%s tuned: %s", in.Model.Name,
			strings.TrimPrefix(strings.Join(nonEmpty(in.tuneNote, specWhy(in.specNote)), ", "), ", "))
	}
	cmd := quiet(exec.Command(argv[0], argv[1:]...))
	cmd.Stdout, cmd.Stderr = fh, fh
	if err := cmd.Start(); err != nil {
		fh.Close()
		return err
	}
	in.cmd = cmd
	in.exited = make(chan struct{})
	go func() {
		err := cmd.Wait()
		if e, ok := err.(*exec.ExitError); ok {
			in.exitCode = e.ExitCode()
		}
		close(in.exited)
	}()
	t0 := time.Now()
	client := &http.Client{Timeout: 2 * time.Second}
	lastPct, lastMove := -1.0, time.Now()
	for {
		if pct := in.Progress(); pct > lastPct+0.001 {
			lastPct, lastMove = pct, time.Now()
		}
		// it has stopped reading and stopped growing: that is stuck, however
		// long the whole load has taken
		if time.Since(lastMove) > loadStallTimeout {
			break
		}
		if !in.alive() {
			out := in.tailLog(24000)
			low := strings.ToLower(out)
			if !in.plainArgs && (strings.Contains(low, "unknown argument") || strings.Contains(low, "invalid argument") ||
				strings.Contains(low, "unrecognized argument")) {
				logf("%s: this llama.cpp build rejected an argument, retrying without the optional ones", in.Model.Name)
				in.plainArgs = true
				fh.Close()
				return in.start()
			}
			if in.loadMode != "mmap" && (strings.Contains(low, "direct") || strings.Contains(low, "dio") || strings.Contains(low, "unsupported")) {
				logf("%s: DirectIO unavailable here, retrying with mmap", in.Model.Name)
				in.loadMode = "mmap"
				fh.Close()
				return in.start()
			}
			in.err = explainLoadFailure(out)
			if strings.Contains(in.err, "llama.cpp can't read") {
				markBroken(in.Model.Name)
			}
			logf("FAILED to load %s: %s", in.Model.Name, in.err)
			if len(out) > 1200 {
				out = out[len(out)-1200:]
			}
			logf("%s", out)
			in.markLoaded()
			return errors.New(in.err)
		}
		if resp, err := client.Get(in.URL() + "/health"); err == nil {
			resp.Body.Close()
			if resp.StatusCode == 200 {
				// llama.cpp warns and carries on when it finds no device to
				// offload to, so a load that succeeded is not a load that
				// used the GPU. Ask the log, once, rather than assume.
				low := strings.ToLower(in.tailLog(24000))
				onGPU := !strings.Contains(low, "no usable gpu") &&
					!strings.Contains(low, "failed to initialize cuda")
				in.mu.Lock()
				in.ready = true
				in.onGPU = onGPU
				in.mu.Unlock()
				if !onGPU {
					logf("%s: no usable GPU, the weights are in system RAM", in.Model.Name)
				}
				logf("ready %s in %.1fs", in.Model.Name, time.Since(t0).Seconds())
				in.markLoaded()
				return nil
			}
		}
		time.Sleep(250 * time.Millisecond)
	}
	in.stop()
	in.markLoaded()
	return errors.New("timed out waiting for llama-server")
}

func (in *Instance) markLoaded() {
	select {
	case <-in.loaded:
	default:
		close(in.loaded)
	}
}

func (in *Instance) stop() {
	if in.cmd != nil && in.alive() {
		in.cmd.Process.Kill()
		select {
		case <-in.exited:
		case <-time.After(10 * time.Second):
		}
	}
	if in.logfh != nil {
		in.logfh.Close()
		in.logfh = nil
	}
	in.mu.Lock()
	in.ready = false
	in.mu.Unlock()
}

// ----------------------------------------------------------------- manager

type Manager struct {
	mu   sync.Mutex
	live map[string]*Instance
	load sync.Mutex // held across a load, like the Python lock
}

var mgr = &Manager{live: map[string]*Instance{}}

func (mg *Manager) Loaded() []*Instance {
	mg.mu.Lock()
	defer mg.mu.Unlock()
	var out []*Instance
	for _, in := range mg.live {
		if in.Ready() {
			out = append(out, in)
		}
	}
	return out
}

func (mg *Manager) Live() map[string]*Instance {
	mg.mu.Lock()
	defer mg.mu.Unlock()
	out := map[string]*Instance{}
	for k, v := range mg.live {
		out[k] = v
	}
	return out
}

func (mg *Manager) evictFor(needGB float64, keep string) {
	loaded := mg.Loaded()
	used := 0.0
	for _, in := range loaded {
		if in.Model.Name != keep {
			used += in.VRAMGB()
		}
	}
	tight := freeRAMGB() < ramFloorGB
	if used+needGB <= vramBudgetGB && !tight {
		return
	}
	if tight {
		logf("system RAM down to %.1f GB, evicting to make room", freeRAMGB())
	}
	sort.Slice(loaded, func(i, j int) bool {
		a, _, _, _ := loaded[i].Snapshot()
		b, _, _, _ := loaded[j].Snapshot()
		return a < b
	})
	for _, in := range loaded {
		if in.Model.Name == keep || isPinned(in.Model.Name) {
			continue
		}
		last, _, _, _ := in.Snapshot()
		if nowF()-last < busyGrace {
			logf("not evicting %s: used %.0fs ago", in.Model.Name, nowF()-last)
			continue
		}
		logf("evicting %s to free %.1f GB", in.Model.Name, in.VRAMGB())
		in.stop()
		mg.mu.Lock()
		delete(mg.live, in.Model.Name)
		mg.mu.Unlock()
		used -= in.VRAMGB()
		if used+needGB <= vramBudgetGB && freeRAMGB() >= ramFloorGB {
			return
		}
	}
}

func (mg *Manager) dropDead() {
	mg.mu.Lock()
	defer mg.mu.Unlock()
	for name, in := range mg.live {
		if in.cmd != nil && !in.alive() {
			logf("%s died underneath us, dropping it", name)
			in.stop()
			delete(mg.live, name)
		}
	}
}

func (mg *Manager) fitCtx(m *Model, ctx int) int {
	weights := float64(m.Size) / (1 << 30)
	want := weights * (1 + float64(ctx*nParallel)/ctxTrainNative)
	freeV := freeVRAMGB()
	mg.mu.Lock()
	if cur := mg.live[m.Name]; cur != nil {
		freeV += cur.VRAMGB()
	}
	mg.mu.Unlock()
	room := math.Max(4, freeV-vramHeadroomGB)
	if want <= room {
		return ctx
	}
	step := 32768.0
	allowed := math.Max(0, room/math.Max(weights, 0.1)-1) * ctxTrainNative
	fitted := int(math.Max(8192, math.Floor(allowed/step)*step))
	if fitted < ctx {
		logf("ctx %d would need ~%.0f GB on the card with %.0f GB free; using %d instead", ctx, want, freeV, fitted)
	}
	if fitted < ctx {
		return fitted
	}
	return ctx
}

var errModelNotFound = errors.New("model not found")
var errModelMissing = errors.New("missing from disk")

// Get returns a ready instance for the model, loading or reloading as needed.
func (mg *Manager) Get(name string, ctx int, keepAlive any, vision bool) (*Instance, error) {
	mg.dropDead()
	m := reg.Get(name)
	if m == nil {
		return nil, errModelNotFound
	}
	if !fileExists(m.GGUF) {
		return nil, errModelMissing
	}
	if ctx == 0 {
		ctx = defaultCtx
	}
	mg.mu.Lock()
	cur := mg.live[m.Name]
	if cur != nil && vision && !cur.Vision && m.Projector != "" && fileExists(m.Projector) {
		logf("%s: media turn, reloading with projector", m.Name)
		cur.stop()
		delete(mg.live, m.Name)
		cur = nil
	}
	mg.mu.Unlock()
	if cur != nil && cur.Ready() && cur.Ctx >= ctx {
		cur.SetKeepAlive(parseKeepAlive(keepAlive))
		return cur, nil
	}

	native := m.Ctx
	if native == 0 {
		native = defaultCtx
	}
	forced := ctxTarget(name)
	ceiling := ctxCeiling(name, native)
	weights := float64(m.Size) / (1 << 30)
	var need float64
	switch {
	case forced > 0:
		ctx = forced
		need = weights*1.05 + weights*0.4*(float64(ctx)/ctxTrainNative)
	case ceiling > native:
		if ctx > ceiling {
			ctx = ceiling
		}
		ctx = mg.fitCtx(m, ctx)
		need = weights*1.05 + weights*0.4*(float64(ctx)/ctxTrainNative)
	default:
		if ctx > native {
			ctx = native
		}
		if ctx == 0 {
			ctx = defaultCtx
		}
		ctx = mg.fitCtx(m, ctx)
		need = weights * 1.05
	}

	mg.load.Lock()
	mg.mu.Lock()
	inst := mg.live[m.Name]
	if inst != nil && inst.Ctx < ctx {
		logf("reloading %s for a larger context (%d -> %d)", m.Name, inst.Ctx, ctx)
		inst.stop()
		delete(mg.live, m.Name)
		inst = nil
	}
	fresh := inst == nil
	if fresh {
		mg.mu.Unlock()
		mg.evictFor(need, m.Name)
		inst = newInstance(m, ctx, vision)
		mg.mu.Lock()
		mg.live[m.Name] = inst
	}
	mg.mu.Unlock()
	if fresh {
		if err := inst.start(); err != nil {
			inst.stop()
			mg.mu.Lock()
			if mg.live[m.Name] == inst {
				delete(mg.live, m.Name)
			}
			mg.mu.Unlock()
			mg.load.Unlock()
			return nil, err
		}
	}
	mg.load.Unlock()
	if !inst.Ready() {
		<-inst.loaded
		if !inst.Ready() {
			if inst.err != "" {
				return nil, errors.New(inst.err)
			}
			return nil, errors.New("model failed to load")
		}
	}
	inst.SetKeepAlive(parseKeepAlive(keepAlive))
	return inst, nil
}

func (mg *Manager) Unload(name string) bool {
	key := name
	if m := reg.Get(name); m != nil {
		key = m.Name
	}
	mg.mu.Lock()
	inst := mg.live[key]
	delete(mg.live, key)
	mg.mu.Unlock()
	if inst != nil {
		logf("unloading %s", key)
		inst.stop()
		return true
	}
	return false
}

func (mg *Manager) Find(name string) *Instance {
	mg.mu.Lock()
	defer mg.mu.Unlock()
	if in := mg.live[name]; in != nil {
		return in
	}
	if name == "" {
		return nil
	}
	for k, in := range mg.live {
		if strings.Contains(k, name) {
			return in
		}
	}
	return nil
}

func (mg *Manager) reaper(ctx context.Context) {
	t := time.NewTicker(5 * time.Second)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
		}
		now := nowF()
		for name, in := range mg.Live() {
			if isPinned(name) || isPinned(in.Model.Name) {
				continue
			}
			last, exp, _, ready := in.Snapshot()
			if !(ready && now > exp) {
				continue
			}
			if now-last < 5 {
				continue
			}
			logf("%s idle past keep_alive", name)
			in.stop()
			mg.mu.Lock()
			delete(mg.live, name)
			mg.mu.Unlock()
		}
	}
}

func (mg *Manager) Shutdown() {
	mg.mu.Lock()
	all := mg.live
	mg.live = map[string]*Instance{}
	mg.mu.Unlock()
	for _, in := range all {
		in.stop()
	}
}

// parseKeepAlive: Ollama accepts 5m / 1h / seconds / -1 (forever) / 0 (now).
func parseKeepAlive(v any) float64 {
	if v == nil {
		v = defaultKeep
	}
	switch t := v.(type) {
	case float64:
		if t < 0 {
			return math.Inf(1)
		}
		return t
	case int:
		if t < 0 {
			return math.Inf(1)
		}
		return float64(t)
	}
	s := strings.TrimSpace(fmt.Sprint(v))
	if s == "-1" || s == "-1s" {
		return math.Inf(1)
	}
	parse := func(x string) (float64, bool) {
		f, err := strconv.ParseFloat(x, 64)
		return f, err == nil
	}
	switch {
	case strings.HasSuffix(s, "ms"):
		if f, ok := parse(s[:len(s)-2]); ok {
			return f / 1000
		}
	case strings.HasSuffix(s, "s"):
		if f, ok := parse(s[:len(s)-1]); ok {
			return f
		}
	case strings.HasSuffix(s, "m"):
		if f, ok := parse(s[:len(s)-1]); ok {
			return f * 60
		}
	case strings.HasSuffix(s, "h"):
		if f, ok := parse(s[:len(s)-1]); ok {
			return f * 3600
		}
	default:
		if f, ok := parse(s); ok {
			if f < 0 {
				return math.Inf(1)
			}
			return f
		}
	}
	return 300
}

// reapOrphans kills llama-server processes left over from a previous run.
func reapOrphans() int {
	out, _ := hiddenPowerShell("Get-CimInstance Win32_Process -Filter \"Name='llama-server.exe'\" | ForEach-Object { $_.ProcessId }", true)
	killed := 0
	for _, f := range strings.Fields(out) {
		pid, err := strconv.Atoi(f)
		if err != nil {
			continue
		}
		if p, err := os.FindProcess(pid); err == nil && p.Kill() == nil {
			killed++
		}
	}
	if killed > 0 {
		logf("reaped %d orphaned llama-server process(es) from a previous run", killed)
	}
	return killed
}

// Whether this instance's weights actually reached the GPU.
func (in *Instance) OnGPU() bool {
	in.mu.Lock()
	defer in.mu.Unlock()
	return in.onGPU
}

// Whether the llama.cpp beside us is the build llmash ships, which is the
// only one known to carry its flags and to be as new as its models.
func ownRuntime() bool {
	if llamaBin == "" {
		return false
	}
	_, err := os.Stat(filepath.Join(filepath.Dir(llamaBin), "RUNTIME.txt"))
	return err == nil
}

// Where that runtime lives, for a message that tells the user what to replace.
func runtimeDir() string {
	if llamaBin == "" {
		return "the runtime folder"
	}
	return filepath.Dir(llamaBin)
}
