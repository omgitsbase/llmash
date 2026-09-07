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
var vramFree float64

// freeVRAMGB as the driver reports it, cached for two seconds.
func freeVRAMGB() float64 {
	vramMu.Lock()
	defer vramMu.Unlock()
	if time.Since(vramAt) < 2*time.Second {
		return vramFree
	}
	free := vramBudgetGB
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
					free = mb / 1024
				}
			}
		}
	case <-time.After(8 * time.Second):
		cmd.Process.Kill()
	}
	vramAt, vramFree = time.Now(), free
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
	a := []string{llamaBin, "-m", m.GGUF, "--host", "127.0.0.1", "--port", strconv.Itoa(in.Port),
		"-ngl", "999", "--load-mode", in.loadMode, "-c", strconv.Itoa(in.Ctx * nParallel),
		"--jinja", "--no-webui", "-fa", "on", "--cache-type-k", kvType, "--cache-type-v", kvType,
		"--parallel", strconv.Itoa(nParallel), "-bs"}
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
	case strings.Contains(low, "wrong number of tensors") || strings.Contains(low, "dimension_sections") ||
		strings.Contains(low, "unknown model architecture"):
		return "This is an Ollama-packaged build that bundles its vision or audio encoders into one file, which llama.cpp " +
			"does not load. Run `llmash pull` for this model again: it now takes the HuggingFace build instead."
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
	for time.Since(t0) < 600*time.Second {
		if !in.alive() {
			out := in.tailLog(24000)
			low := strings.ToLower(out)
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
				in.mu.Lock()
				in.ready = true
				in.mu.Unlock()
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
