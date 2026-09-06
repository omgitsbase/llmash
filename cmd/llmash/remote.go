package main

import (
	"bufio"
	"context"
	"encoding/json"
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

// Fast-backend routing: a model with a faster OpenAI-compatible server (in a
// Docker container) is proxied there when it is up, else served by llama.cpp.
// Routes live in vllm_routes.json beside the program.

var fastRoutes []map[string]any

// A fast route names a model prefix and an OpenAI-compatible server that
// answers for it: either an executable llmash starts on demand, or a container
// it docker-starts. Without the file every model is served by llama.cpp.
func loadRoutes() {
	var b []byte
	var err error
	for _, name := range []string{"routes.json", "vllm_routes.json"} {
		if b, err = os.ReadFile(filepath.Join(root, name)); err == nil {
			break
		}
	}
	if err != nil {
		return
	}
	var arr []map[string]any
	if json.Unmarshal(b, &arr) == nil {
		fastRoutes = arr
	} else {
		var obj map[string]any
		if json.Unmarshal(b, &obj) == nil {
			for _, r := range list(obj, "routes") {
				if m, ok := r.(map[string]any); ok {
					fastRoutes = append(fastRoutes, m)
				}
			}
		} else {
			logf("bad routes.json (%v); using defaults", err)
		}
	}
	var names []string
	for _, r := range fastRoutes {
		names = append(names, str(r, "match"))
	}
	logf("fast routes: %v", names)
}

func fastRoute(name string) map[string]any {
	low := strings.ToLower(name)
	for _, r := range fastRoutes {
		if m := strings.ToLower(str(r, "match")); m != "" && strings.Contains(low, m) {
			return r
		}
	}
	return nil
}

func routeContainer(route map[string]any) string {
	if c := str(route, "container"); c != "" {
		return c
	}
	return remoteContainer
}

func routeURL(route map[string]any) string { return strings.TrimRight(str(route, "url"), "/") }

// ------------------------------------------------------- reachability

var (
	upMu    sync.Mutex
	upCache = map[string]struct {
		at time.Time
		ok bool
	}{}
	routeUpTTL        = 8 * time.Second
	routeDownTTL      = 120 * time.Second
	routeProbeTimeout = 350 * time.Millisecond
)

func upCacheDrop(url string) {
	upMu.Lock()
	delete(upCache, url)
	upMu.Unlock()
}

func routeUp(route map[string]any) bool {
	url := routeURL(route)
	upMu.Lock()
	c, ok := upCache[url]
	upMu.Unlock()
	if ok {
		ttl := routeDownTTL
		if c.ok {
			ttl = routeUpTTL
		}
		if time.Since(c.at) < ttl {
			return c.ok
		}
	}
	client := &http.Client{Timeout: routeProbeTimeout}
	good := false
	if resp, err := client.Get(url + "/models"); err == nil {
		resp.Body.Close()
		good = resp.StatusCode == 200
	}
	upMu.Lock()
	upCache[url] = struct {
		at time.Time
		ok bool
	}{time.Now(), good}
	upMu.Unlock()
	return good
}

// --------------------------------------------------------- container

func docker(args ...string) (int, string) {
	cmd := quiet(exec.Command("docker", args...))
	out, err := cmd.CombinedOutput()
	code := 0
	if e, ok := err.(*exec.ExitError); ok {
		code = e.ExitCode()
	} else if err != nil {
		code = 1
	}
	return code, strings.TrimSpace(string(out))
}

var (
	remoteMu       sync.Mutex
	remoteLastUsed = map[string]float64{}
	remoteKeep     = map[string]float64{} // container -> idle seconds set by hand (+Inf pins it)
	remoteLocks    = map[string]*sync.Mutex{}
	runningCache   = map[string]struct {
		at time.Time
		ok bool
	}{}
	runningTTL = 5 * time.Second
)

// How long this container may sit idle: its own keep_alive if one was set,
// otherwise the global idle timeout.
func remoteIdleFor(container string) float64 {
	remoteMu.Lock()
	defer remoteMu.Unlock()
	if v, ok := remoteKeep[container]; ok {
		return v
	}
	return remoteIdle
}

func setRemoteKeep(container string, ka float64) {
	remoteMu.Lock()
	remoteKeep[container] = ka
	remoteLastUsed[container] = nowF()
	remoteMu.Unlock()
}

// The expiry a stopped-clock container would show, +Inf when it is pinned.
func remoteExpiry(container string) float64 {
	idle := remoteIdleFor(container)
	remoteMu.Lock()
	last := remoteLastUsed[container]
	remoteMu.Unlock()
	if last == 0 {
		last = nowF()
	}
	if math.IsInf(idle, 1) || idle <= 0 {
		return math.Inf(1)
	}
	return last + idle
}

func remoteLock(container string) *sync.Mutex {
	remoteMu.Lock()
	defer remoteMu.Unlock()
	l := remoteLocks[container]
	if l == nil {
		l = &sync.Mutex{}
		remoteLocks[container] = l
	}
	return l
}

func remoteRunning(container string) bool {
	remoteMu.Lock()
	c, ok := runningCache[container]
	remoteMu.Unlock()
	if ok && time.Since(c.at) < runningTTL {
		return c.ok
	}
	rc, out := docker("inspect", "-f", "{{.State.Running}}", container)
	good := rc == 0 && strings.TrimSpace(out) == "true"
	remoteMu.Lock()
	runningCache[container] = struct {
		at time.Time
		ok bool
	}{time.Now(), good}
	remoteMu.Unlock()
	return good
}

func runningCacheDrop(container string) {
	remoteMu.Lock()
	delete(runningCache, container)
	remoteMu.Unlock()
}

var portRe = regexp.MustCompile(`:(\d+)`)

func remotePortPublished(container string, route map[string]any) bool {
	url := routeURL(route)
	after := url
	if i := strings.Index(url, "//"); i >= 0 {
		after = url[i+2:]
	}
	m := portRe.FindStringSubmatch(after)
	if m == nil {
		return true
	}
	for i := 0; i < 6; i++ {
		rc, out := docker("port", container)
		if rc == 0 && strings.Contains(out, ":"+m[1]) {
			return true
		}
		time.Sleep(500 * time.Millisecond)
	}
	return false
}

// ------------------------------------------------------ load progress

type remoteLoadState struct {
	t0, bytes, total, esec, at, doneAt float64
	expect                             float64
	cancel                             context.CancelFunc
}

var (
	remoteLoads  = map[string]*remoteLoadState{}
	remoteExpect = map[string]float64{}
	remoteNames  = map[string][]string{}
	remoteLoadRe = regexp.MustCompile(`load\s+(\w+)\s+([0-9.]+)%\s+([0-9.]+)\s*([KMGTP]?i?B)\s*/\s*([0-9.]+)\s*([KMGTP]?i?B)\s+([0-9.]+)\s*s`)
	remoteDoneRe = regexp.MustCompile(`model loaded in\s+([0-9.]+)\s*s`)
	remoteUnit   = map[string]float64{"B": 1, "KiB": 1024, "MiB": 1 << 20, "GiB": 1 << 30, "TiB": 1 << 40,
		"KB": 1000, "MB": 1e6, "GB": 1e9, "TB": 1e12}
)

func remoteStatePath() string { return filepath.Join(root, "remote_load.json") }

func remoteExpectLoad() {
	b, err := os.ReadFile(remoteStatePath())
	if err != nil {
		return
	}
	json.Unmarshal(b, &remoteExpect)
}

func remoteExpectSave() {
	if b, err := json.Marshal(remoteExpect); err == nil {
		os.WriteFile(remoteStatePath(), b, 0o644)
	}
}

func remoteTail(ctx context.Context, container string) {
	cmd := quiet(exec.CommandContext(ctx, "docker", "logs", "-f", "--tail", "0", container))
	out, err := cmd.StdoutPipe()
	if err != nil {
		return
	}
	cmd.Stderr = cmd.Stdout
	if err := cmd.Start(); err != nil {
		return
	}
	defer cmd.Wait()
	sc := bufio.NewScanner(out)
	sc.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	for sc.Scan() {
		line := sc.Text()
		remoteMu.Lock()
		st := remoteLoads[container]
		if st == nil {
			remoteMu.Unlock()
			return
		}
		if m := remoteLoadRe.FindStringSubmatch(line); m != nil {
			st.bytes = toFloat(m[3]) * unitOf(m[4])
			st.total = toFloat(m[5]) * unitOf(m[6])
			st.esec = toFloat(m[7])
			st.at = nowF()
			remoteMu.Unlock()
			continue
		}
		if d := remoteDoneRe.FindStringSubmatch(line); d != nil {
			remoteExpect[container] = toFloat(d[1])
			remoteExpectSave()
			st.doneAt = nowF()
			remoteMu.Unlock()
			return
		}
		remoteMu.Unlock()
	}
}

func unitOf(u string) float64 {
	if v, ok := remoteUnit[u]; ok {
		return v
	}
	return 1
}

func remoteLoadBegin(container string) {
	remoteMu.Lock()
	if old := remoteLoads[container]; old != nil && old.cancel != nil {
		old.cancel()
	}
	ctx, cancel := context.WithCancel(context.Background())
	remoteLoads[container] = &remoteLoadState{t0: nowF(), expect: remoteExpect[container], cancel: cancel}
	remoteMu.Unlock()
	go remoteTail(ctx, container)
}

func remoteLoadEnd(container string, ok bool) {
	remoteMu.Lock()
	st := remoteLoads[container]
	delete(remoteLoads, container)
	if st != nil && st.cancel != nil {
		st.cancel()
	}
	if ok && st != nil && st.doneAt == 0 {
		if _, have := remoteExpect[container]; !have {
			remoteExpect[container] = float64(int((nowF()-st.t0)*100)) / 100
			remoteExpectSave()
		}
	}
	remoteMu.Unlock()
}

func remoteLoadPct(container string) (pct float64, size int64, elapsed float64, ok bool) {
	remoteMu.Lock()
	defer remoteMu.Unlock()
	st := remoteLoads[container]
	if st == nil || st.doneAt != 0 {
		return 0, 0, 0, false
	}
	now := nowF()
	if st.bytes > 0 && st.total > 0 {
		rate := st.bytes / maxF(st.esec, 1e-3)
		est := st.bytes + rate*maxF(now-st.at, 0)
		pct = 100 * minF(est, st.total*0.995) / st.total
	} else {
		exp := st.expect
		if exp == 0 {
			exp = remoteLoadGuess
		}
		pct = minF(95, 100*(now-st.t0)/maxF(exp, 1))
	}
	pct = float64(int(maxF(0, minF(99.9, pct))*10)) / 10
	return pct, int64(st.total), float64(int((now-st.t0)*10)) / 10, true
}

func remoteLoadNames(container string) []string {
	remoteMu.Lock()
	if v, ok := remoteNames[container]; ok {
		remoteMu.Unlock()
		return v
	}
	remoteMu.Unlock()
	var out []string
	for _, m := range reg.All(false) {
		if r := fastRoute(m.Name); r != nil && routeKey(r) == container {
			out = append(out, m.Name)
		}
	}
	remoteMu.Lock()
	remoteNames[container] = out
	remoteMu.Unlock()
	return out
}

func loadingContainers() []string {
	remoteMu.Lock()
	defer remoteMu.Unlock()
	var out []string
	for c := range remoteLoads {
		out = append(out, c)
	}
	return out
}

func maxF(a, b float64) float64 {
	if a > b {
		return a
	}
	return b
}

func minF(a, b float64) float64 {
	if a < b {
		return a
	}
	return b
}

// ensureRemoteUp starts the container if stopped and waits for it to serve.
func ensureRemoteUp(route map[string]any) bool {
	key := routeKey(route)
	if key == "" {
		upCacheDrop(routeURL(route))
		return routeUp(route)
	}
	l := remoteLock(key)
	l.Lock()
	defer l.Unlock()
	remoteMu.Lock()
	remoteLastUsed[key] = nowF()
	remoteMu.Unlock()
	upCacheDrop(routeURL(route))
	if routeUp(route) {
		return true
	}
	if routeExe(route) != "" {
		return startExeRoute(route, key)
	}
	container := routeContainer(route)
	runningCacheDrop(container)
	rc, out := docker("start", container)
	if rc != 0 {
		if len(out) > 200 {
			out = out[:200]
		}
		logf("%s start failed: %s", container, out)
		return false
	}
	if !remotePortPublished(container, route) {
		logf("%s started without its port; restarting it", container)
		docker("stop", container)
		time.Sleep(2 * time.Second)
		rc, _ = docker("start", container)
		if rc != 0 || !remotePortPublished(container, route) {
			logf("%s will not publish its port; falling back", container)
			docker("stop", container)
			return false
		}
	}
	logf("%s starting", container)
	remoteLoadBegin(container)
	t0 := time.Now()
	for time.Since(t0).Seconds() < remoteBootWait {
		upCacheDrop(routeURL(route))
		if routeUp(route) {
			logf("%s ready in %.1fs", container, time.Since(t0).Seconds())
			remoteLoadEnd(container, true)
			return true
		}
		time.Sleep(time.Second)
	}
	remoteLoadEnd(container, false)
	logf("%s did not become ready; falling back to llama.cpp", container)
	return false
}

func stopRemoteFor(name string) bool {
	route := fastRoute(name)
	if route == nil {
		return false
	}
	key := routeKey(route)
	if key == "" {
		return false
	}
	if routeExe(route) != "" {
		if !exeRouteRunning(route, key) {
			return false
		}
		logf("stopping %s on request", key)
		stopExeRoute(route, key)
	} else {
		container := routeContainer(route)
		if !remoteRunning(container) {
			return false
		}
		logf("stopping %s on request", container)
		docker("stop", container)
		runningCacheDrop(container)
	}
	remoteMu.Lock()
	delete(remoteLastUsed, key)
	delete(remoteKeep, key)
	remoteMu.Unlock()
	upCacheDrop(routeURL(route))
	return true
}

func remoteReaper(ctx context.Context) {
	t := time.NewTicker(30 * time.Second)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
		}
		remoteMu.Lock()
		snap := map[string]float64{}
		for k, v := range remoteLastUsed {
			snap[k] = v
		}
		remoteMu.Unlock()
		for key, last := range snap {
			idle := remoteIdleFor(key)
			if last == 0 || idle <= 0 || math.IsInf(idle, 1) || nowF()-last < idle {
				continue
			}
			var route map[string]any
			for _, r := range fastRoutes {
				if routeKey(r) == key {
					route = r
					break
				}
			}
			switch {
			case route != nil && routeExe(route) != "":
				if exeRouteRunning(route, key) {
					logf("%s idle past %.0fs; stopping", key, idle)
					stopExeRoute(route, key)
					upCacheDrop(routeURL(route))
				}
			case remoteRunning(key):
				logf("%s idle past %.0fs; stopping", key, idle)
				docker("stop", key)
				runningCacheDrop(key)
				for _, r := range fastRoutes {
					if routeContainer(r) == key {
						upCacheDrop(routeURL(r))
					}
				}
			}
			remoteMu.Lock()
			remoteLastUsed[key] = 0
			delete(remoteKeep, key)
			remoteMu.Unlock()
		}
	}
}
