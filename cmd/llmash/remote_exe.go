package main

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"syscall"
	"time"
)

// A fast backend can be a plain executable instead of a Docker container: the
// route says `exe` and `args`, llmash starts it on the first routed request
// and stops it when it goes idle, exactly as it does for a container.

func routeExe(route map[string]any) string { return str(route, "exe") }

func routeArgs(route map[string]any) []string {
	var out []string
	for _, a := range list(route, "args") {
		out = append(out, fmt.Sprint(a))
	}
	return out
}

// Everything about a backend's lifetime is keyed by this: the container name,
// or the address an executable route serves.
func routeKey(route map[string]any) string {
	if routeExe(route) != "" {
		return "exe " + routeURL(route)
	}
	return routeContainer(route)
}

type exeProc struct {
	cmd  *exec.Cmd
	done chan struct{}
}

var remoteProcs = map[string]*exeProc{}

func (p *exeProc) alive() bool {
	select {
	case <-p.done:
		return false
	default:
		return true
	}
}

func trackedProc(key string) *exeProc {
	remoteMu.Lock()
	defer remoteMu.Unlock()
	return remoteProcs[key]
}

// startExeRoute launches the server and waits for its port, tailing stderr for
// load progress the way `docker logs` is tailed for a container.
func startExeRoute(route map[string]any, key string) bool {
	exe := routeExe(route)
	if !fileExists(exe) {
		logf("%s: %s is not there", key, exe)
		return false
	}
	cmd := exec.Command(exe, routeArgs(route)...)
	cmd.Dir = filepath.Dir(exe)
	if extra := str(route, "path"); extra != "" {
		cmd.Env = append(os.Environ(), "PATH="+extra+string(os.PathListSeparator)+os.Getenv("PATH"))
	}
	stderr, err := cmd.StderrPipe()
	if err != nil {
		logf("%s: %v", key, err)
		return false
	}
	cmd.SysProcAttr = &syscall.SysProcAttr{HideWindow: true, CreationFlags: createNoWindow}
	if err := cmd.Start(); err != nil {
		logf("%s start failed: %v", key, err)
		return false
	}
	proc := &exeProc{cmd: cmd, done: make(chan struct{})}
	remoteMu.Lock()
	remoteProcs[key] = proc
	remoteMu.Unlock()
	go func() {
		cmd.Wait()
		close(proc.done)
	}()

	logf("%s starting", key)
	remoteLoadBeginFrom(key, stderr)
	t0 := time.Now()
	for time.Since(t0).Seconds() < remoteBootWait {
		upCacheDrop(routeURL(route))
		if routeUp(route) {
			logf("%s ready in %.1fs", key, time.Since(t0).Seconds())
			remoteLoadEnd(key, true)
			return true
		}
		if !proc.alive() {
			logf("%s exited before it served (%s)", key, cmd.ProcessState)
			remoteLoadEnd(key, false)
			return false
		}
		time.Sleep(500 * time.Millisecond)
	}
	remoteLoadEnd(key, false)
	logf("%s did not become ready; falling back to llama.cpp", key)
	stopExeRoute(route, key)
	return false
}

func stopExeRoute(route map[string]any, key string) {
	remoteMu.Lock()
	proc := remoteProcs[key]
	delete(remoteProcs, key)
	remoteMu.Unlock()
	if proc != nil && proc.alive() {
		killTree(proc.cmd.Process.Pid)
		select {
		case <-proc.done:
		case <-time.After(10 * time.Second):
		}
		return
	}
	// Nothing of ours is running, but something may be holding the port from
	// an earlier llmash: take it down by the executable it was started from.
	if exe := routeExe(route); exe != "" && routeUp(route) {
		killByExe(exe)
	}
	upCacheDrop(routeURL(route))
}

func killTree(pid int) {
	quiet(exec.Command("taskkill", "/PID", fmt.Sprint(pid), "/T", "/F")).Run()
}

func killByExe(exe string) {
	abs, err := filepath.Abs(exe)
	if err != nil {
		abs = exe
	}
	script := "Get-CimInstance Win32_Process -Filter \"Name='" + filepath.Base(abs) + "'\" | " +
		"Where-Object { $_.ExecutablePath -eq '" + psQuote(abs) + "' } | " +
		"ForEach-Object { Stop-Process -Id $_.ProcessId -Force }"
	hiddenPowerShell(script, true)
}

// exeRouteRunning: ours is alive, or something already answers on that address.
func exeRouteRunning(route map[string]any, key string) bool {
	if p := trackedProc(key); p != nil && p.alive() {
		return true
	}
	return routeUp(route)
}

// Orphan sweep at startup: an executable backend left behind by a previous
// llmash holds VRAM that this one cannot account for.
func reapExeRoutes() {
	for _, route := range fastRoutes {
		exe := routeExe(route)
		if exe == "" {
			continue
		}
		if routeUp(route) {
			logf("stopping an orphaned %s from an earlier run", filepath.Base(exe))
			killByExe(exe)
			upCacheDrop(routeURL(route))
		}
	}
}

// remoteLoadBeginFrom is remoteLoadBegin for a backend whose log arrives on a
// pipe rather than from `docker logs`.
func remoteLoadBeginFrom(key string, r io.Reader) {
	remoteMu.Lock()
	if old := remoteLoads[key]; old != nil && old.cancel != nil {
		old.cancel()
	}
	remoteLoads[key] = &remoteLoadState{t0: nowF(), expect: remoteExpect[key]}
	remoteMu.Unlock()
	go scanLoadLog(key, r)
}

func scanLoadLog(key string, r io.Reader) {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		remoteMu.Lock()
		st := remoteLoads[key]
		if st == nil {
			remoteMu.Unlock()
			io.Copy(io.Discard, r) // keep the pipe drained so the child never blocks
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
			remoteExpect[key] = toFloat(d[1])
			remoteExpectSave()
			st.doneAt = nowF()
			remoteMu.Unlock()
			continue
		}
		remoteMu.Unlock()
	}
}
