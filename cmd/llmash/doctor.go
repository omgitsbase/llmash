package main

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
	"unsafe"
)

// `llmash doctor`: everything that has to be true for a model to answer, in
// one pass, with the reason when it is not.

type check struct {
	state  int // 0 ok, 1 warn, 2 fail
	name   string
	detail string
}

const (
	stOK = iota
	stWarn
	stFail
)

type doctorReport struct{ rows []check }

func (d *doctorReport) add(state int, name, format string, a ...any) {
	d.rows = append(d.rows, check{state, name, fmt.Sprintf(format, a...)})
}

var procGetDiskFreeSpaceExW = kernel32.NewProc("GetDiskFreeSpaceExW")

func freeDiskGB(path string) float64 {
	var free, total, totalFree uint64
	r, _, _ := procGetDiskFreeSpaceExW.Call(uintptr(unsafe.Pointer(utf16(path))),
		uintptr(unsafe.Pointer(&free)), uintptr(unsafe.Pointer(&total)),
		uintptr(unsafe.Pointer(&totalFree)))
	if r == 0 {
		return 0
	}
	return float64(free) / (1 << 30)
}

func run(timeout time.Duration, name string, args ...string) (string, bool) {
	if which(name) == "" && !fileExists(name) {
		return "", false
	}
	cmd := quiet(exec.Command(name, args...))
	done := make(chan struct{})
	var out []byte
	var err error
	go func() { out, err = cmd.CombinedOutput(); close(done) }()
	select {
	case <-done:
	case <-time.After(timeout):
		if cmd.Process != nil {
			cmd.Process.Kill()
		}
		return "", false
	}
	return strings.TrimSpace(string(out)), err == nil
}

func cmdDoctor() {
	// Config and route loading log for the server's benefit; this is a report.
	log.SetOutput(io.Discard)
	loadConfig()
	loadRoutes()
	log.SetOutput(os.Stdout)
	d := &doctorReport{}

	// ---- this install ----------------------------------------------------
	d.add(stOK, "version", "%s %s", prog, versionString())
	origin, dev := "dev", true
	if b, err := readJSONFile(filepath.Join(root, "install.json")); err == nil {
		var info map[string]any
		if json.Unmarshal(b, &info) == nil {
			origin = first(str(info, "origin"), "dev")
			dev, _ = info["dev"].(bool)
		}
	}
	kind := "installed from " + origin
	if dev || origin == "dev" {
		kind = "source checkout"
	}
	d.add(stOK, "install", "%s  (%s)", root, kind)

	// ---- the server ------------------------------------------------------
	if v, _, err := callJSON("GET", "/api/version", nil, 3*time.Second); err == nil {
		d.add(stOK, "server", "answering at %s, version %s", host, str(v, "version"))
		if str(v, "version") != versionString() {
			d.add(stWarn, "server version", "server %s but this command is %s; restart the server",
				str(v, "version"), versionString())
		}
	} else {
		d.add(stFail, "server", "not answering at %s (%v)", host, err)
	}
	trayUp, serveUp := 0, 0
	if out, ok := run(6*time.Second, "powershell", "-NoProfile", "-Command",
		"(Get-CimInstance Win32_Process -Filter \"Name='llmashw.exe'\" | ForEach-Object { $_.CommandLine }) -join ';'"); ok {
		if strings.Contains(out, " tray") {
			trayUp = 1
		}
		if strings.Contains(out, " serve") {
			serveUp = 1
		}
	}
	switch {
	case trayUp == 1 && serveUp == 1:
		d.add(stOK, "processes", "tray and server are running")
	case serveUp == 1:
		d.add(stWarn, "processes", "server is running without the tray")
	default:
		d.add(stWarn, "processes", "no llmashw.exe running from this install")
	}

	// ---- the runtime -----------------------------------------------------
	if fileExists(llamaBin) {
		ver := "version unknown"
		if out, ok := run(8*time.Second, llamaBin, "--version"); ok {
			for _, ln := range strings.Split(out, "\n") {
				if strings.Contains(ln, "build") || strings.Contains(ln, "version") {
					ver = strings.TrimSpace(ln)
					break
				}
			}
		}
		d.add(stOK, "llama-server", "%s  (%s)", llamaBin, ver)
	} else {
		d.add(stFail, "llama-server", "missing at %s; reinstall or set LLAMA_BIN", llamaBin)
	}

	// ---- models ----------------------------------------------------------
	if tags, _, err := callJSON("GET", "/api/tags", nil, 20*time.Second); err == nil {
		rows := list(tags, "models")
		var bytes float64
		for _, m := range rows {
			if mm, ok := m.(map[string]any); ok {
				bytes += num(mm, "size")
			}
		}
		d.add(stOK, "models", "%d, %s total", len(rows), humanBytes(int64(bytes)))
		if len(rows) == 0 {
			d.add(stWarn, "models", "the store is empty; pull one with `%s pull <name>`", prog)
		}
	}
	if p, _, err := callJSON("GET", "/api/paths", nil, 5*time.Second); err == nil {
		store := str(p, "models")
		freeGB := freeDiskGB(store)
		state := stOK
		if freeGB > 0 && freeGB < 20 {
			state = stWarn
		}
		d.add(state, "model store", "%s, %.0f GB free", store, freeGB)
	}

	// ---- the card --------------------------------------------------------
	if out, ok := run(6*time.Second, "nvidia-smi",
		"--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu",
		"--format=csv,noheader,nounits"); ok && out != "" {
		f := strings.Split(out, ", ")
		if len(f) >= 5 {
			d.add(stOK, "gpu", "%s, driver %s, %s of %s MiB used, %s%% busy", f[0], f[1], f[3], f[2], f[4])
		}
	} else {
		d.add(stWarn, "gpu", "nvidia-smi did not answer; running on CPU or the driver is busy")
	}
	freeV, freeR := freeVRAMGB(), freeRAMGB()
	vState := stOK
	if freeV < 4 {
		vState = stWarn
	}
	d.add(vState, "headroom", "%.1f GB VRAM free, %.1f GB RAM free", freeV, freeR)

	// ---- what is loaded, and how it was tuned ---------------------------
	if ps, _, err := callJSON("GET", "/api/ps", nil, 5*time.Second); err == nil {
		rows := list(ps, "models")
		if len(rows) == 0 {
			d.add(stOK, "loaded", "nothing resident")
		}
		for _, m := range rows {
			mm, _ := m.(map[string]any)
			d.add(stOK, "loaded", "%s, %s, until %s", str(mm, "name"),
				humanBytes(int64(num(mm, "size_vram"))), humanTimeISO(str(mm, "expires_at"), "never"))
		}
	}
	tuned, why := (&Instance{}).autoTune()
	if len(tuned) > 0 {
		d.add(stOK, "auto-tuning", "%s", why)
	}
	if specFallback != "" && specFallback != "none" {
		d.add(stOK, "speculation", "models without a drafter of their own use %s", specFallback)
	} else {
		d.add(stWarn, "speculation", "no fallback drafter; set LLMASH_SPEC_FALLBACK=ngram-mod")
	}

	// ---- fast backends ---------------------------------------------------
	for _, route := range fastRoutes {
		name := str(route, "match")
		switch {
		case routeExe(route) != "":
			exe := routeExe(route)
			if !fileExists(exe) {
				d.add(stFail, "route "+name, "%s is missing", exe)
			} else if routeUp(route) {
				d.add(stOK, "route "+name, "native, serving at %s", routeURL(route))
			} else {
				d.add(stOK, "route "+name, "native, starts on demand from %s", filepath.Base(exe))
			}
		default:
			c := routeContainer(route)
			if routeUp(route) {
				d.add(stOK, "route "+name, "container %s, serving", c)
			} else {
				d.add(stOK, "route "+name, "container %s, starts on demand", c)
			}
		}
	}

	// ---- reachability of the commands -----------------------------------
	for _, n := range shimNames {
		p := which(n)
		switch {
		case p == "":
			if n != "ollama" {
				d.add(stWarn, "command "+n, "not on PATH")
			}
		case strings.EqualFold(filepath.Dir(filepath.Dir(p)), root) || strings.EqualFold(filepath.Dir(p), root):
			d.add(stOK, "command "+n, "%s", p)
		default:
			state := stWarn
			note := "%s points at a different install"
			if dev || origin == "dev" {
				state = stOK
				note = "%s (the installed copy, not this checkout)"
			}
			d.add(state, "command "+n, note, p)
		}
	}

	// ---- updates ---------------------------------------------------------
	switch rel, err := latestRelease(); {
	case dev || origin == "dev":
		d.add(stOK, "updates", "source checkout; rebuild with python build.py --here")
	case err != nil:
		d.add(stWarn, "updates", "%v", err)
	case rel.version() == versionString():
		d.add(stOK, "updates", "%s has %s; up to date", repoSlug(), rel.version())
	default:
		d.add(stWarn, "updates", "%s has %s; run `%s update`", repoSlug(), rel.version(), prog)
	}

	// ---- print -----------------------------------------------------------
	width := 0
	for _, r := range d.rows {
		if len(r.name) > width {
			width = len(r.name)
		}
	}
	bad := 0
	for _, r := range d.rows {
		mark, color := "ok  ", green
		switch r.state {
		case stWarn:
			mark, color = "warn", orange
			bad++
		case stFail:
			mark, color = "FAIL", "\x1b[38;5;203m"
			bad++
		}
		fmt.Printf("%s%s%s  %-*s  %s\n", color, mark, reset, width, r.name, r.detail)
	}
	fmt.Println()
	switch {
	case bad == 0:
		fmt.Println("everything checks out")
	default:
		fmt.Printf("%d thing(s) worth a look\n", bad)
	}
	for _, r := range d.rows {
		if r.state == stFail {
			exit(1)
		}
	}
}
