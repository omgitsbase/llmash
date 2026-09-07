package main

import (
	"context"
	"crypto/rand"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"syscall"
	"time"
)

// ------------------------------------------------------------ list / ps

// The server keeps the finished tables in cache\ beside a heartbeat; while
// the heartbeat is fresh the file is the answer and no socket is opened.
func cachedTable(name string) (string, bool) {
	st, err := os.Stat(filepath.Join(root, "cache", "alive"))
	if err != nil || time.Since(st.ModTime()) > 15*time.Second {
		return "", false
	}
	b, err := os.ReadFile(filepath.Join(root, "cache", name))
	if err != nil || len(b) == 0 {
		return "", false
	}
	return string(b), true
}

// The pre-rendered table, or false if this server doesn't serve one (an
// actual ollama, say), in which case the caller renders it itself.
func serverTable(path string) bool {
	r, err := call("GET", path, nil, 60*time.Second)
	if err != nil {
		needServer()
		die("error: %v", err)
	}
	b, _ := readAll(r)
	text := string(b)
	if r.StatusCode != 200 || !strings.HasPrefix(text, "NAME") {
		return false
	}
	fmt.Print(text)
	return true
}

// A name prefix filters the table, the way ollama's does; without one the
// server's pre-rendered copy is the answer.
func cmdList(args []string) {
	if len(args) == 0 {
		if t, ok := cachedTable("list.txt"); ok {
			fmt.Print(t)
			return
		}
		if serverTable("/cli/list") {
			return
		}
	}
	needServer()
	d, _, err := callJSON("GET", "/api/tags", nil, 60*time.Second)
	if err != nil {
		die("error: %v", err)
	}
	fmt.Print(renderList(filterRows(d, arg0(args), true)))
}

func cmdPs(args []string) {
	if len(args) == 0 {
		if t, ok := cachedTable("ps.txt"); ok {
			fmt.Print(t)
			return
		}
		if serverTable("/cli/ps") {
			return
		}
	}
	needServer()
	d, _, err := callJSON("GET", "/api/ps", nil, 60*time.Second)
	if err != nil {
		die("error: %v", err)
	}
	fmt.Print(renderPs(filterRows(d, arg0(args), false)))
}

func arg0(args []string) string {
	if len(args) > 0 {
		return args[0]
	}
	return ""
}

func filterRows(d map[string]any, prefix string, fold bool) []map[string]any {
	var out []map[string]any
	for _, v := range list(d, "models") {
		m, ok := v.(map[string]any)
		if !ok {
			continue
		}
		name := str(m, "name")
		if fold {
			name, prefix = strings.ToLower(name), strings.ToLower(prefix)
		}
		if strings.HasPrefix(name, prefix) {
			out = append(out, m)
		}
	}
	return out
}

// ------------------------------------------------------------------ show

func cmdShow(o showOpts) {
	needServer()
	set := 0
	for _, b := range []bool{o.lic, o.modelfile, o.parameters, o.system, o.template} {
		if b {
			set++
		}
	}
	if set > 1 {
		die("Error: only one of '--license', '--modelfile', '--parameters', '--system', or '--template' can be specified")
	}
	d, code := showModel(o.model)
	if code != 200 {
		die("Error: %s", first(str(d, "error"), "model not found"))
	}
	switch {
	case o.lic:
		fmt.Println(str(d, "license"))
	case o.modelfile:
		fmt.Println(str(d, "modelfile"))
	case o.parameters:
		fmt.Println(str(d, "parameters"))
	case o.system:
		fmt.Print(str(d, "system"))
	case o.template:
		fmt.Print(str(d, "template"))
	default:
		showInfo(d, o.verbose, os.Stdout)
	}
}

// The blocks `ollama show` prints, in its order and its table layout.
func showInfo(resp map[string]any, verbose bool, w io.Writer) {
	render := func(header string, rows [][]string) {
		fmt.Fprintln(w, " ", header)
		t := newTable()
		switch header {
		case "Template", "System", "License":
			t.wrapAt = 100
		}
		for _, r := range rows {
			t.add(r...)
		}
		fmt.Fprint(w, t.String())
		fmt.Fprintln(w)
	}

	info := sub(resp, "model_info")
	det := sub(resp, "details")
	numStr := func(v any) (string, bool) {
		f, ok := v.(float64)
		if !ok {
			return "", false
		}
		return strconv.FormatFloat(f, 'f', -1, 64), true
	}

	var rows [][]string
	arch := str(info, "general.architecture")
	if len(info) > 0 {
		if arch != "" {
			rows = append(rows, []string{"", "architecture", arch})
		}
		param := str(det, "parameter_size")
		if param == "" {
			if f, ok := info["general.parameter_count"].(float64); ok {
				param = humanNumber(uint64(f))
			}
		}
		if param != "" {
			rows = append(rows, []string{"", "parameters", param})
		}
		if s, ok := numStr(info[arch+".context_length"]); ok {
			rows = append(rows, []string{"", "context length", s})
		}
		if s, ok := numStr(info[arch+".embedding_length"]); ok {
			rows = append(rows, []string{"", "embedding length", s})
		}
	} else {
		rows = append(rows, []string{"", "architecture", str(det, "family")})
		rows = append(rows, []string{"", "parameters", str(det, "parameter_size")})
	}
	rows = append(rows, []string{"", "quantization", str(det, "quantization_level")})
	render("Model", rows)

	if cs := list(resp, "capabilities"); len(cs) > 0 {
		rows = nil
		for _, c := range cs {
			rows = append(rows, []string{"", fmt.Sprint(c)})
		}
		render("Capabilities", rows)
	}

	if p := str(resp, "parameters"); p != "" {
		rows = nil
		for _, ln := range strings.Split(p, "\n") {
			if strings.TrimSpace(ln) != "" {
				rows = append(rows, append([]string{""}, strings.Fields(ln)...))
			}
		}
		render("Parameters", rows)
	}

	if verbose && len(info) > 0 {
		keys := make([]string, 0, len(info))
		for k := range info {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		rows = nil
		for _, k := range keys {
			var v string
			switch t := info[k].(type) {
			case bool:
				v = fmt.Sprintf("%t", t)
			case string:
				v = t
			case float64:
				v = fmt.Sprintf("%g", t)
			case []any:
				v = elide(t, 10)
			default:
				v = fmt.Sprintf("%T", t)
			}
			rows = append(rows, []string{"", k, v})
		}
		render("Metadata", rows)
	}

	// The first two lines of each, with "..." when there is more.
	head := func(s string, n int) (rows [][]string) {
		count := 0
		for _, ln := range strings.Split(s, "\n") {
			text := strings.TrimSpace(ln)
			if text == "" {
				continue
			}
			count++
			if n < 0 || count <= n {
				rows = append(rows, []string{"", text})
			}
		}
		if n >= 0 && count > n {
			rows = append(rows, []string{"", "..."})
		}
		return
	}
	if s := str(resp, "system"); s != "" {
		render("System", head(s, 2))
	}
	if s := str(resp, "license"); s != "" {
		render("License", head(s, 2))
	}
}

// A long array value is cut to what fits a narrow column.
func elide(v []any, target int) string {
	total, show := 1, 0
	for i := range v {
		w := dispWidth(fmt.Sprintf("%v", v[i]))
		if i > 0 {
			w += 2
		}
		if total+w > target && i > 0 {
			break
		}
		total += w
		show++
	}
	if show < len(v) {
		s := fmt.Sprintf("%v", v[:show])
		return strings.TrimSuffix(s, "]") + fmt.Sprintf(" ...+%d more]", len(v)-show)
	}
	return fmt.Sprintf("%v", v)
}

// ------------------------------------------------------------- rm / stop

func cmdRm(models []string) {
	needServer()
	for _, name := range models {
		r, err := call("DELETE", "/api/delete", map[string]any{"model": name}, 60*time.Second)
		if err != nil {
			die("Error: %v", err)
		}
		b, _ := readAll(r)
		if r.StatusCode >= 400 {
			var d map[string]any
			msg := strings.TrimSpace(string(b))
			if json.Unmarshal(b, &d) == nil && str(d, "error") != "" {
				msg = str(d, "error")
			}
			die("Error: %s", msg)
		}
		fmt.Printf("deleted '%s'\n", name)
	}
}

// Unloading is a generate request with keep_alive 0, and like ollama it says
// nothing when it worked.
func cmdStop(models []string) {
	needServer()
	for _, name := range models {
		if _, code := showModel(name); code == 404 {
			die("Error: couldn't find model \"%s\" to stop", name)
		}
		d, r, err := callJSON("POST", "/api/generate", map[string]any{"model": name, "keep_alive": 0,
			"prompt": ""}, 60*time.Second)
		if err != nil {
			die("Error: %v", err)
		}
		if r.StatusCode >= 400 {
			die("Error: %s", first(str(d, "error"), r.Status))
		}
	}
}

// ------------------------------------------------------------------ pull

func human(n float64) string {
	for _, u := range []string{"B", "KB", "MB", "GB", "TB"} {
		if n < 1024 {
			if u == "B" || u == "KB" {
				return fmt.Sprintf("%.0f %s", n, u)
			}
			return fmt.Sprintf("%.1f %s", n, u)
		}
		n /= 1024
	}
	return fmt.Sprintf("%.1f PB", n)
}

// One bar per layer and a spinner for every other status, the way
// `ollama pull` draws them.
func cmdPull(model string, quant string) {
	needServer()
	if quant == "" && isHFRef(model) && !strings.Contains(model, "@") && isConsole(os.Stdin) && isConsole(os.Stdout) {
		quant = chooseQuant(model)
	}
	p := newProgress(os.Stderr)
	defer p.stop()
	bars := map[string]*progBar{}
	status := ""
	var spin *progSpinner
	var failed string

	body := map[string]any{"model": model}
	if quant != "" {
		body["quant"] = quant
	}
	err := stream(context.Background(), "/api/pull", body, func(ev map[string]any) bool {
		if e := str(ev, "error"); e != "" {
			failed = e
			return false
		}
		digest := str(ev, "digest")
		if digest != "" {
			completed := int64(num(ev, "completed"))
			if completed == 0 {
				return true // the server's size announcement, before any bytes
			}
			if spin != nil {
				spin.stop()
			}
			bar, ok := bars[digest]
			if !ok {
				name := strings.TrimSpace(strings.TrimPrefix(digest, "sha256:"))
				if strings.HasPrefix(digest, "sha256:") && len(name) > 12 {
					name = name[:12]
				}
				bar = newBar(fmt.Sprintf("pulling %s:", name), int64(num(ev, "total")), completed)
				bars[digest] = bar
				p.add(bar)
			}
			bar.set(completed)
			return true
		}
		if st := str(ev, "status"); st != status {
			if spin != nil {
				spin.stop()
			}
			status = st
			spin = newSpinner(st)
			p.add(spin)
		}
		return true
	})
	if failed != "" {
		p.stop()
		fmt.Fprintln(os.Stderr, "Error: "+failed)
		exit(1)
	}
	if err != nil {
		p.stop()
		die("Error: %v", err)
	}
}

// ------------------------------------------------------------ create / cp

// The server owns the model store, so it does the copying and quantizing;
// this side reads the Modelfile and prints what the server reports.
func streamStatuses(path string, body any) {
	err := stream(context.Background(), path, body, func(ev map[string]any) bool {
		if e := str(ev, "error"); e != "" {
			fmt.Fprintln(os.Stderr, e)
			exit(1)
		}
		if s := str(ev, "status"); s != "" && s != "success" {
			fmt.Println(s)
		}
		return true
	})
	if err != nil {
		die("error: %v", err)
	}
}

func cmdCreate(name, file, quantize, draftQuantize string) {
	b, err := os.ReadFile(file)
	if err != nil {
		die("Modelfile not found: %s", file)
	}
	from := ""
	for _, line := range strings.Split(string(b), "\n") {
		s := strings.TrimSpace(line)
		if strings.HasPrefix(strings.ToUpper(s), "FROM ") {
			from = strings.Trim(strings.TrimSpace(s[5:]), "\"'")
			break
		}
	}
	if from == "" {
		die("%s has no FROM line, so there is nothing to import.", file)
	}
	needServer()
	if !strings.EqualFold(filepath.Ext(from), ".gguf") || !fileExists(from) {
		loose := "the model store's gguf folder"
		if d, _, err := callJSON("GET", "/api/paths", nil, 5*time.Second); err == nil && str(d, "loose_dir") != "" {
			loose = str(d, "loose_dir")
		}
		fmt.Fprintf(os.Stderr, "'%s' isn't a local .gguf, so there's nothing to import.\n"+
			"llmash builds models from GGUF files, not Modelfile layers:\n"+
			"  - local file : put its .gguf in %s (it appears in `%s list` automatically),\n"+
			"                 or point a Modelfile's FROM at the .gguf and re-run create\n"+
			"  - from the hub: %s pull hf:owner/repo\n", from, loose, prog, prog)
		exit(1)
	}
	abs, _ := filepath.Abs(from)
	streamStatuses("/api/create", map[string]any{"model": name, "from": abs, "quantize": quantize,
		"draft_quantize": draftQuantize})
}

func cmdCp(source, destination string) {
	needServer()
	streamStatuses("/api/copy", map[string]any{"source": source, "destination": destination})
}

func cmdPush() {
	fmt.Fprintf(os.Stderr, "llmash has no model registry to push to. It serves local GGUFs only, so "+
		"there's nothing to upload.\nTo share this llmash instead, expose it over Tailscale with `%s link`.\n", prog)
	exit(1)
}

func cmdSignin() {
	fmt.Fprintf(os.Stderr, "llmash is fully local: there is no ollama.com account to sign in to.\n"+
		"To reach this server remotely, use `%s link` instead.\n", prog)
	exit(1)
}

// ---------------------------------------------------------- serve / tray

func cmdServe() {
	serveMain(nil)
}

func cmdTray() {
	exe := filepath.Join(root, "llmashw.exe")
	if !fileExists(exe) {
		die("llmashw.exe was not found in %s; reinstall llmash.", root)
	}
	cmd := exec.Command(exe, "tray")
	cmd.Dir = root
	cmd.SysProcAttr = &syscall.SysProcAttr{HideWindow: true, CreationFlags: 0x08000000 | 0x00000008}
	if err := cmd.Start(); err != nil {
		die("could not start the tray: %v", err)
	}
	cmd.Process.Release()
}

// ------------------------------------------------------------- uninstall

// Commands the installer puts on PATH. `llamash` was this project's old
// name and is only here so an uninstall clears it.
var shimNames = []string{"llmash", "ollama", "llamash"}

func stopServer() int {
	r := psQuote(root)
	script := "Get-CimInstance Win32_Process -Filter \"Name='llmashw.exe' OR Name='llmash.exe' OR Name='llmash-server.exe' OR Name='pythonw.exe'\" " +
		"| Where-Object { $_.CommandLine -like '*" + r + "\\llmashw.exe*' -or $_.CommandLine -like '*" + r + "\\llmash-server.exe*' -or " +
		"$_.CommandLine -like '*" + r + "\\server.py*' -or $_.CommandLine -like '*" + r + "\\tray.py*' -or " +
		"($_.CommandLine -like '*" + r + "\\llmash.exe*' -and $_.CommandLine -like '* serve*') } " +
		"| ForEach-Object { Stop-Process -Id $_.ProcessId -Force; $_.ProcessId }"
	out, _ := hiddenPowerShell(script, true)
	n := 0
	for _, f := range strings.Fields(out) {
		if _, err := fmt.Sscan(f, new(int)); err == nil {
			n++
		}
	}
	return n
}

func cmdUninstall(keep bool) {
	binDir := filepath.Join(root, "bin")
	startupDir := filepath.Join(os.Getenv("APPDATA"), "Microsoft", "Windows", "Start Menu", "Programs", "Startup")
	startupLnk := filepath.Join(startupDir, "llmash.lnk")

	// A directory the installer did not create (a source checkout registered
	// by hand) is never deleted; only the registration goes.
	dev := true
	if b, err := os.ReadFile(filepath.Join(root, "install.json")); err == nil {
		b = []byte(strings.TrimPrefix(string(b), "\ufeff"))
		var info map[string]any
		if json.Unmarshal(b, &info) == nil && len(info) > 0 {
			dev, _ = info["dev"].(bool)
		}
	}
	keepDir := keep || dev
	fmt.Printf("Removing %s from %s\n", prog, root)

	if dev {
		fmt.Println("  left the server running (not an installer directory)")
	} else if n := stopServer(); n > 0 {
		s := ""
		if n != 1 {
			s = "es"
		}
		fmt.Printf("  stopped %d process%s\n", n, s)
	} else {
		fmt.Println("  server was not running")
	}

	var shims []string
	for _, name := range shimNames {
		found := false
		for _, ext := range []string{".exe", ".cmd"} {
			if p := filepath.Join(binDir, name+ext); fileExists(p) {
				shims = append(shims, p)
				found = true
			}
		}
		if found {
			fmt.Printf("  removing command: %s\n", name)
		}
	}
	// Another install may own the startup entry; leave that one alone.
	if target := shortcutTarget(startupLnk); target != "" {
		if under(target, root) {
			os.Remove(startupLnk)
			fmt.Println("  removed the startup entry")
		} else {
			fmt.Printf("  left the startup entry alone (it starts %s)\n", target)
		}
	}
	if disabled := filepath.Join(startupDir, "Ollama.lnk.disabled"); fileExists(disabled) {
		os.Rename(disabled, filepath.Join(startupDir, "Ollama.lnk"))
		fmt.Println("  restored Ollama's startup entry")
	}
	if regDeleteKey(`Software\Microsoft\Windows\CurrentVersion\Uninstall\llmash`) {
		fmt.Println("  removed from Settings > Apps")
	}
	if removeFromUserPath(binDir) {
		fmt.Println("  removed from PATH")
	}

	// The program that started us lives in the directory being deleted, so
	// the last step goes to a hidden PowerShell that waits for this process
	// to exit and retries the delete for a few seconds.
	retry := func(target string, recurse bool) string {
		flag := ""
		if recurse {
			flag = " -Recurse"
		}
		return "for ($i = 0; $i -lt 30; $i++) { Remove-Item -LiteralPath '" + psQuote(target) + "'" + flag +
			" -Force -EA SilentlyContinue; if (-not (Test-Path -LiteralPath '" + psQuote(target) + "')) { break }; Start-Sleep -Milliseconds 300 }"
	}
	var steps []string
	if keepDir {
		for _, s := range shims {
			steps = append(steps, retry(s, false))
		}
		if len(steps) == 0 {
			steps = append(steps, "$null")
		}
		why := " (not an installer directory)"
		if keep {
			why = " (--keep)"
		}
		fmt.Printf("  left %s in place%s\n", root, why)
	} else {
		steps = append(steps, retry(root, true))
		fmt.Printf("  %s will be removed in a moment\n", root)
	}
	script := fmt.Sprintf("Wait-Process -Id %d -EA SilentlyContinue; Start-Sleep -Milliseconds 500; %s",
		os.Getpid(), strings.Join(steps, "; "))
	hiddenPowerShell(script, false)
	fmt.Println("done. Your models were left where they are.")
}

// ----------------------------------------------------------- link/unlink

var (
	publicPort = envInt("LLMASH_PUBLIC_PORT", 11435)
	funnelPort = envInt("LLMASH_FUNNEL_PORT", 8443)
)

func envInt(name string, def int) int {
	if v := env(name); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}
	return def
}

func tailscale() string {
	for _, p := range []string{`C:\Program Files\Tailscale\tailscale.exe`, `C:\Program Files (x86)\Tailscale\tailscale.exe`} {
		if fileExists(p) {
			return p
		}
	}
	return which("tailscale")
}

func linkFile() string { return filepath.Join(root, "link.json") }

func readLink() map[string]any {
	d := map[string]any{}
	if b, err := os.ReadFile(linkFile()); err == nil {
		json.Unmarshal(b, &d)
	}
	return d
}

func writeLink(d map[string]any) {
	b, _ := json.MarshalIndent(d, "", "  ")
	os.WriteFile(linkFile(), b, 0o600)
}

// Same source of truth as the server: env var, else link.json, else mint one.
func linkKey() string {
	if k := env("LLMASH_LINK_KEY"); k != "" {
		return k
	}
	d := readLink()
	if k, _ := d["key"].(string); k != "" {
		return k
	}
	raw := make([]byte, 24)
	rand.Read(raw)
	key := "sk-llmash-" + base64.RawURLEncoding.EncodeToString(raw)
	d["key"] = key
	d["public_port"] = publicPort
	writeLink(d)
	return key
}

func portOpen(port int) bool {
	c, err := net.DialTimeout("tcp", fmt.Sprintf("127.0.0.1:%d", port), 600*time.Millisecond)
	if err != nil {
		return false
	}
	c.Close()
	return true
}

func tailnetName(ts string) string {
	out, err := exec.Command(ts, "status", "--json").Output()
	if err != nil {
		return ""
	}
	var st struct {
		Self struct {
			DNSName string
		}
	}
	json.Unmarshal(out, &st)
	return strings.TrimRight(st.Self.DNSName, ".")
}

func cmdLink(off bool) {
	ts := tailscale()
	if ts == "" {
		die("Tailscale isn't installed (or not on PATH).")
	}
	if !portOpen(publicPort) {
		die("llmash's public port %d isn't up, so the tunnel would point at nothing.\n"+
			"Restart llmash (it opens that port on boot), then run `%s link` again.", publicPort, prog)
	}
	dns := tailnetName(ts)
	if dns == "" {
		die("Couldn't read your Tailscale name. Is `tailscale` logged in and up?")
	}
	key := linkKey()
	if !off {
		out, err := exec.Command(ts, "funnel", "--bg", fmt.Sprintf("--https=%d", funnelPort),
			fmt.Sprintf("127.0.0.1:%d", publicPort)).CombinedOutput()
		if err != nil {
			die("Funnel setup failed:\n%s", strings.TrimSpace(string(out)))
		}
	}
	url := fmt.Sprintf("https://%s:%d", dns, funnelPort)
	bar := dim + strings.Repeat("-", 66) + reset
	fmt.Println()
	fmt.Println(bar)
	fmt.Printf("  %sllmash is public%s  %s- an Ollama-compatible API, keyed%s\n", bold, reset, dim, reset)
	fmt.Println(bar)
	fmt.Printf("  %sURL%s  %s%s%s%s\n", dim, reset, bold, cyan, url, reset)
	fmt.Printf("  %skey%s  %s%s%s\n", dim, reset, green, key, reset)
	fmt.Println(bar)
	fmt.Printf("  %scurl:%s\n", dim, reset)
	short := key
	if len(short) > 14 {
		short = short[:14]
	}
	fmt.Printf("  %scurl %s/api/chat -H \"Authorization: Bearer %s...\" \\%s\n", dim, url, short, reset)
	fmt.Printf("  %s     -d '{\"model\":\"qwen3.6:27b\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}'%s\n", dim, reset)
	fmt.Println(bar)
	if off {
		fmt.Printf("  %s--off: the tunnel was not (re)started. To take the public API down, run `%s unlink`.%s\n", dim, prog, reset)
		fmt.Println(bar)
	}
	if !isConsole(os.Stdin) || !isConsole(os.Stdout) {
		return
	}
	fmt.Printf("  %sc%s copy link   %sk%s copy key   %sq%s done %s(tunnel stays up in the background)%s\n",
		bold, reset, bold, reset, bold, reset, dim, reset)
	for {
		switch ch := getch(); ch {
		case 'c', 'C':
			report(clip(url), "link copied")
		case 'k', 'K':
			report(clip(key), "key copied")
		case 'q', 'Q', '\r', '\n', 0x1b, 0x03:
			return
		}
	}
}

func report(ok bool, what string) {
	if ok {
		fmt.Printf("  %s%s%s\n", green, what, reset)
	} else {
		fmt.Println("  (copy failed)")
	}
}

func cmdUnlink() {
	ts := tailscale()
	if ts == "" {
		die("Tailscale isn't installed (or not on PATH).")
	}
	// Only our port, so any other funnel on the machine is left alone.
	out, err := exec.Command(ts, "funnel", fmt.Sprintf("--https=%d", funnelPort), "off").CombinedOutput()
	if err != nil {
		die("%s", strings.TrimSpace(string(out)))
	}
	fmt.Printf("public API on :%d is off\n", funnelPort)
}

// shortcutTarget reads what a .lnk points at, or "" when there is no shortcut.
func shortcutTarget(lnk string) string {
	if !fileExists(lnk) {
		return ""
	}
	out, _ := hiddenPowerShell("(New-Object -ComObject WScript.Shell).CreateShortcut('"+
		psQuote(lnk)+"').TargetPath", true)
	return strings.TrimSpace(out)
}

func under(path, dir string) bool {
	rel, err := filepath.Rel(dir, path)
	return err == nil && !strings.HasPrefix(rel, "..")
}

func isHFRef(name string) bool {
	for _, p := range hfPrefixes {
		if strings.HasPrefix(name, p) {
			return true
		}
	}
	return false
}

// chooseQuant lists the builds a repository offers and takes a choice. Enter
// keeps the default, which is the Q4_K_M build when there is one.
func chooseQuant(model string) string {
	d, r, err := callJSON("GET", "/api/quants?repo="+url.QueryEscape(model), nil, 60*time.Second)
	if err != nil || r.StatusCode != 200 {
		return ""
	}
	quants := list(d, "quants")
	if len(quants) < 2 {
		return ""
	}
	def := -1
	for i, q := range quants {
		if m, _ := q.(map[string]any); strings.EqualFold(str(m, "name"), "Q4_K_M") {
			def = i
		}
	}
	repo := strings.TrimPrefix(strings.TrimPrefix(model, "hf.co/"), "hf:")
	fmt.Printf("%s offers %d builds:\n", repo, len(quants))
	for i, q := range quants {
		m, _ := q.(map[string]any)
		mark := "  "
		if i == def {
			mark = "* "
		}
		files := ""
		if n := int(num(m, "files")); n > 1 {
			files = fmt.Sprintf("  (%d files)", n)
		}
		fmt.Printf("  %s%2d. %-14s %8s%s\n", mark, i+1, str(m, "name"), humanBytes(int64(num(m, "size"))), files)
	}
	if def >= 0 {
		fmt.Printf("Which one? [%d] ", def+1)
	} else {
		fmt.Print("Which one? ")
	}
	var buf []byte
	for {
		ch := getch()
		switch {
		case ch == '\r' || ch == '\n':
			fmt.Println()
			n := def
			if len(buf) > 0 {
				fmt.Sscanf(string(buf), "%d", &n)
				n--
			}
			if n >= 0 && n < len(quants) {
				m, _ := quants[n].(map[string]any)
				return str(m, "name")
			}
			return ""
		case ch == 0x1b || ch == 0x03:
			fmt.Println()
			return ""
		case ch >= '0' && ch <= '9':
			buf = append(buf, ch)
			fmt.Print(string(ch))
		case ch == 8 || ch == 127:
			if len(buf) > 0 {
				buf = buf[:len(buf)-1]
				fmt.Print("\b \b")
			}
		}
	}
}
