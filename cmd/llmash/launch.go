package main

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

// Real Ollama 0.32.x: bare `ollama` (and bare `ollama launch`) opens an
// interactive menu of integrations; `ollama launch <name>` configures one to
// use the local server and starts it. Same menu, same flags.

type integration struct{ title, exe string }

var integrations = map[string]integration{
	"claude":         {"Claude Code", "claude"},
	"chatgpt":        {"ChatGPT", "chatgpt"},
	"hermes":         {"Hermes Agent", "hermes"},
	"openclaw":       {"OpenClaw", "openclaw"},
	"opencode":       {"OpenCode", "opencode"},
	"codex":          {"Codex", "codex"},
	"hermes-desktop": {"Hermes Desktop", "hermes-desktop"},
	"copilot":        {"Copilot CLI", "copilot"},
	"omp":            {"OMP", "omp"},
	"droid":          {"Droid", "droid"},
	"kimi":           {"Kimi Code CLI", "kimi"},
	"pi":             {"Pi", "pi"},
	"pool":           {"Pool", "pool"},
	"cline":          {"Cline", "cline"},
	"qwen":           {"Qwen Code", "qwen"},
	"vscode":         {"VS Code", "code"},
}

var integrationAliases = map[string]string{
	"codex-app": "chatgpt", "codex-desktop": "chatgpt", "codex-gui": "chatgpt",
	"clawdbot": "openclaw", "moltbot": "openclaw",
	"copilot-cli": "copilot", "code": "vscode",
}

func profileFile() string { return filepath.Join(root, "integrations.json") }

func profiles() map[string]map[string]any {
	p := map[string]map[string]any{}
	if b, err := os.ReadFile(profileFile()); err == nil {
		json.Unmarshal(b, &p)
	}
	return p
}

func saveProfiles(p map[string]map[string]any) {
	b, _ := json.MarshalIndent(p, "", "  ")
	os.WriteFile(profileFile(), b, 0o644)
}

func profileModel(p map[string]map[string]any, key string) string {
	if m, ok := p[key]; ok {
		if s, _ := m["model"].(string); s != "" {
			return s
		}
	}
	return ""
}

func runInherit(name string, args []string, env []string) {
	cmd := exec.Command(name, args...)
	cmd.Stdin, cmd.Stdout, cmd.Stderr = os.Stdin, os.Stdout, os.Stderr
	if env != nil {
		cmd.Env = env
	}
	err := cmd.Run()
	if e, ok := err.(*exec.ExitError); ok {
		exit(e.ExitCode())
	} else if err != nil {
		die("Error: %v", err)
	}
	exit(0)
}

func launchClaude(model string, extra []string) {
	ps1 := filepath.Join(root, "Claude-on-Local.ps1")
	if !fileExists(ps1) {
		die("Error: launcher missing: %s", ps1)
	}
	args := []string{"-NoProfile", "-ExecutionPolicy", "Bypass", "-File", ps1}
	if model != "" {
		args = append(args, "-Model", model)
	}
	args = append(args, extra...)
	runInherit("powershell", args, nil)
}

// Point an OpenAI-compatible CLI at llmash's /v1 and start it.
func launchGeneric(key, model string, extra []string) {
	it := integrations[key]
	path := which(it.exe)
	if path == "" {
		die("Error: %s is not installed (no '%s' on PATH)", it.title, it.exe)
	}
	env := append(os.Environ(), "OPENAI_BASE_URL="+host+"/v1", "OPENAI_API_KEY=ollama", "OLLAMA_HOST="+host)
	with := ""
	if model != "" {
		env = append(env, "OPENAI_MODEL="+model, "OLLAMA_MODEL="+model)
		with = " with " + model
	}
	fmt.Printf("%slaunching %s against %s/v1%s%s\n", dim, it.title, host, with, reset)
	runInherit(path, extra, env)
}

func installedModels() []string {
	d, _, err := callJSON("GET", "/api/tags", nil, 4*time.Second)
	if err != nil {
		return nil
	}
	var out []string
	for _, m := range list(d, "models") {
		if mm, ok := m.(map[string]any); ok {
			out = append(out, str(mm, "name"))
		}
	}
	return out
}

// The interactive menu bare `ollama` opens: up/down to move, enter to launch,
// right-arrow to configure the entry's model, esc to quit. Falls back to the
// help text when there is no console to draw on.
func menu() {
	if !isConsole(os.Stdout) || !isConsole(os.Stdin) {
		fmt.Print(progText(helpText))
		return
	}
	prof := profiles()
	rows := [][3]string{
		{"claude", "Launch Claude Code", "Anthropic's coding tool with subagents"},
		{"opencode", "Launch OpenCode", "Anomaly's open-source coding agent"},
		{"hermes", "Launch Hermes Agent", "Self-improving AI agent built by Nous Research"},
		{"openclaw", "Launch OpenClaw", "Personal AI with 100+ skills"},
	}
	installed := installedModels()
	whichCache := map[string]bool{}
	have := func(exe string) bool {
		if v, ok := whichCache[exe]; ok {
			return v
		}
		whichCache[exe] = which(exe) != ""
		return whichCache[exe]
	}
	contains := func(xs []string, s string) bool {
		for _, x := range xs {
			if x == s {
				return true
			}
		}
		return false
	}
	modelOf := func(k string) string {
		if m := profileModel(prof, k); m != "" {
			return m
		}
		if k == "claude" && len(installed) > 0 {
			return installed[0]
		}
		return ""
	}
	suffix := func(k string) string {
		if k == "claude" {
			m := modelOf(k)
			if m == "" {
				return ""
			}
			if len(installed) > 0 && !contains(installed, m) {
				return " (" + m + ") (not installed)"
			}
			return " (" + m + ")"
		}
		if have(integrations[k].exe) {
			return ""
		}
		return " (install)"
	}
	drawn := 0
	draw := func(lines []string) {
		fmt.Print(strings.Repeat("\x1b[F\x1b[2K", drawn))
		fmt.Print(strings.Join(lines, "\n") + "\n")
		drawn = len(lines)
	}
	frame := func(sel int) []string {
		out := []string{fmt.Sprintf("%s%s %s%s", bold, prog, versionString(), reset), "",
			"  Chat, Code, & Work",
			dim + "    Chat with models, code, search the web, and delegate real work" + reset, ""}
		for i, r := range rows {
			cur, hl := "  ", ""
			if i == sel {
				cur, hl = "▸ ", bold
			}
			out = append(out, cur+hl+r[1]+suffix(r[0])+reset, dim+"    "+r[2]+reset, "")
		}
		out = append(out, dim+"↑/↓ navigate • enter launch • → configure • esc quit"+reset)
		return out
	}
	pickModel := func(k string) {
		if len(installed) == 0 {
			installed = installedModels()
		}
		if len(installed) == 0 {
			draw([]string{fmt.Sprintf("%smodel for %s%s", bold, integrations[k].title, reset), "",
				fmt.Sprintf("  couldn't reach llmash at %s. Start it, then press → again", host), "",
				dim + "press any key to go back" + reset})
			getch()
			return
		}
		msel := 0
		if cur := modelOf(k); cur != "" {
			for i, n := range installed {
				if n == cur {
					msel = i
				}
			}
		}
		for {
			out := []string{fmt.Sprintf("%smodel for %s%s", bold, integrations[k].title, reset), ""}
			for i, n := range installed {
				cur, hl := "  ", ""
				if i == msel {
					cur, hl = "▸ ", bold
				}
				out = append(out, cur+hl+n+reset)
			}
			out = append(out, "", dim+"↑/↓ navigate • enter select • esc back"+reset)
			draw(out)
			switch ch := getch(); ch {
			case 0xE0, 0x00:
				switch getch() {
				case 'H':
					msel = (msel - 1 + len(installed)) % len(installed)
				case 'P':
					msel = (msel + 1) % len(installed)
				}
			case '\r':
				prof[k] = map[string]any{"model": installed[msel]}
				saveProfiles(prof)
				return
			case 0x1b, 'q':
				return
			}
		}
	}
	sel := 0
	fmt.Print("\x1b[?25l")
	defer fmt.Print("\x1b[?25h")
	for {
		draw(frame(sel))
		switch ch := getch(); ch {
		case 0xE0, 0x00:
			switch getch() {
			case 'H':
				sel = (sel - 1 + len(rows)) % len(rows)
			case 'P':
				sel = (sel + 1) % len(rows)
			case 'M':
				pickModel(rows[sel][0])
			}
		case '\r':
			k := rows[sel][0]
			fmt.Print("\x1b[?25h\n")
			if k == "claude" {
				launchClaude(modelOf(k), nil)
			}
			if !have(integrations[k].exe) {
				die("%s is not installed.", integrations[k].title)
			}
			launchGeneric(k, modelOf(k), nil)
		case 0x1b, 'q', 0x03:
			fmt.Println()
			return
		}
	}
}

func cmdLaunch(args []string) {
	var rest, extra []string
	rest = args
	for i, t := range args {
		if t == "--" {
			rest, extra = args[:i], args[i+1:]
			break
		}
	}
	name, model := "", ""
	config, restore := false, false
	for i := 0; i < len(rest); i++ {
		t := rest[i]
		switch {
		case t == "--config":
			config = true
		case t == "--restore":
			restore = true
		case t == "-y" || t == "--yes":
		case t == "--model":
			if i+1 >= len(rest) {
				die("Error: flag needs an argument: --model")
			}
			i++
			model = rest[i]
		case strings.HasPrefix(t, "--model="):
			model = t[len("--model="):]
		case strings.HasPrefix(t, "-"):
			die("Error: unknown flag: '%s'", t)
		case name == "":
			name = t
		default:
			extra = append(extra, t)
		}
	}
	if name == "" {
		if len(extra) > 0 || model != "" || config || restore {
			die("Error: flags and extra arguments require an integration name")
		}
		menu()
		return
	}
	key := name
	if a, ok := integrationAliases[name]; ok {
		key = a
	}
	if _, ok := integrations[key]; !ok {
		die("Error: unknown integration \"%s\"", name)
	}
	prof := profiles()
	if restore {
		delete(prof, key)
		saveProfiles(prof)
		fmt.Printf("restored %s to its default profile\n", integrations[key].title)
		return
	}
	if model != "" {
		p := prof[key]
		if p == nil {
			p = map[string]any{}
		}
		p["model"] = model
		prof[key] = p
		saveProfiles(prof)
	}
	if model == "" {
		model = profileModel(prof, key)
	}
	if config {
		shown, hint := model, ""
		if shown == "" {
			shown, hint = "(default)", "  (set one with --model)"
		}
		fmt.Printf("%s: model = %s%s\n", integrations[key].title, shown, hint)
		return
	}
	if key == "claude" {
		launchClaude(model, extra)
	}
	launchGeneric(key, model, extra)
}
