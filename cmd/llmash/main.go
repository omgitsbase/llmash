// llmash: the command line, native. Talks to the llmash server over HTTP;
// `list`, `ps` and `version` need no server round trip at all.
package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

const fallbackVersion = "0.3.0"

var (
	prog string // what was typed: llmash, llmash, ollama
	root string // the install directory (exe dir, or its parent for bin\)
)

func init() {
	exe, _ := os.Executable()
	exe, _ = filepath.EvalSymlinks(exe)
	dir := filepath.Dir(exe)
	root = dir
	if !fileExists(filepath.Join(dir, "llmashw.exe")) && fileExists(filepath.Join(filepath.Dir(dir), "llmashw.exe")) {
		root = filepath.Dir(dir)
	} else if !fileExists(filepath.Join(dir, "llmashw.exe")) && fileExists(filepath.Join(filepath.Dir(dir), "server.py")) {
		root = filepath.Dir(dir)
	}
	prog = env("LLMASH_PROG")
	if prog == "" {
		prog = strings.ToLower(strings.TrimSuffix(filepath.Base(exe), filepath.Ext(exe)))
		os.Setenv("LLMASH_PROG", prog)
	}
}

// Re-address the real binary's help to whatever name was typed.
func progText(text string) string {
	if prog == "ollama" {
		return text
	}
	r := strings.NewReplacer(
		"\n  ollama ", "\n  "+prog+" ",
		"Use \"ollama [command]", "Use \""+prog+" [command]",
		"help for ollama", "help for "+prog,
		"Start Ollama", "Start "+prog,
		"`ollama ", "`"+prog+" ",
	)
	return r.Replace(text)
}

var aliases = map[string]string{"ls": "list", "start": "serve"}

var thinkLevels = map[string]bool{"true": true, "false": true, "high": true, "medium": true, "low": true, "max": true}

func versionString() string {
	for _, p := range []string{filepath.Join(root, "VERSION")} {
		if b, err := os.ReadFile(p); err == nil && strings.TrimSpace(string(b)) != "" {
			return strings.TrimSpace(string(b))
		}
	}
	if d, _, err := callJSON("GET", "/api/version", nil, 2*time.Second); err == nil && str(d, "version") != "" {
		return str(d, "version")
	}
	return fallbackVersion
}

func cmdVersion() {
	fmt.Printf("%s version is %s\n", prog, versionString())
}

func die(format string, a ...any) {
	fmt.Fprintf(os.Stderr, format+"\n", a...)
	exit(1)
}

func main() {
	enableVT()
	useUTF8()
	defer restoreCP()

	argv := os.Args[1:]

	// The server and the tray are this same program.
	if len(argv) > 0 && argv[0] == "serve" {
		serveMain(argv[1:])
		return
	}
	exeName, _ := os.Executable()
	windowed := strings.HasSuffix(strings.ToLower(strings.TrimSuffix(filepath.Base(exeName), ".exe")), "w")
	if len(argv) > 0 && argv[0] == "tray" && windowed {
		trayMain()
		return
	}

	// cobra accepts the global flags before the subcommand; peel them off.
	gVerbose, gNowrap := false, false
	for len(argv) > 0 && (argv[0] == "--verbose" || argv[0] == "--nowordwrap") {
		gVerbose = gVerbose || argv[0] == "--verbose"
		gNowrap = gNowrap || argv[0] == "--nowordwrap"
		argv = argv[1:]
	}
	_ = gNowrap

	if len(argv) == 0 {
		menu()
		return
	}
	switch argv[0] {
	case "-v", "--version":
		cmdVersion()
		return
	case "-h", "--help":
		fmt.Print(progText(helpText))
		return
	}

	cmd := argv[0]
	if a, ok := aliases[cmd]; ok {
		cmd = a
	}

	if cmd == "help" {
		topic := ""
		if len(argv) > 1 {
			topic = argv[1]
			if a, ok := aliases[topic]; ok {
				topic = a
			}
		}
		if topic == "" {
			fmt.Print(progText(helpText))
		} else if h, ok := commandHelp[topic]; ok {
			fmt.Print(progText(h))
		} else {
			fmt.Fprintf(os.Stderr, "Unknown help topic [`%s`]\n", topic)
			fmt.Fprint(os.Stderr, helpText)
			exit(1)
		}
		return
	}

	if _, ok := commandHelp[cmd]; !ok {
		die("Error: unknown command \"%s\" for \"%s\"", argv[0], prog)
	}

	// -h/--help before any `--` separator prints the cobra-style help.
	head := argv[1:]
	for i, t := range head {
		if t == "--" {
			head = head[:i]
			break
		}
	}
	for _, t := range head {
		if t == "-h" || t == "--help" {
			fmt.Print(progText(commandHelp[cmd]))
			return
		}
	}

	rest := argv[1:]
	switch cmd {
	case "list":
		cmdList(parseSimple(rest, nil, nil).pos)
	case "ps":
		cmdPs(parseSimple(rest, nil, nil).pos)
	case "show":
		cmdShow(parseShow(rest))
	case "pull", "install":
		o := parseSimple(rest, map[string]bool{"--insecure": true, "--draft": true, "--no-draft": true}, map[string]bool{"--quant": true, "-q": true})
		if len(o.pos) < 1 {
			die("Error: requires at least 1 arg(s), only received 0")
		}
		cmdPull(o.pos[0], first(o.vals["--quant"], o.vals["-q"]))
		if !o.flags["--no-draft"] {
			offerDraft(o.pos[0])
		}
	case "rm":
		o := parseSimple(rest, nil, nil)
		if len(o.pos) < 1 {
			die("Error: requires at least 1 arg(s), only received 0")
		}
		cmdRm(o.pos)
	case "stop":
		o := parseSimple(rest, nil, nil)
		if len(o.pos) < 1 {
			die("Error: requires at least 1 arg(s), only received 0")
		}
		cmdStop(o.pos)
	case "serve":
		cmdServe()
	case "tray":
		cmdTray()
	case "pulldraft":
		o := parseSimple(rest, map[string]bool{"--yes": true, "-y": true, "--force": true}, nil)
		if len(o.pos) < 1 {
			die("Error: requires at least 1 arg(s), only received 0")
		}
		cmdPullDraft(o.pos[0], o.flags["--yes"] || o.flags["-y"], o.flags["--force"])
	case "doctor":
		cmdDoctor()
	case "update":
		o := parseSimple(rest, map[string]bool{"--force": true}, nil)
		cmdUpdate(o.flags["--force"])
	case "uninstall":
		o := parseSimple(rest, map[string]bool{"--keep": true}, nil)
		cmdUninstall(o.flags["--keep"])
	case "create":
		o := parseSimple(rest, map[string]bool{"--experimental": true},
			map[string]bool{"-f": true, "--file": true, "-q": true, "--quantize": true, "--draft-quantize": true})
		if len(o.pos) < 1 {
			die("Error: requires at least 1 arg(s), only received 0")
		}
		file := first(o.vals["-f"], o.vals["--file"], "Modelfile")
		cmdCreate(o.pos[0], file, first(o.vals["-q"], o.vals["--quantize"], ""), o.vals["--draft-quantize"])
	case "cp":
		o := parseSimple(rest, nil, nil)
		if len(o.pos) < 2 {
			die("Error: requires at least 2 arg(s), only received %d", len(o.pos))
		}
		cmdCp(o.pos[0], o.pos[1])
	case "push":
		cmdPush()
	case "signin":
		cmdSignin()
	case "signout":
		fmt.Println("not signed in to ollama.com (llmash is fully local)")
	case "link":
		o := parseSimple(rest, map[string]bool{"--off": true}, nil)
		cmdLink(o.flags["--off"])
	case "unlink":
		cmdUnlink()
	case "launch":
		cmdLaunch(rest)
	case "run":
		r := parseRun(rest)
		r.verbose = r.verbose || gVerbose
		cmdRun(r)
	}
}

func first(vals ...string) string {
	for _, v := range vals {
		if v != "" {
			return v
		}
	}
	return ""
}

type opts struct {
	pos   []string
	flags map[string]bool
	vals  map[string]string
}

// Flags in any position, `--x=v` or `--x v`; anything else starting with `-`
// is an error, the way cobra reports it.
func parseSimple(args []string, boolFlags map[string]bool, valFlags map[string]bool) opts {
	o := opts{flags: map[string]bool{}, vals: map[string]string{}}
	for i := 0; i < len(args); i++ {
		t := args[i]
		if !strings.HasPrefix(t, "-") || t == "-" {
			o.pos = append(o.pos, t)
			continue
		}
		name, val, hasVal := t, "", false
		if eq := strings.Index(t, "="); eq > 0 {
			name, val, hasVal = t[:eq], t[eq+1:], true
		}
		if boolFlags[name] {
			o.flags[name] = true
			continue
		}
		if valFlags[name] {
			if !hasVal {
				if i+1 >= len(args) {
					die("Error: flag needs an argument: %s", name)
				}
				i++
				val = args[i]
			}
			o.vals[name] = val
			continue
		}
		die("Error: unknown flag: '%s'", t)
	}
	return o
}

type showOpts struct {
	model                                                 string
	modelfile, template, parameters, system, lic, verbose bool
}

func parseShow(args []string) showOpts {
	o := parseSimple(args, map[string]bool{"--modelfile": true, "--template": true, "--parameters": true,
		"--system": true, "--license": true, "-v": true, "--verbose": true}, nil)
	if len(o.pos) < 1 {
		die("Error: requires at least 1 arg(s), only received 0")
	}
	return showOpts{model: o.pos[0], modelfile: o.flags["--modelfile"], template: o.flags["--template"],
		parameters: o.flags["--parameters"], system: o.flags["--system"], lic: o.flags["--license"],
		verbose: o.flags["-v"] || o.flags["--verbose"]}
}

type runOpts struct {
	model, prompt, format, keepalive, think string
	verbose, nowordwrap, hidethinking       bool
	thinkSet                                bool
	ctx, dimensions                         int
	truncate                                *bool
	temperature                             *float64
}

func parseRun(args []string) runOpts {
	var r runOpts
	var pos []string
	boolFlags := map[string]bool{"--verbose": true, "--nowordwrap": true, "--hidethinking": true, "--insecure": true}
	valFlags := map[string]bool{"--format": true, "--keepalive": true, "--dimensions": true, "--width": true,
		"--height": true, "--steps": true, "--seed": true, "--negative": true, "--ctx": true, "--temperature": true}
	for i := 0; i < len(args); i++ {
		t := args[i]
		if !strings.HasPrefix(t, "-") || t == "-" {
			pos = append(pos, t)
			continue
		}
		name, val, hasVal := t, "", false
		if eq := strings.Index(t, "="); eq > 0 {
			name, val, hasVal = t[:eq], t[eq+1:], true
		}
		switch {
		case name == "--think":
			// cobra's optional-value flag: bare --think is true, and only the
			// --think=level form carries a value.
			r.thinkSet = true
			r.think = "true"
			if hasVal {
				if !thinkLevels[val] {
					die("Error: invalid value for --think: %q (must be true, false, high, medium, low, or max)", val)
				}
				r.think = val
			}
		case name == "--truncate":
			// also an optional-value flag
			b := true
			if hasVal {
				var err error
				if b, err = strconv.ParseBool(val); err != nil {
					die("Error: invalid argument %q for \"--truncate\" flag", val)
				}
			}
			r.truncate = &b
		case boolFlags[name]:
			switch name {
			case "--verbose":
				r.verbose = true
			case "--nowordwrap":
				r.nowordwrap = true
			case "--hidethinking":
				r.hidethinking = true
			}
		case valFlags[name]:
			if !hasVal {
				if i+1 >= len(args) {
					die("Error: flag needs an argument: %s", name)
				}
				i++
				val = args[i]
			}
			switch name {
			case "--format":
				r.format = val
			case "--keepalive":
				r.keepalive = val
			case "--ctx":
				fmt.Sscan(val, &r.ctx)
			case "--dimensions":
				fmt.Sscan(val, &r.dimensions)
			case "--temperature":
				var f float64
				if _, err := fmt.Sscan(val, &f); err == nil {
					r.temperature = &f
				}
			}
		default:
			die("Error: unknown flag: '%s'", t)
		}
	}
	if len(pos) < 1 {
		die("Error: requires at least 1 arg(s), only received 0")
	}
	r.model = pos[0]
	r.prompt = strings.Join(pos[1:], " ")
	return r
}
