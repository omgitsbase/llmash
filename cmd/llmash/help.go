package main

// The help text, byte-for-byte what the real ollama.exe 0.32.5 prints, with
// llmash's own commands slotted in above `help`.
const helpText = `Large language model runner

Usage:
  ollama [flags]
  ollama [command]

Available Commands:
  serve        Start Ollama
  create       Create a model
  show         Show information for a model
  run          Run a model
  stop         Stop a running model
  pull         Pull a model from a registry
  install      Same as pull
  push         Push a model to a registry
  signin       Sign in to ollama.com
  signout      Sign out from ollama.com
  list         List models
  ps           List running models
  cp           Copy a model
  rm           Remove a model
  launch       Launch the Ollama menu or an integration
  link         Expose the server publicly over Tailscale Funnel
  unlink       Take the public API back down
  tray         Show the notification-area icon
  doctor       Check this machine over and report what is wrong
  update       Update llmash from the host it was installed from
  uninstall    Remove llmash from this machine
  help         Help about any command

Flags:
  -h, --help         help for ollama
      --nowordwrap   Don't wrap words to the next line automatically
      --verbose      Show timings for response
  -v, --version      Show version information

Use "ollama [command] --help" for more information about a command.
`

const hostEnv = `
Environment Variables:
      OLLAMA_HOST                   IP Address for the ollama server (default 127.0.0.1:11434)
`

var commandHelp = map[string]string{
	"install": `Same as pull: download a model

Usage:
  llmash install MODEL [flags]
`,
	"doctor": `Check this machine over and report what is wrong

Usage:
  llmash doctor

Reports the install, the server, the llama.cpp runtime, the model store, the
card and its headroom, what is resident, the optimizations being applied, the
fast-backend routes, the commands on PATH, and whether an update is waiting.
Exits non-zero when something is broken rather than merely worth a look.
`,
	"update": `Update llmash from the host it was installed from

Usage:
  llmash update [flags]

Flags:
      --force   reinstall even when the installed version is already current

Reads the latest release from GitHub and runs its installer when it is newer.
`,
	"uninstall": `Remove llmash from this machine

Usage:
  llmash uninstall [flags]

Flags:
      --keep   leave the install directory in place

Models are never removed; they stay in Ollama's model store.
`,
	"tray": `Show the notification-area icon

Usage:
  llmash tray
`,
	"serve": `Start Ollama

Usage:
  ollama serve [flags]

Aliases:
  serve, start

Flags:
  -h, --help   help for serve

Environment Variables:
      OLLAMA_HOST                   IP Address for the ollama server (default 127.0.0.1:11434)
      OLLAMA_KEEP_ALIVE             The duration that models stay loaded in memory (default "15m")
      OLLAMA_MODELS                 The path to the models directory
      LLMASH_GGUF                  The path to the loose GGUF directory (default: gguf/ inside the model store)
      LLMASH_PORT                  Port for the local API (default 11434)
      LLMASH_PUBLIC_PORT           Keyed public listener for ` + "`ollama link`" + ` (default 11435, 0 = off)
      LLMASH_CTX                   Default context length (default 8192)
      LLMASH_KV                    Quantization type for the K/V cache (default "q8_0")
      LLMASH_LOAD_MODE             How weights reach VRAM: dio or mmap (default "dio")
      LLMASH_VRAM_GB               VRAM budget for resident models (default 80)
      LLMASH_PIN                   Comma-separated models never evicted
      LLAMA_BIN                     Path to llama-server.exe
`,
	"create": `Create a model

Usage:
  ollama create MODEL [flags]

Flags:
      --draft-quantize string   Quantize draft model to this level
      --experimental            Enable experimental safetensors model creation
  -f, --file string             Name of the Modelfile (default "Modelfile")
  -h, --help                    help for create
  -q, --quantize string         Quantize model to this level (e.g. q4_K_M)
` + hostEnv,
	"show": `Show information for a model

Usage:
  ollama show MODEL [flags]

Flags:
  -h, --help         help for show
      --license      Show license of a model
      --modelfile    Show Modelfile of a model
      --parameters   Show parameters of a model
      --system       Show system message of a model
      --template     Show template of a model
  -v, --verbose      Show detailed model information
` + hostEnv,
	"run": `Run a model

Usage:
  ollama run MODEL [PROMPT] [flags]

Flags:
      --dimensions int          Truncate output embeddings to specified dimension (embedding models only)
      --format string           Response format (e.g. json)
  -h, --help                    help for run
      --hidethinking            Hide thinking output (if provided)
      --insecure                Use an insecure registry
      --keepalive string        Duration to keep a model loaded (e.g. 5m)
      --nowordwrap              Don't wrap words to the next line automatically
      --think string[="true"]   Enable thinking mode: true/false or high/medium/low for supported models
      --truncate                For embedding models: truncate inputs exceeding context length (default: true). Set --truncate=false to error instead
      --verbose                 Show timings for response

Image Generation Flags (experimental):
      --width int      Image width
      --height int     Image height
      --steps int      Denoising steps
      --seed int       Random seed
      --negative str   Negative prompt
` + hostEnv,
	"stop": `Stop a running model

Usage:
  ollama stop MODEL [flags]

Flags:
  -h, --help   help for stop
` + hostEnv,
	"pull": `Pull a model from a registry

Usage:
  ollama pull MODEL [flags]

Flags:
  -h, --help       help for pull
      --insecure   Use an insecure registry
` + hostEnv,
	"push": `Push a model to a registry

Usage:
  ollama push MODEL [flags]

Flags:
  -h, --help       help for push
      --insecure   Use an insecure registry
` + hostEnv,
	"signin": `Sign in to ollama.com

Usage:
  ollama signin [flags]

Flags:
  -h, --help   help for signin
`,
	"signout": `Sign out from ollama.com

Usage:
  ollama signout [flags]

Flags:
  -h, --help   help for signout
`,
	"list": `List models

Usage:
  ollama list [flags]

Aliases:
  list, ls

Flags:
  -h, --help   help for list
` + hostEnv,
	"ps": `List running models

Usage:
  ollama ps [flags]

Flags:
  -h, --help   help for ps
` + hostEnv,
	"cp": `Copy a model

Usage:
  ollama cp SOURCE DESTINATION [flags]

Flags:
  -h, --help   help for cp
` + hostEnv,
	"rm": `Remove a model

Usage:
  ollama rm MODEL [MODEL...] [flags]

Flags:
  -h, --help   help for rm
` + hostEnv,
	"launch": `Launch the Ollama interactive menu, or directly launch a specific integration.

Without arguments, this is equivalent to running 'ollama' directly.
Flags and extra arguments require an integration name.

Supported integrations:
  claude          Claude Code
  chatgpt         ChatGPT (aliases: codex-app, codex-desktop, codex-gui)
  hermes          Hermes Agent
  openclaw        OpenClaw (aliases: clawdbot, moltbot)
  opencode        OpenCode
  codex           Codex
  hermes-desktop  Hermes Desktop
  copilot         Copilot CLI (aliases: copilot-cli)
  omp             OMP
  droid           Droid
  kimi            Kimi Code CLI
  pi              Pi
  pool            Pool
  cline           Cline
  qwen            Qwen Code
  vscode          VS Code (aliases: code)

Examples:
  ollama launch
  ollama launch claude
  ollama launch claude --model <model>
  ollama launch chatgpt
  ollama launch chatgpt --restore
  ollama launch hermes
  ollama launch hermes-desktop
  ollama launch droid --config (does not auto-launch)
  ollama launch codex --restore
  ollama launch codex -- --sandbox workspace-write

Usage:
  ollama launch [INTEGRATION] [-- [EXTRA_ARGS...]] [flags]

Flags:
      --config         Configure without launching
  -h, --help           help for launch
      --model string   Model to use
      --restore        Restore an integration to its default profile
  -y, --yes            Automatically answer yes to confirmation prompts
`,
	"link": `Expose the server publicly over Tailscale Funnel

Usage:
  ollama link [flags]

Flags:
  -h, --help   help for link
      --off    show the link without (re)starting the tunnel
`,
	"unlink": `Take the public API back down

Usage:
  ollama unlink [flags]

Flags:
  -h, --help   help for unlink
`,
	"help": `Help provides help for any command in the application.
Simply type ollama help [path to command] for full details.

Usage:
  ollama help [command] [flags]

Flags:
  -h, --help   help for help
`,
}
