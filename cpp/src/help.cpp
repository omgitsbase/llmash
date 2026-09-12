#include "help.h"

#include <unordered_map>
#include <utility>
#include <vector>

namespace llmash {

namespace {

// A single left-to-right pass trying each pair at the current position in
// argument order, same semantics as Go's strings.Replacer.
std::string multi_replace(const std::string & text, const std::vector<std::pair<std::string, std::string>> & pairs) {
    std::string out;
    out.reserve(text.size());
    size_t i = 0;
    while (i < text.size()) {
        bool matched = false;
        for (const auto & pr : pairs) {
            const std::string & from = pr.first;
            if (!from.empty() && text.compare(i, from.size(), from) == 0) {
                out += pr.second;
                i += from.size();
                matched = true;
                break;
            }
        }
        if (!matched) {
            out += text[i];
            i++;
        }
    }
    return out;
}

const std::string k_help_text = R"HELP(Large language model runner

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
  pulldraft    Find and install a draft model to make a model faster
  models       Show where models are read from, or point llmash at a folder
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
)HELP";

const std::string k_host_env = R"HOSTENV(
Environment Variables:
      OLLAMA_HOST                   IP Address for the ollama server (default 127.0.0.1:11434)
)HOSTENV";

const std::string k_install_help = R"HELP(Same as pull: download a model

Usage:
  llmash install MODEL [flags]
)HELP";

const std::string k_pulldraft_help = R"HELP(Find and install a draft model to make a model faster

Usage:
  llmash pulldraft MODEL [flags]

Flags:
  -y, --yes     do not ask before downloading
      --force   fetch it again when one is already installed

A draft model guesses the next few tokens so the real model can check several
at once. Hugging Face is searched for one trained against this exact model;
candidates built for a fine-tune, or packaged for another runtime, are refused.
A model with an MTP head of its own is left alone. No account is needed.
)HELP";

const std::string k_models_help = R"HELP(Show where models are read from, or point llmash at a folder

Usage:
  llmash models
  llmash models set DIR

With no folder, shows where models are read from and how many are there.

set reads a folder for what it is. Ollama's store, found on its own when
there is one, or any folder with a manifests folder inside, becomes the model
store, where pulls go. Any other folder is taken as a folder of GGUFs, the
kind llama.cpp keeps, and is read where it is, subfolders included: nothing
in it is copied, moved or deleted, and rm will not touch it.

It is the same setting as OLLAMA_MODELS, which llmash takes the way Ollama
does, pointed at either kind of folder. LLMASH_MODELS is the same variable
under this program's name.
)HELP";

const std::string k_doctor_help = R"HELP(Check this machine over and report what is wrong

Usage:
  llmash doctor

Reports the install, the server, the llama.cpp runtime, the model store, the
card and its headroom, what is resident, the optimizations being applied, the
fast-backend routes, the commands on PATH, and whether an update is waiting.
Exits non-zero when something is broken rather than merely worth a look.
)HELP";

const std::string k_update_help = R"HELP(Update llmash from the host it was installed from

Usage:
  llmash update [flags]

Flags:
      --force   reinstall even when the installed version is already current

Reads the latest release from GitHub and runs its installer when it is newer.
)HELP";

const std::string k_uninstall_help = R"HELP(Remove llmash from this machine

Usage:
  llmash uninstall [flags]

Flags:
      --keep   leave the install directory in place

Models are never removed; they stay in Ollama's model store.
)HELP";

const std::string k_tray_help = R"HELP(Show the notification-area icon

Usage:
  llmash tray
)HELP";

const std::string k_serve_help_body = R"HELP(Start Ollama

Usage:
  ollama serve [flags]

Aliases:
  serve, start

Flags:
  -h, --help   help for serve

Environment Variables:
      OLLAMA_HOST                   IP Address for the ollama server (default 127.0.0.1:11434)
      OLLAMA_KEEP_ALIVE             The duration that models stay loaded in memory (default "15m")
      OLLAMA_MODELS                 The path to the models directory: an Ollama store, or a folder of GGUFs read in place
      LLMASH_MODELS                Same as OLLAMA_MODELS
      LLMASH_GGUF                  The path to the loose GGUF directory (default: gguf/ inside the model store)
      LLMASH_EXTRA_ROOTS           Other Ollama stores read alongside the models directory, ';' between them
      LLMASH_PORT                  Port for the local API (default 11434)
      LLMASH_PUBLIC_PORT           Keyed public listener for )HELP" "`ollama link`" R"HELP( (default 11435, 0 = off)
      LLMASH_CTX                   Default context length (default 8192)
      LLMASH_KV                    Quantization type for the K/V cache (default "q8_0")
      LLMASH_LOAD_MODE             How weights reach VRAM: dio or mmap (default "dio")
      LLMASH_VRAM_GB               VRAM budget for resident models (default 80)
      LLMASH_PIN                   Comma-separated models never evicted
      LLAMA_BIN                     Path to llama-server.exe
)HELP";

const std::string k_create_help = R"HELP(Create a model

Usage:
  ollama create MODEL [flags]

Flags:
      --draft-quantize string   Quantize draft model to this level
      --experimental            Enable experimental safetensors model creation
  -f, --file string             Name of the Modelfile (default "Modelfile")
  -h, --help                    help for create
  -q, --quantize string         Quantize model to this level (e.g. q4_K_M)
)HELP";

const std::string k_show_help = R"HELP(Show information for a model

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
)HELP";

const std::string k_run_help = R"HELP(Run a model

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
)HELP";

const std::string k_stop_help = R"HELP(Stop a running model

Usage:
  ollama stop MODEL [flags]

Flags:
  -h, --help   help for stop
)HELP";

const std::string k_pull_help = R"HELP(Pull a model from a registry

Usage:
  ollama pull MODEL [flags]

Flags:
  -h, --help       help for pull
  -q, --quant Q    which build of a Hugging Face repository to take (Q4_K_M, IQ4_XS, Q8_0 ...)
      --insecure   Use an insecure registry
      --no-draft   do not offer to fetch a draft model afterwards

A Hugging Face repository is written hf.co/<org>/<repo>. Without --quant, the
builds it offers are listed and one is chosen; hf.co/<org>/<repo>@<quant> names
one directly. Files come down over several connections at once.

RCO-<bits> asks for a build assembled here at that many bits a weight. The
list offers a ladder of widths; --quant takes any of them, RCO-4.2 included.
)HELP";

const std::string k_push_help = R"HELP(Push a model to a registry

Usage:
  ollama push MODEL [flags]

Flags:
  -h, --help       help for push
      --insecure   Use an insecure registry
)HELP";

const std::string k_signin_help = R"HELP(Sign in to ollama.com

Usage:
  ollama signin [flags]

Flags:
  -h, --help   help for signin
)HELP";

const std::string k_signout_help = R"HELP(Sign out from ollama.com

Usage:
  ollama signout [flags]

Flags:
  -h, --help   help for signout
)HELP";

const std::string k_list_help = R"HELP(List models

Usage:
  ollama list [flags]

Aliases:
  list, ls

Flags:
  -h, --help   help for list
)HELP";

const std::string k_ps_help = R"HELP(List running models

Usage:
  ollama ps [flags]

Flags:
  -h, --help   help for ps
)HELP";

const std::string k_cp_help = R"HELP(Copy a model

Usage:
  ollama cp SOURCE DESTINATION [flags]

Flags:
  -h, --help   help for cp
)HELP";

const std::string k_rm_help = R"HELP(Remove a model

Usage:
  ollama rm MODEL [MODEL...] [flags]

Flags:
  -h, --help   help for rm
)HELP";

const std::string k_launch_help = R"HELP(Launch the Ollama interactive menu, or directly launch a specific integration.

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
)HELP";

const std::string k_link_help = R"HELP(Expose the server publicly over Tailscale Funnel

Usage:
  ollama link [flags]

Flags:
  -h, --help   help for link
      --off    show the link without (re)starting the tunnel
)HELP";

const std::string k_unlink_help = R"HELP(Take the public API back down

Usage:
  ollama unlink [flags]

Flags:
  -h, --help   help for unlink
)HELP";

const std::string k_help_topic_help = R"HELP(Help provides help for any command in the application.
Simply type ollama help [path to command] for full details.

Usage:
  ollama help [command] [flags]

Flags:
  -h, --help   help for help
)HELP";

const std::unordered_map<std::string, std::string> & command_help_map() {
    static const std::unordered_map<std::string, std::string> m = {
        {"install", k_install_help},
        {"pulldraft", k_pulldraft_help},
        {"models", k_models_help},
        {"doctor", k_doctor_help},
        {"update", k_update_help},
        {"uninstall", k_uninstall_help},
        {"tray", k_tray_help},
        {"serve", k_serve_help_body},
        {"create", k_create_help + k_host_env},
        {"show", k_show_help + k_host_env},
        {"run", k_run_help + k_host_env},
        {"stop", k_stop_help + k_host_env},
        {"pull", k_pull_help + k_host_env},
        {"push", k_push_help + k_host_env},
        {"signin", k_signin_help},
        {"signout", k_signout_help},
        {"list", k_list_help + k_host_env},
        {"ps", k_ps_help + k_host_env},
        {"cp", k_cp_help + k_host_env},
        {"rm", k_rm_help + k_host_env},
        {"launch", k_launch_help},
        {"link", k_link_help},
        {"unlink", k_unlink_help},
        {"help", k_help_topic_help},
    };
    return m;
}

} // namespace

const std::string & help_text() { return k_help_text; }

const std::string * command_help(const std::string & topic) {
    const auto & m = command_help_map();
    const auto it = m.find(topic);
    return it == m.end() ? nullptr : &it->second;
}

std::string prog_text(const std::string & text, const std::string & prog) {
    if (prog == "ollama") {
        return text;
    }
    return multi_replace(text, {
        {"\n  ollama ", "\n  " + prog + " "},
        {"Use \"ollama [command]", "Use \"" + prog + " [command]"},
        {"help for ollama", "help for " + prog},
        {"Start Ollama", "Start " + prog},
        {"`ollama ", "`" + prog + " "},
    });
}

} // namespace llmash
