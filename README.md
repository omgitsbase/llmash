# llmash

An Ollama-compatible server and command line for Windows, built on llama.cpp.

Same commands, same API, same model store. It runs your GGUF files through
`llama-server` instead of Ollama's runtime, keeps weights out of system RAM,
and turns on the llama.cpp options that are off by default.

```powershell
irm https://raw.githubusercontent.com/itsTurdle/llmash/main/install.ps1 | iex
```

Nothing has to be installed first. The download is two compiled executables and
an icon; no Python, no runtime of any kind. The installer fetches the llama.cpp
build that matches your GPU, puts `llmash` on your PATH, and starts the server.

## What it does differently

**Weights stay out of RAM.** llama.cpp memory-maps a model by default, so a
model living entirely in VRAM still counts against your working set. llmash
loads with `--load-mode dio`, which reads weights straight to the card. A 20 GB
model went from 20.0 GB of RAM in use to 0.8 GB, and loaded faster.

**Every model gets speculative decoding.** Models with a draft head (MTP,
DSpark, EAGLE-3) use it. Everything else gets `ngram-mod`, which drafts from the
text already in the context and needs no draft model and no extra VRAM.

**Tuned at launch, not by hand.** Prompt-prefix reuse, a host-RAM prompt cache
sized from what is free, a prompt batch wide enough to keep a large card busy,
and a priority bump. Each is overridable, and what was chosen is in the log.

**A tray icon.** Unload a model, set how long it stays loaded, restart the
server, start at login.

**One binary.** `llmash.exe` is the command line and the server. `llmashw.exe`
is the same program without a console, for the tray.

## Commands

Everything Ollama's CLI does, plus a few of its own.

| | |
|---|---|
| `llmash list`, `ps`, `show`, `run`, `pull`, `rm`, `cp`, `stop` | as in Ollama |
| `llmash serve` | start the server (the tray does this for you) |
| `llmash doctor` | check the install, runtime, GPU, models and routes |
| `llmash update` | install the latest release |
| `llmash launch` | point Claude Code, Codex, Droid and others at this server |
| `llmash link` | expose the API over a Tailscale funnel, with a key |
| `llmash uninstall` | remove everything the installer created |

The `ollama` command is installed as an alias, so anything that shells out to
Ollama keeps working.

## Configuration

Optional. `local.json` next to the program, or environment variables:

| | |
|---|---|
| `LLMASH_PORT` | local API port (default 11434) |
| `LLMASH_CTX` | default context length (default 8192) |
| `LLMASH_KV` | K/V cache type, `f16` or `q8_0` |
| `LLMASH_VRAM_GB` | budget for resident models |
| `LLMASH_PIN` | comma-separated models never evicted |
| `LLMASH_SPEC_FALLBACK` | drafter for models without one (default `ngram-mod`) |
| `LLMASH_TUNE_OFF` | turn off individual auto-tuning: `cache-reuse,cache-ram,batch,prio` |

`llmash serve --help` lists the rest.

### Fast routes

A `routes.json` beside the program sends a model to another
OpenAI-compatible server when one is faster for it, and back to llama.cpp when
that server is not up. The route can name an executable to start on demand or a
Docker container to start. See `routes.example.json`.

## Building

Go 1.26 and MinGW's `windres` for the icon.

```powershell
python build.py           # dist/llmash-win-x64.zip
python build.py --here    # ...and run this checkout on it
```

## License

MIT.
