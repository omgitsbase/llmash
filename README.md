# llmash

An Ollama-compatible server and command line for Windows, on llama.cpp.

Same commands, same API, same model store. It serves your GGUF files through
`llama-server`, and turns on the llama.cpp options that are off by default.

```powershell
irm https://raw.githubusercontent.com/itsTurdle/llmash/main/install.ps1 | iex
```

Nothing needs to be installed first. The download is two executables and an
icon. The installer fetches the llama.cpp build that matches your GPU, puts
`llmash` on your PATH, and starts the server.

## Speed

Tokens per second, same GPU, same prompts, same quantisation.

<!-- BENCHMARK -->

## What it does differently

**Speculative decoding for every model.** Models with a draft head (MTP,
DSpark, EAGLE-3) use it. The rest get `ngram-mod`, which drafts from the text
already in the context and needs no second model and no extra VRAM.

**Settings chosen at launch.** Prompt-prefix reuse, a host-RAM prompt cache
sized from what is free, a prompt batch wide enough to keep a large card busy,
and a priority bump. Each is overridable and each decision is logged.

**DirectIO loading.** llama.cpp memory-maps weights by default. `--load-mode
dio` reads them straight to the card instead, which loads faster and keeps the
working set off the page cache.

**A tray icon.** Unload a model, set how long it stays loaded, restart the
server, start at login.

**One binary.** `llmash.exe` is the command line and the server. `llmashw.exe`
is the same program without a console, for the tray.

## Commands

| | |
|---|---|
| `list` `ps` `show` `run` `pull` `rm` `cp` `stop` | as in Ollama |
| `serve` | start the server; the tray does this for you |
| `doctor` | check the install, runtime, GPU, models and routes |
| `update` | install the latest release |
| `launch` | point Claude Code, Codex, Droid and others at this server |
| `link` | expose the API over a Tailscale funnel, with a key |
| `uninstall` | remove everything the installer created |

`ollama` is installed as an alias, so anything already pointed at Ollama keeps
working.

## Configuration

Optional. `local.json` next to the program, or environment variables.

| | |
|---|---|
| `LLMASH_PORT` | local API port (default 11434) |
| `LLMASH_CTX` | default context length (default 8192) |
| `LLMASH_KV` | K/V cache type, `f16` or `q8_0` |
| `LLMASH_VRAM_GB` | budget for resident models |
| `LLMASH_PIN` | comma-separated models never evicted |
| `LLMASH_SPEC_FALLBACK` | drafter for models without one (default `ngram-mod`) |
| `LLMASH_TUNE_OFF` | disable individual tuning: `cache-reuse,cache-ram,batch,prio` |

`llmash serve --help` lists the rest.

### Fast routes

A `routes.json` beside the program sends a model to another OpenAI-compatible
server when one is faster for it, and back to llama.cpp when that server is
not up. A route names either an executable to start on demand or a Docker
container. See `routes.example.json`.

## Building

Go 1.26, and MinGW's `windres` for the icon.

```powershell
python build.py           # dist/llmash-win-x64.zip
python build.py --here    # ...and run this checkout on it
```

`python bench/bench.py --help` reproduces the table above.

## License

MIT.
