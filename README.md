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

**gemma-4 26B-A4B (Q4_K_M)**

| backend | conversation | coding | thinking |
|---|--:|--:|--:|
| Ollama | 173.0 | 189.5 | 118.5 |
| **llmash** | **177.6** | **179.1** | **179.8** |

**Qwen3.6 35B-A3B (IQ4_XS)**

| backend | conversation | coding | thinking |
|---|--:|--:|--:|
| Ollama | 129.8 | 120.3 | 226.8 |
| **llmash** | **513.2** | **599.6** | **596.7** |

**Qwen3.8 27B (Q4_K_XL)**

| backend | conversation | coding | thinking |
|---|--:|--:|--:|
| Ollama | 63.1 | 62.3 | 63.4 |
| **llmash** | **136.5** | **135.5** | **147.0** |

Tokens per second while generating, median of three runs, excluding model load and prompt processing.

## What it does differently

**Speculative decoding for every model.** Models with a draft head (MTP,
DSpark, EAGLE-3) use it. The rest get `ngram-mod`, which drafts from the text
already in the context and needs no second model and no extra VRAM.

**Draft models found for you.** A pull offers to fetch the matching draft head
if one exists, and `pulldraft` does it on demand. Candidates are read from
Hugging Face without an account or a token, and each one is checked against the
model it would serve before anything is downloaded: same vocabulary, an encoder
shaped for this model's hidden size, and layers this model actually has. A
model that ships its own head keeps it.

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
| `pulldraft` | find and install a draft model for a model you have |
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
