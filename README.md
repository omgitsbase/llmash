# llmash

[![Release](https://img.shields.io/github/v/release/omgitsbase/llmash?include_prereleases&label=release)](https://github.com/omgitsbase/llmash/releases/latest)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/omgitsbase/llmash/blob/main/LICENSE)
[![Windows](https://img.shields.io/badge/Windows-10%2F11-0078D6.svg?logo=windows)](https://github.com/omgitsbase/llmash/releases/latest)
[![Go](https://img.shields.io/badge/Go-1.26-00ADD8.svg?logo=go)](https://go.dev)
[![CUDA](https://img.shields.io/badge/CUDA-13.3-76B900.svg?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/omgitsbase/llmash)

An Ollama-compatible server and command line for Windows, built on llama.cpp.

It serves your GGUF files through `llama-server` and keeps Ollama's commands,
API and model store, so anything already pointed at Ollama keeps working. What
it changes is the llama.cpp settings, picked per model at launch.

```powershell
irm https://raw.githubusercontent.com/omgitsbase/llmash/main/install.ps1 | iex
```

Nothing needs to be installed first: the installer fetches the llama.cpp build
for your GPU, puts `llmash` on your PATH, and starts the server. With no
NVIDIA card it takes the Vulkan or CPU build instead, and models load into
system RAM, which works but is slower.

## Models you already have

Ollama's store is found and used as it is. A folder of GGUFs from llama.cpp
or LM Studio is one command away:

```powershell
llmash models set D:\models
```

It reads the folder where it is, subfolders included, and says how many
models it found. Nothing is copied, downloaded or deleted, and `rm` will not
touch it. `llmash models` shows what is being read. It is the same setting
as `OLLAMA_MODELS`, which llmash takes the way Ollama does, pointed at
either kind of folder.

## Speed

One GPU, an RTX PRO 6000 Blackwell with 96 GB: same prompts, 4-bit weights in
each engine's own format, every backend run as it comes with no hand tuning.
Your numbers will differ.

<!-- BENCHMARK -->

**gemma-4 26B-A4B (Q4_K_M)**

| backend | conversation | coding | thinking |
|---|--:|--:|--:|
| Ollama | 173.0 | 189.5 | 118.5 |
| **llmash** | **255.6** | **338.6** | **432.0** |

**Qwen3.6 35B-A3B (IQ4_XS)**

| backend | conversation | coding | thinking |
|---|--:|--:|--:|
| Ollama | 129.8 | 120.3 | 226.8 |
| vLLM | 182.6 | 182.0 | 182.1 |
| **llmash** | **385.0** | **482.0** | **514.4** |

**Qwen3.8 27B (Q4_K_XL)**

| backend | conversation | coding | thinking |
|---|--:|--:|--:|
| Ollama | 63.1 | 62.3 | 63.4 |
| vLLM | 72.9 | 73.7 | 73.7 |
| **llmash** | **129.9** | **160.1** | **188.1** |

Tokens per second while generating, median of three runs, excluding model load and prompt processing.

<!-- /BENCHMARK -->

vLLM is the native Windows build on AWQ int4 weights. gemma-4 has no vLLM row
because that build cannot run its mixed head sizes.

## How it fits together

- **`llmash.exe`** is the command line and the server. `llmashw.exe` is the same
  program with no console, for the tray.
- **The server** owns the model store, starts and stops `llama-server`
  processes, and decides how long each one stays resident.
- **The tray** unloads a model, sets the keep-alive, restarts the server, and
  starts it at login.
- **Routes** send a named model to another OpenAI-compatible server, and fall
  back to llama.cpp when that server is not running.

Per model, at launch, it picks the ordinary llama.cpp settings and logs each:
prompt-prefix reuse, a host-RAM prompt cache sized from free RAM, batch width
when the card has room, raised process priority, and DirectIO loading.
`LLMASH_TUNE_OFF` disables any of them.

## Speculation

A model carrying an MTP head drafts for itself, with nothing to fetch or
configure. The whole round is one CUDA graph: the verify batch, the accept
decision and every draft step, with the accept decided on the GPU rather than
read back to the host. Measured at 424 tok/s against 373 for the same model
drafting one token per decode.

For a model with no head, `pulldraft` finds a drafter on Hugging Face, checked
against the weights before anything downloads. gemma-4 26B-A4B went from 244
to 342 tok/s on a fetched EAGLE-3 head.

## Kernels

Decode on this card is launch-bound: a graph is 923 kernel launches at about
2.3 µs each, so close to half of a 4.66 ms graph is dispatch. The CUDA work is
therefore fewer launches, not faster kernels. An elementwise-chain pass
collapses runs of elementwise ops into one launch, and a residual add is folded
into the rms_norm that reads it. Both apply to any model with the pattern, and
both have a switch.

## Commands

| | |
|---|---|
| `list` `ps` `show` `run` `pull` `rm` `cp` `stop` | as in Ollama |
| `serve` | start the server; the tray does this for you |
| `pulldraft` | find and install a draft model for a model you have |
| `models` | show where models are read from, or point llmash at a folder of them |
| `doctor` | check the install, runtime, GPU, models and routes |
| `update` | install the latest release |
| `launch` | point Claude Code, Codex, Droid and others at this server |
| `link` | expose the API over a Tailscale funnel, with a key |
| `uninstall` | remove everything the installer created |

`ollama` is installed as an alias.

## Configuration

Optional. `local.json` next to the program, or environment variables.

| | |
|---|---|
| `OLLAMA_MODELS` | the models directory, as Ollama takes it: a store, or a folder of GGUFs read in place (default `~\.ollama\models`) |
| `LLMASH_MODELS` | same as `OLLAMA_MODELS`; `llmash models set` sets the same thing |
| `LLMASH_PORT` | local API port (default 11434) |
| `LLMASH_CTX` | default context length (default 8192) |
| `LLMASH_KV` | K/V cache type, `f16` or `q8_0` |
| `LLMASH_PARALLEL` | server slots (default 1; raise it to serve several at once) |
| `LLMASH_VRAM_GB` | budget for resident models |
| `LLMASH_PIN` | comma-separated models never evicted |
| `LLMASH_SPEC_FALLBACK` | drafter for models without one (default `ngram-mod`) |
| `LLMASH_TUNE_OFF` | disable individual tuning: `cache-reuse,cache-ram,batch,prio` |

`llmash serve --help` lists the rest. A `routes.json` beside the program
configures fast routes; see `routes.example.json`.

## Building

Go 1.26, and MinGW's `windres` for the icon.

```powershell
python build.py           # dist/llmash-win-x64.zip
python build.py --here    # ...and run this checkout on it
```

`python bench/bench.py --help` reproduces the table above.

## License

MIT.
