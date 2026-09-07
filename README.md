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

## Speed

One GPU, an RTX PRO 6000 Blackwell with 96 GB: same prompts, 4-bit weights in
each engine's own format, every backend run as it comes with no hand tuning.
Your numbers will differ.

<!-- BENCHMARK -->

**gemma-4 26B-A4B (Q4_K_M)**

| backend | conversation | coding | thinking |
|---|--:|--:|--:|
| Ollama | 173.0 | 189.5 | 118.5 |
| **llmash** | **254.2** | **332.1** | **417.3** |

**Qwen3.6 35B-A3B (IQ4_XS)**

| backend | conversation | coding | thinking |
|---|--:|--:|--:|
| Ollama | 129.8 | 120.3 | 226.8 |
| vLLM | 182.6 | 182.0 | 182.1 |
| **llmash** | **382.5** | **473.6** | **506.7** |

**Qwen3.8 27B (Q4_K_XL)**

| backend | conversation | coding | thinking |
|---|--:|--:|--:|
| Ollama | 63.1 | 62.3 | 63.4 |
| vLLM | 72.9 | 73.7 | 73.7 |
| **llmash** | **130.1** | **160.1** | **187.4** |

Tokens per second while generating, median of three runs, excluding model load and prompt processing.

<!-- /BENCHMARK -->

vLLM here is the native Windows build on AWQ int4 weights. gemma-4 has no vLLM
row because that build gives every layer one head_dim, and gemma-4 does not: 25
of its layers are 256 wide and 5 are 512.

## How it fits together

- **`llmash.exe`** is the command line and the server. `llmashw.exe` is the same
  program with no console, for the tray.
- **The server** owns the model store, starts and stops `llama-server`
  processes, and decides how long each one stays resident.
- **The tray** unloads a model, sets the keep-alive, restarts the server, and
  starts it at login.
- **Routes** send a named model to another OpenAI-compatible server, and fall
  back to llama.cpp when that server is not running.

## What it turns on

llama.cpp options that are off by default. llmash sets each per model at launch,
logs the decision, and lets you override it.

| | |
|---|---|
| Speculative decoding | A model with an MTP head uses it. Otherwise a draft model beside it, or `ngram-mod`, which drafts from the context and costs no VRAM. |
| One graph per round | The whole speculative round is one CUDA graph: the replay, the decision about how many tokens were accepted, and every draft step, with the accept decided on the GPU rather than read back. The two graphs hand off device to device, ordered by an event, so nothing waits on a forward pass. Measured at 424 tok/s against 373 for a decode per drafted token. |
| Prompt-prefix reuse | `--cache-reuse`, so a repeated prefix is not processed twice. |
| Host-RAM prompt cache | Sized from free RAM, between 8 and 32 GiB. |
| Batch size | A wider prompt batch when the card has the VRAM for it. |
| Process priority | Raised, so background work does not stall generation. |
| DirectIO loading | `--load-mode dio` reads weights straight to the card instead of memory-mapping them. |

## Draft models

A draft model guesses the next few tokens so the real model can check several at
once. `pulldraft` finds one for a model you have, and a pull offers the same
thing when it finishes.

Candidates come from Hugging Face; no account, no token. Repositories built for
a fine-tune of the model, or packaged for another runtime, are refused, and the
rest are checked against the weights they would serve before anything is
downloaded: matching vocabularies, an encoder shaped for this model's hidden
size, and the layers it reads present. A model with an MTP head of its own is
left alone.

Measured on this machine: gemma-4 26B-A4B went from 244 to 342 tok/s on a
fetched EAGLE-3 head. Qwen3.6 35B-A3B ran 308 tok/s on its own MTP head against
265 on a downloaded one.

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

`ollama` is installed as an alias.

## Configuration

Optional. `local.json` next to the program, or environment variables.

| | |
|---|---|
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
