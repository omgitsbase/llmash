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

It also picks the ordinary llama.cpp settings per model at launch, and logs each
choice: prompt-prefix reuse, a host-RAM prompt cache sized from free RAM, batch
width when the card has room, raised process priority, and DirectIO loading.
`LLMASH_TUNE_OFF` disables any of them.

## The speculative round is one CUDA graph

A model carrying an MTP head drafts for itself: no draft model to fetch, no
pairing to declare, nothing to configure. If the head is in the weights, it is
used.

The round then runs as a single graph rather than a decode per drafted token.
The replay of the verify batch, the decision about how many tokens the target
accepted, and every draft step are one graph; the accept is decided on the GPU
instead of read back to the host; and the target and draft graphs hand off
device to device, ordered by an event, so nothing waits on a forward pass
mid-round. Measured at 424 tok/s against 373 for the same model drafting a
token per decode.

Speculation itself is no longer unusual, and other runners have it. Running the
round without returning to the host is the part that is ours.

What is left is the kernels, and there is less there than it looks. A round
costs 5.8 ms of GPU on this card for 2.9 accepted tokens, so 2.0 ms a token.
Verifying four drafted tokens costs 1.34x a single-token pass, against a floor
near 1.0x if the weights were read once and reused across the batch; for a
mixture-of-experts model the four tokens route to different experts, so much of
that 0.34 is weight the card genuinely has to fetch, not waste.

Three ways at it were measured and all three lost. Handing the batch to the
tensor-core quantized path is 15% slower, so llama.cpp's crossover is already
right. Doubling the warps per block changes nothing. Reordering the inner loop
so that calls sharing a weight block sit together changes nothing either, which
says the compiler was already hoisting those loads.

They lost because they were aimed at the wrong half. A decode graph on this
model is 923 kernel launches, and a launch costs about 2.3 microseconds here,
measured by fusing 38 of them away and watching the graph time move. That puts
roughly 2.1 ms of a 4.66 ms graph in dispatch rather than in arithmetic or
memory. The model is launch-bound, so the lever is fewer and larger kernels,
not faster ones. Making one kernel quicker cannot reach the 45% that is spent
getting to kernels at all.

So that is what the CUDA work here is. An elementwise-chain pass collapses runs
of elementwise ops into one launch, 2141 nodes a graph, and it widens when a
value computed over few elements is broadcast into many, which is how a gate
reaches the tensor it gates. A residual add is folded into the rms_norm and
weight-multiply that read it, another 30 launches; llama.cpp fuses the mirror
image, norm-then-add, but not the add-then-norm every block opens with, and the
sum has to be written out as well because the next block reads it. Both apply
to any model that has the pattern, with no per-model configuration, and both
have a switch. Neither is large alone, 0.8% for the residual fold, but they are
the shape the remaining work takes.

## Draft models

For a model with no MTP head, `pulldraft` finds one on Hugging Face and a pull
offers the same when it finishes. No account, no token. Candidates are checked
against the weights they would serve before anything downloads: matching
vocabularies, an encoder shaped for this model's hidden size, and the layers it
reads present. A model with a head of its own is left alone.

Measured here: gemma-4 26B-A4B went from 244 to 342 tok/s on a fetched EAGLE-3
head.

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
