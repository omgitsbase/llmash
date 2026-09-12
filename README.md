# llmash

[![Release](https://img.shields.io/github/v/release/omgitsbase/llmash?include_prereleases&label=release)](https://github.com/omgitsbase/llmash/releases/latest)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/omgitsbase/llmash/blob/main/LICENSE)
[![Windows](https://img.shields.io/badge/Windows-10%2F11-0078D6.svg?logo=windows)](https://github.com/omgitsbase/llmash/releases/latest)
[![C++](https://img.shields.io/badge/C%2B%2B-17-00599C.svg?logo=cplusplus)](https://isocpp.org)
[![CUDA](https://img.shields.io/badge/CUDA-13.3-76B900.svg?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/omgitsbase/llmash)

An Ollama-compatible server and command line for Windows, built on llama.cpp.
There is an alpha Linux build too.

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

On Linux:

```sh
curl -fsSL https://raw.githubusercontent.com/omgitsbase/llmash/main/install.sh | sh
```

Same idea, under systemd, finding any models an existing Ollama has. `pull` is
not implemented there yet, so models have to already be on disk. llama.cpp
publishes no Linux CUDA build, so a GPU gets Vulkan and everything else gets
CPU; `--runtime` overrides that, `--user` keeps it in your home directory and
`--uninstall` removes it.

## Models you already have

Ollama's store is found and used as it is. A folder of GGUFs from llama.cpp
or LM Studio is one command away:

```powershell
llmash models set D:\models
```

It reads that folder in place, subfolders included. Nothing is copied or
deleted, and `rm` will not touch it. `llmash models` shows what is being read.
This is the same setting as `OLLAMA_MODELS`.

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
  back to llama.cpp when that server is not running. Not yet in the C++ build.

Per model, at launch, it picks the ordinary llama.cpp settings and logs each:
prompt-prefix reuse, a host-RAM prompt cache sized from free RAM, batch width
when the card has room, raised process priority, and DirectIO loading.
`LLMASH_TUNE_OFF` disables any of them.

## Custom builds

`pull` lists a build assembled here beside the published ones. Every tensor is
measured at each candidate type and given the one that buys the most accuracy
per byte under a size budget, so the bits go where they change the answer. The
tag is the target width: `RCO-3.9` averages 3.9 bits a weight.

```
qwen3-1.7b, which build?
  tiny     IQ3_XS          923 MB
  custom   RCO-3.9         2.2 GB bandwidth,   996 MB finalized
  medium   Q4_K_M          1.1 GB
  large    Q8_0            2.2 GB
```

Picking it opens the widths, since the target is a number rather than a fixed
build: 2.75, 3, 3.4, 3.9, 4.4 and 5 bits a weight, each with what it leaves on
disk. Choosing one shows what it costs against the published build nearest its
size, then asks:

```
                                       download    on disk
  RCO-3.9       ████████████████████     2.2 GB     996 MB
  IQ3_XS        ████████                 923 MB     923 MB
```

Twice the download and a few minutes of CPU. Against llama.cpp's own build of
the same size, on technical problems at temperature 0 with the Q8_0 as the
reference:

| build | size | correct |
|---|---|---|
| Q8_0 | 2.02 GB | 5/6 |
| IQ3_XS | 0.90 GB | 4/6 |
| RCO-3.9 | 0.93 GB | 5/6 |

The source is the repository's Q8_0, read a block at a time and dropped once
quantized, so nothing but the result reaches disk and the conversion holds
about 350 MB. Qwen3.6-35B-A3B goes from 37 GB to 17 GB in 17 minutes. A
finished build reports how long it spent choosing widths, waiting on the
download and quantizing, which is what says whether more cores or a faster
link would change anything.

It needs ggml, which comes with the llama.cpp runtime beside `llama-server`.

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
| `LLMASH_RCO_THREADS` | cores a custom build may quantize on (default: all but one) |
| `LLMASH_RCO_SAMPLE` | weights per tensor the bit-width search measures (default 131072; lower is faster and noisier) |

`llmash serve --help` lists the rest. A `routes.json` beside the program
configures fast routes (see `routes.example.json`); the C++ build reads it but
does not send requests to those backends yet.

## Building

Visual Studio 2022 with the C++ workload; CMake comes with it.

```powershell
python build.py           # dist/llmash-win-x64.zip
python build.py --here    # ...and run this checkout on it
```

`python bench/bench.py --help` reproduces the table above.

## License

MIT.
