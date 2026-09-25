# llmash

[![Release](https://img.shields.io/github/v/release/omgitsbase/llmash?include_prereleases&label=release)](https://github.com/omgitsbase/llmash/releases/latest)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/omgitsbase/llmash/blob/main/LICENSE)
[![Windows](https://img.shields.io/badge/Windows-10%2F11-0078D6.svg?logo=windows)](https://github.com/omgitsbase/llmash/releases/latest)
[![C++](https://img.shields.io/badge/C%2B%2B-17-00599C.svg?logo=cplusplus)](https://isocpp.org)
[![CUDA](https://img.shields.io/badge/CUDA-13.3-76B900.svg?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/omgitsbase/llmash)

An Ollama-compatible server and command line for Windows, built on a fork of
llama.cpp. There is an alpha Linux build too.

It serves your GGUF files through `llama-server` and keeps Ollama's commands,
API and model store, so anything already pointed at Ollama keeps working. What
it changes is how each model is launched: the settings, the context window,
the cache, the drafter, and when a repository has no build at the size you
want, the build itself.

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
This is the same setting as `OLLAMA_MODELS`. A store moved to another drive
survives an upgrade.

## Speed

Same prompts, same GPU (an RTX PRO 6000 Blackwell, 96 GB), each tool running the
model it gives you by default. Your numbers will differ.

The rows are different files on purpose. Ollama pulls a Q8_0. vLLM runs 8-bit
weights of the same size, with speculative decoding. llmash runs a 3-bit build it
puts together itself, with a drafter. Most of the gap is bytes read per token,
not engine: a third of the bytes is most of the speed.

The llmash and vLLM rows move between runs, by up to a third, because speculative
decoding is faster on predictable text. Ollama has no drafter and repeats to a
tenth of a token per second.

<!-- BENCHMARK -->

**gemma-4 26B-A4B**

| backend | build | conversation | coding | thinking |
|---|---|--:|--:|--:|
| Ollama | Q8_0, 28.1 GB | 198.6 | 199.1 | 197.5 |
| **llmash**\* | RCO-3, 8.9 GB | **333.2** | **449.5** | **445.5** |

**Qwen3.6 35B-A3B**

| backend | build | conversation | coding | thinking |
|---|---|--:|--:|--:|
| Ollama | Q8_0, 38.7 GB | 204.5 | 204.6 | 207.2 |
| vLLM | INT8 GPTQ, 43 GB | 302.2 | 353.4 | 457.0 |
| **llmash**\* | RCO-3, 12.5 GB | **455.0** | **534.3** | **569.3** |

**Qwen3.8 27B**

| backend | build | conversation | coding | thinking |
|---|---|--:|--:|--:|
| Ollama | Q8_0, 29.0 GB | 50.1 | 50.1 | 50.0 |
| vLLM | INT8 W8A16, 32 GB | 50.9 | 51.1 | 63.6 |
| **llmash**\* | RCO-3, 9.6 GB | **153.2** | **170.1** | **178.3** |

Tokens per second while generating, median of five runs, excluding model load and prompt processing. Ollama's rows are from one model load; between loads they drift by about a tenth.

\* llmash runs a custom build: one quantization type per tensor, chosen under a size budget and assembled on this machine. No other runtime has an equivalent. It is a third of a Q8_0's bytes, which is where the speed comes from; how close it stays in accuracy has been measured here only on the six problems in the table further down. Ollama runs its own Q8_0 pull, which ships without a draft model; vLLM runs 8-bit weights with speculative decoding; llmash runs what a pull assembles, drafter and launch settings included.

<!-- /BENCHMARK -->

vLLM is the native Windows build on 8-bit weights (INT8, weight-only, the same
bytes as a Q8_0), with the draft model published for Qwen3.6 and n-gram lookup
for Qwen3.8. Its fp8 path is not an option on this card: the wheel carries only
the Ampere w8a8 kernel and aborts in `cutlass_scaled_mm_sm80_epilogue` on
Blackwell, so INT8 through Marlin is what runs. gemma-4 has no vLLM row at all,
because `head_dim` varies per layer in that architecture and the loader refuses
it.

## How it fits together

- **`llmash.exe`** is the command line and the server. `llmashw.exe` is the same
  program with no console, for the tray.
- **The server** owns the model store, starts and stops `llama-server`
  processes, and decides how long each one stays resident. When a load needs
  room it evicts the least recently used model first, leaving one somebody
  used moments ago for a second pass rather than failing the request.
- **The tray** unloads a model, sets the keep-alive, restarts the server, and
  starts it at login.
- **The runtime** is a llama.cpp fork the installer fetches beside the program.
  A release that carries a newer one refreshes it. A model newer than the
  runtime can name another `llama-server` of its own, by name fragment, under
  `runtime` in `local.json`; it gets only the flags every build takes.
- **Updates** come from two places. Every push to `main` is built into the
  `edge` prerelease, which `llmash update` installs; `--stable` takes the latest
  release instead. Releases are cut when the runtime changes.

Per model, at launch, it picks the ordinary llama.cpp settings and logs each:
prompt-prefix reuse, a host-RAM prompt cache sized from free RAM, batch width
when the card has room, raised process priority, and DirectIO loading.
`LLMASH_TUNE_OFF` disables any of them. `serve --verbose` prints a line per
turn with the token counts, the speed, what the cache reused and what the
drafter got accepted.

A client that looks for a `llama-server` rather than an Ollama, as Hermes's
local provider does, finds this one: `/props` and `/models` answer, and
`/v1/models` advertises the context each model actually runs at.

## Context

A model runs at 8192 tokens unless something asks for more: the request's
`num_ctx`, `llmash ctx`, or the `num_ctx` in an Ollama model's Modelfile.
Past its trained context, llmash runs it under YaRN, up to four times the
trained length. `llmash ctx MODEL 1m` sets a million tokens
for every client, including ones that cannot ask for one, `--kv q8_0` halves
the cache, and `--keep` holds the model loaded. The running server takes the
change at once, and `pull` and `run` say what a window costs on the card
before loading it.

```
llmash ctx qwen3.8-27b:rco-3 768k --keep
qwen3.8-27b:rco-3 runs at 768k (YaRN x3 over the trained 256k); 59 GB on the card at f16, 92 free
loading and keeping it ... loaded
```

The cost comes from the model's header, not a guess from its file size: how
many layers attend over the whole context, how many heads they keep, and what
a recurrent or windowed layer holds instead. Weights llama.cpp keeps in RAM,
the token embedding and on some architectures a per-layer embedding table
tens of gigabytes wide, are counted as RAM rather than card, and `ctx` says
so.

On Windows every video allocation is backed by commit charge, RAM plus
pagefile, so one process may hold only what is left of that, however much of
the card is free. The limit is the machine's, not the process's: a second
process draws on the same pool, so splitting a model across two does not
raise the total. llmash reads the figure live and fits within it, and `ctx`
says when a window is over it rather than over the card. With 64 GB of RAM
and a 32 GB pagefile that is about 71 GB of a 96 GB card; a larger pagefile
raises it, and costs nothing at run time, since the reservation is never
written while the data sits on the card. A cache too large for one allocation
is laid out across several.

## Custom builds

`pull` lists a build assembled here beside the published ones. Every tensor is
measured at each candidate type and given the one that buys the most accuracy
per byte under a size budget, so the bits go where they change the answer. The
measure is imatrix-weighted weight error, the budget is met by bisection on one
Lagrange multiplier, and only the output head and the embedding keep a floor.
The tag is the target width, `RCO-3`, and the ladder runs 2.4, 2.75 and 3 bits
a weight, which is where a uniform build has fallen apart and choosing per
tensor earns its keep.

This is the allocation idea from the published GSQ-RCO method, run with ggml's
own quantizers. llmash does not implement GSQ's learned grids, and none of
GSQ-RCO's published scores are numbers for the files llmash makes. Where the
hub carries a real GSQ-RCO build of a model, `pull` looks for it without being
asked and lists it first, as `rco`, because that is the better file; the build
assembled here is for models that have none.

```
qwen3-1.7b, which build?
  custom   3 bits, one type per tensor   2.2 GB bandwidth
  medium   Q4_K_M                        1.1 GB
  large    Q8_0                          2.2 GB
```

Pick it and you choose the width:

```
how small?
  rco 2.4  about 613 MB   2.4 bits a weight
  rco 2.75 about 703 MB   2.75 bits a weight
  rco 3    about 766 MB   3 bits a weight
```

It then shows the cost against the nearest published build and asks:

```
                                       download    on disk
  RCO-3         ████████████████████     2.2 GB    ~766 MB
  IQ3_XS        ████████                 923 MB     923 MB
```

Twice the download, and a minute or two of quantizing on the GPU. What has
been measured of these builds is small: six technical problems at temperature
0 on Qwen3-1.7B, with the Q8_0 as the reference.

| build | size | correct |
|---|---|---|
| Q8_0 | 2.02 GB | 5/6 |
| IQ3_XS | 0.90 GB | 4/6 |
| RCO-3.9 | 0.93 GB | 5/6 |

Six problems is a sample, not a benchmark. Treat the width as a size you
choose, whose cost in accuracy has not been charted here.

The source is the smallest published build that is still comfortably wider than
the target: a 3-bit build reads a Q6_K rather than a Q8_0, a quarter fewer bytes.
It is read a block at a time and dropped once quantized, so only the result
touches disk. Qwen3.6-35B-A3B goes from 32 GB to 13 GB in 12 minutes, 10 of them
the download.

The quantizing runs on the GPU for every type a custom build uses, and the
source crosses the bus still quantized rather than expanded to floats. gemma-4
26B quantizes in 1:12 that way against 9:44 on the CPU, with byte-identical
output. Any type the GPU does not carry falls back to the CPU per tensor, so a
CPU-only runtime still works.

A finished build reports how long it spent choosing widths, downloading and
quantizing, so you can see whether a faster link would help.

`rco convert` does the same from a model already on disk, with no download at
all. It needs ggml, which comes with the llama.cpp runtime beside
`llama-server`.

## Pulling

`pull` takes an Ollama name or a Hugging Face repository as `hf.co/<org>/<repo>`,
lists the builds it holds and asks which; `hf.co/<org>/<repo>@<quant>` or
`--quant` names one directly. A width asked for by name is a requirement: when
the repository does not have it, the pull stops and lists what it does have,
rather than taking the nearest and downloading something you did not ask for.
A repository with no GGUF at all is answered by one of the same model that has
one, preferring the model's own over a fine-tune's. Files come down over
several connections at once. The download runs inside the server, so closing
the terminal does not stop it.

## Speculation

A model carrying an MTP head drafts for itself, with nothing to fetch or
configure. The whole round is one CUDA graph: the verify batch, the accept
decision and every draft step, with the accept decided on the GPU rather than
read back to the host. Measured at 424 tok/s against 373 for the same model
drafting one token per decode.

For a model with no head, `pulldraft` finds a drafter on Hugging Face, checked
against the weights before anything downloads. gemma-4 26B-A4B went from 244
to 342 tok/s on a fetched EAGLE-3 head.

On Qwen3.5, 3.6 and 3.8 models the MTP head drafts through a smaller copy of
the LM head, built when the model loads: the head's rows for the 65,536 tokens
most frequent in English text and code (262 MiB on the 27B). Verification
still reads the full head, so the output is the model's own; each drafted token
reads a quarter of the head instead. Qwen3.8-27B decodes 5-10% faster and
Qwen3.6-35B-A3B 5-7%, with nearly the same acceptance. Most of a Chinese, Japanese,
Korean or Russian text is outside that list, so while more than 15% of recent
tokens are, the drafts go back to the full head, and those languages run as
fast as before. `LLAMA_MTP_FULL_HEAD=1` turns the smaller head off.

A DFlash drafter (a block-diffusion model that drafts seven tokens at once)
beside a model with no MTP head is used for it: `<stem>.dflash.gguf`, or any
file with `dflash` in its name trained for the base model the target's header
names, so one drafter serves every quantization of a model. With an MTP head
too, the head stays the default, since it drafts prose faster;
`LLMASH_PREFER_DFLASH=1` switches, which pays on code and structured output
(Qwen3.8-27B JSON 216 -> 247 tok/s).

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
| `start` | start llmash in the background, with its tray icon |
| `serve` | run the server in this console; `--host`, `--port`, `--ctx`, `--kv`, `--gpu-budget`, `--verbose` |
| `ctx` | the context a model runs at, up to 4x its trained length under YaRN; `--kv` sets its cache type, `--keep` holds it loaded, `--release` lets it go |
| `pulldraft` | find and install a draft model for a model you have |
| `rco convert` | assemble a custom build from a model already on disk |
| `models` | show where models are read from, or point llmash at a folder of them |
| `doctor` | check the install, runtime, GPU, models and routes |
| `update` | install the latest push; `--stable` for the latest release, `--force` to reinstall |
| `launch` | point Claude Code, Codex, Droid and others at this server |
| `link` | expose the API over a Tailscale funnel, with a key; says where the bare host name goes when that is not llmash |
| `uninstall` | remove everything the installer created |

`ollama` is installed as an alias.

## Configuration

Optional. `local.json` next to the program, or environment variables. The
variable wins over the file.

| | |
|---|---|
| `OLLAMA_MODELS` | the models directory, as Ollama takes it: a store, or a folder of GGUFs read in place (default `~\.ollama\models`) |
| `LLMASH_MODELS` | same as `OLLAMA_MODELS`; `llmash models set` sets the same thing |
| `LLMASH_PORT` | local API port (default 11434) |
| `LLMASH_CTX` | default context length (default 8192) |
| `LLMASH_KV` | K/V cache type, `f16` or `q8_0` |
| `LLMASH_YARN_MAX` | how far past its trained context a model may run under YaRN (default 4) |
| `LLMASH_GPU_BUDGET_GB` | what one process may hold on the card, instead of what commit charge allows |
| `LLMASH_VRAM_HEADROOM` | what to leave free on the cards when fitting (default 8% of them, 1 to 6 GB) |
| `LLMASH_FIT_MARGIN_MB` | what llama.cpp leaves free on each card when it places a model (default 512) |
| `LLMASH_PARALLEL` | server slots (default 1; raise it to serve several at once) |
| `LLMASH_PIN` | comma-separated models never evicted |
| `LLMASH_SPEC_FALLBACK` | drafter for models without one (default `ngram-mod`) |
| `LLMASH_PREFER_DFLASH` | draft with a DFlash drafter beside the model even when it has an MTP head |
| `LLMASH_DFLASH_DRAFT` | tokens a DFlash drafter drafts per round (default 7) |
| `LLMASH_TUNE_OFF` | disable individual tuning: `cache-reuse,cache-ram,batch,prio` |
| `LLMASH_RCO_THREADS` | cores a custom build may quantize on (default: all but one) |
| `LLMASH_RCO_SAMPLE` | weights per tensor the bit-width search measures (default 131072; lower is faster and noisier) |

`local.json` takes the same things by name, plus a few that are per model:
`ctx_override` and `ctx_max` by model name, `fit` for a model's cache type,
`pin`, `launch_extra` for extra `llama-server` flags by name fragment,
`no_mmproj` for models whose projector should not load up front, `runtime` for
another `llama-server` by name fragment, `gpu_budget_gb`, `extra_roots` for
further folders to read, and `gguf_dir` for where pulled GGUFs go.

`llmash serve --help` lists the rest.

## Building

Visual Studio 2022 with the C++ workload; CMake comes with it.

```powershell
python build.py           # dist/llmash-win-x64.zip
python build.py --here    # ...and run this checkout on it
```

`python bench/bench.py --help` reproduces the table above.

## License

MIT.
