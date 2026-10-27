"""Decode speed of one build on one provider, measured the same way for llmash, Ollama and vLLM: each one's
OpenAI-compatible API, the three prompts of each workload below, greedy, 600 tokens, streamed, timed from the first
streamed token to the last so model load and prompt processing are left out. A workload's speed is the median of its
three prompts. The row is added to providers.json, which charts.py draws.

    python bench/bench.py --url http://127.0.0.1:11434/v1 --model gemma-4-e4b-it:q4_k_m --provider llmash \\
        --label "Gemma 4 E4B" --build Q4_K_M --format 4-bit --size-gb 4.98

--size-gb is the text model's weights: the GGUF's size for llmash, the model layer of Ollama's manifest, and the
safetensors tensors outside the vision and audio towers for vLLM. Projectors and separate draft models are left out.
"""
import argparse
import json
import pathlib
import statistics
import time
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
PROVIDERS = HERE / "providers.json"
FORMATS = ("3-bit", "4-bit", "NVFP4", "8-bit", "16-bit")

# Three prompts per workload, one per repeat. Repeating a single prompt at
# temperature 0 lets a self-speculating drafter replay its own previous output,
# which measured seven times faster here and says nothing about real use.
WORKLOADS = {
    "conversation": [
        "Explain to a friend why the sky is blue but sunsets are red. "
        "Two or three paragraphs of prose, no lists.",
        "Describe how a vinyl record stores and reproduces sound, for someone who has "
        "never seen one. Two or three paragraphs of prose, no lists.",
        "Explain why bread rises, and what changes when you use a sourdough starter "
        "instead of dried yeast. Two or three paragraphs of prose, no lists.",
    ],
    "coding": [
        "Write a Python class LRUCache with get and put in O(1), using a dict and a "
        "doubly linked list. Include the node class, a docstring on every method, and "
        "a short example of use.",
        "Write a Go function that walks a directory tree concurrently with a worker pool "
        "and returns the ten largest files. Handle errors, bound the goroutines, and "
        "include the struct definitions.",
        "Write a SQL schema for a library lending system: books, copies, members, loans "
        "and reservations. Add the indexes you would want, and one query that finds "
        "overdue loans with the member's name.",
    ],
    "thinking": [
        "A train leaves A at 9:00 travelling 60 km/h. Another leaves B at 9:40 travelling "
        "90 km/h toward A. A and B are 300 km apart. When do they meet, and how far from A? "
        "Work through it step by step, then state the answer.",
        "Three switches downstairs control three bulbs upstairs. You may flip switches as "
        "much as you like, but you may go upstairs only once. How do you tell which switch "
        "controls which bulb? Reason it out, then give the procedure.",
        "You have a 12-litre jug, an 8-litre jug and a 5-litre jug. The 12 is full, the "
        "others empty, and there are no markings. Split the water into two equal parts. "
        "Work through the pourings, then list them.",
    ],
}


def stream(url, model, prompt, max_tokens):
    """Tokens per second between the first streamed token and the last, and the tokens generated."""
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0, "stream": True, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    first = last = None
    chunks, usage = 0, None
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            d = json.loads(data)
            if d.get("usage"):
                usage = d["usage"]
            for c in d.get("choices") or []:
                delta = c.get("delta") or {}
                if delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning"):
                    now = time.perf_counter()
                    first = first or now
                    last = now
                    chunks += 1
    n = (usage or {}).get("completion_tokens") or chunks
    if first is None or last <= first or n < 64:
        return None, n
    return (n - 1) / (last - first), n


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", required=True, help="the provider's OpenAI-compatible base, ending in /v1")
    p.add_argument("--model", required=True, help="the name the provider serves the build under")
    p.add_argument("--provider", required=True, choices=["llmash", "Ollama", "vLLM"])
    p.add_argument("--label", required=True, help="the model, as the chart titles it")
    p.add_argument("--build", required=True, help="the build's own name, e.g. Q4_K_M or NVFP4")
    p.add_argument("--format", required=True, choices=FORMATS, help="the group the build is charted in")
    p.add_argument("--size-gb", type=float, required=True, help="the text model's weights, in GB")
    p.add_argument("--max-tokens", type=int, default=600)
    args = p.parse_args()

    stream(args.url, args.model, "Say ok.", 16)
    res, toks = {}, {}
    for name, prompts in WORKLOADS.items():
        rates, counts = [], []
        for prompt in prompts:
            rate, n = stream(args.url, args.model, prompt, args.max_tokens)
            if rate:
                rates.append(rate)
            counts.append(n)
        res[name] = round(statistics.median(rates), 1) if rates else None
        toks[name] = counts
    print(f"{args.provider} {args.label} {args.build}: " + "  ".join(f"{k} {v}" for k, v in res.items()))

    rows = json.loads(PROVIDERS.read_text("utf-8")) if PROVIDERS.exists() else []
    rows.append({"provider": args.provider, "label": args.label, "tier": args.format, "build": args.build,
                 "size_gb": args.size_gb, "model": args.model, "workloads": res, "tokens": toks,
                 "measured": time.strftime("%Y-%m-%d")})
    PROVIDERS.write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
