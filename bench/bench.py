"""Measure decode speed on three workloads, against any OpenAI-compatible server.

    python bench/bench.py --backend llmash --url http://127.0.0.1:11434/v1 \
        --model gemma4:26b --label "gemma-4 26B-A4B"

Every backend is asked the same questions with the same sampling and the same
token budget, over the OpenAI route all three speak. Speed is the server's own
completion_tokens divided by the time between the first and the last token, so
neither the model load nor the prompt is counted. A run that generates fewer
than --min-tokens is discarded rather than reported: a handful of tokens
arriving in a few milliseconds produces a meaningless rate.

Results append to bench/results.json. `--table` turns that file into the
markdown block in the README.
"""
import argparse
import json
import pathlib
import statistics
import time
import urllib.error
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
RESULTS = HERE / "results.json"

WORKLOADS = {
    "conversation": (
        "Explain to a friend why the sky is blue but sunsets are red. "
        "Write two or three paragraphs of prose, no lists."
    ),
    "coding": (
        "Write a Python class LRUCache with get and put in O(1), using a dict and a "
        "doubly linked list. Include the node class, a docstring on every method, and "
        "a short example of use at the end."
    ),
    "thinking": (
        "A train leaves A at 9:00 travelling 60 km/h. Another leaves B at 9:40 travelling "
        "90 km/h toward A. A and B are 300 km apart. At what time do they meet, and how far "
        "from A? Work through it step by step, then state the answer."
    ),
}


def stream(url, body, timeout=900):
    """One streamed completion: (tokens, seconds spent generating, characters)."""
    body = dict(body, stream=True, stream_options={"include_usage": True})
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), method="POST",
        headers={"Content-Type": "application/json"})
    chars, chunks, first, last, reported = 0, 0, None, None, None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                ev = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if isinstance(ev.get("usage"), dict):
                reported = ev["usage"].get("completion_tokens") or reported
            for choice in ev.get("choices") or []:
                delta = choice.get("delta") or {}
                piece = (delta.get("content") or "") + (delta.get("reasoning_content") or "")
                if not piece:
                    continue
                now = time.perf_counter()
                if first is None:
                    first = now
                last = now
                chunks += 1
                chars += len(piece)
    if first is None or last is None or last <= first:
        return 0, 0.0, chars
    # Prefer the server's count; fall back to chunks, which is one token per
    # chunk on every backend measured here.
    return (reported or chunks), last - first, chars


def run_one(url, model, prompt, max_tokens, temperature):
    body = {"model": model, "max_tokens": max_tokens, "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}]}
    return stream(url.rstrip("/") + "/chat/completions", body)


def warm(url, model):
    """One short request, so weights, graphs and caches are hot before timing."""
    try:
        run_one(url, model, "Say ok.", 16, 0.0)
    except (urllib.error.URLError, OSError) as e:
        print(f"  warm-up failed: {e}")


def measure(args):
    rows = []
    warm(args.url, args.model)
    for name, prompt in WORKLOADS.items():
        rates, notes = [], []
        for _ in range(args.repeat):
            tokens, secs, chars = run_one(args.url, args.model, prompt,
                                          args.max_tokens, args.temperature)
            if tokens < args.min_tokens or secs <= 0:
                notes.append(f"{tokens}t discarded")
                continue
            rates.append(tokens / secs)
            notes.append(f"{tokens}t in {secs:.2f}s = {tokens / secs:.1f}")
            time.sleep(1)
        print(f"  {name:14} " + ("  ".join(notes) if notes else "no output"))
        if len(rates) < 2:
            print(f"  {'':14} not enough valid runs, skipped")
            continue
        rows.append({"workload": name, "tok_s": round(statistics.median(rates), 1),
                     "runs": [round(x, 1) for x in rates]})
        print(f"  {'':14} median {statistics.median(rates):.1f} tok/s")
    return rows


def load_results():
    if RESULTS.exists():
        return json.loads(RESULTS.read_text("utf-8"))
    return {"models": {}}


def save(args, rows):
    data = load_results()
    model = data["models"].setdefault(args.label or args.model, {})
    model[args.backend] = {"measured": time.strftime("%Y-%m-%d"), "model_id": args.model,
                           "workloads": {r["workload"]: r["tok_s"] for r in rows}}
    RESULTS.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def table():
    data = load_results()
    order = [("ollama", "Ollama"), ("vllm", "vLLM (Docker)"), ("llmash", "llmash")]
    cols = ("conversation", "coding", "thinking")
    out = []
    for label, backends in data["models"].items():
        out.append(f"**{label}**\n")
        out.append("| backend | conversation | coding | thinking |")
        out.append("|---|--:|--:|--:|")
        for key, name in order:
            if key not in backends:
                continue
            w = backends[key]["workloads"]
            bold = "**" if key == "llmash" else ""
            cells = " | ".join(f"{bold}{w[c]:.1f}{bold}" if c in w else "-" for c in cols)
            out.append(f"| {bold}{name}{bold} | {cells} |")
        out.append("")
    out.append("Tokens per second while generating, median of three runs, "
               "excluding model load and prompt processing.")
    return "\n".join(out)


def write_readme():
    readme = HERE.parent / "README.md"
    s = readme.read_text("utf-8")
    start, end = "<!-- BENCHMARK -->", "## What it does differently"
    i, j = s.index(start), s.index(end)
    readme.write_text(s[:i] + start + "\n\n" + table() + "\n\n" + s[j:], encoding="utf-8")
    print(f"wrote the table into {readme}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", choices=["ollama", "vllm", "llmash"])
    p.add_argument("--url", default="http://127.0.0.1:11434/v1")
    p.add_argument("--model")
    p.add_argument("--label", help="how the model is named in the table")
    p.add_argument("--max-tokens", type=int, default=600)
    p.add_argument("--min-tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--table", action="store_true", help="rebuild the README table and exit")
    args = p.parse_args()

    if args.table:
        write_readme()
        return 0
    if not (args.backend and args.model):
        p.error("--backend and --model are required unless --table is given")
    print(f"{args.backend}: {args.model} at {args.url}")
    rows = measure(args)
    if rows:
        save(args, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
