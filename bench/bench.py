import argparse
import json
import pathlib
import statistics
import time
import urllib.error
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
RESULTS = HERE / "results.json"

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


def post(url, body, timeout=900):
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), method="POST",
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def via_ollama(base, model, prompt, max_tokens, temperature):
    body = {"model": model, "stream": False,
            "options": {"num_predict": max_tokens, "temperature": temperature},
            "messages": [{"role": "user", "content": prompt}]}
    d = post(base.rstrip("/") + "/api/chat", body)
    tokens = d.get("eval_count") or 0
    secs = (d.get("eval_duration") or 0) / 1e9
    return tokens, secs


def via_openai(base, model, prompt, max_tokens, temperature):
    body = {"model": model, "max_tokens": max_tokens, "temperature": temperature,
            "stream": True, "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request(
        base.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
        method="POST", headers={"Content-Type": "application/json"})
    chunks, first, last, reported = 0, None, None, None
    with urllib.request.urlopen(req, timeout=900) as r:
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
                piece = "".join(str(delta.get(k) or "") for k in
                                ("content", "reasoning_content", "reasoning", "thinking"))
                if not piece:
                    continue
                now = time.perf_counter()
                first = first if first is not None else now
                last = now
                chunks += 1
    if first is None or last is None or last <= first:
        return 0, 0.0
    return (reported or chunks), last - first


def run_one(base, api, model, prompt, max_tokens, temperature):
    fn = via_ollama if api == "ollama" else via_openai
    return fn(base, model, prompt, max_tokens, temperature)


def warm(base, api, model):
    try:
        run_one(base, api, model, "Say ok.", 16, 0.0)
    except (urllib.error.URLError, OSError) as e:
        print(f"  warm-up failed: {e}")


def measure(args):
    rows = []
    warm(args.url, args.api, args.model)
    for name, prompts in WORKLOADS.items():
        rates, notes = [], []
        for i in range(args.repeat):
            prompt = prompts[i % len(prompts)]
            tokens, secs = run_one(args.url, args.api, args.model, prompt,
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
    order = [("ollama", "Ollama"), ("vllm", "vLLM"), ("llmash", "llmash")]
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
    start, end = "<!-- BENCHMARK -->", "<!-- /BENCHMARK -->"
    i, j = s.index(start), s.index(end)
    readme.write_text(s[:i] + start + "\n\n" + table() + "\n\n" + s[j:], encoding="utf-8")
    print(f"wrote the table into {readme}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", choices=["ollama", "vllm", "llmash"])
    p.add_argument("--url", default="http://127.0.0.1:11434",
                   help="server root, without /v1")
    p.add_argument("--api", choices=["ollama", "openai"], default="ollama",
                   help="ollama: /api/chat with the server's own decode timing. "
                        "openai: /v1 streamed, timed between first and last token.")
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
