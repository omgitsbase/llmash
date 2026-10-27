"""Throughput under concurrent load, measured the same way for llmash and vLLM: C clients at once, each sending
three of bench.py's nine prompts in turn (greedy, 600 tokens), against one provider's OpenAI-compatible API. A level's
throughput is every completion token of the level over the wall time from the first request sent to the last reply
received; a stream's speed is its completion tokens over its own request time. Rows go to batch.json.

    python bench/bench_batch.py --url http://127.0.0.1:11634/v1 --model gemma4:e4b-it-q8_0 --provider llmash \\
        --label "Gemma 4 E4B" --build q8_0 --concurrency 1 2 4 8 16 32
"""
import argparse
import json
import os
import pathlib
import statistics
import threading
import time
import urllib.request

from bench import WORKLOADS

HERE = pathlib.Path(__file__).resolve().parent
OUT = pathlib.Path(os.environ.get("LLMASH_BATCH_OUT", HERE / "batch.json"))
PROMPTS = [p for w in WORKLOADS.values() for p in w]


def ask(url, model, prompt, max_tokens):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0, "stream": False}
    req = urllib.request.Request(url + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.load(r)
    t1 = time.perf_counter()
    return d["usage"]["completion_tokens"], t0, t1


def level(url, model, c, per_client, max_tokens):
    results, errors = [], []
    lock = threading.Lock()

    def client(k):
        for j in range(per_client):
            prompt = PROMPTS[(k * per_client + j) % len(PROMPTS)]
            try:
                n, t0, t1 = ask(url, model, prompt, max_tokens)
                with lock:
                    results.append((n, t0, t1))
            except Exception as e:  # noqa: BLE001
                with lock:
                    errors.append(repr(e))

    threads = [threading.Thread(target=client, args=(k,)) for k in range(c)]
    start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    end = time.perf_counter()
    tokens = sum(n for n, _, _ in results)
    streams = [n / (t1 - t0) for n, t0, t1 in results if t1 > t0]
    return {
        "concurrency": c,
        "requests": len(results),
        "errors": errors[:3],
        "tokens": tokens,
        "seconds": round(end - start, 3),
        "throughput": round(tokens / (end - start), 1),
        "stream_median": round(statistics.median(streams), 1) if streams else None,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--provider", required=True, choices=["llmash", "vLLM"])
    p.add_argument("--label", required=True)
    p.add_argument("--build", required=True)
    p.add_argument("--file", default="")
    p.add_argument("--concurrency", type=int, nargs="+", required=True)
    p.add_argument("--per-client", type=int, default=3)
    p.add_argument("--max-tokens", type=int, default=600)
    a = p.parse_args()

    ask(a.url, a.model, "Say hello.", 16)  # warm-up
    rows = json.loads(OUT.read_text()) if OUT.exists() else []
    for c in a.concurrency:
        r = level(a.url, a.model, c, a.per_client, a.max_tokens)
        r.update({"provider": a.provider, "label": a.label, "build": a.build, "file": a.file,
                  "measured": time.strftime("%Y-%m-%d")})
        rows = [x for x in rows if not (x["provider"] == a.provider and x["label"] == a.label and
                                        x["build"] == a.build and x["concurrency"] == c)]
        rows.append(r)
        OUT.write_text(json.dumps(rows, indent=1))
        print(f"{a.provider} {a.build} c={c:3d}: {r['throughput']:8.1f} tok/s total, {r['stream_median']} per stream, "
              f"{r['requests']} requests, {len(r['errors'])} errors", flush=True)


if __name__ == "__main__":
    main()
