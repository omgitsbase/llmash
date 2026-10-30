"""The README's charts and tables, drawn from the measured results beside this script:

    providers.json     decode speed and size of each build of a model on llmash, Ollama and vLLM
    math_results.json  the 40-question math comparison of RCO-3 against Q8_0 and other builds, with tokens used

One chart per model, written twice each for GitHub's light and dark themes, into docs/; and the tables between the
PROVIDERS and MATH markers in the README, one per model.

    python bench/charts.py
"""
import json
import pathlib
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
DOCS = HERE.parent / "docs"
WORKLOADS = ("conversation", "coding", "thinking")
PROVIDERS = ("llmash", "Ollama", "vLLM")

THEMES = {
    "light": {"surface": "#ffffff", "text": "#1f2328", "muted": "#59636e", "grid": "#d1d9e0",
              "llmash": "#2a78d6", "Ollama": "#eb6834", "vLLM": "#1baf7a", "other": "#8c959f", "fill": 0.22},
    "dark": {"surface": "#0d1117", "text": "#e6edf3", "muted": "#9198a1", "grid": "#3d444d",
             "llmash": "#3987e5", "Ollama": "#d95926", "vLLM": "#199e70", "other": "#6e7681", "fill": 0.32},
}


def slug(name):
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def style(t):
    plt.rcParams.update({
        "font.family": ["DejaVu Sans"], "font.size": 11, "svg.fonttype": "path",
        "text.color": t["text"], "axes.labelcolor": t["muted"], "xtick.color": t["muted"], "ytick.color": t["text"],
        "axes.edgecolor": t["grid"], "axes.facecolor": "none", "figure.facecolor": "none",
    })


def clean(ax, t):
    ax.grid(axis="x", color=t["grid"], linewidth=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="both", length=0)


def mean_speed(entry):
    vals = [entry["workloads"][w] for w in WORKLOADS if entry["workloads"].get(w)]
    return sum(vals) / len(vals)


def bars(rows, groups, t, xlabel, xmax, path, title, value_text, ticks_to=0):
    """Horizontal bars, one per row, in labelled groups; the value and a note at each bar's end."""
    fig, ax = plt.subplots(figsize=(9.6, 0.5 * len(rows) + 0.45 * sum(1 for g in groups if g["name"]) + 1.1))
    y, ys, labels = 0.0, [], []
    for g in groups:
        if g["name"]:
            ax.text(0, -y + 0.05, g["name"], fontsize=11.5, fontweight="bold", color=t.get(g["name"], t["text"]),
                    va="bottom", ha="left", transform=ax.get_yaxis_transform())
            y += 0.9
        for r in g["rows"]:
            ax.barh(-y, r["value"], height=0.66, color=r["color"], zorder=2)
            ax.text(r["value"] + xmax * 0.012, -y, value_text(r), va="center", fontsize=10.5, color=t["text"])
            ys.append(-y)
            labels.append(r["label"])
            y += 1
        y += 0.35
    ax.set_yticks(ys)
    ax.set_yticklabels(labels)
    ax.set_ylim(-y + 0.3, 0.9 if groups[0]["name"] else 0.2)
    # room for the text past the longest bar, with the ticks stopping at the scale's own end
    ax.set_xlim(0, xmax * (1.55 if ticks_to else 1.3))
    if ticks_to:
        ax.set_xticks(range(0, ticks_to + 1, 10))
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", color=t["text"], pad=10)
    clean(ax, t)
    fig.tight_layout()
    fig.savefig(path, transparent=True)
    plt.close(fig)


# Three formats per model, each on every provider that runs it, the providers in one order on every chart. The 3-bit
# builds are llmash's own and run nowhere else; in the 8-bit group llmash runs Ollama's own Q8_0 file.
TIERS = (("3-bit", "RCO 3-bit, llmash only"), ("NVFP4", "NVFP4"), ("8-bit", "8-bit"))


def provider_rows(runs, m):
    for tier, name in TIERS:
        rs = [r for p in PROVIDERS for r in sorted((r for r in runs if r["label"] == m and r["provider"] == p and
                                                   r["tier"] == tier), key=lambda r: r["size_gb"] or 0)]
        if rs:
            yield name, rs


def quality(r):
    """A GGUF row's perplexity as its distance from the model's own BF16 weights: +1.1% says a little worse."""
    if r.get("ppl_ref"):
        return f"{100 * (r['ppl'] / r['ppl_ref'] - 1):+.1f}%"
    return f"{r['ppl']:.2f}"


def providers(theme):
    """Per model: every build's decode speed, one band per format with its bars packed together in a fixed provider
    order, the speed, build and size at each bar's end. The charts share one left margin, so they line up, and each
    has its own scale."""
    t = THEMES[theme]
    style(t)
    runs = json.loads((HERE / "providers.json").read_text("utf-8"))
    step, bar, head, gap = 0.5, 0.38, 0.55, 0.3
    for m in dict.fromkeys(r["label"] for r in runs):
        groups = list(provider_rows(runs, m))
        # each chart on its own scale: one fast model must not flatten the others
        xmax = max(mean_speed(r) for _, rs in groups for r in rs)
        n = sum(len(rs) for _, rs in groups)
        height = step * n + (head + gap) * len(groups) + 1.2
        fig, ax = plt.subplots(figsize=(9.6, height))
        fig.subplots_adjust(left=0.025, right=0.99, top=1 - 0.6 / height, bottom=0.5 / height)
        ax.set_xlim(0, xmax * 1.45)
        y = 0.0
        for gi, (name, rs) in enumerate(groups):
            top = y + head * 0.15
            # the format as a band: its name at the top, its bars under it, a faint fill to hold them together
            ax.text(0.004, -(y + head * 0.5), name.upper(), transform=ax.get_yaxis_transform(), va="center",
                    ha="left", fontsize=9, fontweight="bold", color=t["muted"])
            y += head
            for r in rs:
                v = mean_speed(r)
                ax.barh(-y, v, height=bar, color=t[r["provider"]], linewidth=0, zorder=2)
                ax.text(xmax * 0.012, -y, r["provider"], va="center", ha="left", fontsize=9.5, fontweight="bold",
                        color=t["surface"], zorder=3)
                ax.text(v + xmax * 0.012, -y, f"{v:.0f}", va="center", fontsize=10.5, fontweight="bold",
                        color=t["text"])
                note = f"{r['build']}  ·  {r['size_gb']:.1f} GB" + (f"  ·  ppl {quality(r)}" if r.get("ppl") else "")
                ax.text(v + xmax * 0.085, -y, note, va="center", fontsize=9.5, color=t["muted"])
                y += step
            if gi % 2 == 0:
                ax.axhspan(-(y - step / 2 - 0.02), -top, facecolor=t["text"], alpha=0.035, linewidth=0, zorder=0)
            y += gap
        ax.set_ylim(-(y - gap) - 0.05, 0.05)
        ax.set_yticks([])
        ax.set_xlabel("tokens per second while generating", fontsize=9.5)
        ax.set_title(m, loc="left", fontsize=13, fontweight="bold", color=t["text"], pad=10)
        clean(ax, t)
        ax.grid(axis="x", color=t["grid"], linewidth=0.5, alpha=0.6)
        fig.savefig(DOCS / f"speed-{slug(m)}-{theme}.svg", transparent=True)
        plt.close(fig)


def math(theme):
    """Per model: correct answers out of 40 per build, the size in the label, the tokens spent at the bar's end."""
    t = THEMES[theme]
    style(t)
    data = json.loads((HERE / "math_results.json").read_text("utf-8"))
    for m in data["models"]:
        rows = [{"label": f"{b['label']}  ·  {b['size_gb']:.1f} GB", "value": b["total"][0],
                 "color": t["llmash"] if b["label"] == "RCO-3" else t["other"],
                 "note": f"{b['total'][0]}/{b['total'][1]}  ·  {b['tokens'] / 1000:.0f}k tokens"}
                for b in sorted(m["builds"], key=lambda b: -b["size_gb"])]
        bars(rows, [{"name": "", "rows": rows}], t, "correct answers of 40", 40,
             DOCS / f"math-{slug(m['model'])}-{theme}.svg", m["model"], lambda r: r["note"], ticks_to=40)


def batch(theme):
    """Per model: total tokens per second against concurrent requests, one line per provider and format. The levels
    sit evenly along the axis, and each chart scales to its own fastest point."""
    path = HERE / "batch.json"
    if not path.exists():
        return
    t = THEMES[theme]
    style(t)
    rows = json.loads(path.read_text("utf-8"))
    for m in dict.fromkeys(r["label"] for r in rows):
        rs = [r for r in rows if r["label"] == m and not r.get("errors")]
        levels = sorted({r["concurrency"] for r in rs})
        pos = {c: i for i, c in enumerate(levels)}
        ymax = max(r["throughput"] for r in rs)
        fig, ax = plt.subplots(figsize=(9.6, 4.6))
        for (provider, build) in dict.fromkeys((r["provider"], r["build"]) for r in rs):
            pts = sorted((r["concurrency"], r["throughput"]) for r in rs if r["provider"] == provider and r["build"] == build)
            xs, ys = [pos[c] for c, _ in pts], [v for _, v in pts]
            dashed = "nvfp4" in build.lower()
            ax.plot(xs, ys, color=t.get(provider, t["other"]), linewidth=2.2, linestyle="--" if dashed else "-",
                    marker="o", markersize=4.5, label=f"{provider}  ·  {build}", zorder=3)
            ax.text(xs[-1] + 0.12, ys[-1], f"{ys[-1]:.0f}", va="center", fontsize=10, fontweight="bold", color=t["text"])
        ax.set_xticks(range(len(levels)))
        ax.set_xticklabels([str(c) for c in levels])
        ax.set_xlim(-0.2, len(levels) - 0.4)
        ax.set_ylim(0, ymax * 1.12)
        ax.set_xlabel("requests at once", fontsize=9.5)
        ax.set_ylabel("tokens per second, all requests", fontsize=9.5)
        ax.set_title(f"{m}: concurrent requests", loc="left", fontsize=13, fontweight="bold", color=t["text"], pad=10)
        ax.grid(axis="y", color=t["grid"], linewidth=0.5, alpha=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="both", length=0)
        leg = ax.legend(loc="upper left", frameon=False, fontsize=9.5)
        for txt in leg.get_texts():
            txt.set_color(t["text"])
        fig.tight_layout()
        fig.savefig(DOCS / f"batch-{slug(m)}-{theme}.svg", transparent=True)
        plt.close(fig)


def scaling_levels(rs):
    return sorted({r["concurrency"] for r in rs})


def scaling_series(rs):
    """(provider, build) -> {streams: tokens per second}: the shared model's series first, an extra one (llmash on a
    file only it runs) after them."""
    keys = list(dict.fromkeys((r["provider"], r["build"]) for r in sorted(rs, key=lambda r: bool(r.get("extra") or r.get("table_only")))))
    return {k: dict(sorted((r["concurrency"], r["throughput"]) for r in rs if (r["provider"], r["build"]) == k))
            for k in keys}


def scaling(theme):
    """Per model: tokens per second of all streams together against the streams at once, every provider on the same
    file, the gap between llmash and vLLM filled and llmash's ratio to each provider in rows under the axis. An extra
    dashed llmash line is a smaller file the others cannot run. The levels sit evenly along the axis, and each chart
    scales to its own fastest point."""
    path = HERE / "scaling.json"
    if not path.exists():
        return
    t = THEMES[theme]
    style(t)
    rows = json.loads(path.read_text("utf-8"))
    for m in dict.fromkeys(r["label"] for r in rows):
        # a table-only row (llmash on another provider's copy of the file) stays off the chart
        rs = [r for r in rows if r["label"] == m and not r.get("errors") and r.get("throughput") and not r.get("table_only")]
        levels = scaling_levels(rs)
        pos = {c: i for i, c in enumerate(levels)}
        series = scaling_series(rs)
        main = {p: next((v for (q, _), v in series.items() if q == p), {}) for p in PROVIDERS}
        size = {(r["provider"], r["build"]): r.get("size_gb") for r in rs}
        extra = {(r["provider"], r["build"]) for r in rs if r.get("extra")}
        ymax = max(r["throughput"] for r in rs)
        fig, ax = plt.subplots(figsize=(9.6, 5.8))
        fig.subplots_adjust(left=0.14, right=0.83, top=0.87, bottom=0.2)
        both = [c for c in levels if c in main["llmash"] and c in main["vLLM"]]
        xs = [pos[c] for c in both]
        lo, hi = [main["vLLM"][c] for c in both], [main["llmash"][c] for c in both]
        ax.fill_between(xs, lo, hi, where=[h >= v for h, v in zip(hi, lo)], interpolate=True, color=t["llmash"],
                        alpha=t["fill"], linewidth=0, zorder=1)
        ax.fill_between(xs, lo, hi, where=[h < v for h, v in zip(hi, lo)], interpolate=True, color=t["vLLM"],
                        alpha=t["fill"], linewidth=0, zorder=1)
        # llmash's speed over each of the others at every level, on the same checkpoint, in rows under the axis
        for k, p in enumerate(p for p in ("vLLM", "Ollama") if main[p]):
            y = -0.13 - 0.065 * k
            ax.text(-0.015, y, f"llmash vs {p}", transform=ax.transAxes, ha="right", va="center", fontsize=9,
                    color=t["muted"])
            for c in levels:
                if c in main["llmash"] and c in main[p]:
                    r = main["llmash"][c] / main[p][c]
                    ax.text(pos[c], y, f"{r:.1f}x", transform=ax.get_xaxis_transform(), ha="center", va="center",
                            fontsize=10, fontweight="bold", color=t["llmash"] if r >= 1 else t[p])
        # the lines, with the provider, its last value and the file's build and size at each line's end; the end
        # labels are spread apart where lines end close together
        ends = []
        for (p, build), pts in series.items():
            px, py = [pos[c] for c in pts], list(pts.values())
            dashed = (p, build) in extra
            ax.plot(px, py, color=t[p], linewidth=3.0 if p == "llmash" and not dashed else 2.0,
                    linestyle=(0, (4, 3)) if dashed else "-", marker="o", markersize=5.5 if p == "llmash" else 4.5,
                    zorder=3 if p == "llmash" else 2)
            ends.append([py[-1], px[-1], p, build, size.get((p, build))])
        ends.sort(key=lambda e: e[0])
        gap = ymax * 0.1
        for i, e in enumerate(ends):
            e.append(e[0] if i == 0 else max(e[0], ends[i - 1][-1] + gap))
        for v, x, p, build, gb, ly in ends:
            ax.text(x + 0.16, ly, f"{p}  {v:,.0f}", va="center", fontsize=10.5, fontweight="bold", color=t[p],
                    clip_on=False)
            note = build + (f"  ·  {gb:.1f} GB" if gb else "")
            ax.text(x + 0.16, ly - ymax * 0.042, note, va="center", fontsize=8.5, color=t["muted"], clip_on=False)
        ax.set_xticks(range(len(levels)))
        ax.set_xticklabels([str(c) for c in levels])
        ax.set_xlim(-0.25, len(levels) - 0.75)
        ax.set_ylim(0, ymax * 1.08)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.set_xlabel("streams at once", fontsize=9.5, labelpad=48)
        ax.set_ylabel("tokens per second, all streams", fontsize=9.5)
        ax.set_title(m, loc="left", fontsize=13, fontweight="bold", color=t["text"], pad=22)
        ax.text(0, 1.015, next((r["note"] for r in rs if r.get("note")), ""), transform=ax.transAxes, fontsize=9,
                color=t["muted"], va="bottom")
        ax.grid(axis="y", color=t["grid"], linewidth=0.5, alpha=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="both", length=0)
        fig.savefig(DOCS / f"scale-{slug(m)}-{theme}.svg", transparent=True)
        plt.close(fig)


def scaling_tables():
    """A chart and a table per model between the SCALING markers: tokens per second of all streams at each level on
    each provider, and llmash's over the others' on the same checkpoint."""
    path = HERE / "scaling.json"
    if not path.exists():
        return ""
    rows = json.loads(path.read_text("utf-8"))
    parts = []
    for m in dict.fromkeys(r["label"] for r in rows):
        rs = [r for r in rows if r["label"] == m and not r.get("errors") and r.get("throughput")]
        levels = scaling_levels(rs)
        series = scaling_series(rs)
        main = {p: next((v for (q, _), v in series.items() if q == p), {}) for p in PROVIDERS}
        lines = [f"#### {m}", "",
                 picture(f"scale-{slug(m)}", f"{m}: tokens per second against streams at once on llmash, vLLM and Ollama"),
                 "", "| provider | file | weights | " + " | ".join(str(c) for c in levels) + " |",
                 "|---|---|--:|" + "--:|" * len(levels)]
        for (p, build), pts in series.items():
            r0 = next(r for r in rs if (r["provider"], r["build"]) == (p, build))
            cells = " | ".join(f"{pts[c]:,.0f}" if c in pts else "-" for c in levels)
            gb = f"{r0['size_gb']:.1f} GB" if r0.get("size_gb") else "-"
            lines.append(f"| {p} | {r0['file']} | {gb} | {cells} |")
        for p in ("vLLM", "Ollama"):
            if main[p] and main["llmash"]:
                cells = " | ".join(f"{main['llmash'][c] / main[p][c]:.2f}x"
                                   if c in main[p] and c in main["llmash"] else "-" for c in levels)
                lines.append(f"| llmash over {p}, same checkpoint | | | {cells} |")
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


def batch_tables():
    """A chart and a table per model between the BATCH markers: total tokens per second at each level."""
    path = HERE / "batch.json"
    if not path.exists():
        return ""
    rows = json.loads(path.read_text("utf-8"))
    parts = []
    for m in dict.fromkeys(r["label"] for r in rows):
        rs = [r for r in rows if r["label"] == m and not r.get("errors")]
        levels = sorted({r["concurrency"] for r in rs})
        lines = [f"#### {m}", "", picture(f"batch-{slug(m)}", f"{m}: tokens per second against concurrent requests"), "",
                 "| provider | format | file | " + " | ".join(str(c) for c in levels) + " |",
                 "|---|---|---|" + "--:|" * len(levels)]
        for (provider, build) in dict.fromkeys((r["provider"], r["build"]) for r in rs):
            by = {r["concurrency"]: r for r in rs if r["provider"] == provider and r["build"] == build}
            cells = " | ".join(f"{by[c]['throughput']:.0f}" if c in by else "-" for c in levels)
            file = next(r.get("file", "") for r in by.values())
            lines.append(f"| {provider} | {build} | {file} | {cells} |")
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


def picture(name, alt):
    return (f"<picture>\n  <source media=\"(prefers-color-scheme: dark)\" srcset=\"docs/{name}-dark.svg\">\n"
            f"  <img alt=\"{alt}\" src=\"docs/{name}-light.svg\">\n</picture>")


def replace_block(s, start, end, body):
    i, j = s.index(start), s.index(end)
    return s[:i] + start + "\n\n" + body + "\n\n" + s[j:]


def readme_tables():
    """A chart and a table per model, between the PROVIDERS markers and again between the MATH markers."""
    readme = HERE.parent / "README.md"
    s = readme.read_text("utf-8")

    runs = json.loads((HERE / "providers.json").read_text("utf-8"))
    parts = []
    for m in dict.fromkeys(r["label"] for r in runs):
        lines = [f"#### {m}", "", picture(f"speed-{slug(m)}", f"{m}: decode speed of each build on llmash, Ollama and vLLM"),
                 "", "| format | provider | file | weights | perplexity | vs BF16 | conversation | coding | thinking |",
                 "|---|---|---|--:|--:|--:|--:|--:|--:|"]
        ref = next((r["ppl_ref"] for r in runs if r["label"] == m and r.get("ppl_ref")), None)
        for name, rs in provider_rows(runs, m):
            for r in rs:
                cells = " | ".join(f"{r['workloads'][w]:.0f}" if r["workloads"].get(w) else "-" for w in WORKLOADS)
                ppl = f"{r['ppl']:.2f}" if r.get("ppl") else "-"
                vs = quality(r) if r.get("ppl_ref") else "-"
                lines.append(f"| {name} | {r['provider']} | {r.get('file', r['build'])} | {r['size_gb']:.1f} GB | {ppl} | {vs} | {cells} |")
        if ref:
            lines += ["", f"The model's own BF16 weights measure {ref:.2f} on the same text with the same runtime."]
        parts.append("\n".join(lines))
    s = replace_block(s, "<!-- PROVIDERS -->", "<!-- /PROVIDERS -->", "\n\n".join(parts))

    data = json.loads((HERE / "math_results.json").read_text("utf-8"))
    parts = []
    for m in data["models"]:
        tiers = data["tiers"]
        lines = [f"#### {m['model']}", "",
                 picture(f"math-{slug(m['model'])}", f"{m['model']}: correct answers out of 40 per build"), "",
                 "| build | size | " + " | ".join(tiers) + " | total | tokens |",
                 "|---|--:|" + "--:|" * len(tiers) + "--:|--:|"]
        for b in sorted(m["builds"], key=lambda b: -b["size_gb"]):
            cells = " | ".join(f"{b['tiers'][x][0]}/{b['tiers'][x][1]}" for x in tiers)
            lines.append(f"| {b['label']} | {b['size_gb']:.1f} GB | {cells} | {b['total'][0]}/{b['total'][1]} | "
                         f"{b['tokens'] / 1000:.0f}k |")
        parts.append("\n".join(lines))
    s = replace_block(s, "<!-- MATH -->", "<!-- /MATH -->", "\n\n".join(parts))
    if "<!-- BATCH -->" in s:
        s = replace_block(s, "<!-- BATCH -->", "<!-- /BATCH -->", batch_tables())
    if "<!-- SCALING -->" in s:
        s = replace_block(s, "<!-- SCALING -->", "<!-- /SCALING -->", scaling_tables())
    readme.write_text(s, encoding="utf-8")


if __name__ == "__main__":
    DOCS.mkdir(exist_ok=True)
    for old in list(DOCS.glob("providers-*.svg")) + list(DOCS.glob("math-light.svg")) + list(DOCS.glob("math-dark.svg")):
        old.unlink()
    for theme in THEMES:
        if (HERE / "providers.json").exists():
            providers(theme)
        if (HERE / "math_results.json").exists():
            math(theme)
        batch(theme)
        scaling(theme)
    readme_tables()
    print("wrote", ", ".join(sorted(p.name for p in DOCS.glob("*.svg"))))
