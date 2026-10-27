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
from matplotlib.patches import Patch  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
DOCS = HERE.parent / "docs"
WORKLOADS = ("conversation", "coding", "thinking")
PROVIDERS = ("llmash", "Ollama", "vLLM")

THEMES = {
    "light": {"surface": "#ffffff", "text": "#1f2328", "muted": "#59636e", "grid": "#d1d9e0",
              "llmash": "#2a78d6", "Ollama": "#eb6834", "vLLM": "#1baf7a", "other": "#8c959f"},
    "dark": {"surface": "#0d1117", "text": "#e6edf3", "muted": "#9198a1", "grid": "#3d444d",
             "llmash": "#3987e5", "Ollama": "#d95926", "vLLM": "#199e70", "other": "#6e7681"},
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
    readme_tables()
    print("wrote", ", ".join(sorted(p.name for p in DOCS.glob("*.svg"))))
