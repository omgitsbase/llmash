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


def bars(rows, groups, t, xlabel, xmax, path, title, value_text, ticks_to=0, legend=None, left_in=0.0):
    """Horizontal bars, one per row, in labelled groups; the value and a note at each bar's end. A fixed left margin
    (left_in inches) puts the bars of every chart drawn with it at the same place."""
    height = 0.5 * len(rows) + 0.45 * sum(1 for g in groups if g["name"]) + 1.1 + (0.3 if legend else 0)
    fig, ax = plt.subplots(figsize=(9.6, height))
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
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", color=t["text"], pad=10 + (18 if legend else 0))
    if legend:
        handles = [Patch(color=color, label=name) for name, color in legend]
        ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=len(handles), frameon=False,
                  fontsize=10.5, handlelength=1.0, columnspacing=1.4, borderaxespad=0.2, labelcolor=t["text"])
    clean(ax, t)
    if left_in:
        fig.subplots_adjust(left=left_in / 9.6, right=0.98, top=1 - (0.75 + (0.3 if legend else 0)) / height,
                            bottom=0.6 / height)
    else:
        fig.tight_layout()
    fig.savefig(path, transparent=True)
    plt.close(fig)


# builds grouped by format, so the same build on each provider sits together; within a group the providers keep one
# order on every chart. The 3-bit builds are llmash's own and run nowhere else.
TIERS = (("3-bit", "3-bit, llmash only"), ("4-bit", "4-bit GGUF"), ("NVFP4", "NVFP4"), ("8-bit", "8-bit"),
         ("16-bit", "16-bit"))


def provider_rows(runs, m):
    for tier, name in TIERS:
        rs = [r for p in PROVIDERS for r in sorted((r for r in runs if r["label"] == m and r["provider"] == p and
                                                   r["tier"] == tier), key=lambda r: r["size_gb"] or 0)]
        if rs:
            yield name, rs


def size_text(r):
    return f"{r['size_gb']:.1f} GB" if r.get("size_gb") else "September build"


def providers(theme):
    """Per model: every build's decode speed, grouped by format, the provider, build and size in each row's label.
    Every chart shares one scale and one label margin, so the charts line up with each other."""
    t = THEMES[theme]
    style(t)
    runs = json.loads((HERE / "providers.json").read_text("utf-8"))
    xmax = max(mean_speed(r) for r in runs)
    for m in dict.fromkeys(r["label"] for r in runs):
        groups = [{"name": name, "rows": [{"label": f"{r['provider']}  {r['build']}  ·  {size_text(r)}",
                                           "value": mean_speed(r), "color": t[r["provider"]]} for r in rs]}
                  for name, rs in provider_rows(runs, m)]
        bars([r for g in groups for r in g["rows"]], groups, t, "tokens per second while generating", xmax,
             DOCS / f"speed-{slug(m)}-{theme}.svg", m, lambda r: f"{r['value']:.0f}",
             legend=[(p, t[p]) for p in PROVIDERS], left_in=3.1)


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
                 "", "| format | provider | build | weights | conversation | coding | thinking |",
                 "|---|---|---|--:|--:|--:|--:|"]
        notes = []
        for name, rs in provider_rows(runs, m):
            for r in rs:
                cells = " | ".join(f"{r['workloads'][w]:.0f}" if r["workloads"].get(w) else "-" for w in WORKLOADS)
                size = f"{r['size_gb']:.1f} GB" if r.get("size_gb") else "-"
                build = r["build"] + ("\\*" if r.get("note") else "")
                lines.append(f"| {name} | {r['provider']} | {build} | {size} | {cells} |")
                if r.get("note") and r["note"] not in notes:
                    notes.append(r["note"])
        for n in notes:
            lines += ["", f"\\* {n}."]
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
    readme_tables()
    print("wrote", ", ".join(sorted(p.name for p in DOCS.glob("*.svg"))))
