"""The README's charts, drawn from the measured results beside this script:

    providers.json     decode speed and size of each build of a model on llmash, Ollama and vLLM
    math_results.json  the 40-question math comparison of RCO-3 against Q8_0 and other builds, with tokens used

Each chart is written twice, for GitHub's light and dark themes, into docs/.

    python bench/charts.py
"""
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

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


def style(t):
    plt.rcParams.update({
        "font.family": ["DejaVu Sans"], "font.size": 9.5, "svg.fonttype": "path",
        "text.color": t["text"], "axes.labelcolor": t["muted"], "xtick.color": t["muted"], "ytick.color": t["muted"],
        "axes.edgecolor": t["grid"], "axes.facecolor": "none", "figure.facecolor": "none",
    })


def clean(ax, t):
    ax.grid(color=t["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="both", length=0)


def mean_speed(entry):
    vals = [entry["workloads"][w] for w in WORKLOADS if entry["workloads"].get(w)]
    return sum(vals) / len(vals)


def place_labels(ax, points, t):
    """Each dot's name beside it, on whichever side keeps it clear of the other names and dots."""
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    taken = []
    for x, y, _, _ in points:
        cx, cy = ax.transData.transform((x, y))
        taken.append((cx - 5, cy - 5, cx + 5, cy + 5))
    frame = ax.get_window_extent(renderer)
    # a label crossing another provider's line stays readable
    backing = dict(boxstyle="round,pad=0.12", facecolor=t["surface"], edgecolor="none", alpha=0.85)
    sides = [((7, 0), "left", "center"), ((-7, 0), "right", "center"), ((0, 8), "center", "bottom"),
             ((0, -8), "center", "top"), ((7, 7), "left", "bottom"), ((7, -7), "left", "top"),
             ((-7, 7), "right", "bottom"), ((-7, -7), "right", "top")]
    for x, y, text, color in points:
        best = None
        for off, ha, va in sides:
            a = ax.annotate(text, (x, y), xytext=off, textcoords="offset points", ha=ha, va=va, fontsize=8.5,
                            color=color, bbox=backing, zorder=4)
            bb = a.get_window_extent(renderer)
            box = (bb.x0 - 1, bb.y0 - 1, bb.x1 + 1, bb.y1 + 1)
            inside = bb.x0 >= frame.x0 and bb.x1 <= frame.x1 + 40 and bb.y0 >= frame.y0 and bb.y1 <= frame.y1 + 6
            clear = all(box[2] < o[0] or box[0] > o[2] or box[3] < o[1] or box[1] > o[3] for o in taken)
            if inside and clear:
                best = box
                break
            a.remove()
        if best is None:
            off, ha, va = sides[0]
            a = ax.annotate(text, (x, y), xytext=off, textcoords="offset points", ha=ha, va=va, fontsize=8.5,
                            color=color, bbox=backing, zorder=4)
            bb = a.get_window_extent(renderer)
            best = (bb.x0, bb.y0, bb.x1, bb.y1)
        taken.append(best)


def providers(theme):
    """Per model: every build's decode speed against its size, one line per provider."""
    t = THEMES[theme]
    style(t)
    runs = json.loads((HERE / "providers.json").read_text("utf-8"))
    models = list(dict.fromkeys(r["label"] for r in runs))
    fig, axes = plt.subplots(1, len(models), figsize=(3.3 * len(models) + 0.6, 3.9), squeeze=False)
    for ax, m in zip(axes[0], models):
        points = []
        top_x = max(mean_speed(r) for r in runs if r["label"] == m)
        top_y = max(r["size_gb"] for r in runs if r["label"] == m)
        for p in PROVIDERS:
            rs = sorted((r for r in runs if r["label"] == m and r["provider"] == p), key=lambda r: r["size_gb"])
            if not rs:
                continue
            xs, ys = [mean_speed(r) for r in rs], [r["size_gb"] for r in rs]
            ax.plot(xs, ys, color=t[p], linewidth=2, zorder=2)
            ax.plot(xs, ys, "o", color=t[p], markersize=7, markeredgecolor=t["surface"], markeredgewidth=1.5, zorder=3)
            points += [(x, y, r.get("short", r["build"]), t["text"]) for x, y, r in zip(xs, ys, rs)]
        ax.set_xlim(0, top_x * 1.28)
        ax.set_ylim(0, top_y * 1.15)
        ax.set_title(m, loc="left", fontsize=10.5, fontweight="bold", color=t["text"], pad=8)
        ax.set_xlabel("tokens per second")
        clean(ax, t)
        place_labels(ax, points, t)
    axes[0][0].set_ylabel("size on disk (GB)")
    handles = [plt.Line2D([], [], color=t[p], linewidth=2, marker="o", markersize=7, markeredgecolor=t["surface"])
               for p in PROVIDERS]
    leg = fig.legend(handles, PROVIDERS, loc="upper right", ncol=3, frameon=False, fontsize=9.5,
                     bbox_to_anchor=(1.0, 1.0))
    for txt in leg.get_texts():
        txt.set_color(t["text"])
    fig.tight_layout(rect=(0, 0, 1, 0.93), w_pad=2.0)
    fig.savefig(DOCS / f"providers-{theme}.svg", transparent=True)
    plt.close(fig)


def math(theme):
    """Per model: correct answers out of 40 against the tokens each build spent getting there."""
    t = THEMES[theme]
    style(t)
    data = json.loads((HERE / "math_results.json").read_text("utf-8"))
    models = data["models"]
    fig, axes = plt.subplots(1, len(models), figsize=(8.4, 3.4), squeeze=False)
    for ax, m in zip(axes[0], models):
        points = []
        top = max(b["tokens"] for b in m["builds"]) / 1000
        low = min(b["tokens"] for b in m["builds"]) / 1000
        floor = min(b["total"][0] for b in m["builds"])
        for b in m["builds"]:
            x, y = b["tokens"] / 1000, b["total"][0]
            color = t["llmash"] if b["label"] == "RCO-3" else t["other"]
            ax.plot([x], [y], "o", color=color, markersize=8, markeredgecolor=t["surface"], markeredgewidth=1.5,
                    zorder=3)
            points.append((x, y, f"{b['label']} · {b['size_gb']:.1f} GB", t["text"]))
        # dots, not bars: each panel spans its own data, and the ticks carry the scale
        ax.set_xlim(low * 0.75, top * 1.3)
        ax.set_ylim(max(0, 5 * ((floor - 4) // 5)), 41.5)
        ax.set_title(m["model"], loc="left", fontsize=10.5, fontweight="bold", color=t["text"], pad=8)
        ax.set_xlabel("tokens generated over the 40 (thousands)")
        clean(ax, t)
        place_labels(ax, points, t)
    axes[0][0].set_ylabel("correct of 40")
    fig.tight_layout(w_pad=2.5)
    fig.savefig(DOCS / f"math-{theme}.svg", transparent=True)
    plt.close(fig)


def readme_table():
    """The providers chart as a table, between the PROVIDERS markers in the README."""
    runs = json.loads((HERE / "providers.json").read_text("utf-8"))
    order = list(dict.fromkeys(r["label"] for r in runs))
    runs.sort(key=lambda r: (order.index(r["label"]), PROVIDERS.index(r["provider"]), r["size_gb"]))
    lines = ["| model | provider | build | size | conversation | coding | thinking |", "|---|---|---|--:|--:|--:|--:|"]
    for r in runs:
        cells = " | ".join(f"{r['workloads'][w]:.0f}" if r["workloads"].get(w) else "-" for w in WORKLOADS)
        lines.append(f"| {r['label']} | {r['provider']} | {r['build']} | {r['size_gb']:.1f} GB | {cells} |")
    readme = HERE.parent / "README.md"
    s = readme.read_text("utf-8")
    start, end = "<!-- PROVIDERS -->", "<!-- /PROVIDERS -->"
    i, j = s.index(start), s.index(end)
    readme.write_text(s[:i] + start + "\n\n" + "\n".join(lines) + "\n\n" + s[j:], encoding="utf-8")


if __name__ == "__main__":
    DOCS.mkdir(exist_ok=True)
    for theme in THEMES:
        if (HERE / "providers.json").exists():
            providers(theme)
        if (HERE / "math_results.json").exists():
            math(theme)
    if (HERE / "providers.json").exists():
        readme_table()
    print("wrote", ", ".join(sorted(p.name for p in DOCS.glob("*.svg"))))
