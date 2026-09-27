"""The README's charts, drawn from the measured results beside this script:

    formats.json       decode speed by model and weight format, and the same files on the previous runtime
    math_results.json  the 40-question math comparison of RCO-3 against Q8_0 and other small builds

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

THEMES = {
    "light": {"text": "#1f2328", "muted": "#59636e", "grid": "#d1d9e0",
              "NVFP4": "#0f766e", "GSQ-RCO": "#2563eb", "RCO-3": "#60a5fa", "INT4": "#b45309",
              "old": "#afb8c1", "new": "#0f766e", "ref": "#57606a", "good": "#0f766e", "bad": "#b45309"},
    "dark": {"text": "#e6edf3", "muted": "#9198a1", "grid": "#3d444d",
             "NVFP4": "#2dd4bf", "GSQ-RCO": "#60a5fa", "RCO-3": "#93c5fd", "INT4": "#f0a04b",
             "old": "#484f58", "new": "#2dd4bf", "ref": "#9198a1", "good": "#2dd4bf", "bad": "#f0a04b"},
}
FORMAT_NAMES = {"NVFP4": "NVFP4", "GSQ-RCO": "GSQ-RCO 3-bit", "RCO-3": "RCO-3 (3-bit)", "INT4": "INT4"}


def style(t):
    plt.rcParams.update({
        "font.family": ["DejaVu Sans"], "font.size": 10, "svg.fonttype": "path",
        "text.color": t["text"], "axes.labelcolor": t["muted"], "xtick.color": t["muted"], "ytick.color": t["text"],
        "axes.edgecolor": t["grid"], "axes.facecolor": "none", "figure.facecolor": "none",
    })


def mean_speed(entry):
    vals = [entry["workloads"][w] for w in WORKLOADS if entry["workloads"].get(w)]
    return sum(vals) / len(vals)


def clean(ax, t, axis="x"):
    ax.grid(axis=axis, color=t["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="both", length=0)


def formats(theme):
    """One bar per model and format on the current runtime."""
    t = THEMES[theme]
    style(t)
    runs = [r for r in json.loads((HERE / "formats.json").read_text("utf-8")) if r["runtime"] == "current"]
    models = list(dict.fromkeys(r["label"] for r in runs))
    rows = []
    for m in models:
        for r in runs:
            if r["label"] == m:
                rows.append(r)
    fig, ax = plt.subplots(figsize=(8.4, 0.42 * len(rows) + 0.4 * len(models) + 0.9))
    y, ys, labels, top = 0.0, [], [], max(mean_speed(r) for r in rows)
    for m in models:
        for r in [r for r in rows if r["label"] == m]:
            v = mean_speed(r)
            ax.barh(-y, v, height=0.66, color=t[r["format"]])
            ax.text(v + top * 0.01, -y, f"{v:.0f}", va="center", fontsize=9.5, color=t["text"])
            ys.append(-y)
            labels.append(f"{m}  ·  {r['build']}")
            y += 1
        y += 0.55
    ax.set_yticks(ys)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, top * 1.12)
    ax.set_xlabel("tokens per second while generating (mean of three workloads)")
    clean(ax, t)
    handles = [plt.Rectangle((0, 0), 1, 1, color=t[f]) for f in FORMAT_NAMES if any(r["format"] == f for r in rows)]
    names = [FORMAT_NAMES[f] for f in FORMAT_NAMES if any(r["format"] == f for r in rows)]
    leg = ax.legend(handles, names, loc="lower right", frameon=False, fontsize=9)
    for txt in leg.get_texts():
        txt.set_color(t["text"])
    fig.tight_layout()
    fig.savefig(DOCS / f"formats-{theme}.svg", transparent=True)
    plt.close(fig)


def release(theme):
    """The same files on the previous runtime and on this one."""
    t = THEMES[theme]
    style(t)
    runs = json.loads((HERE / "formats.json").read_text("utf-8"))
    pairs = []
    for old in [r for r in runs if r["runtime"] == "previous"]:
        for new in runs:
            if new["runtime"] == "current" and new["label"] == old["label"] and new["build"] == old["build"]:
                pairs.append((old, new))
    fig, ax = plt.subplots(figsize=(8.4, 0.95 * len(pairs) + 1.0))
    top = max(mean_speed(n) for _, n in pairs)
    ys, labels = [], []
    for i, (old, new) in enumerate(pairs):
        yc = -i * 1.1
        a, b = mean_speed(old), mean_speed(new)
        ax.barh(yc + 0.19, a, height=0.36, color=t["old"])
        ax.barh(yc - 0.19, b, height=0.36, color=t["new"])
        ax.text(a + top * 0.01, yc + 0.19, f"{a:.0f}", va="center", fontsize=9, color=t["muted"])
        ax.text(b + top * 0.01, yc - 0.19, f"{b:.0f}  (+{(b / a - 1) * 100:.0f}%)", va="center", fontsize=9.5,
                color=t["text"], fontweight="bold")
        ys.append(yc)
        labels.append(f"{new['label']}\n{new['build']}")
    ax.set_yticks(ys)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, top * 1.22)
    ax.set_xlabel("tokens per second while generating (mean of three workloads)")
    clean(ax, t)
    handles = [plt.Rectangle((0, 0), 1, 1, color=t["old"]), plt.Rectangle((0, 0), 1, 1, color=t["new"])]
    leg = ax.legend(handles, ["0.4.35", "0.5.0"], loc="lower right", frameon=False, fontsize=9)
    for txt in leg.get_texts():
        txt.set_color(t["text"])
    fig.tight_layout()
    fig.savefig(DOCS / f"release-{theme}.svg", transparent=True)
    plt.close(fig)


def math(theme):
    """Correct answers out of 40 per build, one panel per model."""
    t = THEMES[theme]
    style(t)
    data = json.loads((HERE / "math_results.json").read_text("utf-8"))
    models = data["models"]
    fig, axes = plt.subplots(1, len(models), figsize=(8.4, 3.1), sharex=True)
    for ax, m in zip(axes, models):
        builds = m["builds"]
        ref = next((b for b in builds if b["label"] == "Q8_0"), builds[0])
        for i, b in enumerate(builds):
            c, n = b["total"]
            color = t["ref"] if b is ref else (t["good"] if b["label"] == "RCO-3" else t["old"])
            ax.barh(-i, c, height=0.62, color=color)
            ax.text(c + 0.5, -i, f"{c}/{n}", va="center", fontsize=9.5, color=t["text"])
        ax.set_yticks([-i for i in range(len(builds))])
        ax.set_yticklabels([f"{b['label']}  ·  {b['size_gb']:.1f} GB" for b in builds])
        ax.set_xlim(0, 46)
        ax.set_title(m["model"], loc="left", fontsize=11, fontweight="bold", color=t["text"], pad=6)
        clean(ax, t)
    axes[0].set_xlabel("correct of 40")
    axes[1].set_xlabel("correct of 40")
    fig.tight_layout(w_pad=2.5)
    fig.savefig(DOCS / f"math-{theme}.svg", transparent=True)
    plt.close(fig)


def readme_table():
    """The formats chart as a table, between the FORMATS markers in the README."""
    runs = [r for r in json.loads((HERE / "formats.json").read_text("utf-8")) if r["runtime"] == "current"]
    order = list(dict.fromkeys(r["label"] for r in runs))
    runs.sort(key=lambda r: order.index(r["label"]))
    lines = ["| model | build | conversation | coding | thinking |", "|---|---|--:|--:|--:|"]
    for r in runs:
        cells = " | ".join(f"{r['workloads'][w]:.0f}" if r["workloads"].get(w) else "-" for w in WORKLOADS)
        lines.append(f"| {r['label']} | {r['build']} | {cells} |")
    readme = HERE.parent / "README.md"
    s = readme.read_text("utf-8")
    start, end = "<!-- FORMATS -->", "<!-- /FORMATS -->"
    i, j = s.index(start), s.index(end)
    readme.write_text(s[:i] + start + "\n\n" + "\n".join(lines) + "\n\n" + s[j:], encoding="utf-8")


if __name__ == "__main__":
    DOCS.mkdir(exist_ok=True)
    for theme in THEMES:
        if (HERE / "formats.json").exists():
            formats(theme)
            release(theme)
            readme_table()
        if (HERE / "math_results.json").exists():
            math(theme)
    print("wrote", ", ".join(sorted(p.name for p in DOCS.glob("*.svg"))))
