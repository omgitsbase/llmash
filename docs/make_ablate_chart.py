"""Abliteration before/after chart, light + dark, matching the repo's other docs/ SVGs."""
import matplotlib
matplotlib.use("svg")
import matplotlib.pyplot as plt
from matplotlib import font_manager  # noqa

# refusals out of 100 harmful prompts, base -> abliterated (16-pass search, measured)
MODELS = ["Qwen3.8\n27B", "Qwen3.5\n4B", "Gemma 4\nE4B", "Gemma 4\n26B-A4B"]
BASE = [98, 99, 98, 100]
ABL  = [0, 1, 1, 4]

THEMES = {
    "light": dict(fg="#1f2328", mut="#59636e", before="#eb6834", after="#1baf7a", grid="#d0d7de"),
    "dark":  dict(fg="#e6edf3", mut="#9198a1", before="#d95926", after="#199e70", grid="#30363d"),
}

for name, t in THEMES.items():
    fig, ax = plt.subplots(figsize=(9.6, 5.1))
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    x = range(len(MODELS))
    w = 0.38
    b1 = ax.bar([i - w / 2 for i in x], BASE, w, label="base model", color=t["before"])
    b2 = ax.bar([i + w / 2 for i in x], ABL, w, label="abliterated", color=t["after"])
    for bars in (b1, b2):
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 1.5, f"{int(r.get_height())}",
                    ha="center", va="bottom", color=t["fg"], fontsize=12, fontweight="bold")
    ax.set_ylim(0, 108)
    ax.set_ylabel("refusals on 100 harmful prompts", color=t["fg"], fontsize=13)
    ax.set_title("Abliteration: refusals before and after  (lower = more uncensored)",
                 color=t["fg"], fontsize=15, pad=14, loc="left")
    ax.set_xticks(list(x))
    ax.set_xticklabels(MODELS, color=t["fg"], fontsize=12)
    ax.tick_params(axis="y", colors=t["mut"])
    ax.tick_params(axis="x", length=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(t["grid"])
    ax.yaxis.grid(True, color=t["grid"], linewidth=0.7, alpha=0.6)
    ax.set_axisbelow(True)
    leg = ax.legend(loc="center right", frameon=False, fontsize=12)
    for txt in leg.get_texts():
        txt.set_color(t["fg"])
    fig.tight_layout()
    fig.savefig(f"abliterate-{name}.svg", transparent=True)
    plt.close(fig)
    print(f"wrote abliterate-{name}.svg")
