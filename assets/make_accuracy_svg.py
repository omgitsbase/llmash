"""Accuracy against bit-width: the custom allocation versus uniform builds.

Writes a light and a dark SVG from one source so the two cannot drift. Static,
because a README image cannot carry a hover layer; the README puts the same
numbers in a table beside it.
"""
import pathlib

# Published GSQ-RCO results on Qwen3.8-27B: the mean of AIME25, GPQA-Diamond
# and LiveCodeBench v6, against the fp8 original.
BASE = 91.87
RCO = [(2.50, 86.0, "IQ2_XS"), (2.75, 89.5, "IQ2_S"),
       (3.00, 91.2, "IQ3_XXS"), (3.47, 91.8, "IQ3_S")]
UNIFORM = [(2.50, 78.3, "IQ2_S"), (2.87, 89.7, "Q2_K_XL"), (3.47, 90.1, "IQ3_S")]

W, H = 760, 430
L, R, T, B = 64, 176, 62, 66          # margins; R leaves room for direct labels
X0, X1 = 2.40, 3.52
Y0, Y1 = 76.0, 94.0

LIGHT = dict(surface="#fcfcfb", primary="#0b0b0b", secondary="#52514e",
             muted="#898781", grid="#e1e0d9", axis="#c3c2b7",
             s1="#2a78d6", s2="#eb6834")
DARK = dict(surface="#1a1a19", primary="#ffffff", secondary="#c3c2b7",
            muted="#898781", grid="#2c2c2a", axis="#383835",
            s1="#3987e5", s2="#d95926")

FONT = ("ui-sans-serif,-apple-system,BlinkMacSystemFont,'Segoe UI',"
        "Helvetica,Arial,sans-serif")


def px(v):
    return L + (v - X0) / (X1 - X0) * (W - L - R)


def py(v):
    return T + (Y1 - v) / (Y1 - Y0) * (H - T - B)


def svg(c):
    o = []
    a = o.append
    a(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
      f'role="img" aria-label="Task average against bits per weight. The custom allocation '
      f'holds within a point of the fp8 original down to 3 bits, where a uniform build of the '
      f'same size has already fallen away.">')
    a(f'<rect width="{W}" height="{H}" fill="{c["surface"]}"/>')
    a(f'<g font-family="{FONT}">')

    a(f'<text x="{L}" y="30" font-size="17" font-weight="600" fill="{c["primary"]}">'
      f'Accuracy against bit-width</text>')
    a(f'<text x="{L}" y="48" font-size="12.5" fill="{c["secondary"]}">'
      f'Qwen3.8-27B, mean of AIME25, GPQA-Diamond and LiveCodeBench v6</text>')

    # gridlines and the y scale
    for v in (78, 82, 86, 90, 94):
        y = py(v)
        a(f'<line x1="{L}" y1="{y:.1f}" x2="{px(X1):.1f}" y2="{y:.1f}" stroke="{c["grid"]}" stroke-width="1"/>')
        a(f'<text x="{L - 10}" y="{y + 4:.1f}" font-size="11.5" text-anchor="end" '
          f'fill="{c["muted"]}">{v}</text>')
    for v in (2.5, 2.75, 3.0, 3.25, 3.5):
        x = px(v)
        a(f'<text x="{x:.1f}" y="{py(Y0) + 22:.1f}" font-size="11.5" text-anchor="middle" '
          f'fill="{c["muted"]}">{v:g}</text>')
    a(f'<line x1="{L}" y1="{py(Y0):.1f}" x2="{px(X1):.1f}" y2="{py(Y0):.1f}" '
      f'stroke="{c["axis"]}" stroke-width="1"/>')
    a(f'<text x="{L}" y="{H - 18}" font-size="12" fill="{c["secondary"]}">bits per weight</text>')
    a(f'<text transform="translate(18,{py(85):.1f}) rotate(-90)" font-size="12" '
      f'text-anchor="middle" fill="{c["secondary"]}">task average</text>')

    # the fp8 original, the thing both methods are trying to keep
    yb = py(BASE)
    a(f'<line x1="{L}" y1="{yb:.1f}" x2="{px(X1):.1f}" y2="{yb:.1f}" stroke="{c["muted"]}" '
      f'stroke-width="1.5" stroke-dasharray="6 4"/>')
    a(f'<text x="{L + 8}" y="{yb - 8:.1f}" font-size="12" fill="{c["secondary"]}">'
      f'fp8 original, {BASE}</text>')

    # uniform builds: published points, not a swept curve, so they stay unjoined
    for i, (x, y, name) in enumerate(UNIFORM):
        a(f'<rect x="{px(x) - 5:.1f}" y="{py(y) - 5:.1f}" width="10" height="10" rx="2" '
          f'fill="{c["s2"]}" stroke="{c["surface"]}" stroke-width="2"/>')
        if i < len(UNIFORM) - 1:
            a(f'<text x="{px(x):.1f}" y="{py(y) + 22:.1f}" font-size="10.5" text-anchor="middle" '
              f'fill="{c["muted"]}">{name}</text>')

    # the custom allocation, swept across widths
    pts = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y, _ in RCO)
    a(f'<polyline points="{pts}" fill="none" stroke="{c["s1"]}" stroke-width="2" '
      f'stroke-linejoin="round" stroke-linecap="round"/>')
    for i, (x, y, name) in enumerate(RCO):
        a(f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="5.5" fill="{c["s1"]}" '
          f'stroke="{c["surface"]}" stroke-width="2"/>')
        if i < len(RCO) - 1:
            a(f'<text x="{px(x):.1f}" y="{py(y) - 13:.1f}" font-size="10.5" text-anchor="middle" '
              f'fill="{c["muted"]}">{name}</text>')

    # direct labels, so identity is never colour alone
    lx = px(RCO[-1][0]) + 14
    a(f'<text x="{lx:.1f}" y="{py(RCO[-1][1]) - 2:.1f}" font-size="12.5" font-weight="600" '
      f'fill="{c["s1"]}">custom allocation</text>')
    a(f'<text x="{lx:.1f}" y="{py(RCO[-1][1]) + 14:.1f}" font-size="11" fill="{c["muted"]}">'
      f'{RCO[-1][2]} at {RCO[-1][1]}</text>')
    a(f'<text x="{lx:.1f}" y="{py(UNIFORM[-1][1]) + 22:.1f}" '
      f'font-size="12.5" font-weight="600" fill="{c["s2"]}">uniform build</text>')
    a(f'<text x="{lx:.1f}" y="{py(UNIFORM[-1][1]) + 38:.1f}" font-size="11" fill="{c["muted"]}">'
      f'{UNIFORM[-1][2]} at {UNIFORM[-1][1]}</text>')

    a('</g></svg>')
    return "\n".join(o)


out = pathlib.Path(__file__).resolve().parent
(out / "accuracy-light.svg").write_text(svg(LIGHT), encoding="utf-8")
(out / "accuracy-dark.svg").write_text(svg(DARK), encoding="utf-8")
print("wrote", out / "accuracy-light.svg")
print("wrote", out / "accuracy-dark.svg")
