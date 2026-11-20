"""Figure of the headline result: success rate by hint level and model.

Writes figures/success-light.svg and figures/success-dark.svg (standard
library only, transparent background; the README picks one with <picture>
and prefers-color-scheme).
Successful runs (grade.json) of the current starter, with 95% Wilson
intervals as whiskers, from
summarize_runs.rows(), so the figure and the tables agree.

    python papers/score-centering/plot_results.py
"""
from pathlib import Path

from summarize_runs import MODELS, rows, wilson

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"
# every level that has runs, easiest last
LEVEL_NAMES = [("L0", "problem only"), ("L1", "pre-paper knowledge"), ("L2a", "direction"),
               ("L2b", "per-prefix direction"), ("L3", "decomposition given"), ("L4", "method given")]
# one hue (slate), three steps: Haiku lightest, Opus strongest. Transparent
# background, text in GitHub's own light / dark inks.
THEMES = {
    "light": {"text": "#1f2328", "halo": "#ffffff", "muted": "#59636e", "rule": "#d1d9e0",
              "series": ["#a3b4c7", "#5f7a96", "#1e3550"]},
    "dark": {"text": "#e6edf3", "halo": "#0d1117", "muted": "#9198a1", "rule": "#3d444d",
             "series": ["#3a5470", "#7894b2", "#c8d6e5"]},
}
W, H = 680, 288
LEFT, RIGHT, TOP, BOTTOM = 44, 8, 54, 52
# models not run yet: in the legend, and as empty dashed bars of unknown height
COMING = ["DeepSeek V4 Pro", "DeepSeek V4.1 Flash", "GLM 5.3 Flash"]
BAR, GAP, R = 15, 2, 2
SEP = 8  # between the models run and the models to come
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans', Helvetica, Arial, sans-serif"


def counts():
    """(runs, successes) per (level, model), over the current starter (v2)."""
    rs = [r for r in rows() if r["era"] == "v2"]
    if any(r["success"] is None for r in rs):
        raise SystemExit("some runs are not judged yet (grade.json: success is null)")
    c = {}
    for r in rs:
        k = (r["level"], r["model"])
        n, p = c.get(k, (0, 0))
        c[k] = (n + 1, p + r["success"])
    return c


def bar(x, y, h, color):
    """Column with slightly rounded top corners, square at the baseline; a
    zero is a 2px stub, so its series stays visible."""
    if h <= 0:
        return f'<rect x="{x}" y="{y - 2}" width="{BAR}" height="2" fill="{color}"/>'
    r = min(R, h)
    return (f'<path d="M{x},{y + h} V{y + r} Q{x},{y} {x + r},{y} H{x + BAR - r} Q{x + BAR},{y} {x + BAR},{y + r} '
            f'V{y + h} Z" fill="{color}"/>')


def svg(theme):
    t = THEMES[theme]
    c = counts()
    LEVELS = [(lvl, name) for lvl, name in LEVEL_NAMES if any(k[0] == lvl for k in c)]
    plot_h = H - TOP - BOTTOM
    base = TOP + plot_h
    band = (W - LEFT - RIGHT) / len(LEVELS)
    group = (len(MODELS) + len(COMING)) * BAR + (len(MODELS) + len(COMING) - 2) * GAP + SEP
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
           f'font-family="{FONT}" role="img" aria-labelledby="title desc">',
           '<title id="title">Successful runs by hint level and model</title>',
           '<desc id="desc">' + "; ".join(
               f"{lvl} {m}: {c[(lvl, m)][1]} of {c[(lvl, m)][0]}" for lvl, _ in LEVELS for m in MODELS
               if (lvl, m) in c) + "</desc>"]
    # legend, top left
    x = LEFT + (band - group) / 2
    for m, color in zip(MODELS, t["series"]):
        out.append(f'<rect x="{x}" y="6" width="10" height="10" rx="2" fill="{color}"/>')
        out.append(f'<text x="{x + 15}" y="15" font-size="12" fill="{t["muted"]}">{m}</text>')
        x += 24 + 7.5 * len(m)
    # second legend row: models not run yet, hollow dashed swatches
    x = LEFT + (band - group) / 2
    out.append(f'<text x="{x}" y="35" font-size="12" font-style="italic" fill="{t["muted"]}">More to come:</text>')
    x += 92
    for m in COMING:
        out.append(f'<rect x="{x + 0.5}" y="26.5" width="9" height="9" rx="2" fill="none" stroke="{t["muted"]}" '
                   f'stroke-dasharray="2 1.5"/>')
        out.append(f'<text x="{x + 15}" y="35" font-size="12" fill="{t["muted"]}">{m}</text>')
        x += 28 + 6.6 * len(m)
    # y axis: success rate, light gridlines, the baseline solid
    for pct in (0, 25, 50, 75, 100):
        y = round(base - pct / 100 * plot_h) + 0.5
        dash = "" if pct == 0 else ' stroke-dasharray="2 3"'
        out.append(f'<line x1="{LEFT}" x2="{W - RIGHT}" y1="{y}" y2="{y}" stroke="{t["rule"]}"{dash}/>')
        out.append(f'<text x="{LEFT - 8}" y="{y + 4}" font-size="10.5" text-anchor="end" fill="{t["muted"]}" '
                   f'font-variant-numeric="tabular-nums">{pct}%</text>')
    for i, (lvl, name) in enumerate(LEVELS):
        x0 = LEFT + i * band + (band - group) / 2
        for j, (m, color) in enumerate(zip(MODELS, t["series"])):
            if (lvl, m) not in c:
                continue
            n, p = c[(lvl, m)]
            h = p / n * plot_h
            xb = x0 + j * (BAR + GAP)
            out.append(bar(xb, base - h, h, color))
            lo, hi = wilson(p, n)
            xm, y_lo, y_hi = xb + BAR / 2, base - lo * plot_h, base - hi * plot_h
            whisker = f"M{xm},{y_lo} V{y_hi} M{xm - 3},{y_hi} H{xm + 3} M{xm - 3},{y_lo} H{xm + 3}"
            # a halo in the page's colour keeps the whisker visible over a bar of any shade
            out.append(f'<path d="{whisker}" stroke="{t["halo"]}" stroke-width="3" fill="none"/>')
            out.append(f'<path d="{whisker}" stroke="{t["muted"]}" stroke-width="1" fill="none"/>')
        # models to come: dashed outline over the whole axis and a "?", so no height is implied
        xc = x0 + len(MODELS) * (BAR + GAP) - GAP + SEP
        for j in range(len(COMING)):
            xb = xc + j * (BAR + GAP)
            out.append(f'<rect x="{xb + 0.5}" y="{TOP + 0.5}" width="{BAR - 1}" height="{plot_h - 1}" rx="2" '
                       f'fill="none" stroke="{t["muted"]}" stroke-opacity="0.45" stroke-dasharray="3 3"/>')
            out.append(f'<text x="{xb + BAR / 2}" y="{TOP + plot_h / 2 + 4}" font-size="11" text-anchor="middle" '
                       f'fill="{t["muted"]}" fill-opacity="0.8">?</text>')
        cx = LEFT + i * band + band / 2
        out.append(f'<text x="{cx}" y="{base + 20}" font-size="12.5" font-weight="600" text-anchor="middle" '
                   f'fill="{t["text"]}">{lvl}</text>')
        out.append(f'<text x="{cx}" y="{base + 37}" font-size="11.5" text-anchor="middle" fill="{t["muted"]}">'
                   f'{name}</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for theme in THEMES:
        (OUT / f"success-{theme}.svg").write_text(svg(theme))
        print(OUT / f"success-{theme}.svg")
