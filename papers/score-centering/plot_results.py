"""Figure of the headline result: drift-free runs by hint level and model.

Writes figures/results-light.svg and figures/results-dark.svg (standard
library only, transparent background; the README picks one with <picture>
and prefers-color-scheme).
Successful runs (grade.json) of the current starter, from
summarize_runs.rows(), so the figure and the tables agree.

    python papers/score-centering/plot_results.py
"""
from pathlib import Path

from summarize_runs import MODELS, rows

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"
LEVELS = [("L1", "pre-paper knowledge"), ("L2a", "direction"), ("L2b", "per-prefix direction"),
          ("L3", "decomposition given")]
# one hue (slate), three steps: Haiku lightest, Opus strongest. Transparent
# background, text in GitHub's own light / dark inks.
THEMES = {
    "light": {"text": "#1f2328", "muted": "#59636e", "rule": "#d1d9e0",
              "series": ["#a3b4c7", "#5f7a96", "#1e3550"]},
    "dark": {"text": "#e6edf3", "muted": "#9198a1", "rule": "#3d444d",
             "series": ["#3a5470", "#7894b2", "#c8d6e5"]},
}
W, H = 680, 270
LEFT, RIGHT, TOP, BOTTOM = 8, 8, 44, 52
BAR, GAP, R = 22, 3, 2
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
    plot_h = H - TOP - BOTTOM
    base = TOP + plot_h
    band = (W - LEFT - RIGHT) / len(LEVELS)
    group = len(MODELS) * BAR + (len(MODELS) - 1) * GAP
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
    # baseline only
    out.append(f'<line x1="{LEFT}" x2="{W - RIGHT}" y1="{base + 0.5}" y2="{base + 0.5}" stroke="{t["rule"]}"/>')
    for i, (lvl, name) in enumerate(LEVELS):
        x0 = LEFT + i * band + (band - group) / 2
        for j, (m, color) in enumerate(zip(MODELS, t["series"])):
            if (lvl, m) not in c:
                continue
            n, p = c[(lvl, m)]
            h = p / n * plot_h
            xb = x0 + j * (BAR + GAP)
            out.append(bar(xb, base - h, h, color))
            out.append(f'<text x="{xb + BAR / 2}" y="{base - h - 6}" font-size="10.5" text-anchor="middle" '
                       f'fill="{t["muted"]}" font-variant-numeric="tabular-nums">{p}/{n}</text>')
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
        (OUT / f"results-{theme}.svg").write_text(svg(theme))
        print(OUT / f"results-{theme}.svg")
