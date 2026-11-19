"""Figure of the headline result: drift-free runs by hint level and model.

Writes figures/results-light.svg and figures/results-dark.svg (standard
library only; the README picks one with <picture> and prefers-color-scheme).
Counts come from summarize_runs.rows(), so the figure and the tables agree.

    python papers/score-centering/plot_results.py
"""
from pathlib import Path

from summarize_runs import MODELS, rows

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"
LEVELS = [("L1", "pre-paper knowledge"), ("L2a", "direction"), ("L2b", "per-prefix direction"),
          ("L3", "decomposition given")]
# categorical slots 1-3 of the reference palette, validated in both modes
THEMES = {
    "light": {"surface": "#fcfcfb", "text": "#0b0b0b", "secondary": "#52514e", "muted": "#898781",
              "grid": "#e1e0d9", "series": ["#2a78d6", "#eb6834", "#1baf7a"]},
    "dark": {"surface": "#1a1a19", "text": "#f0efec", "secondary": "#c3c2b7", "muted": "#898781",
             "grid": "#2c2c2a", "series": ["#3987e5", "#d95926", "#199e70"]},
}
W, H = 760, 400
LEFT, RIGHT, TOP, BOTTOM = 56, 24, 114, 64
BAR, GAP, R = 24, 2, 4
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"


def counts():
    c = {}
    for r in rows():
        k = (r["level"], r["model"])
        n, p = c.get(k, (0, 0))
        c[k] = (n + 1, p + r["theory"])
    return c


def bar(x, y, h, color):
    """Column with a 4px rounded top, square at the baseline."""
    if h <= 0:  # a zero still shows which series it is: a 2px stub on the baseline
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
           '<title id="title">Drift-free submissions by hint level and model</title>',
           '<desc id="desc">' + "; ".join(
               f"{lvl} {m}: {c[(lvl, m)][1]} of {c[(lvl, m)][0]}" for lvl, _ in LEVELS for m in MODELS
               if (lvl, m) in c) + "</desc>",
           f'<rect width="{W}" height="{H}" fill="{t["surface"]}"/>',
           f'<text x="{LEFT - 40}" y="30" font-size="17" font-weight="600" fill="{t["text"]}">'
           'Drift-free submissions, by hint level</text>',
           f'<text x="{LEFT - 40}" y="52" font-size="13" fill="{t["secondary"]}">'
           'Share of runs whose loss passes the exact theory check · labels: passing / runs</text>']
    # legend
    x = LEFT - 40
    for m, color in zip(MODELS, t["series"]):
        out.append(f'<rect x="{x}" y="66" width="12" height="12" rx="3" fill="{color}"/>')
        out.append(f'<text x="{x + 18}" y="76.5" font-size="13" fill="{t["secondary"]}">{m} 5.5</text>')
        x += 104
    # grid and y axis
    for frac in (0, 0.5, 1):
        y = base - frac * plot_h
        out.append(f'<line x1="{LEFT}" x2="{W - RIGHT}" y1="{y}" y2="{y}" stroke="{t["grid"]}" stroke-width="1"/>')
        out.append(f'<text x="{LEFT - 8}" y="{y + 4}" font-size="12" text-anchor="end" fill="{t["muted"]}">'
                   f'{int(frac * 100)}%</text>')
    # bars
    for i, (lvl, name) in enumerate(LEVELS):
        x0 = LEFT + i * band + (band - group) / 2
        for j, (m, color) in enumerate(zip(MODELS, t["series"])):
            if (lvl, m) not in c:
                continue
            n, p = c[(lvl, m)]
            h = p / n * plot_h
            xb = x0 + j * (BAR + GAP)
            out.append(bar(xb, base - h, h, color))
            out.append(f'<text x="{xb + BAR / 2}" y="{base - h - 6}" font-size="11" text-anchor="middle" '
                       f'fill="{t["secondary"]}">{p}/{n}</text>')
        cx = LEFT + i * band + band / 2
        out.append(f'<text x="{cx}" y="{base + 22}" font-size="13" font-weight="600" text-anchor="middle" '
                   f'fill="{t["text"]}">{lvl}</text>')
        out.append(f'<text x="{cx}" y="{base + 40}" font-size="12" text-anchor="middle" fill="{t["muted"]}">'
                   f'{name}</text>')
    out.append(f'<text x="{W - RIGHT}" y="{H - 6}" font-size="11" text-anchor="end" fill="{t["muted"]}">'
               'fewer hints ← → more hints</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for theme in THEMES:
        (OUT / f"results-{theme}.svg").write_text(svg(theme))
        print(OUT / f"results-{theme}.svg")
