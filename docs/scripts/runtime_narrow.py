"""docs/_static/runtime_{clustering,peaks}_narrow.svg: the phone layout of the run-time charts of docs/overview.md.

The same stacked bars as two vertical columns, read from the wide, hand-made runtime_*.svg (legend, segment
widths, hover titles, totals), so the two always agree: rerun this after editing a wide chart.
"""

import re
from pathlib import Path

STATIC = Path(__file__).resolve().parents[1] / "_static"


def narrow(src: Path, dst: Path) -> None:
    s = src.read_text()
    style = re.search(r"<style>.*?</style>", s, re.S).group(0).replace("rt-", "rtn-")
    title_id = re.search(r'aria-labelledby="([^"]+)"', s).group(1)
    title = re.search(r"<title id=\"[^\"]+\">([^<]*)</title>", s).group(1)

    legend = re.findall(
        r'<rect x="[\d.]+" y="\d+" width="12" height="12" rx="3" class="rt-(s\d)"/>\s*<text [^>]*>([^<]+)</text>', s
    )
    names = re.findall(r'<text x="90" y="[\d.]+" text-anchor="end" class="rt-text rt-bold">([^<]+)</text>', s)
    totals = re.findall(r'<text x="[\d.]+" y="[\d.]+" class="rt-text rt-bold">([^<]+)</text>', s)
    blocks = s.split('text-anchor="end" class="rt-text rt-bold">')[1:]
    rows = []
    for name, blk in zip(names, blocks):
        segs = []
        for tip, w, cls in re.findall(
            r'<g><title>([^<]+)</title>\s*<rect x="[\d.]+" y="\d+" width="([\d.]+)" height="36" rx="3" class="rt-(s\d)"/>',
            blk,
        ):
            stage = tip.split(":")[0]
            t = re.match(rf"[^:]+: {re.escape(name)} ([^,]+),", tip).group(1)
            segs.append((cls, float(w) + 2, t, tip, stage))
        rows.append((name, segs))
    assert len(rows) == 2 and len(totals) == 2, (names, totals)

    W, CW, MAXH = 360, 104, 300
    XS = [60, 196]
    o = []
    a = o.append
    # legend rows
    leg, x, y = [], 8, 8
    for cls, label in legend:
        wl = 18 + len(label) * 6.2 + 16
        if x + wl - 16 > W - 8:
            x, y = 8, y + 20
        leg.append((x, y, cls, label))
        x += wl
    TOP = y + 12 + 52
    BASE = TOP + MAXH
    H = BASE + 32
    scale = MAXH / max(sum(sg[1] for sg in segs) for _, segs in rows)

    a(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="100%" style="max-width:{W}px" role="img" aria-labelledby="{title_id}-n">'
    )
    a(f'  <title id="{title_id}-n">{title}</title>')
    a("  " + style)
    for x, y, cls, label in leg:
        a(f'  <rect x="{x:.1f}" y="{y}" width="12" height="12" rx="3" class="rtn-{cls}"/>')
        a(f'  <text x="{x + 18:.1f}" y="{y + 10}" class="rtn-muted">{label}</text>')
    for (name, segs), x0, total in zip(rows, XS, totals):
        y = BASE
        for cls, w, t, tip, stage in segs:
            h = w * scale
            i = cls.replace("s", "i")
            a(f"  <g><title>{tip}</title>")
            a(
                f'    <rect x="{x0}" y="{y - h + 1:.1f}" width="{CW}" height="{max(h - 2, 1):.1f}" rx="3" class="rtn-{cls}"/>'
            )
            if h - 2 >= 17:
                label = f"{stage} {t}" if h - 2 >= 34 and len(stage) + len(t) <= 17 else t
                a(
                    f'    <text x="{x0 + CW / 2}" y="{y - h / 2 + 4:.1f}" text-anchor="middle" class="rtn-seg rtn-{i}">{label}</text>'
                )
            a("  </g>")
            y -= h
        parts = [p.strip() for p in total.split("·")]
        for k, t in enumerate(parts):
            a(
                f'  <text x="{x0 + CW / 2}" y="{y - 10 - 16 * (len(parts) - 1 - k):.1f}" text-anchor="middle" class="rtn-text rtn-bold">{t}</text>'
            )
        a(f'  <text x="{x0 + CW / 2}" y="{BASE + 22}" text-anchor="middle" class="rtn-text rtn-bold">{name}</text>')
    a("</svg>")
    dst.write_text("\n".join(o) + "\n")


for name in ("runtime_clustering", "runtime_peaks"):
    narrow(STATIC / f"{name}.svg", STATIC / f"{name}_narrow.svg")
