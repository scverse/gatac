"""Draw the pipeline diagram of docs/overview.md, wide and narrow.

The wide one (desktop) is docs/_static/pipeline_overview.svg, the narrow one (phones) pipeline_overview_narrow.svg.
"""

from pathlib import Path

STATIC = Path(__file__).resolve().parents[1] / "_static"
G, B = "#73b301", "#4c8bf5"  # fills (tints), and lines in dark mode
GS, BS, KS = "#5a8c00", "#3b7ddd", "#848d97"  # light-mode lines: >= 3:1 on the page and the band
KSD = "#6e7781"  # dark-mode grey outlines (theme border is 2.5:1)


def style(p, ts, ms):
    base = "font-family: var(--pst-font-family-base, sans-serif)"
    mono = "font-family: var(--pst-font-family-monospace, monospace)"
    return f"""  <style>
    .{p}-t     {{ fill: var(--pst-color-text-base, #1f2328); {base}; font-size: {ts}px; font-weight: 600; }}
    .{p}-m     {{ fill: var(--pst-color-text-muted, #59636e); {base}; font-size: {ms}px; }}
    .{p}-c     {{ fill: var(--pst-color-text-muted, #59636e); {mono}; font-size: {ms}px; }}
    .{p}-mid   {{ text-anchor: middle; }}
    .{p}-end   {{ text-anchor: end; }}
    .{p}-box   {{ fill: var(--pst-color-surface, #f6f8fa); stroke: {KS}; stroke-width: 1; }}
    .{p}-sheet {{ fill: var(--pst-color-background, #ffffff); stroke: {KS}; stroke-width: 1; }}
    .{p}-flt   {{ fill: {G}; fill-opacity: 0.14; stroke: {GS}; stroke-width: 1.2; }}
    .{p}-fltbg {{ fill: var(--pst-color-background, #ffffff); stroke: {GS}; stroke-width: 1.2; }}
    .{p}-ad    {{ fill: {B}; fill-opacity: 0.14; stroke: {BS}; stroke-width: 1.2; }}
    .{p}-chip  {{ fill: var(--pst-color-background, #ffffff); stroke: {BS}; stroke-width: 1; }}
    .{p}-line  {{ stroke: var(--pst-color-text-muted, #59636e); stroke-width: 1.4; fill: none; }}
    .{p}-head  {{ fill: var(--pst-color-text-muted, #59636e); }}
    .{p}-read  {{ stroke: {GS}; stroke-width: 1.8; fill: none; }}
    .{p}-rhead {{ fill: {GS}; }}
    .{p}-num   {{ fill: var(--pst-color-text-base, #1f2328); }}
    .{p}-band0 {{ fill: var(--pst-color-text-muted, #59636e); fill-opacity: 0.05; stroke: var(--pst-color-border, #d0d7de); stroke-width: 1; }}
    .{p}-band1 {{ fill: {B}; fill-opacity: 0.06; stroke: {B}; stroke-opacity: 0.45; stroke-width: 1; }}
    .{p}-tab0  {{ fill: var(--pst-color-text-muted, #59636e); fill-opacity: 0.16; }}
    .{p}-tab1  {{ fill: {B}; fill-opacity: 0.22; }}
    .{p}-sech  {{ fill: var(--pst-color-text-base, #1f2328); font-family: var(--pst-font-family-base, sans-serif); font-size: {ms - 0.5}px; font-weight: 700; letter-spacing: 0.08em; }}
    .{p}-sec   {{ fill: var(--pst-color-text-muted, #59636e); {base}; font-size: {ms - 0.5}px; font-weight: 700; letter-spacing: 0.06em; }}
    html[data-theme="dark"] .{p}-flt, html[data-theme="dark"] .{p}-fltbg, html[data-theme="dark"] .{p}-read {{ stroke: {G}; }}
    html[data-theme="dark"] .{p}-rhead {{ fill: {G}; }}
    html[data-theme="dark"] .{p}-ad, html[data-theme="dark"] .{p}-chip {{ stroke: {B}; }}
    html[data-theme="dark"] .{p}-box, html[data-theme="dark"] .{p}-sheet {{ stroke: {KSD}; }}
    .{p}-numt  {{ fill: var(--pst-color-background, #ffffff); {base}; font-size: 11px; font-weight: 700; text-anchor: middle; }}
  </style>
  <defs>
    <marker id="{p}-a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="{p}-head"/></marker>
    <marker id="{p}-g" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="{p}-rhead"/></marker>
  </defs>"""


class D:
    def __init__(s, p):
        s.p, s.o = p, []

    def raw(s, x):
        s.o.append("  " + x)

    def note(s, x):
        s.o.append(f"\n  <!-- {x} -->")

    def rect(s, x, y, w, h, cls, rx=6):
        s.raw(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" class="{s.p}-{cls}"/>')

    def text(s, x, y, t, cls, anchor="mid"):
        a = f" {s.p}-{anchor}" if anchor else ""
        s.raw(f'<text x="{x}" y="{y}" class="{s.p}-{cls}{a}">{t}</text>')

    def arrow(s, d, green=False):
        s.raw(
            f'<path d="{d}" class="{s.p}-{"read" if green else "line"}" marker-end="url(#{s.p}-{"g" if green else "a"})"/>'
        )

    def stack(s, x, y, w, h, o, green=False):
        back = "fltbg" if green else "sheet"
        s.rect(x + 2 * o, y - 2 * o, w, h, back)
        s.rect(x + o, y - o, w, h, back)
        if green:
            s.rect(x, y, w, h, "fltbg")
            s.rect(x, y, w, h, "flt")
        else:
            s.rect(x, y, w, h, "box")

    def num(s, x, y, n):
        s.raw(f'<circle cx="{x}" cy="{y}" r="9" class="{s.p}-num"/>')
        s.raw(f'<text x="{x}" y="{y + 4}" class="{s.p}-numt">{n}</text>')

    def out(s, w, h, title, ts, ms):
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="100%" style="max-width:{w}px" role="img" aria-labelledby="{s.p}-title">\n'
            f'  <title id="{s.p}-title">{title}</title>\n' + style(s.p, ts, ms) + "\n" + "\n".join(s.o) + "\n</svg>\n"
        )


TITLE = (
    "GATAC pipeline: fragment files to Parquet, filtered Parquet, one tile matrix of the top tiles, "
    "clusters, peaks and a peak matrix; the tile, gene, peak-calling and peak-matrix steps read the filtered Parquet files"
)

# ───────────────────────── wide ─────────────────────────
d = D("po")
d.raw('<g transform="translate(26,0)">')
d.note("legend")
d.stack(10, 12, 18, 13, 3)
d.text(40, 22, "one file per sample", "m", None)
d.rect(172, 9, 18, 14, "ad", 3)
d.text(198, 22, "one AnnData, all samples", "m", None)
d.arrow("M10,42 H36", True)
d.text(44, 46, "reads the filtered Parquet files", "m", None)
d.note("band: combine all samples")
d.rect(-22, 158, 812, 148, "band1", 8)
d.rect(-22, 158, 18, 148, "tab1", 8)
d.raw('<text transform="translate(-9,232) rotate(-90)" class="po-sec po-mid">COMBINE ALL SAMPLES</text>')


d.note("fragments.tsv.gz, one per sample")
d.stack(-12, 76, 168, 66, 5)
d.text(72, 99, "fragments.tsv.gz", "t")
d.text(72, 117, "one per sample,", "m")
d.text(72, 132, "names as text in every row", "m")
d.arrow("M156,109 H288")
d.text(227, 101, "gatac convert", "c")

d.note("1. Parquet, one per sample")
d.stack(290, 76, 170, 66, 5)
d.text(375, 99, "Parquet fragments", "t")
d.text(375, 117, "integer ids,", "m")
d.text(375, 132, "barcode table in the file", "m")
d.num(290, 76, 1)

d.note("2. QC tables, then the filter")
d.arrow("M440,66 V26 H455")
d.rect(457, 4, 150, 44, "box")
d.text(532, 22, "QC per cell", "t")
d.text(532, 39, "metrics · doublets", "c")
d.arrow("M532,48 V107")
d.text(539, 62, "query on the metrics", "m", None)
d.arrow("M460,109 H598")
d.text(532, 126, "gatac filter", "c")

d.note("2. Filtered Parquet, one per sample")
d.stack(600, 76, 172, 66, 5, True)
d.text(686, 99, "Filtered Parquet", "t")
d.text(686, 117, "the cells that pass QC,", "m")
d.text(686, 132, "one file per sample", "m")
d.num(600, 76, 2)

d.note("reads of the filtered files")
d.arrow("M732,142 V197", True)
d.arrow("M732,168 H39 V205", True)
d.arrow("M163,168 V205", True)
d.arrow("M459,168 V197", True)
d.arrow("M604,168 V197", True)
d.text(46, 186, "pass 1", "m", None)
d.text(156, 186, "pass 2", "m", "end")

d.note("3. One tile matrix, top tiles")
d.rect(10, 198, 182, 98, "ad")
for x, t in ((16, "scan"), (78, "select"), (140, "build")):
    d.rect(x, 206, 46, 24, "chip", 4)
    d.text(x + 23, 222, t, "m")
d.arrow("M62,218 H77")
d.arrow("M124,218 H139")
d.text(101, 251, "Tile matrix, top n tiles", "t")
d.text(101, 268, "gatac tile -n", "c")
d.text(101, 284, "cells × selected tiles", "m")
d.num(10, 198, 3)
d.arrow("M192,247 H208")

d.note("4. Embedding and clusters")
d.rect(210, 198, 166, 98, "box")
d.text(293, 232, "Embedding and clusters", "t")
d.text(293, 251, "dr.spectral", "c")
d.text(293, 266, "→ kNN → Leiden", "c")
d.text(293, 283, "clusters in obs", "m")
d.num(210, 198, 4)
d.arrow("M376,247 H392")

d.note("5. Peaks per cluster")
d.rect(394, 198, 130, 98, "box")
d.text(459, 232, "Peaks per cluster", "t")
d.text(459, 251, "peak.call (MACS3)", "c")
d.text(459, 266, "→ peak.merge", "c")
d.text(459, 283, "one merged peak set", "m")
d.num(394, 198, 5)
d.arrow("M524,247 H540")

d.note("6. Peak matrix")
d.rect(542, 198, 124, 98, "ad")
d.text(604, 236, "Peak matrix", "t")
d.text(604, 255, "peak.make_matrix", "c")
d.text(604, 272, "cells × peaks", "m")
d.num(542, 198, 6)
d.arrow("M604,296 V318")
d.text(604, 334, "markers, motif enrichment, chromVAR", "m")

d.note("Gene matrix")
d.rect(680, 198, 104, 98, "ad")
d.text(732, 236, "Gene matrix", "t")
d.text(732, 255, "gatac gene", "c")
d.text(732, 272, "cells × genes", "m")
d.raw("</g>")
(STATIC / "pipeline_overview.svg").write_text(d.out(816, 342, TITLE, 13, 11))

# ───────────────────────── narrow ─────────────────────────
d = D("pon")
X, W, C = 34, 316, 192
d.raw('<g transform="translate(0,8)">')
d.rect(2, 320, 356, 562, "band1", 8)
d.raw('<path d="M2,342 V328 Q2,320 10,320 H350 Q358,320 358,328 V342 Z" class="pon-tab1"/>')
d.text(180, 335, "COMBINE ALL SAMPLES", "sech")
d.note("fragments.tsv.gz, one per sample")
d.stack(X, 12, W, 50, 4)
d.text(C, 33, "fragments.tsv.gz", "t")
d.text(C, 51, "one per sample, names as text in every row", "m")
d.arrow(f"M{C},62 V88")
d.text(C + 10, 80, "gatac convert", "c", None)

d.note("1. Parquet, one per sample")
d.stack(X, 98, W, 50, 4)
d.text(C, 119, "Parquet fragments", "t")
d.text(C, 137, "integer ids, barcode table in the file", "m")
d.num(X, 98, 1)

d.note("2. QC tables, then the filter")
d.arrow("M263,148 V164")
d.rect(176, 166, 174, 48, "box")
d.text(263, 186, "QC per cell", "t")
d.text(263, 204, "metrics · doublets", "c")
d.arrow("M176,190 H106")
d.text(263, 232, "query on the metrics", "m")
d.arrow("M104,148 V240")
d.text(94, 176, "gatac filter", "c", "end")

d.note("2. Filtered Parquet, one per sample")
d.stack(X, 250, W, 50, 4, True)
d.text(C, 271, "Filtered Parquet fragments", "t")
d.text(C, 289, "the cells that pass QC, one file per sample", "m")
d.num(X, 250, 2)

d.note("reads of the filtered files: a bus down the left edge")
d.arrow("M34,275 H12 V849 H32", True)
for y in (399, 479):
    d.arrow(f"M12,{y} H44", True)
for y in (657, 739):
    d.arrow(f"M12,{y} H32", True)

d.raw('<g transform="translate(0,30)">')
d.note("3. One tile matrix, top tiles")
d.rect(X, 324, W, 170, "ad")
d.text(C, 346, "One tile matrix, top n tiles", "t")
for y, t in (
    (356, "pass 1 · scan: tile statistics"),
    (396, "select the top n tiles"),
    (436, "pass 2 · build: those tiles only"),
):
    d.rect(46, y, 292, 26, "chip", 4)
    d.text(C, y + 17, t, "m")
d.arrow(f"M{C},382 V395")
d.arrow(f"M{C},422 V435")
d.text(C, 482, "gatac tile -n · cells × selected tiles", "c")
d.num(X, 324, 3)
d.arrow(f"M{C},494 V518")

d.note("4. Embedding and clusters")
d.rect(X, 520, W, 50, "box")
d.text(C, 541, "Embedding and clusters", "t")
d.text(C, 559, "dr.spectral → kNN → Leiden", "c")
d.num(X, 520, 4)
d.arrow(f"M{C},570 V600")
d.text(C + 10, 590, "clusters in obs", "m", None)

d.note("5. Peaks per cluster")
d.rect(X, 602, W, 50, "box")
d.text(C, 623, "Peaks per cluster, merged", "t")
d.text(C, 641, "peak.call (MACS3) → peak.merge", "c")
d.num(X, 602, 5)
d.arrow(f"M{C},652 V682")
d.text(C + 10, 672, "peak set", "m", None)

d.note("6. Peak matrix")
d.rect(X, 684, W, 50, "ad")
d.text(C, 705, "Peak matrix", "t")
d.text(C, 723, "peak.make_matrix · cells × peaks", "c")
d.num(X, 684, 6)
d.arrow(f"M{C},734 V754")
d.text(C, 771, "markers, motif enrichment, chromVAR", "m")

d.note("Gene matrix")
d.rect(X, 794, W, 50, "ad")
d.text(C, 815, "Gene matrix", "t")
d.text(C, 833, "gatac gene · cells × genes", "c")

d.raw("</g>")
d.note("legend")
d.arrow("M12,898 H38", True)
d.text(46, 902, "reads the filtered Parquet files", "m", None)
d.stack(12, 916, 18, 13, 3)
d.text(46, 926, "one file per sample", "m", None)
d.rect(186, 914, 18, 14, "ad", 3)
d.text(212, 926, "one AnnData, all samples", "m", None)
d.raw("</g>")
(STATIC / "pipeline_overview_narrow.svg").write_text(d.out(360, 944, TITLE, 14, 12))
