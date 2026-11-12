"""docs/_static/parquet_size_narrow.svg: the size chart as two vertical stacked columns, for phones."""

from pathlib import Path

STATIC = Path(__file__).resolve().parents[1] / "_static"

SEGS = [  # key, label, gzip MB, parquet MB (bottom to top)
    ("start", "start", 145.9, 52.8),
    ("end", "end (as length)", 214.1, 114.5),
    ("bc", "barcode", 458.1, 170.0),
    ("count", "count", 37.1, 31.5),
    ("rows", "row interleaving", 186.0, None),
]
W, BASE, MAXH, CW = 360, 400, 300, 104
SCALE = MAXH / 1042
COLS = [("fragments.tsv.gz", "gzip", 60, ["1,042 MB"]), ("GATAC Parquet", "parquet", 196, ["369 MB", "×2.8 smaller"])]
H = BASE + 32

o = []
a = o.append
a(
    f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="100%" style="max-width:{W}px" role="img" aria-labelledby="szn-title">'
)
a('  <title id="szn-title">Size on PBMC 5k: fragments.tsv.gz 1,042 MB, GATAC Parquet 369 MB</title>')
a("""  <style>
    .szn-text  { fill: var(--pst-color-text-base, #1f2328); font-family: var(--pst-font-family-base, sans-serif); font-size: 13px; }
    .szn-muted { fill: var(--pst-color-text-muted, #59636e); font-family: var(--pst-font-family-base, sans-serif); font-size: 12px; }
    .szn-bold  { font-weight: 600; }
    .szn-seg   { font-family: var(--pst-font-family-base, sans-serif); font-size: 11.5px; }
    .szn-start { fill: #2a78d6; } .szn-i-start { fill: #fff; }
    .szn-end   { fill: #eb6834; } .szn-i-end   { fill: #fff; }
    .szn-bc    { fill: #eda100; } .szn-i-bc    { fill: #0b0b0b; }
    .szn-count { fill: #e87ba4; } .szn-i-count { fill: #0b0b0b; }
    .szn-rows  { fill: #afb8c1; } .szn-i-rows  { fill: #0b0b0b; }
    html[data-theme="dark"] .szn-start { fill: #3987e5; }
    html[data-theme="dark"] .szn-end   { fill: #d95926; }
    html[data-theme="dark"] .szn-bc    { fill: #c98500; }
    html[data-theme="dark"] .szn-count { fill: #e8679a; }
    html[data-theme="dark"] .szn-rows  { fill: #6e7781; } html[data-theme="dark"] .szn-i-rows { fill: #fff; }
  </style>""")

# legend, two rows
rows = [SEGS[:3], SEGS[3:]]
for r, segs in enumerate(rows):
    x = 8
    for key, label, _, _ in segs:
        y = 8 + r * 20
        a(f'  <rect x="{x:.1f}" y="{y}" width="12" height="12" rx="3" class="szn-{key}"/>')
        a(f'  <text x="{x + 18:.1f}" y="{y + 10}" class="szn-muted">{label}</text>')
        x += 18 + len(label) * 6.2 + 16

for name, which, x0, total in COLS:
    y = BASE
    for key, label, g, p in SEGS:
        v = g if which == "gzip" else p
        if v is None:
            continue
        h = v * SCALE
        if p is None:
            tip = f"{label}: {g:g} MB lost by gzip to the rows interleaving the columns"
        else:
            tip = f"{label}: fragments.tsv.gz {g:g} MB, Parquet {p:g} MB (×{g / p:.1f} smaller)"
        a(f"  <g><title>{tip}</title>")
        a(
            f'    <rect x="{x0}" y="{y - h + 1:.1f}" width="{CW}" height="{max(h - 2, 1):.1f}" rx="3" class="szn-{key}"/>'
        )
        if h - 2 >= 17:
            a(
                f'    <text x="{x0 + CW / 2}" y="{y - h / 2 + 4:.1f}" text-anchor="middle" class="szn-seg szn-i-{key}">{v:.0f} MB</text>'
            )
        a("  </g>")
        y -= h
    for k, t in enumerate(total):
        a(
            f'  <text x="{x0 + CW / 2}" y="{y - 10 - 16 * (len(total) - 1 - k):.1f}" text-anchor="middle" class="szn-text szn-bold">{t}</text>'
        )
    a(f'  <text x="{x0 + CW / 2}" y="{BASE + 22}" text-anchor="middle" class="szn-text szn-bold">{name}</text>')
a("</svg>")
(STATIC / "parquet_size_narrow.svg").write_text("\n".join(o) + "\n")
