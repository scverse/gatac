"""docs/_static/parquet_size.svg: per-column size, fragments.tsv.gz vs GATAC Parquet (PBMC 5k),
in the style of runtime_clustering.svg. Chromosome (0.9 MB vs < 0.01 MB) is too small to draw."""

from pathlib import Path

STATIC = Path(__file__).resolve().parents[1] / "_static"

SEGS = [  # key, label, gzip MB, parquet MB
    ("start", "start", 145.9, 52.8),
    ("end", "end (as length)", 214.1, 114.5),
    ("bc", "barcode", 458.1, 170.0),
    ("count", "count", 37.1, 31.5),
    ("rows", "row interleaving", 186.0, None),  # 1,042 MB file - 856 MB of columns gzipped alone
]
TOTAL = {"gzip": 1042, "parquet": 369}
X0, BARW, H = 150, 470, 36
SCALE = BARW / TOTAL["gzip"]

o = []
a = o.append
a(
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 760 136" width="100%" style="max-width:760px" role="img" aria-labelledby="sz-title">'
)
a('  <title id="sz-title">Size on PBMC 5k: fragments.tsv.gz 1,042 MB, GATAC Parquet 369 MB</title>')
a("""  <style>
    .sz-text  { fill: var(--pst-color-text-base, #1f2328); font-family: var(--pst-font-family-base, sans-serif); font-size: 13px; }
    .sz-muted { fill: var(--pst-color-text-muted, #59636e); font-family: var(--pst-font-family-base, sans-serif); font-size: 12px; }
    .sz-bold  { font-weight: 600; }
    .sz-seg   { font-family: var(--pst-font-family-base, sans-serif); font-size: 11.5px; }
    .sz-start { fill: #2a78d6; } .sz-i-start { fill: #fff; }
    .sz-end   { fill: #eb6834; } .sz-i-end   { fill: #fff; }
    .sz-bc    { fill: #eda100; } .sz-i-bc    { fill: #0b0b0b; }
    .sz-count { fill: #e87ba4; } .sz-i-count { fill: #0b0b0b; }
    .sz-rows  { fill: #afb8c1; } .sz-i-rows  { fill: #0b0b0b; }
    html[data-theme="dark"] .sz-start { fill: #3987e5; }
    html[data-theme="dark"] .sz-end   { fill: #d95926; }
    html[data-theme="dark"] .sz-bc    { fill: #c98500; }
    html[data-theme="dark"] .sz-count { fill: #e8679a; }
    html[data-theme="dark"] .sz-rows  { fill: #6e7781; } html[data-theme="dark"] .sz-i-rows { fill: #fff; }
  </style>""")

# legend
x = X0
for key, label, _, _ in SEGS:
    a(f'  <rect x="{x:.1f}" y="8" width="12" height="12" rx="3" class="sz-{key}"/>')
    a(f'  <text x="{x + 18:.1f}" y="18" class="sz-muted">{label}</text>')
    x += 18 + len(label) * 6.4 + 16


def bar(y, name, which, total_label):
    a(f'  <text x="{X0 - 10}" y="{y + 23}" text-anchor="end" class="sz-text sz-bold">{name}</text>')
    x = X0
    for key, label, g, p in SEGS:
        v = g if which == "gzip" else p
        if v is None:
            continue
        w = v * SCALE
        if p is None:
            tip = f"{label}: {g:g} MB lost by gzip to the rows interleaving the columns"
        else:
            tip = f"{label}: fragments.tsv.gz {g:g} MB, Parquet {p:g} MB (×{g / p:.1f} smaller)"
        a(f"  <g><title>{tip}</title>")
        a(f'    <rect x="{x:.1f}" y="{y}" width="{max(w - 2, 1):.1f}" height="{H}" rx="3" class="sz-{key}"/>')
        txt = f"{v:.0f} MB"
        if w - 2 >= len(txt) * 6.6 + 10:
            a(
                f'    <text x="{x + (w - 2) / 2:.1f}" y="{y + 22}" text-anchor="middle" class="sz-seg sz-i-{key}">{txt}</text>'
            )
        a("  </g>")
        x += w
    a(f'  <text x="{x + 8:.1f}" y="{y + 23}" class="sz-text sz-bold">{total_label}</text>')


bar(42, "fragments.tsv.gz", "gzip", "1,042 MB")
bar(92, "GATAC Parquet", "parquet", "369 MB  ·  ×2.8 smaller")
a("</svg>")
(STATIC / "parquet_size.svg").write_text("\n".join(o) + "\n")
