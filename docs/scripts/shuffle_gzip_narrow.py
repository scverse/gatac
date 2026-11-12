"""docs/_static/shuffle_gzip_narrow.svg: the phone layout of shuffle_gzip.svg (CSR panel on top, bytes top to bottom)."""

from pathlib import Path

STATIC = Path(__file__).resolve().parents[1] / "_static"
W = 360
CLS = ["sgn-b0", "sgn-b1", "sgn-b2", "sgn-b3"]
vals = [12, 47, 103, 260]
by = [[(v >> (8 * j)) & 0xFF for j in range(4)] for v in vals]
o = []
a = o.append

# ---- columns geometry
BH, PITCH, GGAP = 22, 25, 10
GH = 4 * BH + 3 * (PITCH - BH)  # height of one group of 4
Y0 = 258


def gy(g, s):
    return Y0 + g * (GH + GGAP) + s * PITCH


C1, C2, C3, BW, RW = 50, 162, 252, 40, 66
H = gy(3, 3) + BH + 56

a(
    f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="100%" style="max-width:{W}px" role="img" aria-labelledby="sgv-title">'
)
a('  <title id="sgv-title">HDF5 shuffle then gzip on the column indices of one cell of a CSR matrix</title>')
a("""  <style>
    .sgn-text  { fill: var(--pst-color-text-base, #1f2328); font-family: var(--pst-font-family-base, sans-serif); font-size: 13px; }
    .sgn-muted { fill: var(--pst-color-text-muted, #59636e); font-family: var(--pst-font-family-base, sans-serif); font-size: 11px; }
    .sgn-byte  { fill: var(--pst-color-text-base, #1f2328); font-family: var(--pst-font-family-monospace, monospace); font-size: 12px; font-weight: 600; text-anchor: middle; }
    .sgn-num   { fill: var(--pst-color-text-base, #1f2328); font-family: var(--pst-font-family-monospace, monospace); font-size: 11px; text-anchor: middle; }
    .sgn-zero  { fill: var(--pst-color-text-muted, #59636e); font-weight: 400; }
    .sgn-bold  { font-weight: 600; }
    .sgn-brace { stroke: var(--pst-color-border, #d0d7de); stroke-width: 1.2; fill: none; }
    .sgn-line  { stroke: var(--pst-color-text-muted, #59636e); stroke-width: 1.2; fill: none; }
    .sgn-head  { fill: var(--pst-color-text-muted, #59636e); }
    .sgn-cell  { fill: var(--pst-color-surface, #f6f8fa); stroke: var(--pst-color-border, #d0d7de); stroke-width: 1; }
    .sgn-hl    { fill: var(--pst-color-text-base, #1f2328); fill-opacity: 0.10; stroke: var(--pst-color-text-base, #1f2328); stroke-width: 1.2; }
    .sgn-focus { fill: none; stroke: var(--pst-color-text-base, #1f2328); stroke-width: 1.4; stroke-dasharray: 4 3; }
    .sgn-b0 { fill: #4c8bf5; fill-opacity: 0.14; stroke: #4c8bf5; stroke-width: 1.2; }
    .sgn-b1 { fill: #73b301; fill-opacity: 0.14; stroke: #73b301; stroke-width: 1.2; }
    .sgn-b2 { fill: #e0a020; fill-opacity: 0.16; stroke: #e0a020; stroke-width: 1.2; }
    .sgn-b3 { fill: #9b6bd3; fill-opacity: 0.14; stroke: #9b6bd3; stroke-width: 1.2; }
    .sgn-rep { stroke-dasharray: 4 3; }
    .sgn-w0 { stroke: #4c8bf5; } .sgn-w1 { stroke: #73b301; } .sgn-w2 { stroke: #e0a020; } .sgn-w3 { stroke: #9b6bd3; }
    .sgn-wire { fill: none; stroke-width: 1.2; stroke-opacity: 0.8; }
  </style>""")
a(
    '  <defs><marker id="sgv-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="sgn-head"/></marker></defs>'
)

# ---- CSR panel, arrays laid out horizontally (scipy order: data, indices, indptr)
a(
    '  <text x="8" y="18" class="sgn-text"><tspan class="sgn-bold">X, stored as CSR</tspan><tspan class="sgn-muted" dx="6">3 cells × tiles</tspan></text>'
)
cells = [([5, 80, 91], [1, 2, 1]), ([12, 47, 103, 260], [1, 1, 3, 2]), ([33, 1204], [1, 4])]
CX0, CW, CP, GG = 64, 29, 31, 4


def cx(k, grp):
    return CX0 + k * CP + grp * GG


k = 0
pos = []
for g, (ix, dv) in enumerate(cells):
    xs = [cx(k + i, g) for i in range(len(ix))]
    a(
        f'  <text x="{(xs[0] + xs[-1] + CW) / 2:g}" y="36" class="sgn-muted{" sgn-bold" if g == 1 else ""}" text-anchor="middle">cell {g}</text>'
    )
    for i, (c, d) in enumerate(zip(ix, dv)):
        pos.append((xs[i], g, c, d))
    k += len(ix)
RD, RI, RP = 42, 66, 96
a(f'  <text x="8" y="{RD + 14}" class="sgn-muted">data</text>')
a(f'  <text x="8" y="{RI + 14}" class="sgn-muted sgn-bold">indices</text>')
a(f'  <text x="8" y="{RP + 14}" class="sgn-muted">indptr</text>')
for x, g, c, d in pos:
    a(f'  <rect x="{x}" y="{RD}" width="{CW}" height="20" rx="3" class="sgn-cell"/>')
    a(f'  <text x="{x + CW / 2:g}" y="{RD + 14}" class="sgn-num">{d}</text>')
    a(f'  <rect x="{x}" y="{RI}" width="{CW}" height="20" rx="3" class="{"sgn-hl" if g == 1 else "sgn-cell"}"/>')
    a(f'  <text x="{x + CW / 2:g}" y="{RI + 14}" class="sgn-num{" sgn-bold" if g == 1 else ""}">{c}</text>')
f0 = [p for p in pos if p[1] == 1]
fx0, fx1 = f0[0][0] - 3, f0[-1][0] + CW + 3
a(f'  <rect x="{fx0}" y="{RI - 3}" width="{fx1 - fx0}" height="26" rx="5" class="sgn-focus"/>')
for n, v in enumerate([0, 3, 7, 9]):
    x = CX0 + n * CP
    hl = v in (3, 7)
    a(f'  <rect x="{x}" y="{RP}" width="{CW}" height="20" rx="3" class="{"sgn-hl" if hl else "sgn-cell"}"/>')
    a(f'  <text x="{x + CW / 2:g}" y="{RP + 14}" class="sgn-num{" sgn-bold" if hl else ""}">{v}</text>')
a(f'  <text x="{CX0}" y="{RP + 36}" class="sgn-muted">cell 1 = entries 3 to 6</text>')
a(f'  <path d="M8,{RP + 52} H{W - 8}" class="sgn-brace"/>')
# arrow from cell 1's indices down to the four values below
fcx = (fx0 + fx1) / 2
a(f'  <path d="M{fcx:g},{RI + 25} V{RP + 62}" class="sgn-line" marker-end="url(#sgv-arrow)"/>')

# ---- legend + column headers
a(f'  <text x="8" y="{RP + 76}" class="sgn-text sgn-bold">cell 1’s column indices, as int32</text>')
lx = 8
for j, name in enumerate(["byte 0 (low)", "byte 1", "byte 2", "byte 3 (high)"]):
    a(f'  <rect x="{lx}" y="{RP + 86}" width="11" height="11" rx="3" class="{CLS[j]}"/>')
    a(f'  <text x="{lx + 15}" y="{RP + 96}" class="sgn-muted">{name}</text>')
    lx += 15 + len(name) * 5.7 + 12
HY = Y0 - 18
for x, t, sub in ((C1 - 40, "As written", "low byte first"), (C2, "Shuffled", "by byte"), (C3, "gzip", "Deflate")):
    a(f'  <text x="{x}" y="{HY - 12}" class="sgn-text sgn-bold">{t}</text>')
    a(f'  <text x="{x}" y="{HY + 2}" class="sgn-muted">{sub}</text>')


def box(x, y, b, p, w=BW, label=None, rep=False):
    a(f'  <rect x="{x}" y="{y}" width="{w}" height="{BH}" rx="4" class="{CLS[p]}{" sgn-rep" if rep else ""}"/>')
    zero = label is None and b == 0
    a(
        f'  <text x="{x + w / 2:g}" y="{y + 15}" class="sgn-byte{" sgn-zero" if zero else ""}">{label or f"{b:02X}"}</text>'
    )


# column 1: as written, value labels on the left
for i, v in enumerate(vals):
    y0, y1 = gy(i, 0), gy(i, 3) + BH
    a(f'  <text x="{C1 - 10}" y="{(y0 + y1) / 2 + 4:g}" class="sgn-text sgn-bold" text-anchor="end">{v}</text>')
    a(f'  <path d="M{C1 - 4},{y0} h-3 V{y1} h3" class="sgn-brace"/>')
    for j in range(4):
        box(C1, gy(i, j), by[i][j], j)
# wires
for i in range(4):
    for j in range(4):
        sy, dy = gy(i, j) + BH / 2, gy(j, i) + BH / 2
        a(
            f'  <path d="M{C1 + BW + 2},{sy:g} C{C1 + BW + 40},{sy:g} {C2 - 40},{dy:g} {C2 - 2},{dy:g}" class="sgn-wire sgn-w{j}"/>'
        )
# column 2: shuffled
for j in range(4):
    for i in range(4):
        box(C2, gy(j, i), by[i][j], j)
# runs and gzip tokens
BX = C2 + BW + 6
runs = [(gy(0, 0), gy(0, 3) + BH), (gy(1, 0), gy(1, 2) + BH), (gy(1, 3), gy(1, 3) + BH), (gy(2, 0), gy(3, 3) + BH)]
for y0, y1 in runs:
    a(f'  <path d="M{BX},{y0} h4 V{y1} h-4" class="sgn-brace"/>')
toks = []
y = Y0
for i in range(4):
    toks.append(y + BH / 2)
    box(C3, y, by[i][0], 0)
    y += PITCH
y += 10
toks.append(y + BH / 2)
box(C3, y, 0, 1, RW, "00 × 3", True)
y += PITCH
toks.append(y + BH / 2)
box(C3, y, 1, 1)
y += PITCH
toks.append(y + BH / 2)
box(C3, y, 0, 2, RW, "00 × 8", True)
targets = [(toks[0] + toks[3]) / 2, toks[4], toks[5], toks[6]]
for (y0, y1), ty in zip(runs, targets):
    my = (y0 + y1) / 2
    a(
        f'  <path d="M{BX + 5},{my:g} C{BX + 26},{my:g} {C3 - 22},{ty:g} {C3 - 3},{ty:g}" class="sgn-line" marker-end="url(#sgv-arrow)"/>'
    )
yb = gy(3, 3) + BH + 24
a(f'  <text x="8" y="{yb}" class="sgn-text">16 bytes → 4 literal bytes + 3 codes</text>')
a(f'  <text x="8" y="{yb + 17}" class="sgn-muted">a real 4 MiB chunk has runs of thousands</text>')
a("</svg>")
(STATIC / "shuffle_gzip_narrow.svg").write_text("\n".join(o) + "\n")
