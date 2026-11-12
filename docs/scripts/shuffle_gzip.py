"""docs/_static/shuffle_gzip.svg: HDF5 shuffle + gzip on one cell's column indices (docs/api/io.md)."""

from pathlib import Path

STATIC = Path(__file__).resolve().parents[1] / "_static"
X0, BW, BH, GAP, GG = 330, 35, 32, 3, 12
GW = 4 * BW + 3 * GAP
CLS = ["sg-b0", "sg-b1", "sg-b2", "sg-b3"]
vals = [12, 47, 103, 260]
by = [[(v >> (8 * j)) & 0xFF for j in range(4)] for v in vals]
W = X0 + 4 * GW + 3 * GG + 14


def bx(g, s):
    return X0 + g * (GW + GG) + s * (BW + GAP)


Y1, Y2, Y3 = 70, 196, 300
H = Y3 + BH + 14
o = []
a = o.append
a(
    f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="100%" style="max-width:{W}px" role="img" aria-labelledby="sg-title">'
)
a('  <title id="sg-title">HDF5 shuffle then gzip on the column indices of one cell of a CSR matrix</title>')
a("""  <style>
    .sg-text  { fill: var(--pst-color-text-base, #1f2328); font-family: var(--pst-font-family-base, sans-serif); font-size: 14px; }
    .sg-muted { fill: var(--pst-color-text-muted, #59636e); font-family: var(--pst-font-family-base, sans-serif); font-size: 12px; }
    .sg-byte  { fill: var(--pst-color-text-base, #1f2328); font-family: var(--pst-font-family-monospace, monospace); font-size: 14px; font-weight: 600; text-anchor: middle; }
    .sg-num   { fill: var(--pst-color-text-base, #1f2328); font-family: var(--pst-font-family-monospace, monospace); font-size: 12px; text-anchor: middle; }
    .sg-zero  { fill: var(--pst-color-text-muted, #59636e); font-weight: 400; }
    .sg-bold  { font-weight: 600; }
    .sg-brace { stroke: var(--pst-color-border, #d0d7de); stroke-width: 1.2; fill: none; }
    .sg-line  { stroke: var(--pst-color-text-muted, #59636e); stroke-width: 1.3; fill: none; }
    .sg-head  { fill: var(--pst-color-text-muted, #59636e); }
    .sg-cell  { fill: var(--pst-color-surface, #f6f8fa); stroke: var(--pst-color-border, #d0d7de); stroke-width: 1; }
    .sg-hl    { fill: var(--pst-color-text-base, #1f2328); fill-opacity: 0.10; stroke: var(--pst-color-text-base, #1f2328); stroke-width: 1.2; }
    .sg-focus { fill: none; stroke: var(--pst-color-text-base, #1f2328); stroke-width: 1.4; stroke-dasharray: 4 3; }
    .sg-b0 { fill: #4c8bf5; fill-opacity: 0.14; stroke: #4c8bf5; stroke-width: 1.2; }
    .sg-b1 { fill: #73b301; fill-opacity: 0.14; stroke: #73b301; stroke-width: 1.2; }
    .sg-b2 { fill: #e0a020; fill-opacity: 0.16; stroke: #e0a020; stroke-width: 1.2; }
    .sg-b3 { fill: #9b6bd3; fill-opacity: 0.14; stroke: #9b6bd3; stroke-width: 1.2; }
    .sg-rep { stroke-dasharray: 4 3; }
    .sg-w0 { stroke: #4c8bf5; } .sg-w1 { stroke: #73b301; } .sg-w2 { stroke: #e0a020; } .sg-w3 { stroke: #9b6bd3; }
    .sg-wire { fill: none; stroke-width: 1.3; stroke-opacity: 0.8; }
  </style>""")
a(
    '  <defs><marker id="sg-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="sg-head"/></marker></defs>'
)

# ---------------------------------------------------------------- CSR panel
a("  <!-- CSR panel: a 3-cell matrix, cell 1 highlighted -->")
cells = [([5, 80, 91], [1, 2, 1]), ([12, 47, 103, 260], [1, 1, 3, 2]), ([33, 1204], [1, 4])]
a('  <text x="16" y="20" class="sg-text sg-bold">X, stored as CSR</text>')
a('  <text x="16" y="37" class="sg-muted">3 cells × tiles</text>')
CX_I, CW_I, CX_D, CW_D = 104, 56, 166, 40  # indices / data columns
RY0, RP, RH = 72, 20, 18
a(f'  <text x="{CX_I + CW_I / 2:g}" y="{RY0 - 8}" class="sg-muted" text-anchor="middle">indices</text>')
a(f'  <text x="{CX_D + CW_D / 2:g}" y="{RY0 - 8}" class="sg-muted" text-anchor="middle">data</text>')
k = 0
for c, (ix, dv) in enumerate(cells):
    y_top = RY0 + k * RP
    y_bot = RY0 + (k + len(ix) - 1) * RP + RH
    a(f'  <path d="M70,{y_top + 1} h-5 V{y_bot - 1} h5" class="sg-brace"/>')
    a(
        f'  <text x="58" y="{(y_top + y_bot) / 2 + 4:g}" class="sg-muted{" sg-bold" if c == 1 else ""}" text-anchor="end">cell {c}</text>'
    )
    for i, d in zip(ix, dv):
        y = RY0 + k * RP
        a(f'  <text x="94" y="{y + 13}" class="sg-muted" text-anchor="end">{k}</text>')
        a(
            f'  <rect x="{CX_I}" y="{y}" width="{CW_I}" height="{RH}" rx="3" class="{"sg-hl" if c == 1 else "sg-cell"}"/>'
        )
        a(f'  <text x="{CX_I + CW_I / 2:g}" y="{y + 13}" class="sg-num{" sg-bold" if c == 1 else ""}">{i}</text>')
        a(f'  <rect x="{CX_D}" y="{y}" width="{CW_D}" height="{RH}" rx="3" class="sg-cell"/>')
        a(f'  <text x="{CX_D + CW_D / 2:g}" y="{y + 13}" class="sg-num">{d}</text>')
        k += 1
fy0, fy1 = RY0 + 3 * RP - 3, RY0 + 6 * RP + RH + 3
a(f'  <rect x="{CX_I - 3}" y="{fy0}" width="{CW_I + 6}" height="{fy1 - fy0}" rx="5" class="sg-focus"/>')
# indptr
IY = RY0 + 9 * RP + 22
a(f'  <text x="16" y="{IY + 13}" class="sg-muted">indptr</text>')
for n, v in enumerate([0, 3, 7, 9]):
    x = 70 + n * 34
    a(f'  <rect x="{x}" y="{IY}" width="30" height="{RH}" rx="3" class="{"sg-hl" if v in (3, 7) else "sg-cell"}"/>')
    a(f'  <text x="{x + 15}" y="{IY + 13}" class="sg-num{" sg-bold" if v in (3, 7) else ""}">{v}</text>')
a(f'  <text x="16" y="{IY + 40}" class="sg-muted">cell 1 = entries 3 to 6:</text>')
a(f'  <text x="16" y="{IY + 56}" class="sg-muted">the tiles where it has reads</text>')
a(f'  <path d="M220,10 V{H - 10}" class="sg-brace"/>')
# arrow from the focus box to the four values
a(
    f'  <path d="M{CX_I + CW_I + 4},{fy0 + 8} C200,{fy0 - 30} 230,34 {X0 - 8},34" class="sg-line" marker-end="url(#sg-arrow)"/>'
)


# ---------------------------------------------------------------- shuffle + gzip
def box(x, y, b, p, w=BW, label=None, rep=False):
    a(f'  <rect x="{x}" y="{y}" width="{w}" height="{BH}" rx="5" class="{CLS[p]}{" sg-rep" if rep else ""}"/>')
    zero = label is None and b == 0
    a(
        f'  <text x="{x + w / 2:g}" y="{y + 21}" class="sg-byte{" sg-zero" if zero else ""}">{label or f"{b:02X}"}</text>'
    )


def side(y, head, sub):
    a(f'  <text x="236" y="{y + 14}" class="sg-text sg-bold">{head}</text>')
    a(f'  <text x="236" y="{y + 30}" class="sg-muted">{sub}</text>')


a("  <!-- legend -->")
for j, name in enumerate(["byte 0 (low)", "byte 1", "byte 2", "byte 3 (high)"]):
    x = X0 + 210 + j * 100
    a(f'  <rect x="{x}" y="6" width="12" height="12" rx="3" class="{CLS[j]}"/>')
    a(f'  <text x="{x + 18}" y="16" class="sg-muted">{name}</text>')
a(f'  <text x="{X0}" y="38" class="sg-text">cell 1’s column indices, as int32</text>')

a("  <!-- 1: as written -->")
side(Y1, "As written", "low byte first")
for i, v in enumerate(vals):
    g0, g1 = bx(i, 0), bx(i, 3) + BW
    a(f'  <path d="M{g0},{Y1 - 6} v-4 H{g1} v4" class="sg-brace"/>')
    a(f'  <text x="{(g0 + g1) / 2:g}" y="{Y1 - 14}" class="sg-text sg-bold" text-anchor="middle">{v}</text>')
    for j in range(4):
        box(bx(i, j), Y1, by[i][j], j)

a("  <!-- shuffle wires -->")
for i in range(4):
    for j in range(4):
        sx, dx = bx(i, j) + BW / 2, bx(j, i) + BW / 2
        a(
            f'  <path d="M{sx:g},{Y1 + BH + 2} C{sx:g},{Y1 + BH + 55} {dx:g},{Y2 - 55} {dx:g},{Y2 - 2}" class="sg-wire sg-w{j}"/>'
        )

a("  <!-- 2: shuffled -->")
side(Y2, "Shuffled", "same 16 bytes")
for j in range(4):
    for i in range(4):
        box(bx(j, i), Y2, by[i][j], j)

YB = Y2 + BH + 6
runs = [(bx(0, 0), bx(0, 3) + BW), (bx(1, 0), bx(1, 2) + BW), (bx(1, 3), bx(1, 3) + BW), (bx(2, 0), bx(3, 3) + BW)]
for x1, x2 in runs:
    a(f'  <path d="M{x1},{YB} v6 H{x2} v-6" class="sg-brace"/>')

a("  <!-- 3: gzip -->")
side(Y3, "gzip", "Deflate")
toks, x = [], X0
for i in range(4):
    toks.append((x, BW))
    box(x, Y3, by[i][0], 0)
    x += BW + GAP
x += 10
toks.append((x, 70))
box(x, Y3, 0, 1, 70, "00 × 3", True)
x += 70 + GAP
toks.append((x, BW))
box(x, Y3, 1, 1)
x += BW + GAP
toks.append((x, 70))
box(x, Y3, 0, 2, 70, "00 × 8", True)
x += 70
targets = [(toks[0][0] + toks[3][0] + BW) / 2, toks[4][0] + 35, toks[5][0] + BW / 2, toks[6][0] + 35]
for (x1, x2), tx in zip(runs, targets):
    mx = (x1 + x2) / 2
    a(
        f'  <path d="M{mx:g},{YB + 7} C{mx:g},{YB + 35} {tx:g},{Y3 - 35} {tx:g},{Y3 - 3}" class="sg-line" marker-end="url(#sg-arrow)"/>'
    )
a(f'  <text x="{x + 22}" y="{Y3 + 13}" class="sg-text">16 bytes → 4 literal bytes + 3 codes</text>')
a(f'  <text x="{x + 22}" y="{Y3 + 29}" class="sg-muted">a real 4 MiB chunk has runs of thousands</text>')
a("</svg>")
(STATIC / "shuffle_gzip.svg").write_text("\n".join(o) + "\n")
