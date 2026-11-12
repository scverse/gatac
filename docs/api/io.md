# I/O — `gatac.io`

The `gatac.io` namespace provides access to the two kinds of files that
GATAC reads from and writes:

- **Fragment files:** `gatac.io` converts raw fragment TSV.GZ files into
  the GATAC Parquet fragment format on the GPU, then reads these files and
  converts them back to TSV.GZ. Every GATAC function reads fragment files
  through `FragmentFile`. See
  [Parquet fragment format](#parquet-fragment-format).

- **Count matrices:** GATAC writes `.h5ad` files for the tile, gene and
  gene-score count matrices, compressed on GPU. GATAC also reads these
  matrices directly from disk into GPU memory, see [count matrices on
  disk](#count-matrices-on-disk).

All GATAC functions that access the genome sequence do so through `Fasta`.
The BED file, GTF file and cell metrics readers are used by every GATAC
function that takes these types of files as argument.

```{eval-rst}
.. currentmodule:: gatac.io

.. rubric:: Fragment files

.. autosummary::
   :toctree: generated/
   :nosignatures:

   make_parquet
   make_parquet_batch
   sample_names
   set_barcode_prefix
   to_tsv
   FragmentFile
   read_fragments_parquet

.. rubric:: Count matrices

.. autosummary::
   :toctree: generated/
   :nosignatures:

   read_h5ad
   write_h5ad

.. rubric:: Reference and metadata files

.. autosummary::
   :toctree: generated/
   :nosignatures:

   Fasta
   read_bed
   read_gtf
   read_metrics
```

---

## Parquet fragment format

GATAC reads the fragments from Parquet files that were written by
`gatac.io.make_parquet` (or `gatac convert`) from a `fragments.tsv.gz`
file. This section explains the file layout and the benefits of it.

### File layout

In a `fragments.tsv.gz` file, the chromosome and barcode are repeated on every
line. A deep library has hundreds of millions of lines with a
few dozen chromosome names and hundreds of thousands of unique barcodes.
Stored as text, they would have to be inflated and parsed every time the
fragments are read, then matched against the cell and chromosome names given
to GATAC. Even with the GPU-accelerated parser used for
the [conversion](#conversion-on-the-gpu), `compute_metrics` took 23s on a
`fragments.tsv.gz` with 484 million fragments (H100), and 1.8s on the
equivalent Parquet file. Converting the file took 24s.

GATAC stores the chromosome and barcode as integer ids instead, with the
names listed only once in the same file:

```{raw} html
<p class="diagram-scroll-hint">Scroll sideways to see the whole diagram →</p>
```

````{div} diagram-scroll
```{raw} html
:file: ../_static/fragment_layout.svg
```
````

The **row groups** of a Parquet file contain the columns of data,
compressed here with ZSTD:

- the fragment data (`chrom_id`, `start`, `length`, `barcode_id`, `count`):
  one line per fragment, in the order of the input file. The `start`
  position is stored as a *delta* value: the fragments are sorted, so each
  line contains the (small) difference to the previous line. The `end`
  position is stored as `length` (`end - start`), which is a few hundred
  bases for ATAC-seq and therefore almost always starts with a byte of 0.
  All readers of Parquet fragment files provide `end`.
- a barcode table: row *i* gives the original name of barcode id *i*
  (`bc_name`), with the number of fragments (`bc_n_unique`) and reads
  (`bc_n_total`) of that barcode. So `FragmentFile.barcode_counts()` does not need to read
  the fragments data. Barcodes are given an id in decreasing order of
  fragments number: the barcode table only fills the first lines, and is null
  afterwards (doesn't take space).

The **file footer** contains key-value metadata for the file, which is small
and can be quickly read every time the file is opened:

- `gatac.format_version`: format version, in case the format needs to be
  changed in the future.
- `gatac.chroms`: the chromosome names in order of `chrom_id`, as a JSON
  array.
- `gatac.n_barcodes`: the number of barcodes (rows in barcode table)
- `gatac.barcode_prefix`: the sample barcode prefix, or an empty string. The
  name of a cell is this prefix followed by the `bc_name` of the cell. The barcode table
  holds only the barcode names of the original `fragments.tsv.gz`. Changing
  the prefix, as `gatac rename` does, rewrites only this part of the footer
  and takes milliseconds whatever the file size
  ({func}`gatac.io.set_barcode_prefix`). {func}`gatac.io.to_tsv` writes the
  original `fragments.tsv.gz`, without the prefix.

```{important}
Every GATAC function, metrics and matrices included, names cells with
their prefix. A reader of the file outside GATAC must add the prefix
itself to get the same names; see
[Reading a file outside GATAC](#reading-a-file-outside-gatac).
```

The Parquet file is 2.8× smaller than the original `fragments.tsv.gz` on
average. The figure compares each column of the text file, gzipped alone,
with the same data in the Parquet file. The data is PBMC 5k, with 101 million
fragments and 477k barcodes. The chromosome column (0.9 MB gzipped, less than
0.01 MB in Parquet) is too small to show.

````{div} diagram-wide
```{raw} html
:file: ../_static/parquet_size.svg
```
````

````{div} diagram-narrow
```{raw} html
:file: ../_static/parquet_size_narrow.svg
```
````

Gzipped one at a time, the columns take 856 MB, against 1,042 MB for the
whole `fragments.tsv.gz`. The 186 MB difference comes from the text
alternating between columns on every line, which gzip compresses less well.
The barcode names only
take 1.9 MB of the 170 MB barcode column, since they are only stored once per
barcode: the rest is the `barcode_id` of each fragment. The chromosome name
is dictionary-encoded. The start positions are sorted and delta-encoded to a
few bits. The lengths, barcode ids and counts are byte-stream-split. All this
happens before ZSTD compression.

```{note}
The integer ids are different for each file: chromosomes are numbered
depending on the order in which they appear in the `fragments.tsv.gz`, and
barcodes are numbered by decreasing number of fragments. All GATAC functions
taking chromosome or cells as argument convert the names given by the user to
the ids in the file, using the `gatac.chroms` metadata and the barcode table
in the file. That way the same chromosome can have different ids in two files
and GATAC's output won't be affected by that.
```

### Conversion on the GPU

Thanks to nvCOMP, `gatac.io.make_parquet`, `gatac.io.make_parquet_batch` and
`gatac convert` perform the conversion on the GPU, producing the same output
file as DuckDB. nvCOMP decompresses the BGZF blocks of the `fragments.tsv.gz`
and then one CUDA thread parses each line in parallel. The resulting lines
are converted to column arrays and written by the cuDF Parquet writer. The GPU
memory usage is therefore limited by the size of chunks loaded from the
input file.

DuckDB is used for conversion in the following cases:

- `fragments.tsv.gz` files compressed with `gzip` instead of `bgzip`.
  Recompress them with `bgzip` to convert them on the GPU.
- Zstandard files (`.zst`), the default compression of SnapATAC2.
- all files if the argument `gpu=False` is passed to `make_parquet*` or if
  `gatac convert --cpu` is used. This is sometimes necessary to accept lines
  with unusual formatting that the stricter GPU parser rejects, such as
  ` 5`, `+5` or `0x5` instead of `5`, or a `#` character in the line.

A BGZF block that does not inflate to the length given in the block footer raises
an error, even with `skip_invalid=True`.
However, the CRC32 checksums contained in the BGZF blocks are not checked.

`gatac.io.make_parquet_batch` performs the conversion of multiple files
simultaneously on the same GPU. The default maximum number of concurrent
files is 4 (one per available CPU core and 7 GB of GPU memory).

### Reading a file outside GATAC

{class}`gatac.io.FragmentFile` exposes the fragments with names as a
DuckDB view (version 2.0 or later):

```python
import duckdb
import gatac as ga

con = duckdb.connect()
ga.io.FragmentFile("pbmc.parquet").register(con)  # view "fragments"
con.sql("SELECT barcode, count(*) AS n FROM fragments GROUP BY barcode")
```

`FragmentFile(path).iter_arrow()` streams Arrow tables with the same
columns. The DuckDB view and `iter_arrow()` include the barcode prefix of
the file. {func}`gatac.io.to_tsv` writes the original, unprefixed barcodes
to a `fragments.tsv.gz` in file order, using BGZF compression so `tabix`
can index it.

Without GATAC, any Parquet reader that decodes byte-stream split
integers (pyarrow, Polars, cuDF, DuckDB 2.0 or later) gets the ids and the
tables:

```python
import json
import pyarrow.compute as pc
import pyarrow.parquet as pq

pf = pq.ParquetFile("pbmc.parquet")
meta = pf.metadata.metadata
chroms = json.loads(meta[b"gatac.chroms"])  # chrom_id -> name
n_barcodes = int(meta[b"gatac.n_barcodes"])
prefix = meta.get(b"gatac.barcode_prefix", b"").decode()
names = pf.read(columns=["bc_name"])["bc_name"][:n_barcodes]
barcodes = pc.binary_join_element_wise(prefix, names, "")  # barcode_id -> cell name
frags = pf.read(columns=["chrom_id", "start", "length", "barcode_id", "count"])
end = pc.add(frags["start"], frags["length"])
```

---

## Count matrices on disk

GATAC writes tile, gene and gene-score matrices (`make_tile_matrix`,
`make_gene_matrix`, `make_gene_score_matrix`, `pp.combine`, ...) as standard
`.h5ad` files. anndata, scanpy and any HDF5 reader can open them.

### File layout

GATAC changes two things in how the CSR matrix `X` is stored:

1. **int32 indices.** Column indices are int32. Row pointers (`indptr`) are
   int32 too, unless the matrix has more than 2³¹ − 1 stored entries.
   GATAC 0.1 wrote both as int64.
2. **Shuffle + gzip compression.** The default, `compression="gzip"`, uses
   the HDF5 shuffle and deflate filters. GATAC compresses `X` on the GPU with
   nvCOMP. anndata gzips the rest of the file (`obs`, `var`, ...).
   `compression=None` (`--compression none`) writes `X` uncompressed.

Before compression, the shuffle groups the bytes of each chunk by their
position in the value. In a count matrix, the high-order bytes of the counts
and column indices are mostly zero. After the shuffle, they form long runs
of zeros that gzip compresses well. The tile matrix of a 5k-cell PBMC
library takes 36 MB this way, against 58 MB with gzip alone and 190 MB
uncompressed.

````{div} diagram-wide
```{raw} html
:file: ../_static/shuffle_gzip.svg
```
````

````{div} diagram-narrow
```{raw} html
:file: ../_static/shuffle_gzip_narrow.svg
```
````

### Reading on the GPU

{func}`gatac.io.read_h5ad` reads an `.h5ad` file like `anndata.io.read_h5ad`,
but decompresses the sparse matrices (`X`, `layers`, `raw.X`) on the GPU.
KvikIO loads the compressed bytes into GPU memory, nvCOMP inflates them, and
the shuffle is undone there. The checksum of each chunk is checked on the GPU, so
a corrupted file raises an error instead of returning wrong counts. anndata
reads the rest of the file.

The matrices come back in host memory with the value dtypes stored in the
file. The index arrays are int32 when they fit, as scipy builds them. With `to_gpu=True`, they stay on the GPU as cupy CSR
matrices, ready for rapids-singlecell. {func}`gatac.io.write_h5ad` writes any
AnnData with `X` stored as described above.

```python
import gatac as ga

adata = ga.io.read_h5ad("tiles.h5ad", to_gpu=True)  # X is a cupy CSR matrix
```

### Backed AnnData

Four functions take an AnnData opened with `anndata.read_h5ad(path,
backed="r")`, or a view of one, and read only what they need from the file
onto the GPU:

| Function | Reads from the file |
|---|---|
| `pp.select_features` | Nothing for a tile matrix that keeps all cells: the per-tile counts in `var` are enough. Otherwise, the matrix once, in row chunks. |
| `dr.spectral` | Only the selected features. |
| `motif.compute_deviations` | One row chunk at a time, so memory does not grow with the number of cells. |
| `peak.markers` | Only the cells it tests. |

Above 100,000 cells, or when `chunk_size` is given, `dr.spectral` keeps row
chunks on the GPU while they fit in half of the free GPU memory. The other
chunks go to page-locked host memory and are copied to the GPU for each
matrix product.

```python
import anndata as ad
import gatac as ga

adata = ad.read_h5ad("tiles.h5ad", backed="r")  # X stays on disk
ga.pp.select_features(adata)  # var["selected"], from the per-tile counts in var
ga.dr.spectral(adata)  # obsm["X_spectral"], from the selected tiles only
adata.write()  # saves var, obsm, ... into the file, not X
```

### Where results go

Results are stored as scanpy stores them, in the AnnData given as input:

| Input | Results |
|---|---|
| in memory | `adata.obsm` / `adata.uns`, as before |
| backed, opened `"r"` | in memory only. `adata.write()` saves them (anndata rewrites obs, var, obsm, uns, ... but not X) |
| backed, opened `"r+"` | in memory and in the file: the new keys only, X untouched |
| a view of a backed AnnData | nothing stored: `spectral(..., inplace=False)` and `compute_deviations(..., return_adata=True)` return them |

```python
import anndata as ad
import gatac as ga

adata = ad.read_h5ad("combined.h5ad", backed="r+")
ga.dr.spectral(adata)  # obsm["X_spectral"] is also written to the file

emb = ga.dr.spectral(adata[keep_cells, keep_tiles], inplace=False)  # a view: returned
```

```{warning}
Never call `adata.write()` on a view of a backed AnnData: anndata writes the
view's `obs` next to the file's whole matrix, and the file then opens with
the wrong shape. Save a view's results to their own file, or store them on
the whole backed AnnData.
```

HDF5 does not reclaim the space of rewritten keys, so saving results many
times grows the file. `h5repack` gives the space back.

### Reading a matrix outside GATAC

anndata, scanpy and other tools read these matrices as usual. Loading a
whole gzip-compressed matrix with `anndata.read_h5ad` takes several times
longer than an uncompressed one, because HDF5 inflates it on one CPU core.
Backed reads (`X[:100]`) are no slower. For files that other tools load
whole, write them with `compression=None` (`--compression none`).
