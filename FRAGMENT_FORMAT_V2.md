# GATAC fragment format, version 2

_Draft, 2026-11-12. Proposal for discussion; nothing here is implemented yet._

## 1. Summary

GATAC stores single-cell ATAC fragments in Parquet files with integer
chromosome and barcode ids, an in-file barcode table, and key-value metadata
in the footer (format version 1, documented in `docs/api/io.md`, "Parquet
fragment format"). Version 1 is a GATAC file layout: it is defined by what
`gatac.io.make_parquet` writes, and some of its choices are workarounds for
the engines GATAC uses.

Version 2 turns the layout into a **specification**, a profile of standard
Parquet in the spirit of GeoParquet: a logical schema, one versioned metadata
object, and a split between what readers and writers MUST do and what they
SHOULD do. It also defines the **converter architecture** that writes the
format from any source that can produce fragments, not only text files.

Goals:

- A file that any Parquet reader can decode, and that a tool other than GATAC
  can read correctly from this document alone.
- Requirements that do not depend on cuDF, DuckDB or pyarrow versions; those
  appear as recommendations with their reason.
- Region queries without an external index, for files sorted by position.
- One conversion pipeline for fragment text files and their variants, other
  tools' formats (SnapATAC2, ArchR, BPCells), and fragments already in memory
  as data frames.
- Cheap migration: a version 1 file becomes a version 2 file by rewriting its
  footer.

Non-goals:

- A new container. The format is Parquet; nothing here needs a custom reader
  to decode bytes.
- Fragment calling as part of the format. The format stores fragments,
  whoever called them; the read filters, Tn5 shift and deduplication behind
  them are recorded in `source`, not prescribed. GATAC's own fragment caller,
  which reads BAM files on the GPU, is a converter (section 8.8).
- Datasets spanning several files. One file holds one sample; a manifest for
  multi-sample datasets is left to a later version (see section 11).

The key words MUST, MUST NOT, SHOULD, SHOULD NOT and MAY are used as in
RFC 2119.

## 2. Model

A file holds the fragments of one sample. A fragment is a genomic interval
from one cell barcode, with the number of reads that support it:

| Logical field | Meaning                                                                  |
| ------------- | ------------------------------------------------------------------------ |
| `chrom`       | Chromosome or contig name                                                |
| `start`       | 0-based start, inclusive                                                 |
| `end`         | 0-based end, exclusive (`start < end`), as in BED and Cell Ranger        |
| `barcode`     | Cell name: the file's barcode prefix followed by the stored barcode name |
| `count`       | Number of reads (duplicates) behind the fragment                         |

The file stores `chrom` and `barcode` as integer ids into lookup tables, and
`end` as a length. The **logical schema** is what readers return; the
**stored schema** is what is in the Parquet file.

## 3. Stored schema

A file MUST be a valid Parquet file with these columns, in any order. Readers
MUST locate columns by name.

### 3.1 Fragment columns

One row per fragment.

| Column       | Type             | Nullable | Meaning                                |
| ------------ | ---------------- | -------- | -------------------------------------- |
| `chrom_id`   | uint16           | no       | Index into `chroms` (section 4)        |
| `start`      | uint32           | no       | 0-based start                          |
| `length`     | uint16 or uint32 | no       | `end - start`, at least 1              |
| `barcode_id` | uint32           | no       | Row of the barcode table (section 3.2) |
| `count`      | uint16 or uint32 | no       | Reads behind the fragment, at least 1  |

- Readers MUST accept both widths of `length` and `count`, and MUST return
  `end = start + length`. A writer MUST NOT write a `count` above 65,535 or a
  `start + length` above 2³² − 1.
- Parquet has no 16-bit physical type: uint16 and uint32 are both stored as
  INT32 and differ only in their logical annotation, so the width changes no
  stored byte. Writers SHOULD annotate `length` and `count` as uint32
  (section 6).
- `chrom_id` MUST be below the number of chromosomes and `barcode_id` below
  `n_barcodes`.

Change from version 1: `count` may be null in version 1 (an empty count field
in the input). Version 2 forbids it; converters fill it (section 8.3).

### 3.2 Barcode table

The per-barcode table shares the file's rows: row _i_ describes barcode id _i_
for _i_ < `n_barcodes`, and the columns are null from row `n_barcodes` on.
Since every barcode has at least one fragment, `n_barcodes` never exceeds the
number of rows.

| Column        | Type             | Nullable                | Meaning                                           |
| ------------- | ---------------- | ----------------------- | ------------------------------------------------- |
| `bc_name`     | string           | after `n_barcodes` only | Barcode name as in the source, without the prefix |
| `bc_n_unique` | uint32           | after `n_barcodes` only | Fragments of the barcode                          |
| `bc_n_total`  | uint32 or uint64 | after `n_barcodes` only | Sum of `count` over those fragments               |

- Names MUST be unique and non-empty.
- `bc_n_unique` and `bc_n_total` MUST match the fragment columns. Readers MAY
  rely on them instead of scanning the fragments (QC, knee plots).
- A reader that needs only the table SHOULD read only the leading row groups
  that cover rows 0 to `n_barcodes - 1`.

Rationale: keeping the table in the same file makes a sample a single file,
and the null padding costs almost nothing once compressed. A sidecar file
would split a sample in two. A footer entry would be parsed on every open,
and the table is about 2 MB for 500k barcodes.

### 3.3 Other columns

Writers MUST NOT add other columns to version 2 files. A later minor version
can define optional columns. Readers MUST ignore columns they do not know, so
that minor versions stay readable.

## 4. Metadata

### 4.1 Keys

The footer's key-value metadata holds two keys:

| Key                    | Value                              |
| ---------------------- | ---------------------------------- |
| `gatac`                | The metadata object, as UTF-8 JSON |
| `gatac.format_version` | `"2"`, the major version alone     |

`gatac.format_version` is kept so that GATAC releases that read only version
1 fail with "fragment format version '2' is not supported" rather than
calling the file not a GATAC file. Version 2 readers MUST use the version in
the JSON object.

Writers MUST NOT write the version 1 keys `gatac.chroms`, `gatac.n_barcodes`
or `gatac.barcode_prefix`, so that each value has one source.

### 4.2 The metadata object

| Field            | Required                  | Type                           | Meaning                                                                                              |
| ---------------- | ------------------------- | ------------------------------ | ---------------------------------------------------------------------------------------------------- |
| `version`        | yes                       | string                         | Format version, `"MAJOR.MINOR"`, here `"2.0"`                                                        |
| `chroms`         | yes                       | array of strings               | Chromosome names, indexed by `chrom_id`; unique                                                      |
| `n_barcodes`     | yes                       | integer                        | Rows of the barcode table                                                                            |
| `barcode_prefix` | yes                       | string                         | Prefix of every cell name; `""` for none                                                             |
| `counts`         | yes                       | string                         | `"reads"`: `count` holds read counts. `"absent"`: the source had none and `count` is 1 everywhere    |
| `barcode_order`  | yes                       | string                         | `"n_unique_desc"`: barcode ids by decreasing `bc_n_unique`, ties by name. `"unspecified"`: any order |
| `sort`           | yes                       | string                         | Row order: `"position"`, `"barcode"` or `"none"` (section 5)                                         |
| `chrom_rows`     | if `sort` is `"position"` | array of `[first_row, n_rows]` | Rows of each `chrom_id` (section 5)                                                                  |
| `chrom_lengths`  | no                        | array of integers              | Chromosome lengths, aligned with `chroms`                                                            |
| `genome`         | no                        | string                         | Assembly name, such as `"GRCh38"` or `"hg38"`                                                        |
| `source`         | no                        | object                         | Where the file came from (section 4.3)                                                               |
| `writer`         | no                        | object                         | `{"name": "gatac", "version": "0.3.0"}`                                                              |

Readers MUST:

- Reject a major version they do not know. Accept any minor version of a
  known major, ignoring fields they do not know.
- Put `barcode_prefix` in front of `bc_name` whenever they report cell names.
  `gatac rename` (`set_barcode_prefix`) changes only this field and rewrites
  only the footer.
- Not assume an order of barcode ids unless `barcode_order` says so, or an
  order of rows unless `sort` says so.

### 4.3 Provenance

`source` records the input, so that a file can be traced back:

```json
"source": {
  "kind": "tsv",
  "path": "runs/A/outs/atac_fragments.tsv.gz",
  "sha256": "…",
  "rejected_lines": 0
}
```

`kind` is the converter adapter's name (section 8.4). Every other field is
specific to that adapter. A file written by `filter_fragments` records the
file it filtered and the filter, so a chain of files stays traceable.
`sha256` is optional: hashing a large input costs a full read, so converters
SHOULD compute it only on request.

## 5. Row order and region queries

`sort` declares the row order:

- `"position"`: rows ordered by `chrom_id`, then `start`, then `length`.
  Fragments of one chromosome are contiguous. Cell Ranger files are already in
  this order.
- `"barcode"`: rows grouped by `barcode_id`. This is the order of sources
  stored per cell (section 8.4).
- `"none"`: no guarantee, for example a file in the order of a TSV whose sort
  was not checked.

When `sort` is `"position"`, `chrom_rows[k]` gives the first row and the number
of rows of `chrom_id` _k_. `[0, 0]` marks a chromosome with no fragments. A
region query maps a chromosome to its rows, the rows to row groups through
the row-group sizes, and within the chromosome uses the min and max
statistics of `start` per row group and per page to skip the rest, since
`start` is sorted there. This replaces a tabix index without changing the row
groups. Splitting row groups at each chromosome was rejected because
assemblies with thousands of small contigs would get as many tiny row groups.

Writers MUST NOT declare a sort they did not check. The converter checks it
during conversion (section 8.3), at the cost of one comparison per row.

GATAC keeps the input row order and does not sort. Sorting a file that is not
already in order is a separate operation (`gatac sort`, out of scope here).

## 6. Physical layout recommendations

Writers SHOULD use the layout below. Readers MUST NOT require it: any valid
Parquet encoding of the stored schema is a valid file.

| Column                          | Encoding                  | Why                                                                                                                                                                               |
| ------------------------------- | ------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `chrom_id`                      | Dictionary                | A few dozen distinct values                                                                                                                                                       |
| `start`                         | `DELTA_BINARY_PACKED`     | Sorted starts become small deltas                                                                                                                                                 |
| `length`, `barcode_id`, `count` | `BYTE_STREAM_SPLIT`       | The high bytes are almost always 0 once barcode ids follow `barcode_order: n_unique_desc`; ZSTD then all but removes them, getting closer to the entropy than dictionary encoding |
| `bc_name`                       | `DELTA_LENGTH_BYTE_ARRAY` | Smaller than `DELTA_BYTE_ARRAY` here: 2.35 MB against 2.67 MB for 400k barcodes. Prefix sharing does not help, since names follow fragment count, not alphabetical order          |
| `bc_n_unique`, `bc_n_total`     | `DELTA_BINARY_PACKED`     | Sorted decreasing                                                                                                                                                                 |

- Compression: ZSTD.
- Row groups of 1,000,000 rows. The GPU writer lays pages out by 5,000 rows,
  so sizes are rounded to a multiple of 5,000.
- The page index SHOULD be written, for the region queries of section 5.
- Data page version 1. Version 2 brings nothing here: 33.0 MB against
  33.1 MB on an 8-million-row file with this layout.
- `length` and `count` annotated as uint32 (section 3.1).
- `bc_name` SHOULD NOT use `DELTA_BYTE_ARRAY`: cuDF 26.02 reads it back with
  wrong values, and no error, when the page index is present.

Engine notes, checked on 2026-11-12 with cuDF 26.02.01, pyarrow 26.0 and
DuckDB 2.0 (RTX 5070). The layout above avoids every one of them at no cost
in size, so none is a requirement of the format:

| Engine     | Constraint                                                                                | Cost of avoiding it                                                                        |
| ---------- | ----------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| cuDF 26.02 | Fails to decode `BYTE_STREAM_SPLIT` on a uint16 annotation ("Parquet data decode failed") | None on disk: same INT32 bytes. Readers get 4 bytes per value instead of 2 until they cast |
| cuDF 26.02 | Chunked (`low_memory`) reader hits an illegal memory access on data page v2               | None: v1 and v2 files are the same size                                                    |
| cuDF 26.02 | Returns wrong strings without an error for `DELTA_BYTE_ARRAY` with the page index         | None: `DELTA_LENGTH_BYTE_ARRAY` is smaller for barcode names                               |
| DuckDB     | Reads `BYTE_STREAM_SPLIT` integers from 2.0                                               | DuckDB 2.0 or later                                                                        |

Readers that cannot avoid such files SHOULD route around the bug. For
example, GATAC can read the barcode table with pyarrow, since it is small.
When an engine fixes a constraint, writers can change the layout without a
format version change.

## 7. Writers

### 7.1 New files

- A writer MUST write to a temporary file next to the target and rename it on
  success, so that an existing file is replaced only by a complete one. GATAC
  already does this.
- A writer MUST validate before declaring: ids in range, barcode table
  consistent with the fragments, and `sort` checked (section 5).
- A writer MUST reject more than 65,535 chromosomes (`chrom_id` is uint16).

### 7.2 Metadata-only updates

Some changes touch only the footer: the barcode prefix (`gatac rename`,
`set_barcode_prefix`) and the version 1 to 2 upgrade (section 9). Rewriting
the footer in place avoids copying a file of several GB. Today's
`update_key_value_metadata` (`gatac/io/_footer.py`) overwrites the footer and
restores it if Python raises an error. A process killed or a machine lost
during the overwrite can still leave the file without a valid footer.

Requirements:

- A metadata-only update MUST be recoverable after a crash at any point: it
  MUST NOT leave a file whose old and new footers are both lost.
- A file at rest MUST be plain Parquet, with no trace of the update method.
  Readers MUST NOT need to know how files are updated.
- Writers MAY use any method that meets these two requirements. Two are known:
  - **A rollback journal** (section 7.3), as SQLite does. It works on any
    local or network filesystem and is what GATAC uses.
  - **Clone, edit, rename**: a copy-on-write clone of the file (btrfs, XFS
    with reflink, ZFS 2.2 or later, APFS) is edited and renamed over the
    original. This is atomic and costs no data copy, but is not available on
    ext4 or on most HPC filesystems such as Lustre and GPFS.

Two methods were rejected:

- **Appending a new footer** after the old one, which stays as unused bytes.
  Every engine tested reads such a file (pyarrow, DuckDB 2.0, cuDF 26.02, with
  the page index intact). A file cut off by a crash can even be repaired from
  the file alone. But it is not a Parquet convention, each edit leaves about
  6 KB of unused bytes, and a strict validator could object to them.
- **Moving the editable metadata to a sidecar file.** A file would no longer
  describe itself.

### 7.3 The rollback journal

_This section describes GATAC's method. It is not part of the format._

The journal is a file next to the data file, named after it with
`-journal` appended: `s1.parquet-journal`. The name is visible, not a
dotfile, so that a journal left by a crash appears in `ls`, and
`mv s1.parquet*` moves it together with its file.

The journal holds:

| Field                     | Purpose                                                                           |
| ------------------------- | --------------------------------------------------------------------------------- |
| Magic and journal version | Recognise the file                                                                |
| Name of the data file     | Report which file a stray journal belongs to                                      |
| Footer offset             | Where the old tail starts: the old footer, its length and `PAR1`                  |
| Anchor                    | CRC32 of the 4 KB before the footer offset, to recognise the data file by content |
| Old tail                  | Undo, and to recognise a file not yet modified                                    |
| New tail                  | Redo                                                                              |
| CRC32 of all the above    | Detect a journal cut off by a crash                                               |

An update takes four steps:

1. **Lock.** Create the journal with `O_CREAT | O_EXCL`, and hold an exclusive
   `flock` on it for the rest of the update. If the journal already exists,
   run recovery first. If its lock is held, another update is running, and
   this one fails.
2. **Journal.** Write the fields, `fsync` the journal, `fsync` the directory.
   From here on the update survives a crash.
3. **Overwrite.** Write the new tail at the footer offset in a single
   `write`, truncate the file after it, and `fsync` the file.
4. **Commit.** Delete the journal and `fsync` the directory.

Recovery runs whenever GATAC opens a file next to which a journal exists. It
also runs from `gatac repair`. It compares the file's content with the
journal, never its inode or modification time, since `cp` and rsync change
those:

| What GATAC finds                                   | Meaning                                                    | Action                                                    |
| -------------------------------------------------- | ---------------------------------------------------------- | --------------------------------------------------------- |
| Journal CRC wrong                                  | Crash during step 2; the data file is untouched            | Delete the journal                                        |
| Anchor matches, tail equals the new tail           | Crash after step 3                                         | Delete the journal                                        |
| Anchor matches, tail equals the old tail           | Crash before step 3                                        | Redo: steps 3 and 4                                       |
| Anchor matches, tail is neither                    | Crash during step 3                                        | Redo: steps 3 and 4                                       |
| Anchor does not match, or the data file is missing | The journal belongs to another file, or its file was moved | Report the journal and offer to delete it. Never apply it |

Recovery completes the update rather than undoing it, since the new tail is
in the journal and the update was what the user asked for. When GATAC cannot
write the file (no permission), it reads the new footer from the journal and
leaves the file as it is.

Two limitations remain:

- **Concurrent readers.** A reader that opens the file during step 3, which
  lasts milliseconds, can see a broken footer and fail. It does not read
  wrong data in practice, though Parquet footers have no checksum to rule it
  out. Readers that already had the file open are not affected, since no data
  byte moves.
- **A file moved without its journal after a crash during step 3.** The file
  is unreadable, and nothing in it holds the chromosome names, the prefix,
  `sort` or `source`, which live only in the footer. The data pages and the
  barcode names are intact. With the schema fixed by this document,
  `gatac repair --rebuild-footer` can rebuild the row groups from the page
  headers, given the chromosome names (from the original fragments file) and
  the prefix. This takes a crash in a window of milliseconds, then a move of
  the file alone, before GATAC opens it again. See open question 7.

## 8. Converters

### 8.1 Architecture

```
source ──► adapter ──► normalised Arrow stream ──► validate ──► stage ──► write
            (8.4)          (8.2)                     (8.3)       (8.5)    (3–7)
```

An adapter turns a source into a stream of Arrow record batches with the
normalised schema. Everything after the adapter is shared: validation,
counting, sort detection, provenance and the writer. A new source therefore
costs one adapter, and gets every check for free.

The Arrow stream is the contract between the readers and the writer, not a
source format. No scATAC tool writes fragments as Arrow or Parquet today:
Cell Ranger, sinto and chromap write `fragments.tsv.gz`, SnapATAC2 writes
`.tsv.zst` or h5ad, ArchR writes HDF5 and BPCells its own binary format.
Adapters therefore target those formats. The only public Arrow entry point is
`from_arrow()`, for fragments a user already holds as a data frame (8.4).

The GPU parser for BGZF and plain-text TSV stays a separate **fast lane** that
produces the same file and the same metadata. It is the common case, and the
largest files, so it keeps its speed. Every other source goes through the
**generic lane** below.

### 8.2 Normalised stream

Adapters yield Arrow batches with exactly these columns:

| Column    | Arrow type | Constraint                                     |
| --------- | ---------- | ---------------------------------------------- |
| `chrom`   | string     | Not null, not empty                            |
| `start`   | uint32     | Not null, 0-based                              |
| `end`     | uint32     | Not null, greater than `start`                 |
| `barcode` | string     | Not null, not empty, without the sample prefix |
| `count`   | uint16     | Nullable: null means "not given"               |

This is `NAMED_SCHEMA` in `gatac/io/fragments.py`. The adapter does any
coordinate change: a 1-based source subtracts 1 from `start` in the adapter,
not later.

An adapter also reports its `source` object (section 4.3) and, when it knows
them, the sort order of its rows, the chromosome order and lengths, and the
genome.

### 8.3 Validation

Validation runs once, on the normalised stream:

- Rows breaking a constraint of section 8.2 are an error naming the first one.
  With `skip_invalid`, they are dropped, and their number is logged and stored
  in `source.rejected_lines`.
- Null counts: when the whole source has no counts, every `count` becomes 1
  and `counts` is `"absent"`. A source where only some rows lack a count is an
  error unless `skip_invalid` drops those rows. A guessed value would mix
  real counts with made-up ones.
- Sort detection: the stream is compared row to row with the previous row; the
  result sets `sort` and `chrom_rows`. A source that declares a sort is still
  checked.

### 8.4 Adapters

Adapters for files, in order of priority:

| Adapter (`kind`) | Source                                                                                                                                                     | Lane                                                    | Notes                                                                                                                                                                                                                                                                |
| ---------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `tsv`            | Fragments TSV from Cell Ranger, sinto, chromap and others: BGZF, gzip, Zstandard or plain                                                                  | Fast for BGZF and plain, generic for gzip and Zstandard | Today's `make_parquet`. SnapATAC2's `.tsv.zst` is already accepted through DuckDB. Variants go through the generic lane with `columns=`, which gives the order of the fields: four columns without a count (`counts: "absent"`), or extra columns, which are ignored |
| `snapatac2`      | SnapATAC2 `.h5ad`, fragments in `obsm['fragment_paired']` or `obsm['fragment_single']`                                                                     | Generic                                                 | Rows come grouped by cell: `sort` is `"barcode"`. Chromosome lengths come from `uns['reference_sequences']`. Still to verify on real files: the storage layout, whether read counts survive, and the coordinate convention                                           |
| `archr`          | ArchR Arrow files: HDF5, one group per chromosome with `Ranges` (start, width) and run-length-encoded barcodes. Unrelated to Apache Arrow despite the name | Generic                                                 | No read counts: `counts` is `"absent"`. Rows are by chromosome, then by cell. The sample name of the file becomes the default `barcode_prefix` (`Sample#`)                                                                                                           |
| `bpcells`        | BPCells fragment directories                                                                                                                               | Generic                                                 | After `archr`, if users ask                                                                                                                                                                                                                                          |
| `bam`            | Aligned reads: BAM files from Cell Ranger or any aligner, with the barcode in a tag or the read name                                                       | Own GPU lane                                            | Calls fragments on the GPU instead of reading them (section 8.8)                                                                                                                                                                                                     |
| `gatac`          | A GATAC fragment file, version 1 or 2                                                                                                                      | Generic                                                 | Rewrites a file, for example to change the layout or the chromosome order. Version 1 files that only need new metadata use `gatac upgrade` instead (section 9)                                                                                                       |

For fragments already in memory, `from_arrow()` takes any object implementing
the Arrow PyCapsule stream interface (`__arrow_c_stream__`): pyarrow tables
and record batch readers, Polars and pandas data frames, DuckDB relations
(`kind: "arrow"`). Users get there after filtering, downsampling, merging or
simulating fragments with these libraries, which today means writing a TSV
first. `columns=` renames source columns, and a missing `count` column makes
`counts` `"absent"`.

There is no adapter for Parquet files from other tools, since no tool writes
them. A Parquet file a user made, for example a DuckDB copy of a TSV, goes
through `from_arrow()`:

```python
ga.io.from_arrow(duckdb.read_parquet("cache.parquet"), "s1.parquet")
# or with pyarrow: a Dataset has no __arrow_c_stream__, its scanner's reader does
ga.io.from_arrow(ds.dataset("cache.parquet").scanner().to_reader(), "s1.parquet")
```

Adapters for formats tied to other ecosystems SHOULD be optional
dependencies, so that the core install does not grow.

### 8.5 Staging

Single-pass sources (record batch readers, streams) cannot be read twice.
The barcode table, though, has to be known before the first row group is
written (ids by decreasing fragment count). The generic lane therefore always
stages: DuckDB copies the normalised stream to a temporary Parquet next to
the output, counts barcodes from it, and the writer streams it. This is what
the DuckDB path of `make_parquet` already does with `read_csv`; generalising
it means replacing that expression by the adapter's relation. Files larger
than memory keep working.

### 8.6 API

```python
# Generic entry point: the adapter is chosen from the source
ga.io.convert(source, output, *, barcode_prefix=None, columns=None,
              genome=None, chrom_sizes=None, skip_invalid=False, gpu=None)

# Fragments in memory: any object with __arrow_c_stream__
ga.io.from_arrow(obj, output, **kwargs)

# Existing functions keep working, routed through the same pipeline
ga.io.make_parquet(...)
ga.io.make_parquet_batch(...)
```

`gatac convert` selects the adapter from the file's magic bytes and
extension. `--adapter` overrides the choice and `--columns` gives the
mapping.

### 8.7 Export

`FragmentFile` implements `__arrow_c_stream__` on top of `iter_arrow()`, so
`polars.from_arrow(ff)`, `duckdb.sql("FROM ff")` and pyarrow read a file with
names and the prefix applied and no GATAC-specific code. `to_tsv` keeps
writing BGZF, which `tabix` indexes and SnapATAC2's `import_fragments` reads.

### 8.8 Fragment calling from BAM on the GPU

_A converter, not part of the format._

**When it is useful.** Data that never went through Cell Ranger, whose
pipelines stop at a BAM file: SHARE-seq, sci-ATAC-seq, SNARE-seq, dscATAC,
plate-based and spatial ATAC protocols, and in-house protocols aligned with
bwa or bowtie2. Public datasets shared as BAM without a fragments file. And
reprocessing with other choices than Cell Ranger's: another MAPQ threshold,
raw barcodes (`CR`) instead of corrected ones (`CB`), or chrM handling.

Today this takes a CPU tool (`snapatac2.pp.make_fragment_file`,
`sinto fragments`, or ArchR's `createArrowFiles`, which reads BAM files),
then `gatac convert`. BAM files are BGZF, which GATAC already decompresses on
the GPU, and most of the work after that is parallel per read. A GPU caller
would make the BAM a first-class input, and the slowest step of the
preprocessing of deep libraries a fast one.

**Fragment definition.** By default the caller reproduces Cell Ranger's
fragments, as 10x documents them:

- A fragment is the alignment interval of a read pair, from the left-most
  aligned position to the right-most, with the start moved forward by 4 bp
  and the end back by 5 bp (Tn5 shift).
- Coordinates are 0-based with an exclusive end, as in BED.
- Read pairs with the same barcode, chromosome, start and end are duplicates
  and become one fragment. `count` is the number of read pairs behind it,
  duplicates included.
- The barcode is the corrected barcode, the `CB` tag.

Each choice is a parameter: the barcode tag (`CB`, `CR`, any tag) or a regex
on the read name, an optional barcode allow list, the minimum MAPQ, the shift,
the chromosomes to drop, and the SAM flags to reject (unmapped read or mate,
secondary, supplementary, QC fail; pairs not properly paired). Defaults that
10x does not document, such as the MAPQ threshold and the handling of soft
clips at the 5' end, are set from a comparison with Cell Ranger output
(validation, below).

**Pipeline.** Each stage runs on a chunk of the BAM in GPU memory:

1. **Decompress.** nvCOMP inflates BGZF blocks, as in `gpu_convert.py`.
2. **Find records.** BAM records are not line-based: each starts with its
   length, so their offsets form a chain, and records cross BGZF block
   boundaries. The first version walks the chain on the CPU, reading 4 bytes
   per record from pinned memory. If that becomes the bottleneck, the GPU can
   do it: every offset that passes structural checks (consistent lengths,
   reference id in range, read name NUL-terminated) is a candidate, and
   pointer jumping from the first known record keeps the true chain.
3. **Decode.** One thread per record reads the fixed fields (reference,
   position, MAPQ, flags, mate position, template length), the CIGAR for the
   alignment end, and walks the auxiliary fields for the barcode tag. Each
   record is independent.
4. **Filter.** Flags, MAPQ, chromosomes, and barcodes missing or outside the
   allow list.
5. **Pair.** For a properly paired read, the template length gives the
   fragment from the left-most mate alone (`start = pos`, `end = pos + tlen`),
   with no join on read names. That needs the mate's MAPQ, from the `MQ` tag
   when the aligner writes it. Otherwise mates are joined on the read name
   with a GPU sort. Which of the two matches Cell Ranger is part of the
   validation.
6. **Shift and deduplicate.** The shift, then a group-by on barcode,
   chromosome, start and end, whose size is `count`. In a coordinate-sorted
   BAM, fragments come out sorted by start, so duplicates are adjacent. Only
   the rows with the last start of a chunk carry over to the next.
7. **Encode.** Barcodes become ids with the GPU dictionary of the fast lane,
   and rows go to the writer.

**What comes for free.** A coordinate-sorted BAM gives `sort: "position"` and
`chrom_rows` with no extra pass. The `@SQ` lines of the header give `chroms`
and `chrom_lengths`, and `genome` when they carry an `AS` field. `source`
records the caller's parameters, the BAM's `@PG` lines (aligner and
versions) and its sort order:

```json
"source": {"kind": "bam", "path": "possorted_bam.bam", "barcode_tag": "CB",
           "min_mapq": 30, "shift": [4, -5], "pairing": "tlen",
           "pg": ["bwa mem …", "samtools sort …"]}
```

**Validation.** The reference is Cell Ranger: run the caller on a
`possorted_bam.bam` and compare with the `fragments.tsv.gz` of the same run,
on the set of (chromosome, start, end, barcode, count). The test reports every
difference, and the defaults are tuned until there is none, or each remaining
class of difference is explained and documented. The same comparison against
`snapatac2.pp.make_fragment_file` and `sinto` shows where those tools differ,
and gives the speed to beat.

**Limits.**

- Coordinate-sorted BAM first. Name-sorted input needs only the pairing step
  to change, and can follow.
- Paired-end first. Single-end data, where the fragment is an insertion
  site, can follow.
- CRAM is out of scope: decoding it needs the reference genome and codecs
  nvCOMP does not provide.
- Several BAM files for one sample (per lane) are merged by the caller, not
  converted one by one, so that duplicates across files are collapsed.

## 9. Compatibility and migration

- GATAC reads version 1 and version 2 files. Version 1 metadata is mapped to
  the object of section 4.2 with `counts: "reads"`, `barcode_order:
  "n_unique_desc"` and `sort: "none"`. Writers write version 2 only.
- `gatac upgrade file.parquet` turns a version 1 file into a version 2 file by
  rewriting its footer, as `set_barcode_prefix` does: the columns are the same.
  It scans `chrom_id`, `start` and `length` once to detect the sort and fill
  `chrom_rows`, and `count` for nulls. A file with null counts can only be
  converted, not upgraded. The footer is rewritten through the rollback
  journal (section 7.3). `gatac/io/_footer.py` needs to remove keys as well
  as set them.
- GATAC releases that only read version 1 refuse version 2 files with a clear
  message (section 4.1).

## 10. Implementation plan

Each step is a pull request on its own:

1. **Crash-safe footer updates.** The rollback journal of section 7.3 in
   `_footer.py`, with key removal, recovery on open and `gatac repair`.
   `set_barcode_prefix` switches to it. It does not depend on version 2 and
   fixes today's in-place rewrite.
2. **Format version 2 metadata.** The JSON object, the reader for versions 1
   and 2, `counts` and `barcode_order`. The writer emits version 2. Add
   `gatac upgrade`, through the journal. Read the barcode table
   with pyarrow in `FragmentFile._gpu_barcodes`, so that files from other
   writers cannot hit the cuDF string bug (section 6).
3. **Sort detection and region pruning.** Detection in both conversion lanes,
   `chrom_rows`, and row-group pruning in `FragmentFile.iter_batches(chroms=…)`
   (it reads every row group today).
4. **Generic lane.** A normalised-stream validator. DuckDB staging that takes
   any relation. Make `write_fragments` public and add `from_arrow`. TSV
   variants through `columns=`.
5. **Export.** `FragmentFile.__arrow_c_stream__`.
6. **Conformance.** This document in the docs, a JSON schema for the metadata
   object, `gatac validate`, and small conformance files (sorted, unsorted,
   counts absent, prefix, upgraded version 1).
7. **Adapters for other tools.** `snapatac2` once its layout is checked on
   real files, then `archr`, then `bpcells` if users ask. Each is an optional
   dependency.
8. **Fragment calling from BAM on the GPU** (section 8.8), in three pull
   requests: the Cell Ranger comparison test and a CPU reference
   implementation of the definition; the GPU caller for coordinate-sorted,
   paired-end BAM; then name-sorted and single-end input. It needs steps 2
   and 3 (version 2 metadata, `chrom_rows`) and is independent of steps 4
   to 7.

## 11. Open questions

1. **Name of the metadata key.** `gatac` ties the format to the tool. A
   neutral name (for example `fragments`) would suit a format meant for other
   tools, at the cost of being less obviously ours. To settle before the
   first version 2 release, since it cannot change afterwards.
2. **Count width.** Is uint16 enough for every pipeline? Raising it to uint32
   changes the logical schema, which only a new major version can do, so it
   is better decided now.
3. **Who writes the format?** Other tools reading it needs sections 2–6.
   Other tools writing it raises the bar on validation, and on whether
   `barcode_order: n_unique_desc` should be required, since it affects size
   but not correctness.
4. **Multi-sample datasets.** A directory of files plus a manifest (sample,
   path, prefix, metadata), or left to the user's sample sheet as today?
5. **Sorting.** Should `gatac sort` exist, so that any file can get
   `sort: "position"` and region queries?
6. **SnapATAC2 Zstandard on the GPU.** nvCOMP decodes Zstandard, but in
   parallel only if the file has many independent frames. Check a real
   SnapATAC2 `.tsv.zst` before deciding on a fast lane for it.
7. **Chromosome names outside the footer.** Storing the chromosome names in a
   padded column like `bc_name` would let a footer be rebuilt from the file
   alone, after the rare crash of section 7.3. Only the prefix would remain a
   single point of failure. It changes the stored schema, so it has to be
   decided before version 2 is released.

## Appendix A. Example metadata

```json
{
    "version": "2.0",
    "chroms": ["chr1", "chr2", "chrX", "chrY", "chrM"],
    "chrom_lengths": [248956422, 242193529, 156040895, 57227415, 16569],
    "genome": "GRCh38",
    "n_barcodes": 477120,
    "barcode_prefix": "s1_",
    "counts": "reads",
    "barcode_order": "n_unique_desc",
    "sort": "position",
    "chrom_rows": [
        [0, 8412233],
        [8412233, 7903512],
        [16315745, 3011247],
        [19326992, 40112],
        [19367104, 1210334]
    ],
    "source": {
        "kind": "tsv",
        "path": "runs/A/outs/atac_fragments.tsv.gz",
        "rejected_lines": 0
    },
    "writer": { "name": "gatac", "version": "0.3.0" }
}
```

## Appendix B. Reading a file without GATAC

```python
import json
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

pf = pq.ParquetFile("s1.parquet")
meta = json.loads(pf.metadata.metadata[b"gatac"])
assert meta["version"].split(".")[0] == "2"

n = meta["n_barcodes"]
names = pf.read(columns=["bc_name"])["bc_name"][:n]
barcodes = pc.binary_join_element_wise(meta["barcode_prefix"], names, "")  # barcode_id -> cell

frags = pf.read(columns=["chrom_id", "start", "length", "barcode_id", "count"])
chrom = pc.take(pa.array(meta["chroms"]), frags["chrom_id"])
end = pc.add(frags["start"], frags["length"])
cell = pc.take(barcodes, frags["barcode_id"])
```
