"""
GPU conversion of ATAC fragment TSV(.gz) files to GATAC Parquet.

:func:`_make_parquet_gpu` writes the same file as :func:`gatac.io.make_parquet`
(same ids, barcode table, layout and footer, through :class:`_GpuWriter`), but
decompresses and parses on the GPU:

1. **Decompress.** Fragment files are BGZF (bgzip, as Cell Ranger writes
   them): independent deflate blocks of at most 64 KiB of text whose sizes
   are in their headers and trailers. The compressed file is read in chunks
   and copied to the GPU; block headers are found by their signature (and
   checked by chaining each block's size to the next header), and all blocks
   of a chunk are inflated by one nvCOMP batched Deflate call. Uncompressed
   files are copied as they are. A gzip file that is not BGZF is a single
   deflate stream that cannot be split, and goes to :func:`make_parquet`.
2. **Parse.** One CUDA thread per line splits it into the five fields and
   checks them: 5 fields; start/end/count non-negative decimal integers of
   their types; non-empty chrom, start, end and barcode; lines starting with
   ``#`` are comments; an empty count is null. This is
   stricter than the DuckDB reader of :func:`make_parquet`, which also reads
   numbers such as `` 5``, ``+5`` or ``0x5``, and ends a line at any ``#``;
   the two never read different values from a line they both accept.
   Chromosome and barcode names become cuDF string columns built from the
   text on the GPU.
3. **Two passes.** The first counts fragments per barcode and records the
   chromosomes in order of first appearance; the second maps names to ids
   (barcodes by decreasing fragment count, :func:`_sort_barcode_stats`) and
   streams the rows to the writer. Decompressing twice costs far less than
   keeping every row on the GPU, and keeps memory bounded by the chunk size.

Requires cupy, cudf and nvCOMP's Python package (``nvidia-nvcomp-cu12`` /
``-cu13``) of the version cuDF's libnvcomp pins (5.x for cuDF 26.02-26.10);
nvCOMP 6's in-place batch decoding is used when available.
"""

from __future__ import annotations

import logging
import os
import zlib
from collections.abc import Iterator
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# BGZF header bytes that never vary (MTIME, XFL and OS, bytes 4-9, may):
# gzip magic, CM=deflate, FLG=FEXTRA, XLEN=6, subfield 'BC' of length 2
_BGZF_FIXED = ((0, 0x1F), (1, 0x8B), (2, 8), (3, 4), (10, 6), (11, 0), (12, 0x42), (13, 0x43), (14, 2), (15, 0))
_BGZF_HEADER, _BGZF_TRAILER = 18, 8
_BGZF_MAX_ISIZE = 1 << 16  # a block holds at most 64 KiB of text
_ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"

_KERNELS = r"""
extern "C" __global__ void copy_spans(const unsigned char* __restrict__ src,
                                      unsigned char* __restrict__ dst,
                                      const long long* __restrict__ src_off,
                                      const long long* __restrict__ dst_off,
                                      const long long* __restrict__ size,
                                      long long n) {
    // one warp per span: dst[dst_off[i] : +size[i]] = src[src_off[i] : +size[i]]
    long long w = (blockIdx.x * (long long)blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (w >= n) return;
    const unsigned char* s = src + src_off[w];
    unsigned char* d = dst + dst_off[w];
    for (long long i = lane; i < size[w]; i += 32) d[i] = s[i];
}

extern "C" __global__ void copy_from_ptrs(const unsigned long long* __restrict__ src,
                                          unsigned char* __restrict__ dst,
                                          const long long* __restrict__ dst_off,
                                          const long long* __restrict__ size,
                                          long long n) {
    // one warp per buffer: dst[dst_off[i] : +size[i]] = *src[i]
    long long w = (blockIdx.x * (long long)blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (w >= n) return;
    const unsigned char* s = (const unsigned char*)src[w];
    unsigned char* d = dst + dst_off[w];
    for (long long i = lane; i < size[w]; i += 32) d[i] = s[i];
}

__device__ long long parse_uint(const unsigned char* t, long long a, long long b,
                                unsigned long long maxval) {
    // -2: empty field, -1: not a non-negative integer within maxval
    if (a == b) return -2;
    unsigned long long v = 0;
    for (long long p = a; p < b; ++p) {
        unsigned c = t[p] - '0';
        if (c > 9) return -1;
        v = v * 10 + c;
        if (v > maxval) return -1;
    }
    return (long long)v;
}

// status: 0 fragment, 1 comment or blank line, 2 not 5 fields of the right
// types, 3 empty chrom/start/end/barcode, 4 fragment with an empty count
extern "C" __global__ void parse_lines(const unsigned char* __restrict__ t,
                                       const long long* __restrict__ ls,
                                       const long long* __restrict__ le,
                                       long long n, unsigned char sep,
                                       long long* chrom_off, long long* chrom_len,
                                       long long* bc_off, long long* bc_len,
                                       unsigned int* start, unsigned int* end,
                                       unsigned int* count, signed char* status) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= n) return;
    long long a = ls[i], b = le[i];
    if (b > a && t[b - 1] == '\r') --b;
    if (a == b || t[a] == '#') { status[i] = 1; return; }
    long long f[5]; int nsep = 0; f[0] = a;
    for (long long p = a; p < b; ++p)
        if (t[p] == sep) { if (++nsep > 4) break; f[nsep] = p + 1; }
    if (nsep != 4) { status[i] = 2; return; }
    long long e[5] = {f[1] - 1, f[2] - 1, f[3] - 1, f[4] - 1, b};
    long long s = parse_uint(t, f[1], e[1], 4294967295ULL);
    long long en = parse_uint(t, f[2], e[2], 4294967295ULL);
    long long c = parse_uint(t, f[4], e[4], 65535ULL);
    if (s == -1 || en == -1 || c == -1) { status[i] = 2; return; }
    if (s == -2 || en == -2 || e[0] == f[0] || e[3] == f[3]) { status[i] = 3; return; }
    chrom_off[i] = f[0]; chrom_len[i] = e[0] - f[0];
    bc_off[i] = f[3]; bc_len[i] = e[3] - f[3];
    start[i] = (unsigned int)s; end[i] = (unsigned int)en;
    count[i] = c < 0 ? 0u : (unsigned int)c;
    status[i] = c == -2 ? 4 : 0;
}
"""
_module = None


def _kernel(name: str):
    import cupy as cp

    global _module
    if _module is None:
        _module = cp.RawModule(code=_KERNELS)
    return _module.get_function(name)


def _copy_spans(src, dst, src_off, dst_off, size) -> None:
    import cupy as cp

    n = int(size.size)
    if n:
        _kernel("copy_spans")(
            ((n * 32 + 255) // 256,),
            (256,),
            (
                src,
                dst,
                cp.asarray(src_off, cp.int64),
                cp.asarray(dst_off, cp.int64),
                cp.asarray(size, cp.int64),
                np.int64(n),
            ),
        )


def _nvcomp():
    # Some environments ship a regular ``nvidia/__init__.py`` that hides other
    # ``nvidia.*`` installs on sys.path: extend the package path if so
    import sys

    import nvidia

    for p in sys.path:
        cand = os.path.join(p, "nvidia")
        if os.path.isdir(os.path.join(cand, "nvcomp")) and cand not in nvidia.__path__:
            nvidia.__path__.append(cand)
    from nvidia import nvcomp

    return nvcomp


def _gpu_available() -> bool:
    """Whether :func:`_make_parquet_gpu` can run here (cupy, cudf and nvCOMP import)."""
    try:
        import cudf  # noqa: F401
        import cupy  # noqa: F401

        _nvcomp()
        return True
    except Exception:  # noqa: BLE001 - any failure to load nvCOMP means no GPU route
        return False


def _input_kind(path: Path) -> str:
    """Return the kind of an input file, from the first bytes.

    ``"bgzf"`` or ``"plain"`` text is read on the GPU, ``"gzip"`` (a single
    deflate stream) or ``"zstd"`` by DuckDB.
    """
    with open(path, "rb") as fh:
        head = fh.read(16)
    if head[:2] == b"\x1f\x8b":
        return "bgzf" if len(head) == 16 and all(head[i] == v for i, v in _BGZF_FIXED) else "gzip"
    return "zstd" if head[:4] == _ZSTD_MAGIC else "plain"


def _cpu_reason(kind: str) -> str | None:
    """Why a file goes to DuckDB instead of the GPU (None: it does not)."""
    if kind == "gzip":
        return "it is gzip but not BGZF (one deflate stream, which cannot be split)"
    if kind == "zstd":
        return "it is Zstandard compressed"
    return None


# ---------------------------------------------------------------------------
# 1. Decompression: text chunks on the GPU
# ---------------------------------------------------------------------------


def _bgzf_blocks(buf, eof: bool):
    """Find the complete BGZF blocks in ``buf``, a cupy uint8 array starting at a block.

    Returns the starts, sizes and uncompressed sizes (numpy int64) of the
    blocks, and the end of the last one.
    """
    import cupy as cp

    n = int(buf.size)
    if n < _BGZF_HEADER:
        return (np.empty(0, np.int64),) * 3 + (0,)
    m = cp.ones(n - 15, dtype=cp.bool_)
    for i, v in _BGZF_FIXED:
        m &= buf[i : n - 15 + i] == v
    cand = cp.flatnonzero(m)
    cand = cand[cand + _BGZF_HEADER <= n]
    bsize = buf[cand + 16].astype(cp.int64) + (buf[cand + 17].astype(cp.int64) << 8) + 1
    cand, bsize = cand.get(), bsize.get()
    if cand.size == 0 or cand[0] != 0:
        raise ValueError("Not a BGZF block where one was expected: the file is corrupt or truncated")
    nxt = cand + bsize
    if cand.size > 1 and not np.array_equal(nxt[:-1], cand[1:]):
        # A header signature inside compressed data: follow the chain instead
        size_at = dict(zip(cand.tolist(), bsize.tolist()))
        keep, p = [], 0
        while p in size_at:
            keep.append(p)
            p += size_at[p]
        cand = np.array(keep, np.int64)
        bsize = np.array([size_at[p] for p in keep], np.int64)
        nxt = cand + bsize
    complete = nxt <= n
    cand, bsize = cand[complete], bsize[complete]
    if cand.size == 0:
        if eof and n:
            raise ValueError("Truncated BGZF file: its last block is incomplete")
        return (np.empty(0, np.int64),) * 3 + (0,)
    tail = cand + bsize - 4  # ISIZE, little-endian uint32, ends each block
    idx = cp.asarray(tail[:, None] + np.arange(4))
    isize = (buf[idx].astype(cp.int64) << cp.asarray([0, 8, 16, 24])).sum(axis=1).get()
    return cand, bsize, isize, int(cand[-1] + bsize[-1])


def _inflate_blocks(buf, starts, bsize, isize, codec, nvcomp):
    """Inflate BGZF blocks of ``buf`` on the GPU into one text buffer."""
    import cupy as cp

    if (bsize < _BGZF_HEADER + _BGZF_TRAILER).any() or (isize > _BGZF_MAX_ISIZE).any():
        raise _corrupt_block_error()
    empty = isize == 0  # the EOF marker, or a block whose trailer is corrupt
    for s, b in zip(starts[empty].tolist(), bsize[empty].tolist()):
        _check_empty_block(buf[s + _BGZF_HEADER : s + b - _BGZF_TRAILER].get().tobytes())
    starts, bsize, isize = starts[~empty], bsize[~empty], isize[~empty]
    if starts.size == 0:
        return cp.empty(0, cp.uint8)
    pay = bsize - _BGZF_HEADER - _BGZF_TRAILER
    # nvCOMP wants each compressed input 4-byte aligned: repack the payloads
    aligned_off = np.zeros(pay.size, np.int64)
    aligned_off[1:] = np.cumsum((pay + 7) // 8 * 8)[:-1]
    aligned = cp.empty(int(aligned_off[-1] + pay[-1] + 8), cp.uint8)
    _copy_spans(buf, aligned, starts + _BGZF_HEADER, aligned_off, pay)
    out_off = np.zeros(isize.size, np.int64)
    out_off[1:] = np.cumsum(isize)[:-1]
    text = cp.empty(int(isize.sum()), cp.uint8)
    stream = cp.cuda.get_current_stream()
    if _nvcomp_major(nvcomp) >= 6:
        # nvCOMP 6: batches carved from one buffer, decoded in place
        srcs = nvcomp.as_arrays(aligned, aligned_off.tolist(), pay.tolist(), stream.ptr)
        outs = nvcomp.as_arrays(text, out_off.tolist(), isize.tolist(), stream.ptr)
        try:
            codec.decode(srcs, out=outs)
        except (ValueError, RuntimeError) as e:  # a block inflates past its trailer's size
            raise _corrupt_block_error(e) from None
        _check_inflated_sizes(np.asarray(outs.buffer_sizes, np.int64), isize)
        return text
    # nvCOMP 5 (what cuDF 26.02-26.10 pin): a list of buffers in, new
    # buffers out, gathered into the text by pointer
    base = aligned.data.ptr
    views = [_DeviceSpan(base + o, p) for o, p in zip(aligned_off.tolist(), pay.tolist())]
    try:
        outs = codec.decode(nvcomp.as_arrays(views, stream.ptr))
        ifaces = [o.__cuda_array_interface__ for o in outs]
    except (ValueError, RuntimeError) as e:  # a block that does not inflate has no buffer
        raise _corrupt_block_error(e) from None
    del views
    _check_inflated_sizes(np.fromiter((i["shape"][0] for i in ifaces), np.int64, len(ifaces)), isize)
    ptrs = cp.asarray(np.fromiter((i["data"][0] for i in ifaces), np.uint64, len(ifaces)))
    n = len(ifaces)
    _kernel("copy_from_ptrs")(
        ((n * 32 + 255) // 256,), (256,), (ptrs, text, cp.asarray(out_off), cp.asarray(isize), np.int64(n))
    )
    stream.synchronize()  # before the decoded buffers are freed
    return text


class _DeviceSpan:
    """A bare ``__cuda_array_interface__`` over device bytes.

    Spans are far cheaper to make by the hundred thousand than cupy slices.
    The caller keeps the underlying buffer alive.
    """

    __slots__ = ("__cuda_array_interface__",)

    def __init__(self, ptr: int, nbytes: int):
        self.__cuda_array_interface__ = {
            "shape": (nbytes,),
            "typestr": "|u1",
            "data": (ptr, True),
            "version": 3,
            "strides": None,
        }


def _nvcomp_major(nvcomp) -> int:
    """Major version of nvCOMP's Python package (6 decodes a batch in place)."""
    return int(nvcomp.__version__.split(".")[0])


def _corrupt_block_error(cause: Exception | None = None) -> ValueError:
    why = f" ({cause})" if cause else ""
    return ValueError(f"Corrupt BGZF block: it does not inflate to the size in its trailer{why}")


def _check_empty_block(payload: bytes) -> None:
    """A block whose trailer gives no text must inflate to nothing (on the CPU: there is about one per file)."""
    d = zlib.decompressobj(-15)
    try:
        out = d.decompress(payload, 1)
    except zlib.error as e:
        raise _corrupt_block_error(e) from None
    if out or not d.eof:
        raise _corrupt_block_error()


def _check_inflated_sizes(got: np.ndarray, isize: np.ndarray) -> None:
    """Inflated sizes against the trailers': a corrupt block can inflate to fewer bytes."""
    if not np.array_equal(got, isize):
        raise _corrupt_block_error()


def _text_chunks(path: Path, kind: str, chunk_bytes: int) -> Iterator:
    """Text of ``path`` on the GPU (cupy uint8), in chunks of whole lines."""
    import cupy as cp

    codec = nvcomp = None
    if kind == "bgzf":
        nvcomp = _nvcomp()
        codec = nvcomp.Codec(
            algorithm="Deflate", bitstream_kind=nvcomp.BitstreamKind.RAW, cuda_stream=cp.cuda.get_current_stream().ptr
        )
    carry = np.empty(0, np.uint8)  # compressed bytes of the incomplete last block
    tail = cp.empty(0, cp.uint8)  # text of the incomplete last line
    with open(path, "rb") as fh:
        while True:
            raw = fh.read(chunk_bytes)
            eof = len(raw) < chunk_bytes
            host = np.concatenate([carry, np.frombuffer(raw, np.uint8)]) if carry.size else np.frombuffer(raw, np.uint8)
            if kind == "bgzf":
                buf = cp.asarray(host)
                starts, bsize, isize, used = _bgzf_blocks(buf, eof)
                text = _inflate_blocks(buf, starts, bsize, isize, codec, nvcomp)
                carry = host[used:].copy()
                del buf
            else:
                text = cp.asarray(host)
            if tail.size:
                text = cp.concatenate([tail, text])
            if eof:
                if carry.size:  # before the text, whose last line is cut
                    raise ValueError("Truncated BGZF file: its last block is incomplete")
                if text.size and int(text[-1]) != 10:
                    text = cp.concatenate([text, cp.asarray([10], cp.uint8)])
                if text.size:
                    yield text
                return
            nl = cp.flatnonzero(text == 10)
            if nl.size == 0:
                tail = text
                continue
            cut = int(nl[-1]) + 1
            tail = text[cut:].copy()
            yield text[:cut]


# ---------------------------------------------------------------------------
# 2. Parsing
# ---------------------------------------------------------------------------


def _strings(text, off, length):
    """A cuDF string column of ``text[off[i] : off[i] + length[i]]``."""
    import cudf
    import cupy as cp
    import pylibcudf as plc

    n = int(off.size)
    offsets = cp.zeros(n + 1, cp.int64)
    cp.cumsum(length, out=offsets[1:])
    total = int(offsets[-1])
    chars = cp.empty(max(total, 1), cp.uint8)
    _copy_spans(text, chars, off, offsets[:-1], length)
    if total < 2**31 - 1:
        offsets = offsets.astype(cp.int32)
    col = plc.Column(
        plc.DataType(plc.TypeId.STRING),
        n,
        plc.gpumemoryview(chars),
        None,
        0,
        0,
        [plc.Column.from_cuda_array_interface(offsets)],
    )
    return cudf.Series.from_pylibcudf(col)


def _bad_line_error(name: str, line: str, sep: str) -> ValueError:
    """The error for a line that is not a fragment, worded as DuckDB's in make_parquet."""
    fields = line.split(sep)
    if len(fields) != 5:
        why = f"Expected Number of Columns: 5 Found: {len(fields)}"
    else:
        limits = {1: 2**32 - 1, 2: 2**32 - 1, 4: 2**16 - 1}
        bad = next(
            fields[i] for i, top in limits.items() if fields[i] and not (fields[i].isdigit() and int(fields[i]) <= top)
        )
        why = f'Could not convert string "{bad}"'
    hint = (
        ""
        if len(fields) != 5
        else "; or gpu=False (gatac convert --cpu): DuckDB reads numbers more loosely (' 5', '+5', '0x5')"
    )
    return ValueError(
        f"{name} has a line that is not a fragment (chrom, start, end, barcode, count): {why}: "
        f"{line!r}. Pass skip_invalid=True (gatac convert --skip-invalid) to skip such lines; "
        f"their number is logged{hint}"
    )


def _parse(text, sep: str, name: str, skip_invalid: bool) -> tuple[dict, int, int]:
    """Parse the fragments of a chunk of whole lines.

    Returns the columns (cupy / cuDF) and the numbers of lines rejected (not 5
    fields of the right types) and with a missing field.
    """
    import cudf
    import cupy as cp

    nl = cp.flatnonzero(text == 10)
    n = int(nl.size)
    ls = cp.empty(n, cp.int64)
    ls[0] = 0
    ls[1:] = nl[:-1] + 1
    out = {k: cp.empty(n, cp.int64) for k in ("chrom_off", "chrom_len", "bc_off", "bc_len")}
    out.update({k: cp.empty(n, cp.uint32) for k in ("start", "end", "count")})
    status = cp.empty(n, cp.int8)
    _kernel("parse_lines")(
        ((n + 255) // 256,),
        (256,),
        (
            text,
            ls,
            nl,
            np.int64(n),
            np.uint8(ord(sep)),
            out["chrom_off"],
            out["chrom_len"],
            out["bc_off"],
            out["bc_len"],
            out["start"],
            out["end"],
            out["count"],
            status,
        ),
    )
    n_rejected = int((status == 2).sum())
    n_missing = int((status == 3).sum())
    if (n_rejected or n_missing) and not skip_invalid:
        first = int(cp.flatnonzero((status == 2) | (status == 3))[0])
        a, b = int(ls[first]), int(nl[first])
        line = text[a:b].get().tobytes().decode(errors="replace").rstrip("\r")
        if int(status[first]) == 2:
            raise _bad_line_error(name, line, sep)
        raise ValueError(
            f"{name}: line(s) have an empty chrom, start, end or barcode, e.g. {line!r}. Pass "
            "skip_invalid=True (gatac convert --skip-invalid) to skip them"
        )
    keep = cp.flatnonzero((status == 0) | (status == 4))
    count = cudf.Series(out["count"][keep])
    null_count = status[keep] == 4
    if bool(null_count.any()):
        count[null_count] = None
    cols = {
        "chrom": _strings(text, out["chrom_off"][keep], out["chrom_len"][keep]),
        "start": out["start"][keep],
        "end": out["end"][keep],
        "barcode": _strings(text, out["bc_off"][keep], out["bc_len"][keep]),
        "count": count,
    }
    return cols, n_rejected, n_missing


def _fragment_chunks(path: Path, kind: str, sep: str, skip_invalid: bool, chunk_bytes: int, tally: dict):
    """Parsed fragment chunks of ``path``; ``tally`` sums the skipped lines."""
    for text in _text_chunks(path, kind, chunk_bytes):
        cols, n_rejected, n_missing = _parse(text, sep, path.name, skip_invalid)
        del text
        tally["rejected"] += n_rejected
        tally["missing"] += n_missing
        if len(cols["start"]):
            yield cols


# ---------------------------------------------------------------------------
# 3. Conversion
# ---------------------------------------------------------------------------


def _ids(names, index):
    """Position of each of ``names`` (cuDF strings) in ``index``, which holds them all."""
    codes, uniques = names.factorize()
    lut = index.get_indexer(uniques)
    lut = lut.values if hasattr(lut, "values") else lut
    if bool((lut < 0).any()):
        raise ValueError("A name of the second pass is missing from the first")
    return lut[codes.values if hasattr(codes, "values") else codes]


def _default_gpu_workers() -> int:
    """Return the default number of conversions that share the GPU.

    The number is at most :func:`gatac._workers.default_workers` (at most 4,
    at most the usable CPUs minus one), and at most one per 7 GB of free GPU
    memory (the peak of one conversion with the default chunk size, much of
    it independent of the file and chunk sizes).
    """
    import cupy as cp

    from gatac._workers import default_workers

    free = cp.cuda.Device().mem_info[0]
    return max(1, min(default_workers(), int(free // (7 << 30))))


def _make_parquet_gpu(
    input_path: str | Path,
    output_path: str | Path | None = None,
    separator: str = "\t",
    barcode_prefix: str | None = None,
    row_group_size: int = 1_000_000,
    skip_invalid: bool = False,
    chunk_bytes: int = 256 << 20,
) -> Path:
    """
    Convert an ATAC fragments TSV.GZ file to GATAC Parquet on the GPU.

    Writes the same file as :func:`make_parquet` (see the module docstring for
    how). BGZF and uncompressed inputs are read on the GPU; a gzip file that
    is not BGZF (it cannot be split) and a Zstandard file go to DuckDB, as
    with ``make_parquet(gpu=False)``.

    Parameters
    ----------
    input_path, output_path, separator, barcode_prefix, row_group_size, skip_invalid
        As in :func:`make_parquet`. cuDF's writer lays row groups out in
        5,000-row page fragments, so a ``row_group_size`` that is not a
        multiple of 5,000 is rounded (the default 1,000,000 is kept).
    chunk_bytes : int
        Bytes of the input read per chunk (default 256 MiB). GPU memory use
        grows with it: a BGZF chunk holds about four times as much text.

    Returns
    -------
    Path
        Path to the created Parquet file.
    """
    from .convert import _fill, _make_parquet_duckdb, _stem

    input_path = Path(input_path)
    output_path = input_path.with_suffix("").with_suffix(".parquet") if output_path is None else Path(output_path)
    if output_path.resolve() == input_path.resolve():
        raise ValueError(f"output_path must differ from the input: {output_path}")
    if len(separator.encode()) != 1:
        raise ValueError("separator must be a single ASCII character on the GPU (gpu=False takes others)")
    if chunk_bytes < 1:
        raise ValueError(f"chunk_bytes must be positive, not {chunk_bytes}")
    kind = _input_kind(input_path)
    reason = _cpu_reason(kind)
    if reason:
        logger.info(f"{input_path.name}: {reason}; converting it with DuckDB on the CPU")
        return _make_parquet_duckdb(input_path, output_path, separator, barcode_prefix, row_group_size, skip_invalid)
    from gatac.pp._utils import cleanup_gpu_memory  # after the fallback: imports RAPIDS

    barcode_prefix = _fill(barcode_prefix, _stem(input_path))
    logger.info(f"Converting {input_path.name} to Parquet on the GPU ({kind})")
    # No RMM async pool here (unlike compute_metrics / filter_fragments): it
    # keeps freed memory next to CuPy's own pool, which doubles the peak GPU
    # memory of a conversion, for little speed, and runs out of memory when
    # several conversions share the GPU.
    try:
        _convert(input_path, output_path, kind, separator, barcode_prefix, row_group_size, skip_invalid, chunk_bytes)
    finally:
        cleanup_gpu_memory()
    logger.info(f"Created {output_path.name}")
    return output_path


def _convert(input_path, output_path, kind, separator, barcode_prefix, row_group_size, skip_invalid, chunk_bytes):
    """Body of :func:`_make_parquet_gpu` for BGZF and uncompressed inputs."""
    import cudf
    import cupy as cp
    import pandas as pd

    from .fragments import _GpuWriter, _sort_barcode_stats

    # Pass 1: chromosomes in order of first appearance, fragments per barcode
    chroms: list[str] = []
    partial, n_partial = [], 0
    tally = {"rejected": 0, "missing": 0}
    for cols in _fragment_chunks(input_path, kind, separator, skip_invalid, chunk_bytes, tally):
        first = (
            cudf.DataFrame({"chrom": cols["chrom"], "row": cp.arange(len(cols["start"]))})
            .groupby("chrom", sort=False)["row"]
            .min()
            .sort_values()
        )
        seen = set(chroms)
        chroms += [c for c in first.index.to_arrow().to_pylist() if c not in seen]
        df = cudf.DataFrame({"barcode": cols["barcode"], "count": cols["count"]})
        g = df.groupby("barcode", sort=False)
        partial.append(cudf.DataFrame({"n_unique": g.size(), "n_total": g["count"].sum()}))
        n_partial += 1
        if n_partial >= 16:  # bound the GPU memory of the partial counts
            partial, n_partial = [cudf.concat(partial).groupby(level=0, sort=False).sum()], 1
        del cols, df, g
    if tally["rejected"] or tally["missing"]:
        logger.warning(
            f"{input_path.name}: skipped {tally['rejected'] + tally['missing']:,} invalid line(s) "
            f"({tally['rejected']:,} not 5 fields of the right types, {tally['missing']:,} with an "
            "empty chrom, start, end or barcode)"
        )
    if len(chroms) > 65535:
        raise ValueError("More than 65,535 chromosomes are not supported (chrom_id is uint16)")
    if partial:
        counts = cudf.concat(partial).groupby(level=0, sort=False).sum().fillna(0)
        stats = counts.reset_index().rename(columns={"index": "barcode"}).to_pandas()
        stats.columns = ["barcode", "n_unique", "n_total"]
    else:
        stats = pd.DataFrame({"barcode": pd.Series([], dtype=object), "n_unique": [], "n_total": []})
    stats = _sort_barcode_stats(stats.astype({"n_unique": np.int64, "n_total": np.int64}))
    del partial

    # Pass 2: ids, lengths, and the rows to the writer
    chrom_index = cudf.Index(cudf.Series(chroms, dtype="str"))
    bc_index = cudf.Index(cudf.Series(stats["barcode"].to_numpy(dtype=object), dtype="str"))
    tmp_path = output_path.with_name(output_path.name + ".tmp")
    try:
        writer = _GpuWriter(tmp_path, chroms, stats, row_group_size, barcode_prefix)
        tally = {"rejected": 0, "missing": 0}
        for cols in _fragment_chunks(input_path, kind, separator, True, chunk_bytes, tally):
            start, end = cols["start"], cols["end"]
            if bool((end < start).any()):
                raise ValueError("A fragment ends before it starts (end < start)")
            df = cudf.DataFrame(
                {
                    "chrom_id": _ids(cols["chrom"], chrom_index).astype(np.uint16),
                    "start": start,
                    "length": end - start,
                    "barcode_id": _ids(cols["barcode"], bc_index).astype(np.uint32),
                    "count": cols["count"],
                }
            )
            writer.write(df)
            del cols, df
        writer.close()
        os.replace(tmp_path, output_path)
    finally:
        tmp_path.unlink(missing_ok=True)
