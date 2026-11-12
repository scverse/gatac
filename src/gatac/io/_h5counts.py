"""
Count matrices of h5ad files read straight into GPU memory.

:func:`open_counts` reads a CSR matrix (``X`` by default) of an ``.h5ad``
file, a backed AnnData or a backed view into a
:class:`~gatac.dr._stream.StreamedCounts`; :func:`iter_row_chunks` yields it
in row chunks. The bytes are read with kvikio's threads and, for gzip files,
inflated on the GPU by nvCOMP, so the matrix never goes through a host copy
in scipy's layout and its column indices are int32 whatever the file holds.

Supported HDF5 layouts (anything else raises before any read):

- contiguous, no filters;
- chunked, no filters, or deflate (gzip), or byte shuffle then deflate;
- compact (small datasets, read through h5py).

Every deflate chunk's zlib header, inflated size and Adler-32 are checked,
so a corrupt chunk raises instead of giving wrong counts.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

from gatac._rapids import INSTALL_HINT

if TYPE_CHECKING:
    from types import ModuleType

    import nvidia.nvcomp

    from gatac.dr._stream import StreamedCounts

logger = logging.getLogger(__name__)

_FILTER_DEFLATE, _FILTER_SHUFFLE = 1, 2
_SUPPORTED_FILTERS: tuple[list[int], ...] = ([], [_FILTER_DEFLATE], [_FILTER_SHUFFLE, _FILTER_DEFLATE])
_VALUE_DTYPES = {np.dtype(t) for t in (np.uint8, np.uint16, np.int32, np.float32)}
_INT32_MAX = np.iinfo(np.int32).max
_ZLIB_FRAMING = 6  # 2-byte header + 4-byte Adler-32 around the raw Deflate body
_MIN_BLOCK_ENTRIES = 1 << 24  # entries per block over the GPU budget, at the least


# ---------------------------------------------------------------------------
# kvikio and nvCOMP
# ---------------------------------------------------------------------------

_threads_set = False


def _kvikio():
    """kvikio, with its thread pool sized to this process's CPUs unless ``KVIKIO_NTHREADS`` is set."""
    global _threads_set
    # cuDF loads the libkvikio and libnvcomp it pins: import it first
    import cudf  # noqa: F401

    try:
        import kvikio
        import kvikio.defaults
    except ModuleNotFoundError as e:
        raise ImportError(f"Reading h5ad files on the GPU needs kvikio, which is missing ({e}). {INSTALL_HINT}") from e
    if not _threads_set:
        # cuDF's import leaves libkvikio at one thread; the environment
        # variable is read only at libkvikio's first use, so it is set here
        if "KVIKIO_NTHREADS" not in os.environ:
            kvikio.defaults.set("num_threads", len(os.sched_getaffinity(0)))
        _threads_set = True
    return kvikio


# ---------------------------------------------------------------------------
# Dataset layouts
# ---------------------------------------------------------------------------


@dataclass
class _Layout:
    """Where the bytes of a 1-D dataset are in the file."""

    name: str
    dtype: np.dtype
    n: int
    kind: str  # "empty", "contiguous", "chunked" or "compact"
    offset: int = 0  # contiguous: byte offset of the first element
    chunk_len: int = 0  # chunked: elements per chunk
    chunk_offsets: np.ndarray | None = None  # chunked: file offset of chunk i, in logical order
    chunk_sizes: np.ndarray | None = None  # chunked: stored bytes of chunk i
    compressed: bool = False
    shuffled: bool = False
    values: np.ndarray | None = None  # compact: the values, read through h5py

    def chunks(self) -> tuple[np.ndarray, np.ndarray]:
        """``chunk_offsets`` and ``chunk_sizes``, set for a chunked dataset."""
        assert self.chunk_offsets is not None and self.chunk_sizes is not None
        return self.chunk_offsets, self.chunk_sizes


def _filter_names(plist) -> list[str]:
    return [
        plist.get_filter(i)[3].decode(errors="replace") or str(plist.get_filter(i)[0])
        for i in range(plist.get_nfilters())
    ]


def _inspect(ds, name: str) -> _Layout:
    """The layout of the 1-D dataset ``ds``, or a ValueError if it cannot be read on the GPU."""
    import h5py

    if ds.ndim != 1:
        raise ValueError(f"{name} is not 1-D (shape {ds.shape})")
    dtype = ds.dtype
    if dtype.kind not in "uif":
        raise ValueError(f"{name} holds {dtype}, not numbers")
    if not dtype.isnative:
        raise ValueError(f"{name} is stored in non-native byte order ({dtype.str}); rewrite the file with anndata")
    n = int(ds.shape[0])
    if n == 0:
        # a valid empty dataset has no storage: offset None or no chunks
        return _Layout(name, dtype, 0, "empty")
    plist = ds.id.get_create_plist()
    if plist.get_external_count():
        raise ValueError(f"{name} is stored in external files")
    layout = plist.get_layout()
    if layout == h5py.h5d.COMPACT:
        return _Layout(name, dtype, n, "compact", values=ds[()])
    if layout == h5py.h5d.CONTIGUOUS:
        offset = ds.id.get_offset()
        if offset is None:
            raise ValueError(f"{name} has {n:,} entries but no storage: it was never written")
        return _Layout(name, dtype, n, "contiguous", offset=int(offset))
    if layout != h5py.h5d.CHUNKED:
        raise ValueError(f"{name} has an HDF5 layout the GPU reader does not support ({layout})")

    filters = [plist.get_filter(i)[0] for i in range(plist.get_nfilters())]
    if filters not in _SUPPORTED_FILTERS:
        raise ValueError(
            f"{name} is compressed with the HDF5 filters {_filter_names(plist)}; the GPU reader reads gzip "
            "(deflate), with or without shuffle, or no compression. Rewrite the file, e.g. with "
            "anndata's write_h5ad(compression='gzip')"
        )
    chunk_len = int(ds.chunks[0])
    n_chunks = -(-n // chunk_len)
    infos: list = []
    if hasattr(ds.id, "chunk_iter"):
        ds.id.chunk_iter(infos.append)
    else:
        infos = [ds.id.get_chunk_info(i) for i in range(ds.id.get_num_chunks())]
    if len(infos) != n_chunks:
        raise ValueError(
            f"{name} has {n_chunks - len(infos):,} of its {n_chunks:,} chunks never written (they would read as the "
            "fill value)"
        )
    starts = np.fromiter((i.chunk_offset[0] for i in infos), np.int64, n_chunks)
    if (starts % chunk_len).any() or not np.array_equal(np.sort(starts), np.arange(n_chunks) * chunk_len):
        raise ValueError(f"{name} has chunks at unexpected positions")
    if any(i.filter_mask for i in infos):
        raise ValueError(
            f"{name} has chunks stored with a filter skipped (HDF5 does that when compression fails); "
            "the GPU reader does not read them. Rewrite the file"
        )
    order = np.argsort(starts)
    offsets = np.fromiter((i.byte_offset for i in infos), np.int64, n_chunks)[order]
    sizes = np.fromiter((i.size for i in infos), np.int64, n_chunks)[order]
    chunk_bytes = chunk_len * dtype.itemsize
    compressed = bool(filters)
    if not compressed and (sizes != chunk_bytes).any():
        raise ValueError(f"{name} has uncompressed chunks of the wrong size")
    if compressed and (sizes <= _ZLIB_FRAMING).any():
        raise ValueError(f"{name} has deflate chunks too small to be zlib streams")
    return _Layout(
        name,
        dtype,
        n,
        "chunked",
        chunk_len=chunk_len,
        chunk_offsets=offsets,
        chunk_sizes=sizes,
        compressed=compressed,
        shuffled=_FILTER_SHUFFLE in filters and dtype.itemsize > 1,
    )


# ---------------------------------------------------------------------------
# Reading element ranges onto the GPU
# ---------------------------------------------------------------------------


class _Reader:
    """Reads element ranges of datasets of one file into device arrays, decoding on the GPU."""

    def __init__(self, path: str | Path, *, block_bytes: int, max_decoded_bytes: int):
        self.path = str(path)
        self.fh = _kvikio().CuFile(self.path, "r")
        self.block_bytes = int(block_bytes)
        self.max_decoded_bytes = int(max_decoded_bytes)
        self._codec: nvidia.nvcomp.Codec | None = None
        self._nvcomp: ModuleType | None = None

    def close(self) -> None:
        self.fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _pread(self, buf, nbytes: int, offset: int) -> None:
        got = self.fh.pread(buf, nbytes, offset).get()
        if got != nbytes:
            raise OSError(f"{self.path}: read {got:,} of {nbytes:,} bytes at offset {offset:,}; is the file truncated?")

    def read(self, lay: _Layout, e0: int, e1: int, dst) -> None:
        """``dst[:] = dataset[e0:e1]``, cast to ``dst``'s dtype on the GPU."""
        import cupy as cp

        if e1 <= e0:
            return
        if lay.kind == "compact":
            assert lay.values is not None
            dst[:] = cp.asarray(lay.values[e0:e1])
        elif lay.kind == "contiguous":
            self._read_contiguous(lay, e0, e1, dst)
        elif lay.kind == "chunked":
            self._read_chunked(lay, e0, e1, dst)
        else:  # pragma: no cover - empty datasets are never read
            raise AssertionError(lay.kind)

    def _read_contiguous(self, lay: _Layout, e0: int, e1: int, dst) -> None:
        import cupy as cp

        isz = lay.dtype.itemsize
        stream = cp.cuda.get_current_stream()
        if dst.dtype == lay.dtype and dst.flags.c_contiguous:
            # kvikio writes outside the stream: no queued kernel may still use this memory
            stream.synchronize()
            self._pread(dst, (e1 - e0) * isz, lay.offset + e0 * isz)
            return
        step = max(1, self.block_bytes // isz)
        stage = cp.empty(min(step, e1 - e0), lay.dtype)
        for a in range(e0, e1, step):
            b = min(a + step, e1)
            stream.synchronize()  # the previous cast is done with ``stage``
            self._pread(stage, (b - a) * isz, lay.offset + a * isz)
            dst[a - e0 : b - e0] = stage[: b - a]
        stream.synchronize()

    def _blocks(self, lay: _Layout, c0: int, c1: int) -> Iterator[tuple[int, int]]:
        """Yield the runs of chunks that are read with one pread each.

        A run is in file order, at most ``block_bytes`` apart in the file and
        ``max_decoded_bytes`` decoded.
        """
        offs, sizes = lay.chunks()
        max_k = max(1, self.max_decoded_bytes // (lay.chunk_len * lay.dtype.itemsize))
        start = c0
        for c in range(c0 + 1, c1):
            if c - start >= max_k or offs[c] < offs[c - 1] or offs[c] + sizes[c] - offs[start] > self.block_bytes:
                yield start, c
                start = c
        yield start, c1

    def _read_chunked(self, lay: _Layout, e0: int, e1: int, dst) -> None:
        import cupy as cp

        from .gpu_convert import _copy_spans

        L, isz = lay.chunk_len, lay.dtype.itemsize
        chunk_offsets, chunk_sizes = lay.chunks()
        cb = L * isz
        stream = cp.cuda.get_current_stream()
        for ca, cz in self._blocks(lay, e0 // L, -(-e1 // L)):
            k = cz - ca
            offs = chunk_offsets[ca:cz]
            lo = int(offs.min())
            span = int((offs + chunk_sizes[ca:cz]).max()) - lo
            stream.synchronize()  # the previous block's kernels are done with the pool's buffers
            comp = cp.empty(span, cp.uint8)
            self._pread(comp, span, lo)
            rel = offs - lo
            if lay.compressed:
                dec = self._inflate(lay, ca, comp, rel, chunk_sizes[ca:cz], cb)
            else:
                dec = cp.empty(k * cb, cp.uint8)
                _copy_spans(comp, dec, rel, np.arange(k, dtype=np.int64) * cb, np.full(k, cb, np.int64))
            del comp
            if lay.shuffled:
                # HDF5's shuffle stores byte b of every element of a chunk together
                dec = cp.ascontiguousarray(dec.reshape(k, isz, L).transpose(0, 2, 1)).reshape(-1)
            vals = dec.view(lay.dtype)
            s, t = max(e0, ca * L), min(e1, cz * L)
            dst[s - e0 : t - e0] = vals[s - ca * L : t - ca * L]
            del dec, vals
        stream.synchronize()

    def _inflate(self, lay: _Layout, first_chunk: int, comp, rel: np.ndarray, sizes: np.ndarray, cb: int):
        """Inflate the zlib streams ``comp[rel[i] : rel[i] + sizes[i]]``, checked, into ``k`` chunks of ``cb`` bytes."""
        import cupy as cp

        from . import _zlib_gpu
        from .gpu_convert import _copy_spans, _DeviceSpan, _kernel, _nvcomp, _nvcomp_major

        k = len(rel)

        def corrupt(i: int, why: str) -> ValueError:
            return ValueError(f"{lay.name}: deflate chunk {first_chunk + i:,} of {self.path} is corrupt ({why})")

        head = comp[cp.asarray(np.concatenate([rel, rel + 1]))].get()
        bad = np.flatnonzero(~_zlib_gpu.check_headers(head[:k], head[k:]))
        if bad.size:
            raise corrupt(int(bad[0]), "not a zlib header HDF5's deflate filter writes")
        trailer = comp[cp.asarray((rel + sizes - 4)[:, None] + np.arange(4))].get().astype(np.uint32)
        expected = (trailer[:, 0] << 24) | (trailer[:, 1] << 16) | (trailer[:, 2] << 8) | trailer[:, 3]

        # nvCOMP wants each raw Deflate body 4-byte aligned: repack them
        pay = sizes - _ZLIB_FRAMING
        aligned_off = np.zeros(k, np.int64)
        aligned_off[1:] = np.cumsum((pay + 7) // 8 * 8)[:-1]
        aligned = cp.empty(int(aligned_off[-1] + pay[-1] + 8), cp.uint8)
        _copy_spans(comp, aligned, rel + 2, aligned_off, pay)
        dec = cp.empty(k * cb, cp.uint8)
        stream = cp.cuda.get_current_stream()
        if self._codec is None:
            self._nvcomp = _nvcomp()
            self._codec = self._nvcomp.Codec(
                algorithm="Deflate", bitstream_kind=self._nvcomp.BitstreamKind.RAW, cuda_stream=stream.ptr
            )
        nvcomp, codec = self._nvcomp, self._codec
        assert nvcomp is not None and codec is not None
        out_off = np.arange(k, dtype=np.int64) * cb
        try:
            if _nvcomp_major(nvcomp) >= 6:
                srcs = nvcomp.as_arrays(aligned, aligned_off.tolist(), pay.tolist(), stream.ptr)
                outs = nvcomp.as_arrays(dec, out_off.tolist(), [cb] * k, stream.ptr)
                codec.decode(srcs, out=outs)
                got = np.asarray(outs.buffer_sizes, np.int64)
                ptrs = None
            else:
                # nvCOMP 5: new output buffers, gathered into ``dec`` by pointer
                base = aligned.data.ptr
                views = [_DeviceSpan(base + o, p) for o, p in zip(aligned_off.tolist(), pay.tolist())]
                outs = codec.decode(nvcomp.as_arrays(views, stream.ptr))
                ifaces = [o.__cuda_array_interface__ for o in outs]
                got = np.fromiter((i["shape"][0] for i in ifaces), np.int64, k)
                ptrs = np.fromiter((i["data"][0] for i in ifaces), np.uint64, k)
        except (ValueError, RuntimeError) as e:
            raise ValueError(f"{lay.name}: a deflate chunk of {self.path} does not inflate ({e})") from None
        bad = np.flatnonzero(got != cb)
        if bad.size:
            raise corrupt(int(bad[0]), f"inflates to {int(got[bad[0]]):,} bytes, not {cb:,}")
        if ptrs is not None:
            _kernel("copy_from_ptrs")(
                ((k * 32 + 255) // 256,),
                (256,),
                (cp.asarray(ptrs), dec, cp.asarray(out_off), cp.asarray(got), np.int64(k)),
            )
            stream.synchronize()  # before nvCOMP's output buffers are freed
        del outs, aligned
        adler = _zlib_gpu.adler32_chunks(dec, cb, k).get()
        bad = np.flatnonzero(adler != expected)
        if bad.size:
            raise corrupt(int(bad[0]), "Adler-32 mismatch")
        return dec


# ---------------------------------------------------------------------------
# Row gather and column remap on the GPU
# ---------------------------------------------------------------------------

_GATHER_SOURCE = r"""
// One warp per output row i: the entries of source row segment
// [start[i], start[i] + len[i]) whose column remap[col] >= 0 (every one when
// remap is null), in order. Count pass (dst_ptr null): count[i] = kept entries.
// Write pass: copied to dst_*[dst_ptr[i] ...] with their new column numbers.
template <typename T>
__global__ void gather_rows(const T* __restrict__ src_data, const int* __restrict__ src_idx,
                            const long long* __restrict__ start, const long long* __restrict__ len,
                            const int* __restrict__ remap, const long long* __restrict__ dst_ptr,
                            long long* __restrict__ count, T* __restrict__ dst_data,
                            int* __restrict__ dst_idx, long long n_rows) {
    long long w = ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (w >= n_rows) return;
    long long s = start[w], n = len[w];
    long long out = dst_ptr ? dst_ptr[w] : 0, kept = 0;
    for (long long b = 0; b < n; b += 32) {
        long long j = b + lane;
        bool in = j < n;
        int col = in ? src_idx[s + j] : -1;
        int nc = (in && remap) ? remap[col] : col;
        bool keep = in && nc >= 0;
        unsigned mask = __ballot_sync(0xffffffffu, keep);
        if (dst_ptr && keep) {
            long long p = out + kept + __popc(mask & ((1u << lane) - 1u));
            dst_data[p] = src_data[s + j];
            dst_idx[p] = nc;
        }
        kept += __popc(mask);
    }
    if (!dst_ptr && lane == 0) count[w] = kept;
}
"""

_CTYPES = {
    np.dtype(np.uint8): "unsigned char",
    np.dtype(np.uint16): "unsigned short",
    np.dtype(np.int32): "int",
    np.dtype(np.float32): "float",
}
_gather_module = None


def _gather_kernel(dtype):
    import cupy as cp

    global _gather_module
    if _gather_module is None:
        _gather_module = cp.RawModule(
            code=_GATHER_SOURCE,
            options=("-std=c++14",),
            name_expressions=[f"gather_rows<{t}>" for t in _CTYPES.values()],
        )
    return _gather_module.get_function(f"gather_rows<{_CTYPES[np.dtype(dtype)]}>")


# ---------------------------------------------------------------------------
# Selections
# ---------------------------------------------------------------------------


def _positions(idx, n: int, what: str) -> np.ndarray | None:
    """Integer positions in ``range(n)`` selected by ``idx``; None for all of them in order."""
    if idx is None:
        return None
    if isinstance(idx, slice):
        r = np.arange(n, dtype=np.int64)[idx]
        return None if len(r) == n and (n == 0 or r[0] == 0) and (n < 2 or r[1] == 1) else r
    a = np.asarray(idx)
    if a.dtype == bool:
        if a.shape != (n,):
            raise ValueError(f"A boolean {what} selection needs {n:,} values, not {a.shape}")
        return None if a.all() else np.flatnonzero(a).astype(np.int64)
    if a.dtype.kind not in "iu" or a.ndim != 1:
        raise TypeError(f"{what} must be selected by a slice, a boolean mask or integer positions")
    a = a.astype(np.int64)
    a = np.where(a < 0, a + n, a)
    if a.size and (a.min() < 0 or a.max() >= n):
        raise IndexError(f"{what} position out of range for {n:,} {what}s")
    return a


def _compose(base: np.ndarray | None, sel: np.ndarray | None) -> np.ndarray | None:
    """Positions in the file of the positions ``sel`` of a view whose own are ``base``."""
    if sel is None:
        return base
    return sel if base is None else base[sel]


def _column_remap(cols: np.ndarray | None, n_cols: int):
    """New column number of each file column (-1: dropped), or None to keep them all as they are."""
    import cupy as cp

    if cols is None or np.array_equal(cols, np.arange(n_cols)):
        return None, n_cols
    if np.unique(cols).size != cols.size:
        raise ValueError(
            "The selected features repeat some columns; the GPU reader reads each column once. "
            "Select each feature at most once"
        )
    remap = np.full(n_cols, -1, np.int32)
    remap[cols] = np.arange(cols.size, dtype=np.int32)
    return cp.asarray(remap), int(cols.size)


# ---------------------------------------------------------------------------
# Opening a source
# ---------------------------------------------------------------------------


class _Matrix:
    """The CSR group ``key`` of an h5ad, from a path or a backed AnnData (view)."""

    def __init__(self, source, key: str):
        self._own = None
        try:
            self._open(source, key)
        except BaseException:
            self.close()
            raise

    def _open(self, source, key: str) -> None:
        import anndata as ad
        import h5py

        self.rows = self.cols = None
        if isinstance(source, ad.AnnData):
            if not source.isbacked:
                raise ValueError("open_counts reads a file: pass its path or a backed AnnData")
            manager = source.file
            if not manager.is_open:
                manager.open()
            h5 = manager._file
            if h5.mode == "r+":
                h5.flush()  # kvikio reads the file through its own descriptor
            assert source.filename is not None  # backed
            self.path = Path(source.filename)
            if source.is_view:
                self.rows, self.cols = source._oidx, source._vidx
        else:
            self.path = Path(source)
            h5 = self._own = h5py.File(self.path, "r")
        if key not in h5:
            raise KeyError(f"{self.path} has no {key!r}")
        group = h5[key]
        enc = group.attrs.get("encoding-type", "")
        if isinstance(group, h5py.Dataset) or (enc and enc != "csr_matrix"):
            raise ValueError(f"{self.path}:{key} is not a CSR matrix (encoding {enc or 'dense'!r})")
        if not all(k in group for k in ("data", "indices", "indptr")):
            raise ValueError(f"{self.path}:{key} is not a CSR matrix (no data / indices / indptr)")
        self.shape = tuple(int(s) for s in group.attrs["shape"])
        n_rows, n_cols = self.shape
        if n_rows > _INT32_MAX or n_cols > _INT32_MAX:
            raise ValueError(f"{self.path}:{key} has more than 2**31 - 1 rows or columns")
        # (1) shape and indptr
        self.indptr_dtype = group["indptr"].dtype
        indptr = group["indptr"][()].astype(np.int64, copy=False)
        nnz = int(group["data"].shape[0])
        if indptr.shape != (n_rows + 1,) or indptr[0] != 0 or indptr[-1] != nnz:
            raise ValueError(
                f"{self.path}:{key} has an indptr of {indptr.shape[0]:,} entries from {indptr[0] if indptr.size else None} "
                f"to {indptr[-1] if indptr.size else None}, which does not match its {n_rows:,} rows and {nnz:,} entries"
            )
        if (np.diff(indptr) < 0).any():
            raise ValueError(f"{self.path}:{key} has a decreasing indptr")
        if group["indices"].shape != (nnz,):
            raise ValueError(f"{self.path}:{key} has {group['indices'].shape} indices for {nnz:,} values")
        self.indptr, self.nnz = indptr, nnz
        # (2) empty arrays need no storage; (3) the rest is checked before any read
        self.data = _inspect(group["data"], f"{key}/data")
        self.indices = _inspect(group["indices"], f"{key}/indices")
        if self.indices.dtype.kind not in "iu":
            raise ValueError(f"{self.path}:{key}/indices holds {self.indices.dtype}, not integers")
        self.value_dtype = self.data.dtype if self.data.dtype in _VALUE_DTYPES else np.dtype(np.float32)

    def close(self) -> None:
        if self._own is not None:
            self._own.close()
            self._own = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class _Selection(NamedTuple):
    starts: np.ndarray  # source entry of each output row
    lens: np.ndarray  # source entries of each output row
    remap: object  # cupy int32 column remap, or None
    n_cols: int


def _select(m: _Matrix, rows, features) -> _Selection:
    n_rows, n_cols = m.shape
    base_r = _positions(m.rows, n_rows, "row")
    view_rows = n_rows if base_r is None else base_r.size
    file_rows = _compose(base_r, _positions(rows, view_rows, "row"))
    base_c = _positions(m.cols, n_cols, "column")
    view_cols = n_cols if base_c is None else base_c.size
    file_cols = _compose(base_c, _positions(features, view_cols, "column"))
    remap, out_cols = _column_remap(file_cols, n_cols)
    if file_rows is None:
        starts, ends = m.indptr[:-1], m.indptr[1:]
    else:
        starts, ends = m.indptr[file_rows], m.indptr[file_rows + 1]
    return _Selection(starts, ends - starts, remap, out_cols)


def _gather(reader: _Reader, m: _Matrix, starts: np.ndarray, lens: np.ndarray, remap):
    """The output rows ``(starts, lens)`` on the GPU: ``(data, indices, counts)``."""
    import cupy as cp

    n_out = len(starts)
    if n_out == 0 or int(lens.sum()) == 0 and remap is None:
        return cp.empty(0, m.value_dtype), cp.empty(0, cp.int32), lens.copy()
    lo, hi = int(starts.min()), int((starts + lens).max())
    src_d = cp.empty(hi - lo, m.value_dtype)
    src_i = cp.empty(hi - lo, cp.int32)
    reader.read(m.data, lo, hi, src_d)
    reader.read(m.indices, lo, hi, src_i)
    if hi > lo and (int(src_i.min()) < 0 or int(src_i.max()) >= m.shape[1]):
        raise ValueError(f"{m.path}: column indices out of range for {m.shape[1]:,} columns")
    if remap is None and _in_order(starts, lens):
        return src_d, src_i, lens.copy()
    d_start, d_len = cp.asarray(starts - lo), cp.asarray(lens)
    kernel = _gather_kernel(m.value_dtype)
    grid = ((n_out * 32 + 255) // 256,)
    null = np.uint64(0)
    if remap is None:
        counts = lens.copy()
    else:
        d_count = cp.empty(n_out, cp.int64)
        kernel(grid, (256,), (src_d, src_i, d_start, d_len, remap, null, d_count, null, null, np.int64(n_out)))
        counts = d_count.get()
    ptr = np.zeros(n_out, np.int64)
    np.cumsum(counts[:-1], out=ptr[1:])
    total = int(counts.sum())
    out_d, out_i = cp.empty(total, m.value_dtype), cp.empty(total, cp.int32)
    kernel(
        grid,
        (256,),
        (
            src_d,
            src_i,
            d_start,
            d_len,
            null if remap is None else remap,
            cp.asarray(ptr),
            null,
            out_d,
            out_i,
            np.int64(n_out),
        ),
    )
    cp.cuda.get_current_stream().synchronize()
    return out_d, out_i, counts


def _in_order(starts: np.ndarray, lens: np.ndarray) -> bool:
    """Whether the output rows are consecutive source entries, so read as they are (no gather)."""
    return len(starts) == 0 or bool((starts[1:] == starts[:-1] + lens[:-1]).all())


def _row_blocks(starts: np.ndarray, lens: np.ndarray, max_entries: int) -> Iterator[tuple[int, int]]:
    """Yield runs of output rows whose source span plus output fit in ``max_entries`` entries.

    A single row always forms a block.
    """
    n = len(starts)
    cum = np.concatenate([[0], np.cumsum(lens)])

    def split(a: int, b: int):
        span = int((starts[a:b] + lens[a:b]).max() - starts[a:b].min())
        if b - a == 1 or span + int(cum[b] - cum[a]) <= max_entries:
            yield a, b
        else:
            m = (a + b) // 2
            yield from split(a, m)
            yield from split(m, b)

    a = 0
    while a < n:
        # the most rows whose output alone fits, then split by their source span
        b = max(a + 1, int(np.searchsorted(cum, cum[a] + max_entries // 2, side="right")) - 1)
        b = min(b, n)
        yield from split(a, b)
        a = b


def _free_gpu_bytes() -> int:
    import cupy as cp

    return int(cp.cuda.Device().mem_info[0] + cp.get_default_memory_pool().free_bytes())


def read_whole(
    m: _Matrix,
    *,
    to_gpu: bool = False,
    memory_fraction: float = 0.25,
    read_block_bytes: int = 1 << 30,
    max_decoded_bytes: int = 4 << 30,
):
    """
    The whole matrix of an open :class:`_Matrix`, for :func:`gatac.io.read_h5ad`.

    A scipy CSR matrix in host memory, read in blocks of at most ``memory_fraction`` of the
    GPU's free memory. Values keep their stored dtype. Index arrays are int32
    when the shape and the number of entries fit, int64 otherwise, as scipy
    builds them (anndata reads them so from 0.14). With ``to_gpu=True``, a cupy CSR matrix (float32, or
    float64 if stored so, and int32 indices, as cupy requires).
    """
    import cupy as cp
    import scipy.sparse as sp

    nnz, (n_rows, n_cols) = m.nnz, m.shape
    stored = m.data.dtype
    reader = None if nnz == 0 else _Reader(m.path, block_bytes=read_block_bytes, max_decoded_bytes=max_decoded_bytes)

    def read(a: int, b: int, d, i) -> None:
        assert reader is not None  # called only when nnz > 0
        reader.read(m.data, a, b, d)
        reader.read(m.indices, a, b, i)
        if int(i.min()) < 0 or int(i.max()) >= n_cols:
            raise ValueError(f"{m.path}: column indices out of range for {n_cols:,} columns")

    try:
        if to_gpu:
            import cupyx.scipy.sparse as cusp

            if nnz > _INT32_MAX:
                raise ValueError(
                    f"{m.path} holds a matrix of {nnz:,} entries; cupy's CSR matrices hold at most 2**31 - 1. "
                    "Read it with to_gpu=False"
                )
            dtype = stored if stored in (np.dtype(np.float32), np.dtype(np.float64)) else np.dtype(np.float32)
            data, indices = cp.empty(nnz, dtype), cp.empty(nnz, cp.int32)
            if nnz:
                read(0, nnz, data, indices)
            return cusp.csr_matrix((data, indices, cp.asarray(m.indptr.astype(np.int32))), shape=m.shape)

        # one index dtype for indices and indptr, as the scipy constructor chooses it
        idx = np.dtype(np.int32 if max(n_rows, n_cols, nnz) <= _INT32_MAX else np.int64)
        data, indices = np.empty(nnz, stored), np.empty(nnz, idx)
        if nnz:
            per_entry = stored.itemsize + m.indices.dtype.itemsize + idx.itemsize
            step = max(_MIN_BLOCK_ENTRIES, int(memory_fraction * _free_gpu_bytes()) // per_entry)
            step = min(step, nnz)
            d, i = cp.empty(step, stored), cp.empty(step, m.indices.dtype)
            for a in range(0, nnz, step):
                b = min(a + step, nnz)
                read(a, b, d[: b - a], i[: b - a])
                d[: b - a].get(out=data[a:b])
                i[: b - a].astype(idx, copy=False).get(out=indices[a:b])
            del d, i
            cp.get_default_memory_pool().free_all_blocks()
    finally:
        if reader is not None:
            reader.close()
    # assigned rather than passed to the constructor, which would scan the arrays to choose idx again
    X = sp.csr_matrix((n_rows, n_cols), dtype=stored)
    X.data, X.indices, X.indptr = data, indices, m.indptr.astype(idx)
    return X


def open_counts(
    source,
    key: str = "X",
    *,
    rows=None,
    features=None,
    chunk_size: int = 50_000,
    memory_fraction: float = 0.5,
    read_block_bytes: int = 1 << 30,
    max_decoded_bytes: int = 4 << 30,
) -> StreamedCounts:
    """
    Read a CSR count matrix of an h5ad onto the GPU, as a :class:`~gatac.dr._stream.StreamedCounts`.

    Parameters
    ----------
    source
        The path of an ``.h5ad``, a backed AnnData, or a view of one
        (``adata[cells, features]``, both axes in any order; columns may not
        repeat). An in-memory AnnData is used as it is, without the file reader.
    key
        The matrix: ``"X"`` or the path of another CSR group, e.g. ``"layers/counts"``.
    rows, features
        Further selections relative to ``source``: a slice, a boolean mask or
        integer positions. Rows may be reordered or repeated; features may be
        reordered but not repeated.
    chunk_size
        Rows per chunk of the result.
    memory_fraction
        Share of the GPU's free memory the matrix may take. A matrix that does
        not fit is decoded block by block into host memory, and the chunks
        that do not fit are streamed from there at every product.
    read_block_bytes, max_decoded_bytes
        File bytes read, and bytes decoded, per batch.

    Returns
    -------
    StreamedCounts
        Values as stored when they are uint8, uint16, int32 or float32 (other
        types become float32), int32 column indices, int64 ``indptr``.
    """
    import anndata as ad
    import cupy as cp
    import scipy.sparse as sp

    from gatac.dr._stream import StreamedCounts

    if isinstance(source, ad.AnnData) and not source.isbacked:
        X = source.layers[key.removeprefix("layers/")] if key != "X" else source.X
        r = _positions(rows, source.n_obs, "row")
        c = _positions(features, source.n_vars, "column")
        X = sp.csr_matrix(X)  # type: ignore[arg-type]  # any array anndata holds, scipy-stubs list fewer
        X = X if r is None else X[r]
        X = X if c is None else X[:, c]
        return StreamedCounts(X, chunk_size, memory_fraction)

    with _Matrix(source, key) as m:
        sel = _select(m, rows, features)
        n_out = len(sel.starts)
        shape = (n_out, sel.n_cols)
        src_entries = int((sel.starts + sel.lens).max() - sel.starts.min()) if n_out else 0
        out_entries = int(sel.lens.sum())
        entry = m.value_dtype.itemsize + 4
        # the matrix's share of the GPU; the decode buffers (read_block_bytes,
        # max_decoded_bytes) are transient and come from the rest
        budget = memory_fraction * _free_gpu_bytes()
        gathered = sel.remap is not None or not _in_order(sel.starts, sel.lens)
        if m.nnz == 0 or out_entries == 0:
            indptr = np.zeros(n_out + 1, np.int64)
            return StreamedCounts.from_arrays(
                indptr, np.empty(0, m.value_dtype), np.empty(0, np.int32), shape, chunk_size, memory_fraction
            )
        with _Reader(m.path, block_bytes=read_block_bytes, max_decoded_bytes=max_decoded_bytes) as reader:
            if (src_entries * gathered + out_entries) * entry <= budget:
                data, indices, counts = _gather(reader, m, sel.starts, sel.lens, sel.remap)
                indptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
                logger.info(
                    f"Read {m.path.name}:{key} onto the GPU: {shape[0]:,} × {shape[1]:,}, {len(data):,} entries"
                )
                return StreamedCounts.from_arrays(indptr, data, indices, shape, chunk_size, memory_fraction)
            # over the GPU budget: block by block into host memory
            data = np.empty(out_entries, m.value_dtype)
            indices = np.empty(out_entries, np.int32)
            counts = np.empty(n_out, np.int64)
            pos = 0
            max_entries = max(_MIN_BLOCK_ENTRIES, int(budget // entry))
            for a, b in _row_blocks(sel.starts, sel.lens, max_entries):
                d, i, c = _gather(reader, m, sel.starts[a:b], sel.lens[a:b], sel.remap)
                k = len(d)
                d.get(out=data[pos : pos + k])
                i.get(out=indices[pos : pos + k])
                counts[a:b] = c
                pos += k
                del d, i
                cp.get_default_memory_pool().free_all_blocks()
            logger.info(
                f"Read {m.path.name}:{key} block by block into host memory (over the GPU budget): "
                f"{shape[0]:,} × {shape[1]:,}, {pos:,} entries"
            )
            indptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
            return StreamedCounts.from_arrays(indptr, data[:pos], indices[:pos], shape, chunk_size, memory_fraction)


def gpu_csr(source, key: str = "X", *, rows=None, features=None):
    """
    The float32 cupy CSR matrix of ``source`` (as in :func:`open_counts`), read straight onto the GPU.

    cupy's CSR matrices index entries with int32: more than 2**31 - 1
    entries raise; use :func:`open_counts` or :func:`iter_row_chunks`.
    """
    import cupy as cp
    import cupyx.scipy.sparse as cusp

    C = open_counts(source, key, rows=rows, features=features, chunk_size=np.iinfo(np.int32).max)
    try:
        if C.indptr.size and int(C.indptr[-1]) > np.iinfo(np.int32).max:
            raise ValueError("More than 2**31 - 1 entries: read the matrix in chunks (open_counts, iter_row_chunks)")
        data, indices = cp.asarray(C._data), cp.asarray(C._indices)
        return cusp.csr_matrix((data.astype(cp.float32), indices, C.indptr.astype(cp.int32)), shape=C.shape)
    finally:
        C.close()


class RowChunk(NamedTuple):
    """Rows ``start:stop`` of a matrix on the GPU, in CSR arrays relative to ``start``."""

    start: int
    stop: int
    data: Any  # cupy array: values as stored (uint8, uint16, int32 or float32)
    indices: Any  # cupy int32
    indptr: Any  # cupy int64, from 0
    n_cols: int


def iter_row_chunks(
    source,
    key: str = "X",
    rows_per_chunk: int = 50_000,
    *,
    rows=None,
    features=None,
    read_block_bytes: int = 1 << 30,
    max_decoded_bytes: int = 4 << 30,
) -> Iterator[RowChunk]:
    """
    The rows of a CSR matrix of an h5ad on the GPU, ``rows_per_chunk`` at a time.

    ``source``, ``key``, ``rows`` and ``features`` are as in :func:`open_counts`;
    one chunk is in GPU memory at a time.
    """
    import cupy as cp

    with _Matrix(source, key) as m:
        sel = _select(m, rows, features)
        n_out = len(sel.starts)
        reader = (
            None if m.nnz == 0 else _Reader(m.path, block_bytes=read_block_bytes, max_decoded_bytes=max_decoded_bytes)
        )
        try:
            for a in range(0, n_out, rows_per_chunk):
                b = min(a + rows_per_chunk, n_out)
                if reader is None:
                    d, i, c = cp.empty(0, m.value_dtype), cp.empty(0, cp.int32), np.zeros(b - a, np.int64)
                else:
                    d, i, c = _gather(reader, m, sel.starts[a:b], sel.lens[a:b], sel.remap)
                indptr = cp.asarray(np.concatenate([[0], np.cumsum(c)]).astype(np.int64))
                yield RowChunk(a, b, d, i, indptr, sel.n_cols)
        finally:
            if reader is not None:
                reader.close()
