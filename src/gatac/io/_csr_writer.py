"""
Writing a CSR count matrix to an ``.h5ad`` block by block, compressed on the GPU.

:class:`CSRWriter` takes the rows of a matrix in order, in blocks, and the
AnnData metadata at the end (:meth:`CSRWriter.finalize`). With
``compression="gzip"`` (the default) each HDF5 chunk is byte-shuffled,
Deflate-compressed by nVIDIA nvCOMP and framed as the zlib stream of HDF5's
standard shuffle + deflate filters, so h5py, anndata and any HDF5 reader
read the file; a child process writes the chunks (:mod:`gatac._h5_chunk_writer`)
while the GPU compresses the next ones. The output only replaces an existing
file once it is complete.
"""

from __future__ import annotations

import logging
import mmap
import os
import subprocess
import sys
import uuid
from pathlib import Path

import numpy as np

from .._files import csr_disk_dtypes, replacing

logger = logging.getLogger(__name__)

_N_BUFFERS = 3
_GZIP_DECLARED_LEVEL = 1  # the level HDF5 records for its deflate filter; it does not affect reading


def _chunk_bound(chunk_bytes: int) -> int:
    """Upper bound of one compressed, framed chunk."""
    return chunk_bytes + chunk_bytes // 8 + 4096


def _widen(group, key: str, dtype, chunks) -> None:
    """Replace the 1-D dataset ``group[key]`` by a copy of it in ``dtype`` (uncompressed files)."""
    old = group[key]
    kw = {"maxshape": (None,), "chunks": chunks} if old.chunks else {}
    new = group.create_dataset(key + "_wide", shape=old.shape, dtype=dtype, **kw)
    step = 1 << 24
    for i in range(0, old.shape[0], step):
        new[i : i + step] = old[i : i + step]
    del group[key]
    group.move(key + "_wide", key)


class _Child:
    """The writer process and the page-locked shared-memory buffers it writes from."""

    def __init__(self, cap: int):
        from multiprocessing.connection import Connection

        import cupy as cp

        self.cap = cap
        self.bufs: list[np.ndarray] = []
        self._maps: list[mmap.mmap] = []
        self._paths: list[str] = []
        self._registered: list[int] = []
        self.proc = None
        self.conn_in = self.conn_out = None
        self.free: list[int] = []
        try:
            for _ in range(_N_BUFFERS):
                path = f"/dev/shm/gatac-csr-{os.getpid()}-{uuid.uuid4().hex}"
                fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
                self._paths.append(path)
                try:
                    os.ftruncate(fd, cap)
                    m = mmap.mmap(fd, cap)
                finally:
                    os.close(fd)
                self._maps.append(m)
                buf = np.frombuffer(m, np.uint8)
                self.bufs.append(buf)
                cp.cuda.runtime.hostRegister(buf.ctypes.data, cap, 0)  # fast device -> host copies
                self._registered.append(buf.ctypes.data)
            child_in, parent_out = os.pipe()
            parent_in, child_out = os.pipe()
            script = str(Path(__file__).resolve().parents[1] / "_h5_chunk_writer.py")
            code = "import runpy, sys; runpy.run_path(sys.argv.pop(1), run_name='__main__')"
            try:
                self.proc = subprocess.Popen(
                    [sys.executable, "-c", code, script, str(child_in), str(child_out), *self._paths],
                    pass_fds=(child_in, child_out),
                )
            finally:
                os.close(child_in)
                os.close(child_out)
            self.conn_out = Connection(parent_out, readable=False)
            self.conn_in = Connection(parent_in, writable=False)
            self.expect("ready")
            # the child has mapped the buffers: nothing is left in /dev/shm, whatever happens next
            for path in self._paths:
                os.unlink(path)
            self._paths = []
            self.free = list(range(_N_BUFFERS))
        except BaseException:
            self.stop()
            raise

    def recv(self):
        try:
            msg = self.conn_in.recv()
        except (EOFError, OSError):
            code = self.proc.poll() if self.proc is not None else None
            raise RuntimeError(f"The HDF5 writer process of CSRWriter exited (code {code})") from None
        if msg[0] == "error":
            raise RuntimeError(f"The HDF5 writer process of CSRWriter failed:\n{msg[1]}")
        return msg

    def expect(self, what: str):
        while True:
            msg = self.recv()
            if msg[0] == "free":
                self.free.append(msg[1])
            elif msg[0] == what:
                return msg
            else:  # pragma: no cover - protocol error
                raise RuntimeError(f"CSRWriter: unexpected reply {msg[0]!r} from the writer process")

    def acquire(self) -> int:
        while not self.free:
            msg = self.recv()
            if msg[0] != "free":  # pragma: no cover - protocol error
                raise RuntimeError(f"CSRWriter: unexpected reply {msg[0]!r} from the writer process")
            self.free.append(msg[1])
        return self.free.pop()

    def send(self, *msg) -> None:
        try:
            self.conn_out.send(msg)
        except OSError:
            while True:  # the child has gone: raise its error, or that it exited
                self.recv()

    def stop(self) -> None:
        """End the process and release the buffers (also after an error)."""
        import cupy as cp

        if self.proc is not None:
            try:
                if self.proc.poll() is None:
                    self.conn_out.send(("exit",))
                self.proc.wait(timeout=60)
            except Exception:
                self.proc.kill()
                self.proc.wait()
            self.proc = None
        for conn in (self.conn_in, self.conn_out):
            if conn is not None:
                conn.close()
        self.conn_in = self.conn_out = None
        for ptr in self._registered:
            cp.cuda.runtime.hostUnregister(ptr)
        self._registered = []
        self.bufs = []
        for m in self._maps:
            m.close()
        self._maps = []
        for path in self._paths:
            Path(path).unlink(missing_ok=True)
        self._paths = []


class _Stream:
    """The entries of one dataset (``data`` or ``indices``) not yet written, and its chunking."""

    def __init__(self, key: str, dtype, chunk_bytes: int):
        self.key = key
        self.set_dtype(dtype, chunk_bytes)

    def set_dtype(self, dtype, chunk_bytes: int) -> None:
        self.dtype = np.dtype(dtype)
        self.elems = max(1, chunk_bytes // self.dtype.itemsize)
        self.pending: list = []
        self.n_pending = 0
        self.chunks_written = 0
        self.n_written = 0  # uncompressed files

    @property
    def path(self) -> str:
        return f"X/{self.key}"


class CSRWriter:
    """
    Write a CSR matrix to an ``.h5ad`` as its rows come, then its metadata.

    A context manager: the file is written to a temporary path next to
    ``output_path`` and moved onto it only when the block ends after
    :meth:`finalize`. Leaving the block with an error, or without
    :meth:`finalize`, leaves an existing ``output_path`` as it was, and no
    temporary file, writer process or shared memory behind.

    Parameters
    ----------
    output_path : str or Path
        The ``.h5ad`` to write.
    compression : {"gzip", None}
        ``"gzip"`` (default): HDF5's shuffle + deflate filters, compressed on
        the GPU. ``None``: uncompressed chunked datasets.
    value_dtype : dtype, optional
        The dtype of the stored values. By default the first block's; a
        later block with a wider dtype then promotes the values written so
        far (re-encoded on the GPU).
    level : int
        nvCOMP's Deflate algorithm (0 fastest, 2 default).
    chunk_bytes : int
        Uncompressed bytes per HDF5 chunk (default 4 MiB).
    batch_bytes : int
        Uncompressed bytes compressed per nvCOMP call (default 2 GiB); the
        writer process uses three shared host buffers of a sixth of this size.
    nnz : int, optional
        The number of entries, when known. Uncompressed datasets are then
        contiguous (the fastest to read), and a different count raises.

    Examples
    --------
    >>> with CSRWriter("combined.h5ad") as w:
    ...     for block in blocks:  # scipy or cupy CSR matrices, rows in order
    ...         w.append(block)
    ...     w.finalize(obs=obs, var=var, uns=uns)

    Column indices are stored as int32, ``indptr`` as int32 while the
    matrix has at most 2**31 - 1 entries.
    """

    def __init__(
        self,
        output_path: str | os.PathLike,
        *,
        compression: str | None = "gzip",
        value_dtype=None,
        level: int = 2,
        chunk_bytes: int = 4 << 20,
        batch_bytes: int = 2 << 30,
        nnz: int | None = None,
    ):
        if compression not in ("gzip", None):
            raise ValueError(f"compression must be 'gzip' or None, not {compression!r}")
        self.output_path = Path(output_path)
        self.compression = compression
        self.value_dtype = None if value_dtype is None else np.dtype(value_dtype)
        self.level = int(level)
        self.chunk_bytes = int(chunk_bytes)
        self.batch_bytes = max(int(batch_bytes), self.chunk_bytes)
        self.expected_nnz = None if nnz is None else int(nnz)
        self._contiguous = compression is None and nnz is not None
        self._replacing = None
        self._tmp: Path | None = None
        self._h5 = None  # the parent's own handle (uncompressed files)
        self._child: _Child | None = None
        self._child_open = False
        self._codec = None
        self._streams: dict[str, _Stream] = {}
        self._n_cols: int | None = None
        self._n_rows = 0
        self._nnz = 0
        self._indptr: list[np.ndarray] = [np.zeros(1, np.int64)]
        self._finalized = False
        self._closed = False

    # -- session -----------------------------------------------------------

    def __enter__(self) -> CSRWriter:
        self._replacing = replacing(self.output_path)
        self._tmp = self._replacing.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        error = exc
        if exc_type is None and not self._finalized:
            error = RuntimeError("CSRWriter: the block ended without finalize(); nothing was written")
        try:
            self._release()
        except BaseException as e:
            if error is None:
                error = e
        if error is None:
            self._replacing.__exit__(None, None, None)  # publishes the file
            return False
        self._replacing.__exit__(type(error), error, error.__traceback__)  # removes the temporary file
        if error is not exc:
            raise error
        return False

    def _release(self) -> None:
        self._closed = True
        self._streams = {}
        try:
            if self._h5 is not None:
                self._h5.close()
                self._h5 = None
        finally:
            if self._child is not None:
                child, self._child = self._child, None
                child.stop()

    def _check_open(self) -> None:
        if self._tmp is None:
            raise RuntimeError("CSRWriter is used as a context manager: `with CSRWriter(path) as w:`")
        if self._closed or self._finalized:
            raise RuntimeError("CSRWriter: finalize() was called; no more blocks can be appended")

    # -- blocks ------------------------------------------------------------

    def append(self, block) -> None:
        """
        Append the rows of ``block``, a scipy or cupy sparse matrix (converted to CSR).

        Every block has the number of columns of the first one.
        """
        import scipy.sparse as sp

        self._check_open()
        on_gpu = False
        if sp.issparse(block):
            block = block.tocsr()
        else:
            try:
                import cupyx.scipy.sparse as csp
            except ImportError:  # pragma: no cover
                csp = None
            if csp is None or not csp.issparse(block):
                raise TypeError(f"CSRWriter.append takes a scipy or cupy sparse matrix, not {type(block).__name__}")
            block = block.tocsr()
            on_gpu = True
        n_rows, n_cols = block.shape
        indptr = np.asarray(block.indptr.get() if on_gpu else block.indptr, np.int64)
        lo, hi = int(indptr[0]), int(indptr[-1])
        if self._n_cols is None:
            self._start(n_cols, block.dtype)
        elif n_cols != self._n_cols:
            raise ValueError(f"CSRWriter: a block has {n_cols:,} columns, the first one {self._n_cols:,}")
        target = self._value_target(block.dtype)
        if target != self._streams["data"].dtype:
            self._promote(target)
        self._indptr.append(self._nnz + indptr[1:] - lo)
        self._nnz += hi - lo
        self._n_rows += n_rows
        if hi > lo:
            self._push("data", block.data[lo:hi])
            self._push("indices", block.indices[lo:hi])

    def _value_target(self, dtype) -> np.dtype:
        dtype = np.dtype(dtype)
        if self.value_dtype is not None:
            if not np.can_cast(dtype, self.value_dtype, "safe"):
                raise TypeError(f"CSRWriter: {dtype} values do not fit value_dtype={self.value_dtype}")
            return self.value_dtype
        return np.result_type(self._streams["data"].dtype, dtype)

    def _start(self, n_cols: int, first_dtype) -> None:
        """Create the file and the ``X`` datasets: ``n_cols`` and the value dtype are now known."""
        import h5py

        self._n_cols = int(n_cols)
        _, idx = csr_disk_dtypes(0, self._n_cols)
        value = self.value_dtype if self.value_dtype is not None else np.dtype(first_dtype)
        self._streams = {
            "data": _Stream("data", value, self.chunk_bytes),
            "indices": _Stream("indices", idx, self.chunk_bytes),
        }
        self._h5 = h5py.File(self._tmp, "w")
        group = self._h5.create_group("X")
        for s in self._streams.values():
            self._create(group, s)
        if self.compression == "gzip":
            self._h5.close()  # the writer process opens it
            self._h5 = None

    def _create(self, group, s: _Stream) -> None:
        if self._contiguous:
            group.create_dataset(s.key, shape=(self.expected_nnz,), dtype=s.dtype)
            return
        kw = (
            {"compression": "gzip", "compression_opts": _GZIP_DECLARED_LEVEL, "shuffle": True}
            if self.compression
            else {}
        )
        group.create_dataset(s.key, shape=(0,), maxshape=(None,), dtype=s.dtype, chunks=(s.elems,), **kw)

    def _push(self, key: str, values) -> None:
        s = self._streams[key]
        if self.compression is None:
            values = values.get() if hasattr(values, "get") else values
            d, n = self._h5[s.path], s.n_written
            if self._contiguous:
                if n + len(values) > self.expected_nnz:
                    raise ValueError(f"CSRWriter: more than the nnz={self.expected_nnz:,} entries given")
            else:
                d.resize((n + len(values),))
            d[n : n + len(values)] = values.astype(s.dtype, copy=False)
            s.n_written += len(values)
            return
        import cupy as cp

        s.pending.append(cp.asarray(values).astype(s.dtype, copy=False))
        s.n_pending += len(values)
        if s.n_pending * s.dtype.itemsize >= self.batch_bytes:
            self._flush(s, final=False)

    # -- GPU compression ---------------------------------------------------

    def _flush(self, s: _Stream, final: bool) -> None:
        """Compress and send the complete chunks of ``s`` (with ``final``, also the last, zero-padded)."""
        import cupy as cp

        if not s.n_pending:
            return
        vals = s.pending[0] if len(s.pending) == 1 else cp.concatenate(s.pending)
        s.pending = []
        n_full = len(vals) // s.elems
        k = n_full + (1 if final and len(vals) % s.elems else 0)
        if k > n_full:  # the edge chunk is stored full size, zero padded, as HDF5 does
            padded = cp.zeros(k * s.elems, s.dtype)
            padded[: len(vals)] = vals
            body, rest = padded, vals[:0]
        else:
            body, rest = vals[: n_full * s.elems], vals[n_full * s.elems :]
        per_batch = max(1, self.batch_bytes // self.chunk_bytes)
        if self._child is None:
            # compressed batches are a fraction of their bound: a larger one is
            # sent in several buffers
            need = _chunk_bound(self.chunk_bytes) * min(per_batch, max(k, self._pending_chunks()))
            self._child = _Child(min(need, max(_chunk_bound(self.chunk_bytes), self.batch_bytes // 6)))
        self._open_child()
        for c in range(0, k, per_batch):
            kk = min(per_batch, k - c)
            self._send(s, s.chunks_written + c, body[c * s.elems : (c + kk) * s.elems], kk)
        s.chunks_written += k
        s.pending = [rest.copy()] if len(rest) else []
        s.n_pending = len(rest)

    def _pending_chunks(self) -> int:
        return max((-(-s.n_pending // s.elems) for s in self._streams.values()), default=0)

    def _open_child(self) -> None:
        if not self._child_open:
            self._child.send("open", str(self._tmp), {s.path: s.elems for s in self._streams.values()})
            self._child.expect("opened")
            self._child_open = True

    def _close_child_file(self) -> None:
        if self._child_open:
            self._child.send("close")
            self._child.expect("closed")
            self._child_open = False

    def _encoder(self):
        if self._codec is None:
            import cudf  # noqa: F401  # cuDF loads the libnvcomp it pins: before nvidia.nvcomp
            import cupy as cp

            from .gpu_convert import _nvcomp

            nvcomp = _nvcomp()
            stream = cp.cuda.get_current_stream()
            self._codec = (
                nvcomp,
                nvcomp.Codec(
                    algorithm="Deflate",
                    bitstream_kind=nvcomp.BitstreamKind.RAW,
                    algorithm_type=self.level,
                    uncomp_chunk_size=self.chunk_bytes,
                    cuda_stream=stream.ptr,
                ),
                stream,
            )
        return self._codec

    def _send(self, s: _Stream, c0: int, vals, k: int) -> None:
        """Compress ``k`` chunks of ``vals`` on the GPU and hand them to the writer process."""
        import cupy as cp

        from . import _zlib_gpu
        from .gpu_convert import _DeviceSpan

        nvcomp, codec, stream = self._encoder()
        isz = s.dtype.itemsize
        cb = s.elems * isz
        with stream:
            raw = vals.view(cp.uint8)
            if isz > 1:  # HDF5's shuffle: byte b of every element of a chunk together
                raw = cp.ascontiguousarray(raw.reshape(k, s.elems, isz).transpose(0, 2, 1)).reshape(-1)
            adler = _zlib_gpu.adler32_chunks(raw, cb, k)
            base = raw.data.ptr
            bodies = codec.encode(nvcomp.as_arrays([_DeviceSpan(base + i * cb, cb) for i in range(k)], stream.ptr))
            packed, offs, sizes = _zlib_gpu.zlib_frame(bodies, adler)
            # into the shared buffers, at most one buffer per message
            cap = self._child.cap
            i = 0
            while i < k:
                j = i + 1
                while j < k and offs[j] + sizes[j] - offs[i] <= cap:
                    j += 1
                n = int(offs[j - 1] + sizes[j - 1] - offs[i])
                if n > cap:  # pragma: no cover - a chunk is never larger than its bound
                    raise RuntimeError(f"CSRWriter: a compressed chunk of {n:,} bytes exceeds the {cap:,}-byte buffers")
                b = self._child.acquire()
                packed[int(offs[i]) : int(offs[i]) + n].get(stream=stream, out=self._child.bufs[b][:n])
                stream.synchronize()
                self._child.send("batch", s.path, b, c0 + i, offs[i:j] - offs[i], sizes[i:j])
                i = j
            del bodies, raw, packed

    # -- value promotion ---------------------------------------------------

    def _promote(self, dtype) -> None:
        """Store the values in the wider ``dtype``, including those already written."""
        import h5py

        s = self._streams["data"]
        logger.debug(f"CSRWriter: values {s.dtype} -> {dtype}")
        if self.compression is None:
            n = s.n_written
            s.set_dtype(dtype, self.chunk_bytes)
            s.n_written = n
            _widen(self._h5["X"], "data", dtype, (s.elems,))
            return
        import cupy as cp

        from ._h5counts import _inspect, _Reader

        n_written = s.chunks_written * s.elems
        old = cp.empty(n_written, dtype)
        self._close_child_file()
        if n_written:
            with h5py.File(self._tmp, "r") as f:
                lay = _inspect(f[s.path], s.path)
            with _Reader(self._tmp, block_bytes=1 << 30, max_decoded_bytes=1 << 30) as r:
                r.read(lay, 0, n_written, old)
        pending = [p.astype(dtype) for p in s.pending]
        with h5py.File(self._tmp, "r+") as f:
            del f[s.path]
            s.set_dtype(dtype, self.chunk_bytes)
            self._create(f["X"], s)
        s.pending = ([old] if n_written else []) + pending
        s.n_pending = sum(len(p) for p in s.pending)

    # -- metadata ----------------------------------------------------------

    def finalize(self, *, obs, var, uns=None, **elems) -> None:
        """
        Write the last chunks, ``indptr`` and the AnnData metadata.

        ``obs`` and ``var`` must have one row per row and column of the
        matrix. ``elems`` (``obsm``, ``varm``, ``obsp``, ``varp``,
        ``layers``, ``raw``) are written by anndata, gzip-compressed on the
        CPU when ``compression`` is ``"gzip"``. The file replaces
        ``output_path`` when the ``with`` block ends.
        """
        import anndata as ad
        import h5py

        self._check_open()
        if self._n_cols is None:
            if len(obs):
                raise ValueError(f"CSRWriter: no block was appended, but obs has {len(obs):,} rows")
            self._start(len(var), np.float32 if self.value_dtype is None else self.value_dtype)
        if self.expected_nnz is not None and self._nnz != self.expected_nnz:
            raise ValueError(
                f"CSRWriter: {self._nnz:,} entries were given, not the nnz={self.expected_nnz:,} announced"
            )
        if len(obs) != self._n_rows or len(var) != self._n_cols:
            raise ValueError(
                f"CSRWriter: obs and var have {len(obs):,} × {len(var):,} rows; "
                f"the matrix is {self._n_rows:,} × {self._n_cols:,}"
            )
        if self.compression == "gzip":
            for s in self._streams.values():
                self._flush(s, final=True)
            if self._child is not None:
                self._close_child_file()
                child, self._child = self._child, None
                child.stop()
            self._h5 = h5py.File(self._tmp, "r+")
        h5 = self._h5
        group = h5["X"]
        if not self._contiguous:
            for s in self._streams.values():
                group[s.key].resize((self._nnz,))
        ptr_dtype, _ = csr_disk_dtypes(self._nnz, self._n_cols)
        group.create_dataset("indptr", data=np.concatenate(self._indptr).astype(ptr_dtype, copy=False))
        group.attrs.update(
            {
                "encoding-type": "csr_matrix",
                "encoding-version": "0.1.0",
                "shape": np.array([self._n_rows, self._n_cols]),
            }
        )
        h5.close()
        self._h5 = None
        # obs, var, ... from an AnnData without X written alongside, so that X is never rewritten
        meta = self._tmp.with_name(self._tmp.name + ".meta.h5ad")
        try:
            ad.AnnData(obs=obs, var=var, uns=uns, **elems).write_h5ad(str(meta), compression=self.compression)
            with h5py.File(meta, "r") as src, h5py.File(self._tmp, "r+") as dst:
                for key in src:
                    if key != "X":
                        src.copy(src[key], dst, key)
                dst.attrs.update(src.attrs)
        finally:
            meta.unlink(missing_ok=True)
        self._finalized = True
