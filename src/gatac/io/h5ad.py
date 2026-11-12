"""``.h5ad`` files read and written with their sparse matrices decoded on the GPU."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import anndata as ad

logger = logging.getLogger(__name__)


def read_h5ad(
    filename: str | Path,
    backed: Literal["r", "r+"] | bool | None = None,
    *,
    to_gpu: bool = False,
    **kwargs,
) -> ad.AnnData:
    """
    Read an ``.h5ad`` file, its sparse matrices read and decompressed on the GPU.

    As :func:`anndata.read_h5ad`, except that the CSR matrices (``X``,
    ``layers``, ``raw.X``) are read with KvikIO and, when gzip-compressed,
    inflated on the GPU by nvCOMP, each chunk's checksum checked. anndata
    reads everything else, and also reads a matrix the GPU reader does not
    (dense, CSC, an HDF5 filter other than gzip, such as lzf) and files
    written before anndata 0.7.

    Parameters
    ----------
    filename
        The ``.h5ad`` file.
    backed
        ``"r"`` or ``"r+"`` (``True`` is ``"r+"``): open the file backed,
        with :func:`anndata.read_h5ad`. The matrix stays on disk; GATAC's
        functions read it from there onto the GPU (see
        [count matrices on disk](/api/io.md#count-matrices-on-disk)).
    to_gpu
        Return ``X`` and the CSR layers as cupy CSR matrices in GPU memory,
        as rapids-singlecell takes them, rather than scipy CSR matrices in
        host memory. cupy's sparse matrices hold floats, int32 indices and
        at most 2³¹ − 1 entries: integer counts become float32. ``raw.X``
        stays in host memory.
    **kwargs
        Other arguments of :func:`anndata.read_h5ad` (``as_sparse``, ...):
        anndata then reads the whole file.

    Returns
    -------
    AnnData
        As :func:`anndata.read_h5ad` returns it: the same values, value
        dtypes and types. The index arrays of a CSR matrix are int32 when
        they fit, as scipy builds them (anndata reads them so from 0.14).
    """
    import anndata as ad
    import h5py

    if backed not in (None, False) or kwargs:
        if to_gpu:
            raise ValueError("anndata reads this file (backed or with its own arguments): to_gpu does not apply")
        return ad.read_h5ad(filename, backed=backed, **kwargs)

    try:
        from anndata.experimental import read_dispatched
    except ImportError:  # pragma: no cover - anndata moved it
        from anndata.io import read_dispatched  # type: ignore[attr-defined,no-redef]  # only in other anndata versions

    from . import _h5counts

    def matrix(func, name: str, elem):
        key = name.lstrip("/")
        try:
            m = _h5counts._Matrix(filename, key)
        except ValueError as e:
            logger.info(f"{Path(filename).name}:{key} read by anndata ({e})")
            return func(elem)
        with m:
            return _h5counts.read_whole(m, to_gpu=to_gpu and name != "/raw/X")

    def callback(func, name: str, elem, iospec):
        if iospec.encoding_type == "csr_matrix" and (name in ("/X", "/raw/X") or name.startswith("/layers/")):
            return matrix(func, name, elem)
        return func(elem)

    with h5py.File(filename, "r") as f:
        if f.attrs.get("encoding-type") != "anndata":
            # written before anndata 0.7: anndata's own reader handles its layout
            return ad.read_h5ad(filename)
        adata = read_dispatched(f, callback)
    assert isinstance(adata, ad.AnnData)  # the root group of an h5ad file is an AnnData
    return adata


def write_h5ad(filepath: str | Path, adata: ad.AnnData, *, compression: Literal["gzip"] | None = "gzip") -> None:
    """
    Write an AnnData to an ``.h5ad`` file as GATAC's commands write their matrices.

    The arguments come in the order of ``anndata.io.write_h5ad``; ``X`` is
    stored as ``gatac tile``, ``gatac combine``, ... store it (``--compression``),
    and anndata writes everything else. Any HDF5 reader, anndata included,
    reads the file as usual, and :func:`gatac.io.read_h5ad` reads it with its
    matrices decompressed on the GPU.

    Parameters
    ----------
    filepath
        The ``.h5ad`` file. It is replaced only once the new file is complete.
    adata
        The AnnData to write.
    compression
        ``"gzip"`` (default): a CSR ``X`` with HDF5's shuffle + deflate
        filters, compressed on the GPU by nvCOMP, and every other dataset
        gzip-compressed by anndata on the CPU. ``None``: uncompressed.

    Notes
    -----
    Differences from ``anndata.io.write_h5ad`` and ``adata.write_h5ad``:

    - The file is gzip-compressed by default (anndata: uncompressed).
    - A gzip ``X`` is byte-shuffled before deflate (anndata does not
      shuffle), and its column indices are int32 on disk, ``indptr`` too
      while the matrix has at most 2³¹ − 1 entries (anndata writes them as
      held in memory). The values read back are the same.
    - Only ``X`` is compressed on the GPU: large CSR ``layers`` or ``raw.X``
      are compressed at anndata's speed.
    - ``X`` may be a cupy CSR matrix.
    - No ``as_dense``, ``compression_opts`` or ``dataset_kwargs``; ``lzf`` is
      not offered.

    An ``X`` that is not CSR, and the AnnData of a backed file, are written
    by anndata alone, with the same ``compression``.
    """
    from gatac._files import write_h5ad as _write

    if compression not in ("gzip", None):
        raise ValueError(f"compression is 'gzip' or None, not {compression!r}")
    _write(adata, filepath, compression=compression)
