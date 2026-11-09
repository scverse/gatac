"""
``.h5ad`` files read with their sparse matrices decoded on the GPU.
"""

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
        As :func:`anndata.read_h5ad` returns it: the same values, dtypes
        (index arrays included) and types.
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
        from anndata.io import read_dispatched

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
        return read_dispatched(f, callback)
