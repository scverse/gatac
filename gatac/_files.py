"""Output files that are only replaced by complete ones."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np

_INT32_MAX = np.iinfo(np.int32).max


@contextmanager
def replacing(output_path: str | Path) -> Iterator[Path]:
    """
    A temporary path next to ``output_path``, moved onto it when the block succeeds.

    On an error the temporary file is removed and ``output_path``, if it
    exists, is left as it was: a reader never sees a partial output.
    """
    output_path = Path(output_path)
    # hidden, per process, and with the output's suffix for writers that check it
    tmp_path = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp{output_path.suffix}")
    try:
        yield tmp_path
        os.replace(tmp_path, output_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def csr_disk_dtypes(nnz: int, n_cols: int) -> tuple[np.dtype, np.dtype]:
    """
    The dtypes of ``indptr`` and ``indices`` of a CSR matrix written to disk.

    Each is int32 whenever its values fit, independently of the other:
    ``indptr`` holds offsets up to ``nnz``, ``indices`` column numbers below
    ``n_cols``. scipy instead gives both arrays one dtype, int64 as soon as
    ``nnz`` exceeds 2**31 - 1, which makes the column indices of a large
    matrix twice as big as needed.
    """
    indptr = np.int32 if nnz <= _INT32_MAX else np.int64
    indices = np.int32 if n_cols <= _INT32_MAX else np.int64
    return np.dtype(indptr), np.dtype(indices)


def write_csr(parent, key: str, X, step: int = 1 << 26) -> None:
    """
    Write the scipy CSR matrix ``X`` as ``parent[key]`` in anndata's ``csr_matrix`` encoding.

    ``indptr`` and ``indices`` take :func:`csr_disk_dtypes`; ``indices`` is
    converted ``step`` entries at a time, so no full-size copy is made.
    """
    indptr_dtype, indices_dtype = csr_disk_dtypes(X.nnz, X.shape[1])
    group = parent.create_group(key)
    group.attrs.update({"encoding-type": "csr_matrix", "encoding-version": "0.1.0", "shape": np.array(X.shape)})
    group.create_dataset("data", data=X.data)
    indices = group.create_dataset("indices", shape=(X.nnz,), dtype=indices_dtype)
    for i in range(0, X.nnz, step):
        indices[i : i + step] = X.indices[i : i + step]
    group.create_dataset("indptr", data=X.indptr.astype(indptr_dtype, copy=False))


def write_h5ad(adata, output_path: str | Path, compression: str | None = "gzip", block_entries: int = 1 << 27) -> None:
    """
    ``adata.write_h5ad(output_path)``, replacing ``output_path`` only once the file is complete.

    A CSR ``X``, scipy or cupy, is written by GATAC: its column indices are
    int32 on disk even when scipy holds them as int64, and with
    ``compression="gzip"`` (default) it is shuffle + gzip compressed on the
    GPU (``CSRWriter``), ``block_entries`` entries at a time.
    Everything else, and an ``X`` of another kind, is written by anndata,
    with anndata's own gzip compression unless ``compression`` is None.
    """
    import anndata as ad
    import h5py
    import scipy.sparse as sp

    X = adata.X
    on_gpu = False
    if not sp.issparse(X):
        try:
            import cupyx.scipy.sparse as csp
        except ImportError:  # pragma: no cover
            csp = None
        on_gpu = csp is not None and csp.issparse(X)
    own_x = not adata.isbacked and (sp.issparse(X) or on_gpu) and X.format == "csr"
    if not own_x:
        with replacing(output_path) as tmp_path:
            adata.write_h5ad(str(tmp_path), compression=compression)
        return
    # as adata.write_h5ad does to the AnnData it writes
    adata.strings_to_categoricals()
    elems = {
        "obsm": dict(adata.obsm),
        "varm": dict(adata.varm),
        "obsp": dict(adata.obsp),
        "varp": dict(adata.varp),
        # anndata >= 0.13 also lists X under the key None
        "layers": {k: v for k, v in adata.layers.items() if k is not None},
    }
    if adata.raw is not None:
        elems["raw"] = adata.raw
    if compression is None:
        with replacing(output_path) as tmp_path:
            ad.AnnData(obs=adata.obs, var=adata.var, uns=adata.uns, **elems).write_h5ad(
                str(tmp_path), compression=compression
            )
            with h5py.File(tmp_path, "a") as h5:
                write_csr(h5, "X", X.get() if on_gpu else X)
        return
    from .io._csr_writer import CSRWriter

    # row blocks of about block_entries entries
    indptr = X.indptr.get() if on_gpu else X.indptr
    rows = np.searchsorted(indptr, np.arange(0, X.nnz, max(1, block_entries)), side="left")
    bounds = np.unique(np.r_[0, rows, X.shape[0]])
    with CSRWriter(output_path, compression=compression) as writer:
        writer.append(X[:0])  # the number of columns and the value dtype, also of an empty matrix
        for a, b in zip(bounds[:-1], bounds[1:]):
            writer.append(X[a:b])
        writer.finalize(obs=adata.obs, var=adata.var, uns=adata.uns, **elems)
