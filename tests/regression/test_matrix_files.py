"""
Count matrices written to h5ad: int32 column indices whatever scipy holds,
``indptr`` int32 while the entries fit, and files that anndata reads back.
"""

from __future__ import annotations

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from gatac._files import csr_disk_dtypes, write_csr, write_h5ad


def disk_index_dtypes(path, key="X") -> tuple[np.dtype, np.dtype]:
    """(indptr, indices) dtypes of the CSR matrix ``key`` in an h5ad."""
    with h5py.File(path, "r") as f:
        return f[key]["indptr"].dtype, f[key]["indices"].dtype


def _wide(X: sp.csr_matrix) -> sp.csr_matrix:
    """``X`` with int64 index arrays, as scipy holds a matrix of more than 2**31 - 1 entries."""
    X = X.copy()
    X.indices, X.indptr = X.indices.astype(np.int64), X.indptr.astype(np.int64)
    return X


def _adata(X) -> ad.AnnData:
    n, m = X.shape
    rng = np.random.default_rng(1)
    return ad.AnnData(
        X,
        obs=pd.DataFrame({"group": pd.Categorical(["a", "b"] * (n // 2))}, index=[f"c{i}" for i in range(n)]),
        var=pd.DataFrame({"chrom": ["chr1"] * m}, index=[f"t{j}" for j in range(m)]),
        uns={"tile_scan": {"tile_size": 500}},
        obsm={"X_emb": rng.standard_normal((n, 3))},
        layers={"raw": X.copy()},
    )


@pytest.mark.parametrize(
    "nnz, n_cols, expected",
    [
        (2**31 - 1, 50_000, (np.int32, np.int32)),
        (2**31, 50_000, (np.int64, np.int32)),
    ],
)
def test_csr_disk_dtypes_are_chosen_per_array(nnz, n_cols, expected):
    """indptr follows the entry count and indices the column count, independently."""
    assert csr_disk_dtypes(nnz, n_cols) == tuple(np.dtype(t) for t in expected)


def test_write_csr_narrows_wide_indices(tmp_path, step=3):
    """int64 index arrays in memory become int32 on disk, converted ``step`` entries at a time."""
    X = sp.random(40, 30, density=0.2, format="csr", dtype=np.float32, random_state=0)
    path = tmp_path / "x.h5"
    with h5py.File(path, "w") as f:
        write_csr(f, "X", _wide(X), step=step)
    assert disk_index_dtypes(path) == (np.dtype(np.int32), np.dtype(np.int32))
    with h5py.File(path, "r") as f:
        g = f["X"]
        assert g.attrs["encoding-type"] == "csr_matrix" and list(g.attrs["shape"]) == [40, 30]
        got = sp.csr_matrix((g["data"][:], g["indices"][:], g["indptr"][:]), shape=(40, 30))
    assert (got != X).nnz == 0


@pytest.mark.parametrize("compression", ["gzip", None])
def test_write_h5ad_writes_int32_indices_and_keeps_the_rest(tmp_path, compression):
    """A CSR X held with int64 indices is written with int32 ones, compressed or not; obs, var,
    uns, obsm and layers are written as anndata writes them, and anndata reads the file back,
    also backed."""
    X = sp.random(40, 30, density=0.3, format="csr", dtype=np.float32, random_state=0)
    adata = _adata(_wide(X))
    path = tmp_path / "m.h5ad"
    write_h5ad(adata, path, compression=compression, block_entries=50)
    assert disk_index_dtypes(path) == (np.dtype(np.int32), np.dtype(np.int32))
    with h5py.File(path, "r") as f:
        assert f["X/data"].compression == compression

    got = ad.read_h5ad(path)
    assert (got.X != X).nnz == 0 and got.X.dtype == np.float32
    pd.testing.assert_frame_equal(got.obs, adata.obs)
    pd.testing.assert_frame_equal(got.var, adata.var)
    assert got.uns["tile_scan"]["tile_size"] == 500
    np.testing.assert_array_equal(got.obsm["X_emb"], adata.obsm["X_emb"])
    assert (got.layers["raw"] != X).nnz == 0

    backed = ad.read_h5ad(path, backed="r")
    assert (backed.X.to_memory() != X).nnz == 0
    backed.file.close()
    assert [p.name for p in tmp_path.iterdir()] == ["m.h5ad"]


def test_write_h5ad_leaves_other_matrices_to_anndata(tmp_path):
    """A dense X is written by anndata, unchanged."""
    adata = _adata(np.arange(12, dtype=np.float32).reshape(4, 3))
    write_h5ad(adata, tmp_path / "d.h5ad")
    np.testing.assert_array_equal(ad.read_h5ad(tmp_path / "d.h5ad").X, adata.X)
