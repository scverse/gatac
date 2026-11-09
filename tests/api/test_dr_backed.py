"""
``gatac.dr.spectral`` on a backed AnnData: the matrix read from its file
straight into GPU memory, the result stored as anndata and scanpy store it:
in memory for a file opened ``r``, in the file too for one opened ``r+``,
never on a view.
"""

from __future__ import annotations

import anndata as ad
import h5py
import numpy as np
import pytest

import gatac as ga
from gatac.io import _h5counts as h5counts

N_COMPS = 8


@pytest.fixture
def selected(tiles):
    ga.pp.select_features(tiles, n_features=2_000)
    return tiles


@pytest.fixture
def written(selected, tmp_path):
    """The matrix gzip-compressed by anndata; the GPU reader's layouts are tested in test_h5counts.py."""
    path = tmp_path / "tiles.h5ad"
    selected.write_h5ad(path, compression="gzip")
    return path, selected


def _abs_corr(a, b):
    return np.array([abs(np.corrcoef(a[:, i], b[:, i])[0, 1]) for i in range(a.shape[1])])


def _x_bytes(path):
    with h5py.File(path, "r") as f:
        return {k: f["X"][k][()].tobytes() for k in ("data", "indices", "indptr")}


@pytest.fixture
def no_reads(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("the file was read")

    monkeypatch.setattr(h5counts, "_Reader", refuse)


def test_backed_r_gives_the_in_memory_result_and_leaves_the_file(written):
    """The same matrix on the GPU gives the same embedding; ``r`` keeps it in memory until ``adata.write()``."""
    path, selected = written
    evals, emb = ga.dr.spectral(selected.copy(), N_COMPS, inplace=False)
    backed = ad.read_h5ad(path, backed="r")
    ga.dr.spectral(backed, N_COMPS)
    np.testing.assert_array_equal(backed.obsm["X_spectral"], emb)
    np.testing.assert_array_equal(backed.uns["spectral_eigenvalue"], evals)
    assert "X_spectral" not in backed.file._file["obsm"]

    backed.write()  # anndata's own save: obs, var, obsm, uns ... but not X
    backed.file.close()
    again = ad.read_h5ad(path)
    np.testing.assert_array_equal(again.obsm["X_spectral"], emb)
    assert (again.X != selected.X).nnz == 0


def test_backed_r_plus_writes_only_the_results_into_the_file(written):
    path, selected = written
    _, emb = ga.dr.spectral(selected.copy(), N_COMPS, inplace=False)
    x_before = _x_bytes(path)
    backed = ad.read_h5ad(path, backed="r+")
    backed.obs["unsaved"] = 1  # an edit the user has not saved stays unsaved
    ga.dr.spectral(backed, N_COMPS)
    np.testing.assert_array_equal(backed.obsm["X_spectral"], emb)
    backed.file.close()
    assert _x_bytes(path) == x_before
    again = ad.read_h5ad(path, backed="r")
    np.testing.assert_array_equal(again.obsm["X_spectral"], emb)
    assert "spectral_eigenvalue" in again.uns and "unsaved" not in again.obs
    again.file.close()


def test_backed_view_returns_the_result_of_the_subset(written):
    """Rows and columns of the view, then ``features`` relative to the view."""
    path, selected = written
    rng = np.random.default_rng(0)
    rows = np.sort(rng.choice(selected.n_obs, selected.n_obs * 2 // 3, replace=False))
    cols = np.sort(rng.choice(selected.n_vars, selected.n_vars // 2, replace=False))
    expected = ga.dr.spectral(selected[rows, cols].copy(), N_COMPS, inplace=False)
    backed = ad.read_h5ad(path, backed="r")
    got = ga.dr.spectral(backed[rows, cols], N_COMPS, inplace=False)
    np.testing.assert_array_equal(got[0], expected[0])
    np.testing.assert_array_equal(got[1], expected[1])
    backed.file.close()


def test_backed_view_cannot_store_the_result(written, no_reads):
    path, _ = written
    backed = ad.read_h5ad(path, backed="r+")
    with pytest.raises(ValueError, match="inplace=False"):
        ga.dr.spectral(backed[:50], N_COMPS)
    backed.file.close()


def test_backed_chunked_matches_in_memory(written):
    path, selected = written
    ref = selected.copy()
    ga.dr.spectral(ref, N_COMPS, chunk_size=50)
    backed = ad.read_h5ad(path, backed="r")
    ga.dr.spectral(backed, N_COMPS, chunk_size=50)
    assert _abs_corr(backed.obsm["X_spectral"], ref.obsm["X_spectral"]).min() > 0.9999
    backed.file.close()
