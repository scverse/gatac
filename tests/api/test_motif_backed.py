"""
``gatac.motif.compute_deviations`` on a backed AnnData: the peak matrix read
from its file into GPU memory one row chunk at a time, every chunk read once
for all motifs, with the same scores as in memory; results stored as
anndata and scanpy store them.

GPU chromVAR is not bitwise reproducible: two identical in-memory runs
differ by up to about 1e-6 now and then (also before the file route), so
scores are compared with the tolerance of ``test_motif.py``.
"""

from __future__ import annotations

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest

import gatac as ga
from gatac._files import write_h5ad
from gatac.io import _h5counts as h5counts

BATCH = 37  # cell batches; row chunks are whole numbers of them


def _same(got, expected):
    np.testing.assert_allclose(np.asarray(got), np.asarray(expected), rtol=1e-5, atol=1e-5)


@pytest.fixture
def opened():
    """Open a file backed; always closed after the test."""
    handles = []

    def open_(path, mode="r"):
        handles.append(ad.read_h5ad(path, backed=mode))
        return handles[-1]

    yield open_
    for h in handles:
        h.file.close()


@pytest.fixture(scope="module")
def prepared(_peaks, world, tmp_path_factory):
    """A peak matrix with background peaks and motif matches, its in-memory deviations, and its file."""
    a = _peaks.copy()
    ga.motif.chromvar(a, world["fasta"], world["meme"], method="knn", batch_size=BATCH)
    path = tmp_path_factory.mktemp("chromvar") / "peaks.h5ad"
    write_h5ad(a, path)  # shuffle + gzip, as GATAC writes matrices
    return a, path


@pytest.fixture
def no_reads(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("the file was read")

    monkeypatch.setattr(h5counts, "_Reader", refuse)


def _x_bytes(path):
    with h5py.File(path, "r") as f:
        return {k: f["X"][k][()].tobytes() for k in ("data", "indices", "indptr")}


def test_in_memory_scores_do_not_depend_on_the_row_chunks(prepared):
    a, _ = prepared
    b = a.copy()
    ga.motif.compute_deviations(b, batch_size=BATCH, rows_per_chunk=2 * BATCH + 1, key_added="again")
    _same(b.obsm["again"], a.obsm["chromvar"])


def test_backed_r_gives_the_in_memory_scores_and_leaves_the_file(prepared, opened):
    a, path = prepared
    backed = opened(path)
    ga.motif.compute_deviations(backed, batch_size=BATCH, rows_per_chunk=3 * BATCH, key_added="from_file")
    got = backed.obsm["from_file"]
    _same(got, a.obsm["chromvar"])
    assert list(got.columns) == list(a.obsm["chromvar"].columns) and list(got.index) == list(a.obs_names)
    assert "from_file" not in backed.file._file["obsm"]


def test_backed_r_plus_writes_the_scores_into_the_file(prepared, tmp_path):
    a, path = prepared
    copy = tmp_path / "copy.h5ad"
    copy.write_bytes(path.read_bytes())
    x_before = _x_bytes(copy)
    backed = ad.read_h5ad(copy, backed="r+")
    try:
        ga.motif.compute_deviations(backed, batch_size=BATCH, key_added="from_file")
    finally:
        backed.file.close()
    assert _x_bytes(copy) == x_before
    _same(ad.read_h5ad(copy).obsm["from_file"], a.obsm["chromvar"])


def test_backed_view_returns_the_scores_of_its_cells(prepared, opened):
    a, path = prepared
    rows = np.flatnonzero(np.arange(a.n_obs) % 3 != 1)
    sub = a[rows].copy()
    expected = ga.motif.compute_deviations(sub, batch_size=BATCH, return_adata=True)
    got = ga.motif.compute_deviations(opened(path)[rows], batch_size=BATCH, return_adata=True)
    _same(got.X, expected.X)
    assert list(got.obs_names) == list(sub.obs_names)


def test_backed_view_needs_return_adata(prepared, opened, no_reads):
    _, path = prepared
    with pytest.raises(ValueError, match="return_adata=True"):
        ga.motif.compute_deviations(opened(path)[:20])


def test_background_peaks_outside_the_matrix_are_refused(prepared):
    a, _ = prepared
    b = a[:, : a.n_vars // 2].copy()
    with pytest.raises(ValueError, match="sample_bg_peaks"):
        ga.motif.compute_deviations(b)


# -- the other readers of peak matrices ---------------------------------------


def test_markers_of_a_backed_matrix(prepared, opened):
    """Only the sampled cells are read; the tables are those of the in-memory matrix."""
    a, path = prepared
    expected = ga.peak.markers(a.copy(), "group", max_cells=30, min_log2_fc=0, min_pct=0)
    got = ga.peak.markers(opened(path), "group", max_cells=30, min_log2_fc=0, min_pct=0)
    assert expected.keys() == got.keys()
    for g in expected:
        pd.testing.assert_frame_equal(got[g], expected[g])


def test_select_features_of_a_backed_matrix(prepared, opened):
    a, path = prepared
    mem = a.copy()
    ga.pp.select_features(mem, n_features=50)
    backed = opened(path)
    ga.pp.select_features(backed, n_features=50)
    np.testing.assert_array_equal(backed.var["selected"], mem.var["selected"])
    np.testing.assert_array_equal(backed.var["accessibility_count"], mem.var["accessibility_count"])
