"""
``gatac.pp`` feature selection and the multi-sample helpers.

The fragment-level builders (tile, gene and metric matrices, filtering,
doublets) are covered in ``tests/regression/test_fragment_files.py`` and the
reproducibility suite; this file covers what operates on built matrices.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scanpy as sc

import gatac as ga


@pytest.fixture
def split_h5ad(tiles, tmp_path):
    """The tile matrix written as two h5ad files, alternate cells in each, and the matrix in that cell order."""
    paths = [tmp_path / "A_tile.h5ad", tmp_path / "B_tile.h5ad"]
    tiles[0::2].write_h5ad(paths[0])
    tiles[1::2].write_h5ad(paths[1])
    order = np.r_[np.arange(0, tiles.n_obs, 2), np.arange(1, tiles.n_obs, 2)]
    return paths, tiles[order].copy()


def _binarized_tile_matrix(tiles, world):
    from pyfaidx import Fasta

    sizes = {c: len(s) for c, s in Fasta(str(world["fasta"])).items()}
    a = ga.pp.make_tile_matrix(
        world["parquet"], chrom_sizes=sizes, tile_size=5_000, min_fragments_per_cell=100, count_strategy="binarize"
    )
    return a[tiles.obs_names].copy()


@pytest.mark.parametrize("source", ["bool", "make_tile_matrix"])
def test_select_features_ignores_quantiles_on_binary_input(tiles, world, source):
    """Binary matrices: plain top-N by accessibility, quantile filters "ignored for binary matrices"."""
    a = _binarized_tile_matrix(tiles, world) if source == "make_tile_matrix" else tiles
    if source == "bool":
        a.X = a.X.astype(bool)
    n = a.n_vars // 3
    ga.pp.select_features(a, n_features=n, filter_lower_quantile=0.1, filter_upper_quantile=0.1)
    sel = a.var["selected"].to_numpy()
    count = a.var["accessibility_count"].to_numpy()
    assert sel.sum() == n
    assert count[sel].min() >= count[~sel].max()


def test_combine_concatenates_files_and_rejects_duplicate_barcodes(split_h5ad, tmp_path):
    """combine stacks the files' cells in order with ``source_file``, and refuses colliding barcodes."""
    paths, ordered = split_h5ad
    out = tmp_path / "combined.h5ad"
    ga.pp.combine(paths, out)
    got = sc.read_h5ad(out)
    assert list(got.obs_names) == list(ordered.obs_names)
    assert list(got.var_names) == list(ordered.var_names)
    assert (got.X != ordered.X).nnz == 0
    n_a = (ordered.n_obs + 1) // 2
    assert got.obs["source_file"].tolist() == ["A_tile.h5ad"] * n_a + ["B_tile.h5ad"] * (ordered.n_obs - n_a)
    with pytest.raises(ValueError, match="cell names are in more than one of the files"):
        ga.pp.combine([paths[0], paths[0]], tmp_path / "dup.h5ad")


def _one_cell(path, cell, features, values):
    import anndata as ad
    import scipy.sparse as sp
    ad.AnnData(sp.csr_matrix(np.array([values], dtype=np.float32)),
               obs=pd.DataFrame(index=[cell]), var=pd.DataFrame(index=features)).write_h5ad(path)
    return path


@pytest.mark.parametrize("features_b, match", [
    (["geneB", "geneA"], "same 2 features in another order"),
    (["geneA", "geneC"], "1 of them not among the 2"),
])
def test_combine_rejects_other_features_and_keeps_the_output(tmp_path, features_b, match):
    """Columns are stacked by position: files whose features differ, even only in order, are refused
    before the existing output is touched."""
    a = _one_cell(tmp_path / "a.h5ad", "c1", ["geneA", "geneB"], [1, 2])
    b = _one_cell(tmp_path / "b.h5ad", "c2", features_b, [10, 20])
    out = tmp_path / "out.h5ad"
    ga.pp.combine([a], out)
    before = out.read_bytes()
    with pytest.raises(ValueError, match=match):
        ga.pp.combine([a, b], out)
    assert out.read_bytes() == before
    assert sorted(f.name for f in tmp_path.iterdir()) == ["a.h5ad", "b.h5ad", "out.h5ad"]


def test_tile_matrix_from_interval_matrix_sums_overlapping_features(peaks, world, tmp_path):
    """An .h5ad of peaks (var names "chrom:start-end"): each tile is the sum of the peaks it overlaps."""
    peaks.write_h5ad(tmp_path / "peaks.h5ad")
    with ga.io.Fasta(world["fasta"]) as fa:
        sizes = fa.chrom_sizes
    got = ga.pp.make_tile_matrix(tmp_path / "peaks.h5ad", chrom_sizes=sizes, tile_size=5_000,
                                 min_fragments_per_cell=0)
    assert list(got.obs_names) == list(peaks.obs_names)
    tile = {(c, int(s)): j for j, (c, s) in enumerate(zip(got.var["chrom"], got.var["start"]))}
    X = peaks.X.toarray().astype(np.int64)
    expected = np.zeros((peaks.n_obs, got.n_vars), np.int64)
    coords = pd.Series(peaks.var_names).str.extract(r"^(\w+):(\d+)-(\d+)$")
    for k, (c, s, e) in enumerate(coords.itertuples(index=False)):
        for t in range(int(s) // 5_000 * 5_000, int(e), 5_000):
            expected[:, tile[(c, t)]] += X[:, k]
    np.testing.assert_array_equal(got.X.toarray(), expected)
