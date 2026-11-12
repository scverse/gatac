"""
The GPU reader of h5ad count matrices (``gatac.io._h5counts``): every HDF5
layout GATAC and anndata write, read byte for byte as h5py reads it; layouts
it cannot read refused before any read; corrupt deflate chunks refused while
decoding; row and column selections, from views and arguments, composed.
"""

from __future__ import annotations

import zlib

import numpy as np
import pandas as pd
import pytest

cp = pytest.importorskip("cupy")
pytest.importorskip("cudf")
pytest.importorskip("kvikio")

import anndata as ad  # noqa: E402
import h5py  # noqa: E402
import scipy.sparse as sp  # noqa: E402

from gatac.dr._stream import StreamedCounts  # noqa: E402
from gatac.io import _h5counts as h5counts  # noqa: E402
from gatac.io._h5counts import open_counts  # noqa: E402

N_ROWS, N_COLS = 300, 70


def _matrix(dtype=np.uint16, density=0.15, seed=0) -> sp.csr_matrix:
    rng = np.random.default_rng(seed)
    X = sp.random(N_ROWS, N_COLS, density=density, format="csr", random_state=seed, dtype=np.float64)
    X.data = (rng.integers(1, 200, X.nnz)).astype(dtype)
    return sp.csr_matrix(X, dtype=dtype)


def _skeleton(path, n_rows=N_ROWS, n_cols=N_COLS) -> None:
    ad.AnnData(
        obs=pd.DataFrame(index=[f"c{i}" for i in range(n_rows)]),
        var=pd.DataFrame(index=[f"f{j}" for j in range(n_cols)]),
    ).write_h5ad(path)


LAYOUTS = {
    "contiguous": {},
    "chunked": {"chunks": (64,)},
    "gzip": {"chunks": (64,), "compression": "gzip"},
    "shuffle_gzip": {"chunks": (64,), "compression": "gzip", "shuffle": True},
    "big_chunks_gzip": {"chunks": (1 << 14,), "maxshape": (None,), "compression": "gzip", "shuffle": True},
}


def _write(path, X, layout="contiguous", indices_dtype=np.int32, indptr_dtype=np.int64, data_kw=None):
    """An h5ad whose X is ``X`` in the given HDF5 layout."""
    _skeleton(path, *X.shape)
    kw = LAYOUTS[layout] if data_kw is None else data_kw
    with h5py.File(path, "a") as f:
        g = f.create_group("X")
        g.attrs.update({"encoding-type": "csr_matrix", "encoding-version": "0.1.0", "shape": np.array(X.shape)})
        for name, arr in (("data", X.data), ("indices", X.indices.astype(indices_dtype))):
            if arr.size == 0 and "chunks" in kw:
                g.create_dataset(name, shape=(0,), dtype=arr.dtype, maxshape=(None,), **kw)
            else:
                g.create_dataset(name, data=arr, **kw)
        g.create_dataset("indptr", data=X.indptr.astype(indptr_dtype), **kw)
    return path


def _host(C: StreamedCounts) -> sp.csr_matrix:
    """The matrix of ``C`` back on the host."""
    data = cp.asnumpy(C._data) if isinstance(C._data, cp.ndarray) else C._data
    indices = cp.asnumpy(C._indices) if isinstance(C._indices, cp.ndarray) else C._indices
    return sp.csr_matrix((data, indices, C.indptr.get()), shape=C.shape)


def _same(C: StreamedCounts, X: sp.csr_matrix) -> None:
    got = _host(C)
    assert got.shape == X.shape
    expected_dtype = X.dtype if X.dtype in (np.uint8, np.uint16, np.int32, np.float32) else np.float32
    assert got.dtype == expected_dtype and C._indices.dtype == np.int32
    assert (got != X.astype(expected_dtype)).nnz == 0
    # the products see the same matrix
    x = np.random.default_rng(1).random(X.shape[1]).astype(np.float32)
    np.testing.assert_allclose(C.dot(cp.asarray(x)).get(), X.astype(np.float64) @ x, rtol=1e-4)


@pytest.mark.parametrize(
    "layout, value_dtype",
    # every layout, and every value type where its size matters: shuffled
    [(layout, np.uint16) for layout in LAYOUTS]
    + [("shuffle_gzip", t) for t in (np.uint8, np.int32, np.float32, np.float64)],
)
def test_reads_every_layout(tmp_path, layout, value_dtype):
    X = _matrix(value_dtype)
    for idx_dtype, ptr_dtype in ((np.int32, np.int32), (np.int64, np.int64)):
        path = _write(tmp_path / f"{idx_dtype.__name__}.h5ad", X, layout, idx_dtype, ptr_dtype)
        _same(open_counts(path, chunk_size=64), X)


# -- refused before any read ---------------------------------------------------


@pytest.fixture
def no_reads(monkeypatch):
    """Fail the test if the reader is ever created."""

    def refuse(*args, **kwargs):
        raise AssertionError("the file was read")

    monkeypatch.setattr(h5counts, "_Reader", refuse)


def test_refuses_other_filters(tmp_path, no_reads):
    path = _write(tmp_path / "f.h5ad", _matrix(), data_kw={"chunks": (64,), "compression": "lzf"})
    with pytest.raises(ValueError, match="filters"):
        open_counts(path)


# -- corrupt chunks refused while decoding -----------------------------------------


def _with_chunk(tmp_path, X, chunk: bytes):
    """A gzip file whose first data chunk is replaced by the raw stored bytes ``chunk``."""
    path = _write(tmp_path / "c.h5ad", X, "gzip")
    with h5py.File(path, "a") as f:
        f["X/data"].id.write_direct_chunk((0,), chunk)
    return path


def test_refuses_corrupt_chunks(tmp_path):
    """A chunk whose bytes changed raises instead of giving wrong counts."""
    X = _matrix()
    good = zlib.compress(np.ascontiguousarray(X.data[:64]).tobytes(), 6)
    with pytest.raises(ValueError, match="Adler-32"):
        open_counts(_with_chunk(tmp_path, X, good[:-1] + bytes([good[-1] ^ 0xFF])))


# -- selections ----------------------------------------------------------------------


@pytest.fixture
def written(tmp_path):
    X = _matrix(np.uint16)
    return _write(tmp_path / "w.h5ad", X, "shuffle_gzip"), X


@pytest.mark.parametrize(
    "rows, features",
    [
        (None, None),
        (slice(10, 200), slice(5, 60)),
        (slice(None, None, 3), np.array([60, 2, 9, 0])),
        (np.arange(N_ROWS) % 4 != 0, np.arange(N_COLS) % 2 == 0),
        (np.array([5, 3, 299, 0, 3, 3, 150]), None),  # reordered and repeated
    ],
)
def test_selections(written, rows, features):
    path, X = written
    expected = X if rows is None else X[np.arange(N_ROWS)[rows]]
    if features is not None:
        expected = expected[:, np.arange(N_COLS)[features]]
    _same(open_counts(path, rows=rows, features=features), sp.csr_matrix(expected))


def test_backed_views_compose_with_selections(written):
    """A backed view's rows and columns, then ``rows=`` / ``features=`` relative to the view."""
    path, X = written
    backed = ad.read_h5ad(path, backed="r")
    rows, cols = np.array([7, 1, 250, 1, 40, 100]), np.array([30, 3, 50, 10, 0])
    view = backed[rows, cols]
    _same(open_counts(view), sp.csr_matrix(X[rows][:, cols]))
    sub = np.array([True, False, True, True, False])
    _same(open_counts(view, rows=[4, 0], features=sub), sp.csr_matrix(X[rows[[4, 0]]][:, cols[sub]]))
    _same(open_counts(backed[10:50:2]), sp.csr_matrix(X[10:50:2]))
    _same(open_counts(backed), X)
    backed.file.close()


def test_over_the_gpu_budget_reads_block_by_block(written, monkeypatch):
    """Too big for the GPU: decoded in small blocks into host memory, chunks streamed; same matrix."""
    path, X = written
    monkeypatch.setattr(h5counts, "_free_gpu_bytes", lambda: 0)
    monkeypatch.setattr(h5counts, "_MIN_BLOCK_ENTRIES", 200)
    for rows, features in ((None, None), (np.array([299, 4, 4, 120, 0]), np.array([9, 3]))):
        C = open_counts(path, rows=rows, features=features, chunk_size=40, memory_fraction=1e-12)
        assert not isinstance(C._data, cp.ndarray) and C._streamed
        expected = X if rows is None else X[rows][:, features]
        _same(C, sp.csr_matrix(expected))


def test_kvikio_threads_and_import_order():
    """In a fresh process: nvCOMP after cuDF, and kvikio's threads set to the CPUs."""
    import os
    import subprocess
    import sys

    code = (
        "import os, gatac\n"
        "from gatac.io._h5counts import _kvikio\n"
        "from gatac.io.gpu_convert import _nvcomp\n"
        "import kvikio.defaults as d\n"
        "_kvikio(); _nvcomp()\n"
        "print(d.get('num_threads'), len(os.sched_getaffinity(0)))\n"
    )
    env = {k: v for k, v in os.environ.items() if k != "KVIKIO_NTHREADS"}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True)
    threads, cpus = map(int, out.stdout.split()[-2:])
    assert threads == cpus


def _rich_adata() -> ad.AnnData:
    """X, CSR layers of three value and two index types, a dense layer, raw, obsm, uns and a categorical obs column."""
    X = _matrix()
    a = ad.AnnData(
        X,
        obs=pd.DataFrame(
            {"group": pd.Categorical(["a", "b", "c"] * (N_ROWS // 3))}, index=[f"c{i}" for i in range(N_ROWS)]
        ),
        var=pd.DataFrame(index=[f"f{j}" for j in range(N_COLS)]),
        obsm={"X_emb": np.arange(N_ROWS * 2, dtype=np.float32).reshape(N_ROWS, 2)},
        uns={"params": {"k": 3}},
    )
    a.layers["normalized"] = sp.csr_matrix(X / 7.0)  # float64
    wide = sp.csr_matrix(X, dtype=np.int64)  # int64 values, indices and indptr, as stored by anndata
    wide.indices, wide.indptr = wide.indices.astype(np.int64), wide.indptr.astype(np.int64)
    a.layers["int64"] = wide
    a.layers["dense"] = X.toarray()
    a.raw = a[:, :40].copy()
    return a


@pytest.mark.parametrize("compression", [None, "gzip", "lzf"])
def test_read_h5ad_reads_what_anndata_reads(tmp_path, monkeypatch, compression):
    """``gatac.io.read_h5ad`` gives what ``anndata.read_h5ad`` gives: X, the CSR layers and raw.X
    through the GPU reader with their stored value dtypes and the index dtype scipy chooses (as
    anndata from 0.14), a filter it does not read (lzf) and dense layers through anndata."""
    from gatac.io import read_h5ad

    path = tmp_path / "a.h5ad"
    _rich_adata().write_h5ad(path, compression=compression)
    calls = []
    real = h5counts.read_whole
    monkeypatch.setattr(h5counts, "read_whole", lambda m, **kw: calls.append(m.path) or real(m, **kw))

    got, expected = read_h5ad(path), ad.read_h5ad(path)
    assert len(calls) == (0 if compression == "lzf" else 4)  # X, two CSR layers, raw.X
    for g, e in [(got.X, expected.X), (got.raw.X, expected.raw.X)] + [
        (got.layers[k], expected.layers[k]) for k in expected.layers
    ]:
        assert type(g) is type(e) and g.dtype == e.dtype
        if sp.issparse(e):
            # lzf: anndata reads the whole file, with the index dtypes of the installed version
            ref = e if compression == "lzf" else sp.csr_matrix((e.data, e.indices, e.indptr), shape=e.shape)
            assert (g.indices.dtype, g.indptr.dtype) == (ref.indices.dtype, ref.indptr.dtype)
        assert (g != e).sum() == 0
    pd.testing.assert_frame_equal(got.obs, expected.obs)
    pd.testing.assert_frame_equal(got.raw.var, expected.raw.var)
    np.testing.assert_array_equal(got.obsm["X_emb"], expected.obsm["X_emb"])
    assert got.uns["params"]["k"] == 3


def test_read_h5ad_to_gpu_and_backed(tmp_path):
    """``to_gpu=True``: cupy CSR matrices, float32 for integer counts, raw.X kept on the host;
    ``backed``: anndata's backed AnnData."""
    import cupyx.scipy.sparse as cusp

    from gatac.io import read_h5ad

    path = tmp_path / "a.h5ad"
    a = _rich_adata()
    a.write_h5ad(path, compression="gzip")
    got = read_h5ad(path, to_gpu=True)
    assert cusp.isspmatrix_csr(got.X) and got.X.dtype == np.float32
    assert (got.X.get() != a.X).nnz == 0
    assert got.layers["normalized"].dtype == np.float64
    assert sp.issparse(got.raw.X)

    backed = read_h5ad(path, backed="r")
    assert backed.isbacked and (backed.X.to_memory() != a.X).nnz == 0
    backed.file.close()
    with pytest.raises(ValueError, match="to_gpu does not apply"):
        read_h5ad(path, backed="r", to_gpu=True)
