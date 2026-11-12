"""
``gatac.io._csr_writer.CSRWriter``: a matrix written block by block, shuffle + gzip
compressed on the GPU or uncompressed, reads back the same through h5py,
anndata and the GPU reader; values are promoted when a later block needs a
wider dtype; any failure leaves an existing output as it was, with no
temporary file, writer process or shared memory left behind.
"""

from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd
import pytest

cp = pytest.importorskip("cupy")
pytest.importorskip("cudf")
pytest.importorskip("kvikio")

import anndata as ad  # noqa: E402
import cupyx.scipy.sparse as csp  # noqa: E402
import h5py  # noqa: E402
import scipy.sparse as sp  # noqa: E402

from gatac.io import _csr_writer as cw  # noqa: E402
from gatac.io._csr_writer import CSRWriter  # noqa: E402
from gatac.io._h5counts import open_counts  # noqa: E402

N_COLS = 97
SMALL = {"chunk_bytes": 1024, "batch_bytes": 8 * 1024}  # many chunks, several batches, blocks across chunks


def _block(n_rows, dtype=np.uint16, seed=0, density=0.1):
    rng = np.random.default_rng(seed)
    X = sp.random(n_rows, N_COLS, density=density, format="csr", random_state=seed)
    X.data = rng.integers(1, min(np.iinfo(dtype).max, 1000) if np.issubdtype(dtype, np.integer) else 1000, X.nnz)
    return sp.csr_matrix(X, dtype=dtype)


def _meta(n_rows):
    obs = pd.DataFrame({"g": np.arange(n_rows) % 3}, index=[f"c{i}" for i in range(n_rows)])
    var = pd.DataFrame(index=[f"f{j}" for j in range(N_COLS)])
    return obs, var


def _write(path, blocks, **kw):
    n = sum(b.shape[0] for b in blocks)
    obs, var = _meta(n)
    with CSRWriter(path, **{**SMALL, **kw}) as w:
        for b in blocks:
            w.append(b)
        w.finalize(obs=obs, var=var, uns={"note": "kept"})
    return sp.vstack([b.get() if csp.issparse(b) else b for b in blocks], format="csr") if blocks else None


def _streamed(C) -> sp.csr_matrix:
    data = cp.asnumpy(C._data) if isinstance(C._data, cp.ndarray) else C._data
    indices = cp.asnumpy(C._indices) if isinstance(C._indices, cp.ndarray) else C._indices
    return sp.csr_matrix((data, indices, C.indptr.get()), shape=C.shape)


def _check(path, X, dtype, compression="gzip"):
    with h5py.File(path, "r") as f:
        g = f["X"]
        assert g["data"].dtype == dtype
        assert g["indices"].dtype == np.int32 and g["indptr"].dtype == np.int32
        for k in ("data", "indices"):
            assert g[k].compression == compression and g[k].shuffle == (compression == "gzip")
    a = ad.read_h5ad(path)
    assert a.X.dtype == dtype and a.X.shape == X.shape
    assert (a.X != X.astype(dtype)).nnz == 0
    assert a.uns["note"] == "kept" and list(a.obs["g"]) == list(np.arange(X.shape[0]) % 3)
    got = _streamed(open_counts(path, chunk_size=200))
    assert (got != X).nnz == 0


@pytest.fixture(autouse=True)
def nothing_left(tmp_path):
    """No temporary file and no shared memory of this process after any test."""
    yield
    assert not glob.glob(f"/dev/shm/gatac-csr-{os.getpid()}-*")
    assert not [p for p in os.listdir(tmp_path) if p.startswith(".")]


@pytest.mark.parametrize(
    "compression, dtype", [("gzip", t) for t in (np.uint8, np.uint16, np.int32, np.float32)] + [(None, np.uint16)]
)
def test_round_trip(tmp_path, compression, dtype):
    """Blocks of every size, empty ones included, written over an existing file."""
    (tmp_path / "m.h5ad").write_bytes(b"the previous output")
    blocks = [_block(n, dtype, seed=i) for i, n in enumerate([120, 1, 0, 333, 57])]
    X = _write(tmp_path / "m.h5ad", blocks, compression=compression)
    _check(tmp_path / "m.h5ad", X, dtype, compression)


def test_values_are_promoted(tmp_path):
    """uint8, then uint16, then int32 blocks: every value is kept, in the widest dtype."""
    blocks = [_block(400, np.uint8, 0), _block(10, np.uint8, 1), _block(300, np.uint16, 2), _block(200, np.int32, 3)]
    blocks[2].data[:] = 60_000
    blocks[3].data[:5] = 2**30
    X = _write(tmp_path / "m.h5ad", blocks)
    assert X.dtype == np.int32
    _check(tmp_path / "m.h5ad", X, np.int32)


def test_given_value_dtype(tmp_path):
    blocks = [_block(50, np.uint8), _block(50, np.uint16, 1)]
    X = _write(tmp_path / "m.h5ad", blocks, value_dtype=np.uint16)
    _check(tmp_path / "m.h5ad", X, np.uint16)


def test_known_nnz_gives_contiguous_uncompressed_datasets(tmp_path):
    blocks = [_block(100, np.uint8), _block(80, np.uint16, 1)]
    nnz = sum(b.nnz for b in blocks)
    X = _write(tmp_path / "m.h5ad", blocks, compression=None, nnz=nnz)
    _check(tmp_path / "m.h5ad", X, np.uint16, None)
    with h5py.File(tmp_path / "m.h5ad", "r") as f:
        assert f["X/data"].chunks is None and f["X/indices"].chunks is None


# -- failures leave an existing output as it was ------------------------------


@pytest.fixture
def existing(tmp_path):
    path = tmp_path / "out.h5ad"
    path.write_bytes(b"the previous output")
    return path


def _intact(path):
    assert path.read_bytes() == b"the previous output"


def test_error_in_the_block(existing):
    with pytest.raises(KeyError), CSRWriter(existing, **SMALL) as w:
        w.append(_block(500))
        raise KeyError("a later feature mismatch")
    _intact(existing)


def test_writer_process_error_is_reported(existing, monkeypatch):
    """An HDF5 error in the child comes back with its traceback."""
    real = cw._Child.send

    def bad_dataset(self, *msg):
        if msg[0] == "batch":
            msg = (msg[0], "X/missing", *msg[2:])
        real(self, *msg)

    monkeypatch.setattr(cw._Child, "send", bad_dataset)
    with pytest.raises(RuntimeError, match="KeyError"), CSRWriter(existing, **SMALL) as w:
        for i in range(4):
            w.append(_block(500, seed=i))
        obs, var = _meta(2000)
        w.finalize(obs=obs, var=var)
    _intact(existing)


def test_write_h5ad_takes_cupy_x_and_raw(tmp_path):
    """``gatac.io.write_h5ad``: a cupy X is written as a scipy one, raw is kept and compressed
    as X is, and ``gatac.io.read_h5ad`` reads the file back."""
    from gatac.io import read_h5ad, write_h5ad

    X = _block(300, dtype=np.float32)  # cupy's sparse matrices hold floats
    a = ad.AnnData(csp.csr_matrix(X), obs=pd.DataFrame(index=[f"c{i}" for i in range(300)]))
    a.raw = ad.AnnData(X[:, :20], obs=a.obs)
    path = tmp_path / "c.h5ad"
    write_h5ad(path, a)
    with h5py.File(path, "r") as f:
        assert f["X/data"].compression == "gzip" and f["X/indices"].dtype == np.int32
        assert f["raw/X/data"].compression == "gzip"  # the rest as anndata compresses it
    for got in (ad.read_h5ad(path), read_h5ad(path)):
        assert (got.X != X).nnz == 0 and got.X.dtype == np.float32
        assert (got.raw.X != X[:, :20]).nnz == 0
    with pytest.raises(ValueError, match="compression"):
        write_h5ad(path, a, compression="lzf")
