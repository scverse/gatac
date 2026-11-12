"""
``_make_parquet_gpu`` writes the file ``make_parquet`` writes.

Each test converts the same input with both and compares the stored tables
(ids, barcode table, encodings' values) and the footer. Synthetic data, GPU
and nvCOMP required.
"""

from __future__ import annotations

import gzip

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("cupy")
pytest.importorskip("cudf")

import pyarrow.parquet as pq

import gatac as ga
from gatac.io.fragments import _BgzfWriter
from gatac.io.gpu_convert import _gpu_available, _make_parquet_gpu

if not _gpu_available():
    pytest.skip("nvCOMP is not installed", allow_module_level=True)


def _fragments(n: int = 60_000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    contigs = ["chr1", "chr2", "chr10", "chrX", "chrM", "GL000194.1"]
    chrom = rng.choice(contigs, n, p=[0.3, 0.25, 0.2, 0.15, 0.05, 0.05])
    start = rng.integers(0, 2_000_000, n)
    length = rng.integers(40, 700, n)
    weights = rng.pareto(1.2, 250) + 1
    barcodes = np.array([f"{''.join(rng.choice(list('ACGT'), 16))}-1" for _ in range(250)])
    df = pd.DataFrame(
        {
            "chrom": chrom,
            "start": start,
            "end": start + length,
            "barcode": rng.choice(barcodes, n, p=weights / weights.sum()),
            "count": rng.integers(1, 5, n),
        }
    )
    return df.sort_values(["chrom", "start"], kind="stable").reset_index(drop=True)


def _write(path, text: str, kind: str) -> None:
    data = text.encode()
    if kind == "bgzf":
        with _BgzfWriter(path) as fh:
            fh.write(data)
    elif kind == "gzip":
        with gzip.open(path, "wb") as fh:
            fh.write(data)
    elif kind == "zstd":
        import pyarrow as pa

        with pa.CompressedOutputStream(str(path), "zstd") as fh:
            fh.write(data)
    else:
        path.write_bytes(data)


def _text(df: pd.DataFrame, header: bool = True) -> str:
    return ("# a header comment\n" if header else "") + df.to_csv(sep="\t", header=False, index=False)


def _assert_same_file(a, b) -> None:
    ta, tb = pq.read_table(a), pq.read_table(b)
    assert ta.schema == tb.schema
    for name in ta.column_names:
        assert ta[name].equals(tb[name]), name
    ma, mb = pq.ParquetFile(a).metadata.metadata, pq.ParquetFile(b).metadata.metadata
    keys = [k for k in ma if k.startswith(b"gatac.")]
    assert keys and {k: ma[k] for k in keys} == {k: mb.get(k) for k in keys}
    rows = lambda p: [pq.ParquetFile(p).metadata.row_group(i).num_rows for i in range(pq.ParquetFile(p).num_row_groups)]
    assert rows(a) == rows(b)


@pytest.fixture(scope="module")
def frags():
    return _fragments()


@pytest.mark.parametrize("kind", ["bgzf", "plain", "gzip"])
@pytest.mark.parametrize("chunk_bytes", [3_000, 1 << 20, 256 << 20])
def test_same_file_as_make_parquet(frags, tmp_path, kind, chunk_bytes):
    """Small chunks split BGZF blocks and lines across chunks; gzip falls back to the CPU."""
    src = tmp_path / ("frags.tsv" if kind == "plain" else "frags.tsv.gz")
    _write(src, _text(frags), kind)
    cpu = ga.io.make_parquet(src, tmp_path / "cpu.parquet", barcode_prefix="S1_", row_group_size=10_000, gpu=False)
    gpu = _make_parquet_gpu(
        src, tmp_path / "gpu.parquet", barcode_prefix="S1_", row_group_size=10_000, chunk_bytes=chunk_bytes
    )
    _assert_same_file(cpu, gpu)


@pytest.mark.parametrize(
    "line, match",
    [
        ("chr1\t5\t60\tBCX\t1\textra", "Expected Number of Columns: 5 Found: 6"),
        ("chr1\t5\t60\tBCX", "Expected Number of Columns: 5 Found: 4"),
        ("chr1\tabc\t60\tBCX\t1", 'Could not convert string "abc"'),
        ("chr1\t-5\t60\tBCX\t1", 'Could not convert string "-5"'),
        ("chr1\t5\t60\tBCX\t70000", 'Could not convert string "70000"'),
        ("chr1\t5\t60\t\t1", "empty chrom, start, end or barcode"),
    ],
)
def test_invalid_lines_as_make_parquet(frags, tmp_path, caplog, line, match):
    src = tmp_path / "frags.tsv.gz"
    _write(src, _text(frags.head(500)) + line + "\n" + _text(frags.iloc[500:1000], header=False), "bgzf")
    with pytest.raises(ValueError, match=match):
        _make_parquet_gpu(src, tmp_path / "gpu.parquet")
    assert sorted(p.name for p in tmp_path.iterdir()) == [src.name]
    cpu = ga.io.make_parquet(src, tmp_path / "cpu.parquet", skip_invalid=True, gpu=False)
    caplog.clear()
    gpu = _make_parquet_gpu(src, tmp_path / "gpu.parquet", skip_invalid=True)
    assert "skipped 1 invalid line(s)" in caplog.text
    _assert_same_file(cpu, gpu)


def test_empty_count_is_null_and_crlf(frags, tmp_path):
    """An empty count is null (and not summed); CRLF line ends are read as LF."""
    df = frags.head(2_000).astype({"count": object})
    # Empty counts on every other row of the most frequent barcode, which
    # keeps some counts: make_parquet cannot store a barcode whose counts
    # are all empty (its n_total is a NULL sum)
    top = df["barcode"].value_counts().index[0]
    rows = df.index[df["barcode"] == top][::2]
    df.loc[rows, "count"] = ""
    src = tmp_path / "frags.tsv.gz"
    _write(src, _text(df).replace("\n", "\r\n"), "bgzf")
    cpu = ga.io.make_parquet(src, tmp_path / "cpu.parquet", gpu=False)
    gpu = _make_parquet_gpu(src, tmp_path / "gpu.parquet")
    assert pq.read_table(gpu)["count"].null_count == len(rows)
    _assert_same_file(cpu, gpu)


def test_barcode_with_only_empty_counts(tmp_path):
    """Its n_total is 0 (make_parquet fails on such a barcode)."""
    src = tmp_path / "frags.tsv.gz"
    _write(src, "chr1\t1\t50\tA\t2\nchr1\t5\t60\tB\t\nchr1\t9\t70\tA\t3\n", "bgzf")
    counts = ga.io.FragmentFile(_make_parquet_gpu(src, tmp_path / "gpu.parquet")).barcode_counts()
    got = dict(zip(counts["barcode"], zip(counts["n_unique"], counts["n_total"])))
    assert got == {"A": (2, 5), "B": (1, 0)}


def test_end_before_start_is_rejected(tmp_path):
    src = tmp_path / "bad.tsv.gz"
    _write(src, "chr1\t100\t200\tA\t1\nchr1\t500\t400\tA\t1\n", "bgzf")
    with pytest.raises(ValueError, match="ends before it starts"):
        _make_parquet_gpu(src, tmp_path / "gpu.parquet")
    assert sorted(p.name for p in tmp_path.iterdir()) == [src.name]


def test_truncated_bgzf_is_an_error(frags, tmp_path):
    src = tmp_path / "frags.tsv.gz"
    _write(src, _text(frags), "bgzf")
    data = src.read_bytes()
    src.write_bytes(data[: len(data) // 2])
    with pytest.raises(ValueError, match="Truncated BGZF"):
        _make_parquet_gpu(src, tmp_path / "gpu.parquet")


def test_corrupt_bgzf_block_is_an_error(frags, tmp_path):
    """A block that does not inflate to its trailer's size fails, even with skip_invalid, and
    leaves the output as it was; an empty block between two others is read."""
    from gatac.io.fragments import _BGZF_EOF

    src, out = tmp_path / "frags.tsv.gz", tmp_path / "gpu.parquet"
    _write(src, _text(frags), "bgzf")
    good = src.read_bytes()
    first = good[16] + (good[17] << 8) + 1  # size of the first block
    flipped = bytearray(good)
    flipped[40] ^= 0xFF  # in its deflate data
    no_size = bytearray(good)
    no_size[first - 4 : first] = bytes(4)  # its trailer says it holds no text
    out.write_bytes(b"previous")
    for bad in (flipped, no_size):
        src.write_bytes(bytes(bad))
        with pytest.raises(ValueError, match="Corrupt BGZF block"):
            _make_parquet_gpu(src, out, skip_invalid=True)
        assert out.read_bytes() == b"previous"
        assert sorted(p.name for p in tmp_path.iterdir()) == [src.name, out.name]
    src.write_bytes(good[:first] + _BGZF_EOF + good[first:])
    _assert_same_file(ga.io.make_parquet(src, tmp_path / "cpu.parquet", gpu=False), _make_parquet_gpu(src, out))


def test_batch_and_cli_gpu(frags, tmp_path, monkeypatch, caplog):
    """make_parquet_batch(gpu=True), BGZF on the GPU next to plain gzip and Zstandard on
    DuckDB, and `gatac convert --gpu` write what the CPU route writes."""
    import logging

    from gatac.cli import main

    caplog.set_level(logging.INFO)
    srcs = []
    for name, part, kind in (
        ("s1", frags.iloc[:30_000], "bgzf"),
        ("s2", frags.iloc[30_000:45_000], "gzip"),
        ("s3", frags.iloc[45_000:], "zstd"),
    ):
        src = tmp_path / f"{name}.tsv.{'zst' if kind == 'zstd' else 'gz'}"
        _write(src, _text(part), kind)
        srcs.append(src)
    cpu = ga.io.make_parquet_batch(srcs, output_dir=tmp_path / "cpu", barcode_prefix="{sample}_", workers=1, gpu=False)
    for workers in (1, 2):
        gpu = ga.io.make_parquet_batch(
            srcs, output_dir=tmp_path / f"gpu{workers}", barcode_prefix="{sample}_", gpu=True, workers=workers
        )
        for a, b in zip(cpu, gpu):
            _assert_same_file(a, b)
    assert "1 on the GPU with 1 worker(s), 2 with DuckDB" in caplog.text
    monkeypatch.setattr("sys.argv", ["gatac", "convert", str(srcs[0]), "-o", str(tmp_path / "cli.parquet"), "--gpu"])
    main()
    ref = ga.io.make_parquet(srcs[0], tmp_path / "ref.parquet", gpu=False)
    _assert_same_file(ref, tmp_path / "cli.parquet")
    monkeypatch.setattr(
        "sys.argv", ["gatac", "convert", str(srcs[0]), "-o", str(tmp_path / "cli_cpu.parquet"), "--cpu"]
    )
    main()
    _assert_same_file(ref, tmp_path / "cli_cpu.parquet")


@pytest.mark.parametrize(
    "workers, cpus, gpu_auto, files, expected",
    [
        ("auto", 20, 4, (70, 30), (4, 15, 1)),  # gzip files on the CPUs the GPU workers leave
        ("auto", 20, 4, (70, 1), (4, 1, 15)),  # as threads when they are few
        (6, 20, 4, (70, 30), (6, 6, 2)),  # an int sets both pools
    ],
)
def test_batch_workers(monkeypatch, workers, cpus, gpu_auto, files, expected):
    from gatac import _workers
    from gatac.io import convert, gpu_convert

    monkeypatch.setattr(_workers.os, "process_cpu_count", lambda: cpus)
    monkeypatch.setattr(gpu_convert, "_default_gpu_workers", lambda: gpu_auto)
    assert convert._batch_workers(workers, *files) == expected


def test_gpu_is_the_default(frags, tmp_path, caplog):
    """make_parquet converts on the GPU unless gpu=False; plain gzip stays on DuckDB."""
    import logging

    caplog.set_level(logging.INFO)
    bgzf, plain_gz = tmp_path / "a.tsv.gz", tmp_path / "b.tsv.gz"
    _write(bgzf, _text(frags.head(3_000)), "bgzf")
    _write(plain_gz, _text(frags.head(3_000)), "gzip")
    ga.io.make_parquet(bgzf, tmp_path / "a.parquet")
    assert "a.tsv.gz to Parquet on the GPU" in caplog.text
    caplog.clear()
    ga.io.make_parquet(bgzf, tmp_path / "a_cpu.parquet", gpu=False)
    assert "on the GPU" not in caplog.text
    _assert_same_file(tmp_path / "a.parquet", tmp_path / "a_cpu.parquet")
    caplog.clear()
    ga.io.make_parquet(plain_gz, tmp_path / "b.parquet")
    assert "not BGZF (one deflate stream, which cannot be split); converting it with DuckDB" in caplog.text


@pytest.mark.parametrize("cpus, expected", [(1, (1, 1)), (2, (1, 1)), (4, (3, 3)), (8, (4, 7)), (64, (4, 63))])
def test_default_workers(monkeypatch, cpus, expected):
    """auto: usable CPUs minus one, capped at 4 except for the DuckDB import."""
    from gatac import _workers

    monkeypatch.setattr(_workers.os, "process_cpu_count", lambda: cpus)
    assert (_workers.default_workers(), _workers.default_workers(cap=None)) == expected
    assert _workers.resolve_workers("auto", 3) == 3 and _workers.resolve_workers(None, 3) == 3
    assert _workers.resolve_workers(2, 3) == 2
    with pytest.raises(ValueError):
        _workers.resolve_workers(0, 3)
