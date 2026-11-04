"""Every ``gatac`` command on the shared synthetic data (more convert cases: test_fragment_files)."""
import shutil

import anndata as ad
import numpy as np
import pandas as pd
import pytest

import gatac as ga
from gatac import cli


@pytest.fixture
def files(world, tmp_path):
    tsv = tmp_path / "a.tsv.gz"
    shutil.copy(world["tsv"], tsv)
    with ga.io.Fasta(world["fasta"]) as fa:
        sizes = fa.chrom_sizes
    (tmp_path / "sizes.txt").write_text("".join(f"{c}\t{n}\n" for c, n in sizes.items()))
    rng = np.random.default_rng(0)
    rows = []
    for i in range(20):
        c = ("chr1", "chr2")[i % 2]
        s = int(rng.integers(10_000, sizes[c] - 60_000))
        attrs = f'gene_id "G{i}"; gene_name "GENE{i}"; transcript_id "T{i}";'
        for feature in ("gene", "transcript"):
            rows.append(f"{c}\ttest\t{feature}\t{s}\t{s + 20_000}\t.\t{'+-'[i % 2]}\t.\t{attrs}")
    (tmp_path / "genes.gtf").write_text("\n".join(rows) + "\n")
    return {"tsv": tsv, "sizes": sizes, "dir": tmp_path}


def test_commands(files, monkeypatch):
    d, frags = files["dir"], files["dir"] / "a.parquet"
    gtf, sizes = d / "genes.gtf", d / "sizes.txt"

    def gatac(*argv):
        monkeypatch.setattr("sys.argv", ["gatac", *map(str, argv)])
        cli.main()

    # TSV -> Parquet -> TSV gives the original file's fragments back
    gatac("convert", files["tsv"])
    gatac("convert", frags, "-o", d / "back.tsv.gz")
    cols = ["chrom", "start", "end", "barcode", "count"]
    pd.testing.assert_frame_equal(pd.read_csv(d / "back.tsv.gz", sep="\t", header=None, names=cols),
                                  pd.read_csv(files["tsv"], sep="\t", header=None, names=cols))

    gatac("rename", frags, "--barcode-prefix", "B_", "-o", d / "b.parquet")
    assert ga.io.FragmentFile(d / "b.parquet").barcode_prefix == "B_"

    gatac("tile", frags, "-g", sizes, "-o", d / "a.h5ad", "-m", 100)
    gatac("tile", d / "b.parquet", "-g", sizes, "-o", d / "b.h5ad", "-m", 100)
    tiles = ad.read_h5ad(d / "a.h5ad")
    ref = ga.pp.make_tile_matrix(frags, chrom_sizes=files["sizes"], min_fragments_per_cell=100)
    assert tiles.obs_names.equals(ref.obs_names) and (tiles.X != ref.X).nnz == 0

    gatac("combine", d / "a.h5ad", d / "b.h5ad", "-o", d / "ab.h5ad")
    both = ad.read_h5ad(d / "ab.h5ad")
    assert list(both.obs_names) == list(tiles.obs_names) + ["B_" + n for n in tiles.obs_names]

    gatac("metrics", frags, "-g", gtf, "-o", d / "m.csv", "--min-frags", 100)
    metrics = pd.read_csv(d / "m.csv")
    assert {"barcode", "n_unique", "tsse_score"} <= set(metrics.columns)
    assert set(metrics["barcode"]) == set(tiles.obs_names)

    gatac("filter", frags, "--metrics", d / "m.csv", "--filter", "n_unique >= 1500", "-g", sizes,
          "-o", d / "f.parquet")
    kept = ga.io.FragmentFile(d / "f.parquet").barcode_counts()["barcode"]
    assert set(kept) == set(metrics.loc[metrics["n_unique"] >= 1500, "barcode"]) and len(kept) > 0

    for cmd in ("gene", "genescore"):
        gatac(cmd, frags, "-g", gtf, "-o", d / f"{cmd}.h5ad", "-m", 100)
        genes = ad.read_h5ad(d / f"{cmd}.h5ad")
        assert genes.n_vars == 20 and set(genes.obs_names) == set(tiles.obs_names)

    gatac("doublets", frags, "-g", sizes, "-m", 100)              # default output next to the input
    doublets = pd.read_csv(d / "a_doublets.csv")
    assert set(doublets["cell_id"]) == set(tiles.obs_names)


def test_version_and_errors(tmp_path, monkeypatch, capsys, caplog):
    """--version; an error is one line and exit status 1, and -v adds its traceback."""
    def gatac(*argv):
        monkeypatch.setattr("sys.argv", ["gatac", *map(str, argv)])
        with pytest.raises(SystemExit) as e:
            cli.main()
        return e.value.code

    assert gatac("--version") == 0
    assert capsys.readouterr().out.strip() == f"gatac {ga.__version__}"
    bad = tmp_path / "sizes.txt"
    bad.write_text("chr1\tnot-a-number\n")
    frags = tmp_path / "f.parquet"
    frags.touch()
    assert gatac("tile", frags, "-g", bad) == 1
    assert "Error reading the chromosome sizes file" in caplog.text
    assert "Traceback" not in caplog.text
    caplog.clear()
    assert gatac("-v", "tile", frags, "-g", bad) == 1
    assert "gatac tile failed" in caplog.text and "Traceback" in caplog.text
