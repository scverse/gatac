"""The plots draw from the inputs they document; their content is not checked."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import gatac as ga


def test_qc_metrics():
    rng = np.random.default_rng(0)
    metrics = pd.DataFrame({"barcode": [f"c{i}" for i in range(200)], "tsse_score": rng.gamma(4, 2, 200),
                            "n_unique": rng.integers(100, 50_000, 200)})
    fig = ga.pl.qc_metrics(metrics)
    assert len(fig.axes) >= 3                                # TSSe, fragments, TSSe vs fragments
    plt.close(fig)


def test_fragment_size_distribution(world):
    ax = ga.pl.fragment_size_distribution(world["parquet"])
    assert len(ax.patches) > 0
    plt.close("all")


def test_motif_enrichment():
    table = pd.DataFrame({"name": ["M0", "M1", "M2", "DECOY"], "adjusted p-value": [1e-8, 1e-3, 0.2, 0.9]})
    fig, axs = ga.pl.motif_enrichment({"g0": table, "g1": table}, top_motifs=3)
    assert np.size(axs) >= 2
    plt.close(fig)
