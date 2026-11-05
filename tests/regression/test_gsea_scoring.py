"""GSEA's hits-only enrichment score against the full running sum it replaces."""

import cupy as cp
import numpy as np

from gatac.motif._gsea import _enrichment_scores_gpu, _es_from_hits_gpu


def test_hits_only_score_matches_running_sum():
    rng = np.random.default_rng(0)
    n = 2_000
    sets = [np.sort(rng.choice(n, k, replace=False)) for k in (1, 2, 5, 50, 300, n // 2)]
    sets.append(np.array([0, n - 1]))  # hits at both ends
    hit_pos = np.full((len(sets), max(map(len, sets))), n, np.int32)  # padding sorts last
    tags = np.zeros((len(sets), n))
    for i, s in enumerate(sets):
        hit_pos[i, : len(s)] = s
        tags[i, s] = 1
    n_hits = cp.asarray([[len(s)] for s in sets], dtype=cp.float32)
    # A metric, and an all-zero one whose clamped normalisation puts the extremum at the end
    for metric in (np.abs(rng.normal(size=n)), np.zeros(n)):
        got = _es_from_hits_gpu(cp.asarray(metric, cp.float32), cp.asarray(hit_pos), n_hits)
        ref = _enrichment_scores_gpu(cp.asarray(metric), cp.asarray(tags))  # float64 reference
        cp.testing.assert_allclose(got, ref, rtol=0, atol=1e-5)
