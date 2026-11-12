"""Motifs: reading and scanning motifs, enrichment in peak sets, GSEA on motif rankings, and chromVAR TF-activity scores."""

from gatac._rapids import require as _require_rapids

_require_rapids()

from .chromvar import (
    chromvar,
    compute_deviations,
    compute_peak_bias,
    sample_bg_peaks,
    scan,
)
from .motif import (
    DNAMotif,
    enrichment,
    enrichment_regression,
    gsea,
    presence_matrix,
    read,
    sample_gc_matched_background,
)

__all__ = [
    "read",
    "DNAMotif",
    "sample_gc_matched_background",
    "enrichment",
    "presence_matrix",
    "enrichment_regression",
    "gsea",
    "chromvar",
    "compute_deviations",
    "scan",
    "sample_bg_peaks",
    "compute_peak_bias",
]
