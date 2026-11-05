from ._utils import cleanup_gpu_memory
from .amulet import detect_doublets
from .features import select_features
from .filter import filter_fragments
from .gene import make_gene_matrix
from .genescore import make_gene_score_matrix
from .genome import HG19 as _HG19
from .genome import HG38 as _HG38
from .genome import MM10 as _MM10
from .genome import MM39 as _MM39
from .metrics import compute_metrics
from .process import combine
from .tile import make_tile_matrix
from .tile_scan import build_tile_matrix, scan_tiles

#: GRCh38 / hg38 chromosome sizes.
HG38 = _HG38
#: GRCh37 / hg19 chromosome sizes.
HG19 = _HG19
#: GRCm38 / mm10 chromosome sizes.
MM10 = _MM10
#: GRCm39 / mm39 chromosome sizes.
MM39 = _MM39

__all__ = [
    "make_tile_matrix",
    "scan_tiles",
    "build_tile_matrix",
    "make_gene_matrix",
    "make_gene_score_matrix",
    "select_features",
    "combine",
    "compute_metrics",
    "cleanup_gpu_memory",
    "filter_fragments",
    "detect_doublets",
    "HG38",
    "HG19",
    "MM10",
    "MM39",
]
