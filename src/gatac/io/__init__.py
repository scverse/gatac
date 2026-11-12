from gatac._rapids import require as _require_rapids

_require_rapids()

from gatac._readers import read_bed, read_gtf, read_metrics

from .convert import make_parquet, make_parquet_batch, sample_names, set_barcode_prefix, to_tsv
from .fasta import Fasta
from .fragments import FragmentFile, read_fragments_parquet
from .h5ad import read_h5ad, write_h5ad

__all__ = [
    "make_parquet",
    "make_parquet_batch",
    "sample_names",
    "set_barcode_prefix",
    "to_tsv",
    "FragmentFile",
    "Fasta",
    "read_fragments_parquet",
    "read_bed",
    "read_gtf",
    "read_metrics",
    "read_h5ad",
    "write_h5ad",
]
