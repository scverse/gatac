"""Dimensionality reduction: spectral embedding and latent semantic indexing of the cell × feature matrix."""

from gatac._rapids import require as _require_rapids

_require_rapids()

from .lsi import (
    LSIModel,
    iterative_lsi,
    lsi,
    model_from_adata,
    project_lsi,
)
from .spectral import spectral

__all__ = [
    "spectral",
    "lsi",
    "iterative_lsi",
    "project_lsi",
    "model_from_adata",
    "LSIModel",
]
