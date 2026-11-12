"""Peaks: calling, merging, the cell × peak matrix, and marker peaks."""

from gatac._rapids import require as _require_rapids

_require_rapids()

from .markers import get_markers, markers
from .peaks import call, make_matrix, merge

__all__ = [
    "call",
    "merge",
    "make_matrix",
    "markers",
    "get_markers",
]
