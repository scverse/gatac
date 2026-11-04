"""
Peaks: calling, merging, the cell × peak matrix, and marker peaks.
"""

from .._rapids import require as _require_rapids

_require_rapids()

from .peaks import call, merge, make_matrix
from .markers import markers, get_markers

__all__ = [
    "call",
    "merge",
    "make_matrix",
    "markers",
    "get_markers",
]
