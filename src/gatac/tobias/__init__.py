"""
TOBIAS on the GPU: Tn5 bias correction and footprint scores per pseudobulk,
and differential TF binding (BINDetect).
"""

from .._rapids import require as _require_rapids

_require_rapids()

from .bindetect import BINDetectResult, SiteSet, bindetect, bindetect_sites
from .footprint import FootprintTracks, TrackLayout, footprint

__all__ = [
    "footprint",
    "FootprintTracks",
    "TrackLayout",
    "bindetect",
    "bindetect_sites",
    "BINDetectResult",
    "SiteSet",
]
