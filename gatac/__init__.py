"""
GATAC - GPU-accelerated ATAC-seq processing toolkit.

Public API:
    - io: Fragment files (reading, writing, conversion)
    - pp: Preprocessing functions
    - dr: Dimensionality reduction
    - peak: Peak calling and marker peaks
    - motif: Motif enrichment and chromVAR
    - tobias: TOBIAS footprinting and differential TF binding
    - pl: Plotting
    - gpu: GPU helpers for code built on GATAC

Installed extensions (packages with a ``gatac.ext`` entry point) are reached
as ``gatac.<name>``, imported on first use; see :func:`extensions`.
"""

import logging
import os
from importlib.metadata import EntryPoint, entry_points, version

# libcudf means to default KvikIO to POSIX reads, but opens the first file
# before applying that, so each process probes cuFile/GPUDirect Storage once
# (seconds on nodes without it). Apply libcudf's intended default up front.
os.environ.setdefault("KVIKIO_COMPAT_MODE", "ON")
# KvikIO 26.10 writes files with direct I/O by default, which fails with an
# illegal memory access on some systems (WSL2); cuDF's to_csv / to_parquet go
# through it. Writing through the page cache, as before 26.10, works everywhere.
os.environ.setdefault("KVIKIO_AUTO_DIRECT_IO_WRITE", "OFF")

# `pip install gatac` without a CUDA extra installs no RAPIDS; say how to fix it
# instead of failing on `import cudf` deep inside gatac.io
try:
    import cudf  # noqa: F401
    import cupy  # noqa: F401
except ModuleNotFoundError as e:
    raise ImportError(
        f"GATAC runs on the GPU through RAPIDS, which is missing ({e}). Install it with "
        "the CUDA extra matching the CUDA version nvidia-smi reports: "
        "pip install 'gatac[cuda12]' or 'gatac[cuda13]'"
    ) from e

from . import io
from . import pp
from . import dr
from . import peak
from . import motif
from . import tobias
from . import pl
from . import gpu

__version__ = version("gatac")
__all__ = ["io", "pp", "dr", "peak", "motif", "tobias", "pl", "gpu", "extensions", "set_verbosity"]


# ---------------------------------------------------------------------------
# Extensions
# ---------------------------------------------------------------------------
# A package extends GATAC by declaring an entry point in the "gatac.ext"
# group, e.g. in its pyproject.toml:
#
#     [project.entry-points."gatac.ext"]
#     chromafold = "gatac_chromafold"
#
# and ``gatac.chromafold`` then imports gatac_chromafold on first access.
# Core names always win: an extension cannot replace gatac.io, gatac.pp, ...


def extensions() -> dict[str, EntryPoint]:
    """
    Installed GATAC extensions, by the name they are reached at (``gatac.<name>``).

    Examples
    --------
    >>> import gatac as ga
    >>> ga.extensions()
    {'chromafold': EntryPoint(name='chromafold', value='gatac_chromafold', group='gatac.ext')}
    """
    return {ep.name: ep for ep in entry_points(group="gatac.ext")}


def __getattr__(name: str):
    ep = extensions().get(name) if not name.startswith("_") else None
    if ep is None:
        raise AttributeError(f"module 'gatac' has no attribute {name!r}")
    module = ep.load()
    globals()[name] = module
    return module


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(extensions()))


# ---------------------------------------------------------------------------
# Package-level logging
# ---------------------------------------------------------------------------
# GATAC's INFO messages are shown by default: on stderr while logging is not
# configured, and through the application's own handlers once it is (root
# logger handlers, e.g. logging.basicConfig or pytest's caplog), so they are
# never printed twice. gatac.set_verbosity("WARNING") silences them.


class _DefaultHandler(logging.StreamHandler):
    """GATAC's messages on stderr, while the root logger has no handler of its own."""

    def emit(self, record: logging.LogRecord) -> None:
        if not logging.getLogger().handlers:
            super().emit(record)


_logger = logging.getLogger("gatac")
_logger.setLevel(logging.INFO)
_handler = _DefaultHandler()
_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                      datefmt="%H:%M:%S")
)
_logger.addHandler(_handler)


def set_verbosity(level: str | int) -> None:
    """
    Set the logging verbosity for all GATAC submodules.

    Parameters
    ----------
    level
        A standard logging level name (``"DEBUG"``, ``"INFO"``,
        ``"WARNING"``, ``"ERROR"``) or the corresponding integer constant.

    Examples
    --------
    >>> import gatac
    >>> gatac.set_verbosity("DEBUG")   # show all messages
    >>> gatac.set_verbosity("WARNING") # silence info messages
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    logging.getLogger("gatac").setLevel(level)


