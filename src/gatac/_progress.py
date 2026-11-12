"""GATAC's progress bars: shown in a terminal or a notebook, not in logs."""

from __future__ import annotations

import os
import sys


def show_progress() -> bool:
    """
    Whether progress bars are shown.

    They are shown when stderr is a terminal or in a Jupyter kernel, so that
    pipelines (e.g. Snakemake) keep clean logs, and never with
    ``TQDM_DISABLE`` set.
    """
    if os.environ.get("TQDM_DISABLE", "").lower() in ("1", "true", "yes"):
        return False
    if sys.stderr is not None and sys.stderr.isatty():
        return True
    ipython = sys.modules.get("IPython")
    shell = ipython.get_ipython() if ipython is not None else None
    return getattr(shell, "kernel", None) is not None


def tqdm(*args, disable: bool = False, **kwargs):
    """:class:`tqdm.auto.tqdm`, disabled unless :func:`show_progress`."""
    from tqdm.auto import tqdm as _tqdm

    # in kwargs, as mypy matches a keyword after *args against the positional parameters too
    kwargs["disable"] = disable or not show_progress()
    return _tqdm(*args, **kwargs)


def trange(*args, **kwargs):
    """``tqdm(range(*args), **kwargs)``."""
    return tqdm(range(*args), **kwargs)
