"""Default numbers of parallel workers, from the CPUs this process may use."""

from __future__ import annotations

import os

# Files processed at once by default: more than 4 used more GPU memory without
# being faster, for the GPU import as for compute_metrics + filter_fragments.
MAX_DEFAULT_WORKERS = 4


def n_cpus() -> int:
    """CPUs this process may use: a SLURM allocation, not the node's cores."""
    return os.process_cpu_count() or 1


def default_workers(cap: int | None = MAX_DEFAULT_WORKERS) -> int:
    """``n_cpus() - 1`` (one left for the main process), at least 1, at most ``cap``."""
    n = max(1, n_cpus() - 1)
    return n if cap is None else min(cap, n)


def resolve_workers(workers: int | str | None, auto: int) -> int:
    """``workers`` as an int: ``"auto"`` (or None) gives ``auto``."""
    if workers is None or workers == "auto":
        return auto
    if isinstance(workers, str):
        raise ValueError(f'workers must be a positive int or "auto", not {workers!r}')
    if workers < 1:
        raise ValueError(f"workers must be at least 1, not {workers}")
    return int(workers)
