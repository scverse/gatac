"""Output files that are only replaced by complete ones."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def replacing(output_path: str | Path) -> Iterator[Path]:
    """
    A temporary path next to ``output_path``, moved onto it when the block succeeds.

    On an error the temporary file is removed and ``output_path``, if it
    exists, is left as it was: a reader never sees a partial output.
    """
    output_path = Path(output_path)
    # hidden, per process, and with the output's suffix for writers that check it
    tmp_path = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp{output_path.suffix}")
    try:
        yield tmp_path
        os.replace(tmp_path, output_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def write_h5ad(adata, output_path: str | Path) -> None:
    """``adata.write_h5ad(output_path)``, replacing ``output_path`` only once the file is complete."""
    with replacing(output_path) as tmp_path:
        adata.write_h5ad(str(tmp_path))
