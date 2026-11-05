"""
Merging the matrices of several samples into one ``.h5ad``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from ._combine import _feature_mismatch, _shared_names_error, combine_file_matrices

logger = logging.getLogger(__name__)


def combine(
    input_paths: list[str | Path],
    output_path: str | Path,
) -> None:
    """
    Merge several ``.h5ad`` matrices of the same features into one file.

    The cells are stacked file after file, and ``obs["source_file"]`` names
    each cell's file. Every file must have the same features (``var_names``)
    in the same order, and cell names must be distinct across files; both
    are checked before anything is written. The files are read and written
    one at a time, and ``output_path`` is only replaced once the combined
    file is complete. ``X`` takes the widest dtype of the files; ``var`` is
    the first file's, and ``uns``, ``obsm`` and layers are not kept.

    Parameters
    ----------
    input_paths : list of str or Path
        List of paths to h5ad files
    output_path : str or Path
        Output path for combined h5ad file

    Raises
    ------
    ValueError
        When the files' features differ, or a cell name is in several files.

    Examples
    --------
    >>> import gatac as ga
    >>> ga.pp.combine(
    ...     ["sampleA.h5ad", "sampleB.h5ad"],
    ...     "combined.h5ad",
    ... )
    """
    input_paths = [Path(p) for p in input_paths]
    output_path = Path(output_path)
    if len(input_paths) == 0:
        raise ValueError("No input files provided")
    if any(p.resolve() == output_path.resolve() for p in input_paths):
        raise ValueError(f"output_path must differ from the inputs: {output_path}")
    names = [p.name for p in input_paths]
    if len(set(names)) != len(names):
        names = [str(p) for p in input_paths]

    logger.info(f"Combining {len(input_paths)} h5ad files")
    # Features and cell names first, from each file's obs and var only
    var_names, cells = None, []
    for path, name in zip(input_paths, names):
        adata = ad.read_h5ad(path, backed="r")
        try:
            if var_names is None:
                var_names = adata.var_names
            elif not adata.var_names.equals(var_names):
                raise _feature_mismatch(adata.var_names, var_names, name, names[0])
            cells.append(adata.obs_names)
        finally:
            adata.file.close()
    all_cells = pd.Index(np.concatenate(cells))
    if not all_cells.is_unique:
        files = np.repeat(np.arange(len(cells)), [len(c) for c in cells])
        raise _shared_names_error(all_cells, files, names)

    def parts():
        for path in input_paths:
            yield ad.read_h5ad(path)

    combined = combine_file_matrices(parts(), names, output_path)
    logger.info(f"Saved combined matrix ({combined.n_obs:,} cells × {combined.n_vars:,} features) to {output_path}")
    combined.file.close()
