"""
Combine several fragment files into one matrix.

The matrix holds the cells of each file in turn, named and tagged with their
file as :func:`~gatac.pp.make_tile_matrix` does.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

logger = logging.getLogger(__name__)


def _file_names(paths: Sequence[Path]) -> list[str]:
    """The files' names, which must be distinct: they identify each cell's file."""
    if not paths:
        raise ValueError("No fragment files given")
    names = [p.name for p in paths]
    if len(set(names)) != len(names):
        raise ValueError(
            "Fragment files must have distinct names, which identify each cell's file: "
            'convert them with gatac convert --file-prefix "{sample}_" '
            '(make_parquet_batch(file_prefix="{sample}_")), or rename them'
        )
    return names


def _shared_names_error(cells: pd.Index, files: np.ndarray, names: Sequence[str]) -> ValueError:
    """The error for cell names found in several files."""
    dup = cells.duplicated(keep=False)
    shared = list(dict.fromkeys(np.asarray(names, dtype=object)[files[dup]]))
    return ValueError(
        f"{cells[dup].nunique():,} cell names are in more than one of the files "
        f"{shared}, e.g. {list(cells[dup].unique()[:3])}. Give each sample its own barcode "
        "prefix, which only rewrites the file's footer: `gatac rename FILE --barcode-prefix "
        "PREFIX` or `gatac.io.set_barcode_prefix(FILE, PREFIX)`"
    )


def _check_distinct_cells(paths: Sequence[Path], names: Sequence[str]) -> None:
    """Raise, before any work, if fragment files share barcode names."""
    from gatac.io.fragments import FragmentFile

    barcodes = [FragmentFile(p).barcodes for p in paths]
    cells = pd.Index(np.concatenate(barcodes))
    if not cells.is_unique:
        files = np.repeat(np.arange(len(paths)), [len(b) for b in barcodes])
        raise _shared_names_error(cells, files, names)


def _feature_mismatch(got: pd.Index, ref: pd.Index, name: str, ref_name: str) -> ValueError:
    """The error for a file whose features (``var_names``) are not those of the first file."""
    if len(got) == len(ref) and got.sort_values().equals(ref.sort_values()):
        i = int(np.flatnonzero(got.to_numpy() != ref.to_numpy())[0])
        detail = f"the same {len(ref):,} features in another order (feature {i} is {got[i]!r}, not {ref[i]!r})"
    else:
        detail = f"{len(got):,} features, {len(got.difference(ref)):,} of them not among the {len(ref):,} of {ref_name}"
    return ValueError(
        f"{name} has other features than {ref_name}: {detail}. Matrices are combined "
        "column by column, so every file must have the same features in the same order"
    )


def _combined_obs(obs_parts: list[pd.DataFrame], names: Sequence[str]) -> pd.DataFrame:
    """The files' obs in order, with ``source_file`` categorical; cell names must be distinct."""
    obs = pd.concat(obs_parts)
    if not obs.index.is_unique:
        files = np.repeat(np.arange(len(obs_parts)), [len(o) for o in obs_parts])
        raise _shared_names_error(obs.index, files, names)
    obs["source_file"] = pd.Categorical(obs["source_file"], categories=list(names))
    return obs


def combine_file_matrices(
    parts: Iterable[ad.AnnData],
    names: Sequence[str],
    output_path: str | Path | None = None,
    compression: str | None = "gzip",
) -> ad.AnnData:
    """
    One AnnData of the per-file matrices ``parts``, cells of the same features.

    The cells keep their order, file after file; ``obs["source_file"]`` is
    the cell's file among ``names``, cell names must be distinct across
    files, and ``X`` takes the widest dtype of the parts. ``parts`` is consumed once: with ``output_path``, each part's
    rows are written to the ``.h5ad`` as it comes, so only one part is held,
    and the result is opened from the file in backed mode; ``compression``
    is that of ``CSRWriter`` (shuffle + gzip on the GPU by
    default, or None).
    """
    obs_parts: list[pd.DataFrame] = []
    var: pd.DataFrame | None = None

    def rows(part: ad.AnnData, name: str) -> sp.csr_matrix:
        nonlocal var
        # in-memory parts: pandas obs and var, not the Dataset2D of a lazy AnnData
        assert isinstance(part.var, pd.DataFrame) and isinstance(part.obs, pd.DataFrame)
        if var is None:
            var = part.var
        elif not part.var_names.equals(var.index):
            raise _feature_mismatch(part.var_names, var.index, name, names[0])
        obs = part.obs.copy()
        obs.insert(min(1, obs.shape[1]), "source_file", name)
        obs_parts.append(obs)
        # the X union of AnnData includes backed and array API types an in-memory part does not have
        return sp.csr_matrix(part.X)  # type: ignore[arg-type]

    if output_path is None:
        X = [rows(part, name) for part, name in zip(parts, names)]
        X = sp.vstack(X, format="csr", dtype=np.result_type(*(x.dtype for x in X)))
        return ad.AnnData(X=X, obs=_combined_obs(obs_parts, names), var=var)

    from gatac.io._csr_writer import CSRWriter

    # Written next to the output and moved onto it once complete: an error in
    # a later part (other features, shared cell names) leaves an existing
    # output as it was
    with CSRWriter(output_path, compression=compression) as writer:
        n = 0
        for part, name in zip(parts, names):
            x = rows(part, name)
            del part
            writer.append(x)
            n += 1
            del x
        if not n:
            raise ValueError("No matrices to combine")
        writer.finalize(obs=_combined_obs(obs_parts, names), var=var)
    return ad.read_h5ad(output_path, backed="r")
