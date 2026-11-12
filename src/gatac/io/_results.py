"""
Results stored in an AnnData as anndata and scanpy do; in its file too when it is backed ``r+``.

A function such as :func:`gatac.dr.spectral` assigns its results into the
AnnData it was given. When that AnnData is backed and was opened ``r+``,
the same keys are also written into its file at once, through its own
handle: only those keys, all or none of them, and never ``X``. A backed
AnnData opened ``r`` keeps them in memory, and ``adata.write()`` (anndata's
own save, which skips ``X``) puts them in the file.
"""

from __future__ import annotations

from collections.abc import Mapping

_TMP_PREFIX = "__gatac_tmp_"


def _write_elem():
    try:
        from anndata.io import write_elem
    except ImportError:  # anndata < 0.11
        from anndata.experimental import write_elem
    return write_elem


def _write_keys(h5, groups: Mapping[str, Mapping]) -> None:
    """Write ``groups[g][key]`` as ``h5[g][key]``: each under a temporary name first, then all moved into place."""
    write_elem = _write_elem()
    written = []
    try:
        for name, items in groups.items():
            if not items:
                continue
            if name not in h5:
                write_elem(h5, name, {})
            g = h5[name]
            for key, value in items.items():
                tmp = _TMP_PREFIX + key
                if tmp in g:
                    del g[tmp]
                written.append((g, tmp, key))
                write_elem(g, tmp, value)
    except BaseException:
        for g, tmp, _ in written:
            if tmp in g:
                del g[tmp]
        raise
    for g, tmp, key in written:
        if key in g:
            del g[key]
        g.move(tmp, key)
    h5.flush()


def _store_results(adata, *, obsm: Mapping | None = None, uns: Mapping | None = None, varm: Mapping | None = None):
    """
    ``adata.obsm[k] = v`` etc. for each result; in the file too when ``adata`` is backed ``r+``.

    ``obsm`` values must have ``n_obs`` rows and ``varm`` values ``n_vars``;
    ``uns`` values are not checked. On a view of a backed AnnData, anndata
    refuses the assignment; callers check that before computing.
    """
    obsm, uns, varm = dict(obsm or {}), dict(uns or {}), dict(varm or {})
    for what, items, n in (("obsm", obsm, adata.n_obs), ("varm", varm, adata.n_vars)):
        for key, value in items.items():
            if len(value) != n:
                raise ValueError(f"{what}[{key!r}] has {len(value):,} rows, not {n:,}")
    if adata.isbacked and not adata.is_view:
        manager = adata.file
        if not manager.is_open:
            manager.open()
        if manager._file.mode == "r+":
            _write_keys(manager._file, {"obsm": obsm, "uns": uns, "varm": varm})
    for key, value in obsm.items():
        adata.obsm[key] = value
    for key, value in uns.items():
        adata.uns[key] = value
    for key, value in varm.items():
        adata.varm[key] = value
