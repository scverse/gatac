# Quickstart

GATAC reads fragment files and builds {class}`~anndata.AnnData` objects on an
NVIDIA GPU. One sample, from the fragment file to a peak matrix:

```python
import gatac as ga
import rapids_singlecell as rsc

# Fragments to Parquet, then quality metrics per cell
parquet = ga.io.make_parquet("fragments.tsv.gz")
metrics = ga.pp.compute_metrics(parquet, "GRCh38.gtf.gz")

# Tile matrix of the cells that pass QC, and the most accessible tiles
adata = ga.pp.make_tile_matrix(parquet, chrom_sizes="hg38", metrics=metrics, filter_query="tsse_score > 5")
ga.pp.select_features(adata, n_features=500_000)

# Embedding and clusters
ga.dr.spectral(adata, n_comps=30)
rsc.pp.neighbors(adata, use_rep="X_spectral", n_neighbors=30)
rsc.tl.leiden(adata, resolution=0.5)

# Peaks per cluster, then the peak matrix
ga.peak.call(adata, groupby="leiden", parquet_path=parquet, genome="hg38")
ga.peak.merge(adata, chrom_sizes="hg38")
peaks = ga.peak.make_matrix(adata, parquet_path=parquet, genome="hg38")
```

The {doc}`overview <overview>` runs the same steps on several samples, with the
{doc}`command line <cli/index>` for the first ones. The
{doc}`tutorials <tutorials/index>` go into more detail, and the
{doc}`API reference <api/index>` lists every function.
