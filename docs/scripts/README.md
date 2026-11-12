# Docs figure scripts

Each script writes SVG figures into `docs/_static/`. They only need the Python
standard library; run them from anywhere, e.g. `python3 docs/scripts/pipeline_overview.py`,
then rebuild the docs.

| Script                   | Writes                                                      | Used in       |
| ------------------------ | ----------------------------------------------------------- | ------------- |
| `pipeline_overview.py`   | `pipeline_overview.svg`, `pipeline_overview_narrow.svg`     | `overview.md` |
| `runtime_narrow.py`      | `runtime_clustering_narrow.svg`, `runtime_peaks_narrow.svg` | `overview.md` |
| `parquet_size.py`        | `parquet_size.svg`                                          | `api/io.md`   |
| `parquet_size_narrow.py` | `parquet_size_narrow.svg`                                   | `api/io.md`   |
| `shuffle_gzip.py`        | `shuffle_gzip.svg`                                          | `api/io.md`   |
| `shuffle_gzip_narrow.py` | `shuffle_gzip_narrow.svg`                                   | `api/io.md`   |

`runtime_narrow.py` reads the wide `runtime_clustering.svg` and `runtime_peaks.svg`,
which are edited by hand: rerun it after changing them. `fragment_layout.svg` is
also edited by hand and has no script.

## Conventions

- Colours come from the theme (`--pst-color-*` variables, with light-mode
  fallbacks); hard-coded colours get an `html[data-theme="dark"]` override when
  their contrast falls short in dark mode.
- The SVGs are inlined, so their `<style>` rules apply to the whole page: give
  each figure its own class and id prefix.
- A `*_narrow.svg` is the phone layout of the figure of the same name. The page
  includes both, in `{div} diagram-wide` and `{div} diagram-narrow` blocks;
  `_static/custom.css` shows only the narrow one below 640 px.
