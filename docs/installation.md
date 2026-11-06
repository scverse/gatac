# Installation

## Prerequisites

GATAC requires **Python ≥ 3.13**, Linux, and a CUDA-capable GPU with drivers
compatible with CUDA 12 or 13.

The following system libraries must be available:

- CUDA Toolkit 12.x or 13.x

---

## Install from PyPI

Choose the extra of the CUDA version that `nvidia-smi` reports; it installs
RAPIDS (cuDF, cuML, cuGraph) and CuPy for that version:

```bash
pip install 'gatac[rapids-cu12]'
# or
pip install 'gatac[rapids-cu13]'
```

With uv: `uv pip install 'gatac[rapids-cu12]'`, or `uv add 'gatac[rapids-cu12]'` in a
project. Without a CUDA extra, and unless RAPIDS is installed otherwise (see
Conda below), the first use of a GATAC module (`ga.pp`, `gatac tile`, ...)
fails with a message naming the extras.

After installation the `gatac` command is available:

```bash
gatac --version
gatac --help
```

### Optional extras

| Extra | For |
|-------|-----|
| `archr-compat` | Seurat's approximate neighbour search (Annoy), to reproduce ArchR's clustering |
| `bigwig` | Writing footprint tracks as bigWig (`FootprintResult.to_bigwig`) |

Combine them with the CUDA extra: `pip install 'gatac[rapids-cu12,bigwig]'`.

---

## Install from source

```bash
git clone https://github.com/scverse/gatac.git
cd gatac

# With CUDA 12 support
uv sync --extra rapids-cu12

# Or with CUDA 13 support
uv sync --extra rapids-cu13
```

---

## Optional: documentation dependencies

To build the docs locally, install the docs dependency group:

```bash
uv sync --group docs
```

Then build with:

```bash
cd docs
make html
```

---

## Verifying your installation

```python
import gatac as ga

print(ga.__version__)

# Check that GPU is available
import cudf

print(cudf.get_device_info())
```

---

## Hardware requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM  | 8 GB    | 24–80 GB    |
| RAM       | 32 GB   | 128+ GB     |
| Storage   | —       | NVMe SSD    |

:::{note}
GATAC needs an NVIDIA GPU: it imports cuDF and CuPy, which the
`rapids-cu12`/`rapids-cu13` extras install.
:::

---

## Conda / Mamba

There is no conda package of GATAC yet. To use RAPIDS from conda, create
an environment with RAPIDS 26.02 or newer (the
[RAPIDS install selector](https://docs.rapids.ai/install/) gives the command
for your CUDA version), then install GATAC into it from PyPI, without a CUDA
extra so that pip does not install a second RAPIDS:

```bash
conda create -n gatac -c rapidsai -c conda-forge \
    rapids=26.10 python=3.13 cuda-version=12.9
conda activate gatac
pip install gatac
```
