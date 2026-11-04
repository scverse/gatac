# Contributing

Thank you for your interest in contributing to GATAC!

---

## Development setup

```bash
git clone https://github.com/scverse/gatac.git
cd GATAC

# Install in editable mode with the CUDA extra of your driver (cuda12 or
# cuda13), the test and lint tools
uv sync --extra cuda12 --group test --group dev

# Lint each commit (ruff check)
uv run pre-commit install
```

---

## Running tests

Tests are located in `tests/` (unit tests) and in the
[`reproducibility/`](https://github.com/scverse/gatac-reproducibility)
submodule (tracked at `reproducibility/`) for integration/benchmark tests.

After cloning, initialise the submodule once:

```bash
git submodule update --init --recursive
```

```bash
# Unit tests
uv run pytest tests/

# Reproducibility benchmarks (requires GPU + data)
cd reproducibility
pixi run pytest test/
```

After any change to a preprocessing or analysis function, run the related
reproducibility test:

```bash
cd reproducibility
pixi run pytest test/tile_matrix.py
pixi run pytest test/spectral_embedding.py
```

### GPU tests in CI

GATAC imports cuDF at module level, so the whole suite needs a GPU. The
`GPU Tests` workflow runs it on an AWS T4 through [Cirun](https://cirun.io)
on every push to `main`. On a pull request, it runs only once a maintainer
adds the `run-gpu-ci` label (removed again after the run); `skip-gpu-ci`
skips it.

### On a Modal GPU

Without a local GPU, `ci/modal_gpu.py` runs the workflow's test job on a
[Modal](https://modal.com) GPU, against your working tree as it is,
uncommitted changes included. It reads the machine (a T4, 4 cores, 16 GiB),
the Python version, the CUDA extra, the environment and the steps from
`.github/workflows/tests.yml` and `.cirun.yml`, so it runs the tests as CI
does. You need a Modal account (`uvx modal setup` once), then:

```bash
uvx --with pyyaml modal run ci/modal_gpu.py                 # the CI test job
uvx --with pyyaml modal run ci/modal_gpu.py --pytest-args "tests/api/test_pp.py -x"
uvx --with pyyaml modal run ci/modal_gpu.py --gpu L4        # another GPU
GATAC_CUDA=cuda12 uvx --with pyyaml modal run ci/modal_gpu.py   # the cuda12 extra
uv run --no-project --with modal --with pyyaml python ci/modal_gpu.py   # show what it read; runs nothing
```

`--pytest-args` are appended to the workflow's pytest command. The first run
builds the image with the RAPIDS dependencies, which takes several minutes;
later runs reuse it until a dependency list in `pyproject.toml` changes.
The CuPy, Numba and CUDA JIT caches persist in the Modal Volume
`gatac-gpu-tests-jit-cache`. See the script's docstring for the rest of the
options.

---

## Code style

- **Linter**: `ruff check`, with the pyflakes and bugbear rules (`[tool.ruff]`
  in `pyproject.toml`). The pre-commit hook runs it on each commit, and the
  *Checks* workflow on each pull request. There is no formatter yet: keep to
  the style of the surrounding code.
- **Type hints**: encouraged for all public functions
- **Docstrings**: NumPy style

```bash
uv run ruff check            # or: uv run pre-commit run --all-files
```

The *Checks* workflow also builds the package (`uv build`, `twine check
--strict`), installs the wheel without a CUDA extra, and builds the docs with
warnings as errors. It runs on GitHub's free runners, on every pull request;
the GPU tests need the `run-gpu-ci` label.

---

## Submitting changes

1. Fork the repository and create a feature branch.
2. Make your changes with appropriate tests.
3. Ensure all existing tests pass.
4. Open a pull request with a clear description of the changes.

---

## Releasing

The version is the git tag (hatch-vcs): `v0.1.0` builds `0.1.0`. Publishing a
GitHub release uploads it to PyPI through trusted publishing
(`.github/workflows/release.yml`).

Once, before the first release: on PyPI, add a *pending trusted publisher* for
the project `gatac` (owner `scverse`, repository `gatac`, workflow
`release.yml`, environment `pypi`), and create the `pypi` environment in the
repository's settings, ideally with required reviewers.

For each release:

1. In `docs/changelog.md`, replace *unreleased* in the version's heading by
   the date, and start a new *unreleased* section above it for the next one.
2. Run the GPU tests on the release commit (`run-gpu-ci` label, or the Modal
   script) and check that the *Checks* workflow passes.
3. Tag the commit and push the tag: `git tag v0.1.0 && git push origin v0.1.0`.
4. Create a GitHub release of the tag, with the changelog section as notes.
   The *Release* workflow builds the sdist and wheel, checks that their
   version is the tag's, and publishes them to PyPI.

---

## Reporting issues

Please open a GitHub issue with:

- A minimal reproducible example
- GATAC version (`gatac --version`)
- GPU model and CUDA version
- Error traceback (`gatac -v <command> ...` prints it for the CLI)

---

## Documentation

The docs live in `docs/`.  To build locally:

```bash
uv sync --group docs
cd docs
make html
# Open _build/html/index.html
```

### Citing publications

When a docstring or page names a published tool or method, cite it so the
reader can click through to the [References](references.md) page. Add the
entry to `docs/references.bib` (key: first author + year, e.g. `Granja2021`),
then cite it with ``:cite:p:`Granja2021` `` in a docstring or
``{cite:p}`Granja2021` `` in a Markdown page. Use `cite:t` instead of `cite:p`
when the authors are part of the sentence (it renders as "Thibodeau et al.
[TEM+21]"). Keep citations out of a docstring's first line, which becomes the
summary in the API tables.

Analysis notebooks live in the separate
[`gatac-tutorials`](https://github.com/scverse/gatac-tutorials)
repository — see [Tutorials](tutorials/index.md) for details.
