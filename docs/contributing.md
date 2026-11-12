# Contributing

Thank you for your interest in contributing to GATAC!

---

## Development setup

```bash
git clone https://github.com/scverse/gatac.git
cd GATAC

# Install in editable mode with the CUDA extra of your driver (rapids-cu12 or
# rapids-cu13), the test and lint tools
uv sync --extra rapids-cu12 --group test --group dev

# Lint, format and type check each commit. The mypy hook runs
# `hatch check types`, so hatch has to be on the PATH
uv tool install hatch
uv run prek install
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
PYTHONPATH=../src pixi run pytest test/
```

The reproducibility environment installs the released GATAC from PyPI;
`PYTHONPATH=../src` makes the tests use your checkout instead. After any change to
a preprocessing or analysis function, run the related reproducibility test:

```bash
cd reproducibility
PYTHONPATH=../src pixi run pytest test/tile_matrix.py
PYTHONPATH=../src pixi run pytest test/spectral_embedding.py
```

### GPU tests in CI

GATAC imports cuDF at module level, so the whole suite needs a GPU. The
`GPU Tests` workflow runs it on an AWS T4 through [Cirun](https://cirun.io)
on every push to `main`. On a pull request, it runs only once a maintainer
adds the `run-gpu-ci` label (removed again after the run); `skip-gpu-ci`
skips it.

It runs in the hatch-test environments of `hatch.toml`, all on CUDA 13:
`stable` (Python 3.13, the oldest RAPIDS the extras allow) and `dev` (Python
3.14, the latest RAPIDS, anndata and scanpy from git) must pass; the
`rapids_prerelease` one (RAPIDS nightlies) does not run in CI; run it on
Modal (`GATAC_ENV=hatch-test.rapids_prerelease-13`, below). Locally, the same
environments run with hatch, e.g. `hatch run hatch-test.stable-13:run`.

### On a Modal GPU

Without a local GPU, `ci/modal_gpu.py` runs the workflow's required test job
on a [Modal](https://modal.com) GPU, against your working tree as it is,
uncommitted changes included. It reads the machine (a T4, 4 cores, 16 GiB),
the environment and the steps from `.github/workflows/test-gpu.yml` and
`.cirun.yml`, and the Python, CUDA extra and dependencies of the hatch-test
environment from `hatch.toml` (`hatch-test.stable-13` by default), so it runs
the tests as CI does. You need a Modal account (`uvx modal setup` once), then:

```bash
uvx --with pyyaml modal run ci/modal_gpu.py                 # the CI test job
uvx --with pyyaml modal run ci/modal_gpu.py --pytest-args "tests/api/test_pp.py -x"
uvx --with pyyaml modal run ci/modal_gpu.py --gpu L4        # another GPU
GATAC_ENV=hatch-test.dev-13 uvx --with pyyaml modal run ci/modal_gpu.py   # another environment
uv run --no-project --with modal --with pyyaml python ci/modal_gpu.py   # show what it read; runs nothing
```

`--pytest-args` are appended to the workflow's pytest command. The first run
builds the image with the environment's dependencies, which takes several
minutes; later runs reuse it until a dependency list in `pyproject.toml` or
`hatch.toml` changes.
The CuPy, Numba and CUDA JIT caches persist in the Modal Volume
`gatac-gpu-tests-jit-cache`. See the script's docstring for the rest of the
options.

---

## Code style

GATAC follows the [cookiecutter-scverse](https://github.com/scverse/cookiecutter-scverse)
template, linked with cruft (`.cruft.json`): `cruft update` proposes the
template's later changes.

- **Hooks**: [prek](https://github.com/j178/prek) runs the hooks of
  `.pre-commit-config.yaml` on each commit: biome, pyproject-fmt, ruff, mypy,
  the pre-commit-hooks checks and zizmor. [autofix.ci](https://autofix.ci) runs
  them on each pull request and commits the fixes it can make.
  `.git-blame-ignore-revs` lists the reformat commits:
  `git config blame.ignoreRevsFile .git-blame-ignore-revs` once.
- **Linter and formatter**: `ruff check` and `ruff format`, 120 characters a
  line, with the template's rules (`[tool.ruff]` in `pyproject.toml`). Imports
  are absolute.
- **Type hints**: mypy checks `src` and `tests`. RAPIDS and the other GPU
  libraries count as untyped (`[tool.mypy]`); a `type: ignore` names its error
  code and why.
- **Docstrings**: NumPy style, each with a one-line summary

```bash
uv run prek run --all-files   # all hooks
hatch check                   # the same, by group: code (lint), fmt and types; --fix to apply fixes
```

The _Checks_ workflow builds the package (`uv build`, `twine check
--strict`), installs the wheel without a CUDA extra, and builds the docs with
warnings as errors. It runs on GitHub's free runners, on every pull request.
The GPU tests run on pushes to `main`; a pull request needs the `run-gpu-ci`
label to run them, or `skip-gpu-ci` when it cannot affect GPU code. Pull
requests that only change docs or Markdown skip them.

---

## Submitting changes

1. Fork the repository and create a feature branch.
2. Make your changes with appropriate tests.
3. Ensure all existing tests pass.
4. Open a pull request with a clear description of the changes.

---

## Release notes

The release notes follow [rapids-singlecell's](https://rapids-singlecell.readthedocs.io/en/latest/release-notes/index.html).
Each version has its own file in `docs/release-notes/`, `X.Y.Z.md`, started
from `blank.md`. `index.md` includes it at the top, under its minor version
(`## Version 0.2.0` holds 0.2.1, then 0.2.0). The file of the next version is
always on `main`, dated `the-future`, and **every pull request that users would
notice adds its line to it**, in the same pull request.

- Put each change under one of the rubrics, in this order: _Features_,
  _Performance_, _Bug fixes_, _Misc_, _Removals_. Remove the rubrics left
  empty when releasing.
- One bullet per change, on one line, starting with a verb: _Add_, _Speed
  up_, _Fix_, _Match_, _Raise_, _Deprecate_, _Remove_. Say what changes for
  users, including results that differ from the previous version.
- Link what it is about with its full path, and `~` to show only the name:
  `` {func}`~gatac.peak.call` ``, `` {class}`~gatac.io.FragmentFile` ``,
  `` {meth}`~gatac.tobias.BINDetectResult.site_log2fc` ``.
- End with the pull request(s), then the author(s) as initial and surname:
  `` {pr}`12` {pr}`15` {smaller}`L Faure & S Dicks` ``.

```md
- Add {func}`~gatac.peak.call`, MACS3 peak calling on the GPU {pr}`12` {smaller}`L Faure`
- Speed up {func}`~gatac.pp.make_gene_matrix` 3x on large annotations {pr}`14` {smaller}`L Faure`
- Fix {func}`~gatac.dr.spectral` with int64 indices {pr}`15` {smaller}`L Faure`
```

## Releasing

The version is the git tag (hatch-vcs): `v0.1.0` builds `0.1.0`. Publishing a
GitHub release uploads it to PyPI through trusted publishing
(`.github/workflows/release.yml`).

Once, before the first release: on PyPI and on TestPyPI, add a _pending
trusted publisher_ for the project `gatac` (owner `scverse`, repository
`gatac`, workflow `release.yml`, environment `pypi` on PyPI and `testpypi` on
TestPyPI), and create both environments in the repository's settings. As in
scanpy, only version tags (`v[0-9]*.[0-9]*.[0-9]*`) may deploy to `pypi`;
`testpypi` stays open to the branches run by hand.

For each release:

1. In the version's release notes (`docs/release-notes/X.Y.Z.md`), replace
   `the-future` in its heading by the date (`2026-11-20`).
2. Run the GPU tests on the release commit (`run-gpu-ci` label, or the Modal
   script) and check that the _Checks_ workflow passes.
3. Optionally, try the upload on TestPyPI first: _Actions → Release → Run
   workflow_ on `main`, with a version that is not used yet (`0.1.0rc1`), then
   `pip install -i https://test.pypi.org/simple/ --extra-index-url
   https://pypi.org/simple/ gatac==0.1.0rc1`.
4. Tag the commit and push the tag: `git tag v0.1.0 && git push origin v0.1.0`.
5. Create a GitHub release of the tag, whose notes link to the release notes
   page: `https://gatac.readthedocs.io/en/latest/release-notes/index.html`.
   The _Release_ workflow builds the sdist and wheel, checks that their
   version is the tag's, and publishes them to PyPI.
6. On `main`, start the next version's notes: copy `blank.md` to the next
   version's file (`0.1.1.md`), and include it at the top of `index.md`.

---

## Reporting issues

Please open a GitHub issue with:

- A minimal reproducible example
- GATAC version (`gatac --version`)
- GPU model and CUDA version
- Error traceback (`gatac -v <command> ...` prints it for the CLI)

---

## Documentation

The docs live in `docs/`. To build locally:

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
