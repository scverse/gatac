"""Sphinx configuration for GATAC documentation."""

# `scverse_doc` sets the theme and shared defaults, see https://scverse-doc.readthedocs.io
# For the full list of options, see https://www.sphinx-doc.org/page/usage/configuration.html

import sys
from datetime import datetime
from importlib.metadata import metadata
from pathlib import Path

from scverse_doc.config import DEFAULTS
from scverse_doc.registry import intersphinx

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))

# Import the packages that must not load inside the sphinx mock context.
# sphinx mocks cupy/cupyx, and the compat layer of anndata imports them at load
# time: inside the mock context, that first import deadlocks. Imported here,
# they are in sys.modules before any mock context starts.
# As in scverse/rapids-singlecell docs/conf.py.
import anndata  # noqa: E402, F401
import scanpy  # noqa: E402, F401

# -- Project information -----------------------------------------------------
info = metadata("gatac")
project = "GATAC"
author = info["Author"] or "GATAC contributors"
copyright = f"{datetime.now():%Y}, GATAC contributors"
version = release = info["Version"]
source_repository = "https://github.com/scverse/gatac"

# -- General configuration ---------------------------------------------------
extensions = [
    "scverse_doc",  # sets up napoleon, so it comes before sphinx_autodoc_typehints
    "scverse_misc.sphinx_ext",
    "sphinx.ext.githubpages",
    "sphinx.ext.extlinks",
    "sphinx_autodoc_typehints",
    "sphinxcontrib.mermaid",
    # Citations: {cite:p}`key` in pages and :cite:p:`key` in docstrings,
    # resolved against references.bib and listed on references.md
    "sphinxcontrib.bibtex",
    "IPython.sphinxext.ipython_console_highlighting",
]

bibtex_bibfiles = ["references.bib"]
typehints_defaults = "braces"
always_use_bars_union = True

# Release notes: {pr}`12`, {issue}`34`
extlinks = {
    "issue": ("https://github.com/scverse/gatac/issues/%s", "issue%s"),
    "pr": ("https://github.com/scverse/gatac/pull/%s", "pr%s"),
}

myst_enable_extensions = [*DEFAULTS.myst_enable_extensions, "substitution"]
myst_heading_anchors = 4
myst_substitutions = {"version": release}

# The tutorials show the logs of each step: keep stderr, merged in order below
nb_output_stderr = "show"
nb_merge_streams = False

napoleon_preprocess_types = True
napoleon_use_ivar = True

# The API pages list members in the order of the source
autodoc_member_order = "bysource"
autodoc_default_options = {"show-inheritance": True}
# Short type names in docstrings
napoleon_type_aliases = {
    "AnnData": "anndata.AnnData",
    "DataFrame": "pandas.DataFrame",
    "Path": "pathlib.Path",
    "np.ndarray": "numpy.ndarray",
    "BINDetectResult": "gatac.tobias.BINDetectResult",
    "DNAMotif": "gatac.motif.DNAMotif",
    "FootprintTracks": "gatac.tobias.FootprintTracks",
}

# Mock GPU / optional imports that are not available in the docs environment
autodoc_mock_imports = [
    "cudf",
    "cupy",
    "cuml",
    "cupyx",
    "rmm",
    "MOODS",
]

# scverse core packages, Python, NumPy, SciPy, pandas and Matplotlib come from the registry
intersphinx_mapping = intersphinx() | {
    "cudf": ("https://docs.rapids.ai/api/cudf/stable/", None),
    "cupy": ("https://docs.cupy.dev/en/stable/", None),
    "pyarrow": ("https://arrow.apache.org/docs/", None),
}

# Copy button settings
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

templates_path = ["_templates"]
exclude_patterns = [
    *DEFAULTS.exclude_patterns,
    "notebooks/README.md",  # the gatac-tutorials repo readme
    "scripts/**",  # the scripts that draw the figures of _static/
    # each version's notes are included by release-notes/index.md, not pages
    "release-notes/[0-9]*.md",
    "release-notes/blank.md",
]

# -- Options for HTML output -------------------------------------------------
html_title = "GATAC: GPU-Accelerated scATACseq Analysis"
html_logo = "_static/logo_outlines.svg"
html_theme_options = {
    "navigation_with_keys": False,
    "show_toc_level": 2,
}
html_show_sphinx = False

# OpenGraph metadata for link previews. Generated cards are off: they cannot use the SVG logo.
ogp_site_url = "https://gatac.readthedocs.io/"
ogp_site_name = "GATAC"
ogp_social_cards = {"enable": False}
html_static_path = ["_static"]
html_css_files = ["custom.css"]

xref_overrides = {
    # If building the documentation fails because a type annotation links to a private path
    # (e.g. `igraph._igraph.GraphBase`), map it to the documented name here.
    # For a missing link that can't be fixed this way, add it to `nitpick_ignore` instead.
    "gatac.dr.lsi.LSIModel": "gatac.dr.LSIModel",
}

# Private classes in public signatures
nitpick_ignore = [
    ("py:class", "gatac.tobias.bindetect._Motif"),
    ("py:class", "gatac.tobias.footprint.TrackLayout"),
]


# The tutorials live in the gatac-tutorials submodule and enter the docs
# through the docs/notebooks symlink. MyST resolves a notebook's link to a
# sibling notebook through the symlink's real path, outside the source tree,
# and names the target by that absolute path; map it back to its docname.
_NOTEBOOKS = str((Path(__file__).parent / "notebooks").resolve()) + "/"


def _relink_notebooks(app, doctree):
    from sphinx.addnodes import pending_xref

    for node in doctree.findall(pending_xref):
        target = node.get("reftarget", "")
        if node.get("reftype") == "myst" and target.startswith(_NOTEBOOKS):
            node["reftarget"] = "notebooks/" + target[len(_NOTEBOOKS) :]


# myst-nb renders every stdout/stderr chunk of a cell as its own block, so a
# cell that logs line by line shows a stack of boxes. nb_merge_streams only
# joins chunks of the same stream, and moves stdout after stderr; instead
# join each run of consecutive stream chunks, in order, into one stdout block.
def _merge_stream_outputs(app, doctree):
    from docutils import nodes

    def is_stream(node):
        return isinstance(node, nodes.literal_block) and ("stream" in node["classes"] or "stderr" in node["classes"])

    for container in doctree.findall(nodes.container):
        if "cell_output" not in container["classes"]:
            continue
        children, run = [], []
        for child in [*container.children, None]:
            if child is not None and is_stream(child):
                run.append(child)
                continue
            if run:
                text = "".join(node.astext() for node in run)
                block = run[0]
                block["classes"] = ["stream" if c == "stderr" else c for c in block["classes"]]
                block.rawsource = text
                block.children = []
                block += nodes.Text(text)
                children.append(block)
                run = []
            if child is not None:
                children.append(child)
        container.children = []
        container.extend(children)


def _class_role(name):
    """An inline role whose text gets the CSS class ``name``: {small}`date`, {smaller}`author`."""
    from docutils import nodes

    def role(_name, rawtext, text, _lineno, _inliner, options=None, content=None):
        return [nodes.inline(rawtext, text, classes=[name])], []

    return role


def setup(app):
    """Register the release-note roles and the notebook post-processing."""
    app.add_role("small", _class_role("small"))
    app.add_role("smaller", _class_role("smaller"))
    app.connect("doctree-read", _relink_notebooks)
    app.connect("doctree-read", _merge_stream_outputs)
