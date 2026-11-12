"""The check, first in every GPU module of GATAC, that RAPIDS is installed."""

INSTALL_HINT = (
    "With pip, install the extra matching the CUDA version nvidia-smi reports: "
    "pip install 'gatac[rapids-cu12]' or 'gatac[rapids-cu13]'. "
    "With conda, install RAPIDS from the rapidsai channel: "
    "https://gatac.readthedocs.io/en/latest/installation.html"
)


def require() -> None:
    """
    Import cuDF and CuPy, or say how to install them.

    ``pip install gatac`` without a RAPIDS extra installs no RAPIDS; this tells
    how to fix it instead of failing on ``import cudf`` deep inside GATAC.
    """
    try:
        import cudf  # noqa: F401
        import cupy  # noqa: F401
    except ModuleNotFoundError as e:
        raise ImportError(f"GATAC runs on the GPU through RAPIDS, which is missing ({e}). {INSTALL_HINT}") from e
