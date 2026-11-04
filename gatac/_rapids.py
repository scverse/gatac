"""The check, first in every GPU module of GATAC, that RAPIDS is installed."""


def require() -> None:
    """
    Import cuDF and CuPy, or say how to install them.

    ``pip install gatac`` without a CUDA extra installs no RAPIDS; this tells
    how to fix it instead of failing on ``import cudf`` deep inside GATAC.
    """
    try:
        import cudf  # noqa: F401
        import cupy  # noqa: F401
    except ModuleNotFoundError as e:
        raise ImportError(
            f"GATAC runs on the GPU through RAPIDS, which is missing ({e}). Install it with "
            "the CUDA extra matching the CUDA version nvidia-smi reports: "
            "pip install 'gatac[cuda12]' or 'gatac[cuda13]'"
        ) from e
