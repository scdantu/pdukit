"""Choose the numerical backend (NumPy, torch on CPU or GPU) for a run."""

from __future__ import annotations


def _resolve_backend(device):
    """Pick numpy or torch.

    Returns (backend, torch_module_or_None, device_str).

    ``device`` None or "cpu" -> numpy.  "cuda"/"mps"/"auto" -> torch when it is
    installed and the device is actually visible, else numpy: a silent fallback
    to CPU on a GPU job would be far worse than a slow one you know about, so
    `FluxWorkspace.backend` records what was really used and the pilot prints
    it.
    """
    if device in (None, "cpu", "numpy"):
        return "numpy", None, "cpu"
    try:
        import torch
    except ImportError:
        return "numpy", None, "cpu"
    if device == "auto":
        device = (
            "cuda"
            if torch.cuda.is_available()
            else (
                "mps"
                if getattr(torch.backends, "mps", None)
                and torch.backends.mps.is_available()
                else "cpu"
            )
        )
    if device == "cuda" and not torch.cuda.is_available():
        return "numpy", None, "cpu"
    if device == "mps" and not (
        getattr(torch.backends, "mps", None)
        and torch.backends.mps.is_available()
    ):
        return "numpy", None, "cpu"
    if device == "mps":
        # MPS accelerates the covariance build (large float32 matmuls) but not
        # the per-candidate eigendecompositions, which need double precision
        # MPS does not have.  Expect the build to speed up and growth not to.
        return "torch", torch, "mps"
    if device == "cpu":
        return "numpy", None, "cpu"
    return "torch", torch, device
