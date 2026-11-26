"""Choose the numerical backend (NumPy, torch on CPU or GPU) for a run."""

from __future__ import annotations

import os
import re
import shutil

CU121 = (
    'pip install "torch<2.6" --index-url '
    "https://download.pytorch.org/whl/cu121"
)


def _resolve_backend(device):
    """Pick numpy or torch.

    Returns (backend, torch_module_or_None, device_str).

    ``device`` None or "cpu" -> numpy.  "auto" -> CUDA when torch sees a
    CUDA GPU, else numpy (MPS only when asked for by name).
    "cuda"/"mps" -> torch when it is
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
        device = "cuda" if torch.cuda.is_available() else "cpu"
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


def gpu_present():
    """Return True when an NVIDIA driver is installed on this machine."""
    return bool(
        shutil.which("nvidia-smi")
        or os.path.exists("/proc/driver/nvidia/version")
    )


def gpu_note(requested, device):
    """Warning to log when a run that could use the GPU does not, or None.

    ``requested`` is the ``device`` setting, ``device`` what the workspace
    really runs on.  The warning says why and how to fix it.
    """
    if device == "cuda" or requested not in ("auto", "cuda"):
        return None
    if not gpu_present():
        if requested == "cuda":
            return "!!device = cuda but no CUDA GPU is visible: running on CPU"
        return None
    try:
        import torch
    except ImportError:
        return (
            "!!a GPU is present but torch is not installed: running on CPU. "
            f"Install torch for your CUDA, e.g. for CUDA 12.1: {CU121} "
            "(or set device = cpu to silence this)"
        )
    return (
        f"!!a GPU is present but torch {torch.__version__} (CUDA "
        f"{torch.version.cuda}) cannot use it: running on CPU. Install the "
        f"torch build that matches your driver, e.g. for CUDA 12.1: {CU121}"
    )


# GPUs with full-rate double precision (about 1/2 of single precision);
# most others -- GeForce, RTX A-series, Quadro -- run it at 1/32 to 1/64.
FAST_FP64 = re.compile(
    r"\b(A100|A30|A800|H100|H200|H800|GH200|GB200|B100|B200|V100|P100|"
    r"TITAN V|MI[123]\d{2}X?)\b",
    re.IGNORECASE,
)


def fp64_note(name):
    """Warning for a GPU whose double precision is slow, or None.

    pdu scores in double precision, so on such a card the CPU (with
    ``--nt``) is often as fast and has far more memory.
    """
    if not name or FAST_FP64.search(name):
        return None
    return (
        f"!!GPU {name}: double precision on this card runs at about 1/32 to "
        "1/64 of single precision, and pdu scores in double precision; "
        "device=cpu with --nt <cores> is often as fast, and has more memory"
    )


def gpu_name():
    """Return the name of the current CUDA GPU, or None."""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001
        pass
    return None
