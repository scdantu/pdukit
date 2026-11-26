"""Helpers a run needs to record itself.

``_environment`` describes the machine and the code (host, versions, git
commit, GPU) for the run record; ``_parse_sizes`` reads a size list such
as ``6-10,12``;
``_available_cores`` counts the cores this process may use;
``limit_threads`` caps the threads numpy/scipy and torch use.
"""

from __future__ import annotations

import os
import platform
import socket
import subprocess
import sys

import numpy as np


def _available_cores():
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # not on Linux
        return os.cpu_count() or 1


def limit_threads(n):
    """Cap the CPU threads of numpy/scipy (BLAS, OpenMP) and torch at n.

    n <= 0 (the default -1) leaves every library at all the cores this
    process may use.  Returns ``(threads, cores, undo)``; ``undo()`` (if
    not None) puts the previous limits back.
    """
    cores = _available_cores()
    n = int(n if n is not None else -1)
    if n <= 0:
        return cores, cores, None
    from threadpoolctl import threadpool_limits

    old_env = os.environ.get("OMP_NUM_THREADS")
    os.environ["OMP_NUM_THREADS"] = str(n)  # libraries loaded later
    limiter = threadpool_limits(limits=n)
    torch = sys.modules.get("torch")
    old_torch = torch.get_num_threads() if torch is not None else None
    if torch is not None:
        torch.set_num_threads(n)

    def undo():
        limiter.restore_original_limits()
        if old_env is None:
            os.environ.pop("OMP_NUM_THREADS", None)
        else:
            os.environ["OMP_NUM_THREADS"] = old_env
        if old_torch is not None:
            torch.set_num_threads(old_torch)

    return n, cores, undo


def _parse_sizes(text):
    out = []
    for part in str(text).split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return sorted(set(out))


def _environment(repo_root):
    env = {
        "host": socket.gethostname(),
        "python": platform.python_version(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "cores_available": _available_cores(),
        "numpy": np.__version__,
    }
    for mod in ("scipy", "MDAnalysis", "torch", "threadpoolctl"):
        try:
            env[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            env[mod] = None
    try:
        import torch

        env["cuda_available"] = bool(torch.cuda.is_available())
        env["cuda_device"] = (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        )
        env["torch_cuda"] = torch.version.cuda
    except Exception:  # noqa: BLE001
        env["cuda_available"] = False
    for key, cmd in (
        ("git_commit", ["rev-parse", "HEAD"]),
        ("git_branch", ["rev-parse", "--abbrev-ref", "HEAD"]),
    ):
        try:
            env[key] = subprocess.run(
                ["git", "-C", str(repo_root), *cmd],
                capture_output=True,
                text=True,
                timeout=10,
                env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
            ).stdout.strip()
        except Exception:  # noqa: BLE001
            env[key] = None
    return env
