"""Helpers a run needs to record itself.

``_environment`` describes the machine and the code (host, versions, git
commit, GPU) for the run record; ``_parse_sizes`` reads a size list such
as ``6-10,12``;
``_available_cores`` counts the cores this process may use.
"""

from __future__ import annotations

import os
import platform
import socket
import subprocess

import numpy as np


def _available_cores():
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # not on Linux
        return os.cpu_count() or 1


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
