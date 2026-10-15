"""Standalone, GPU-batched reimplementation of the null-threshold calibration
logic in `interface.py`'s `calibrate_null_thresholds`.

Structurally this mirrors `null_calibration_cpu.py` exactly (same public
functions, same arguments and return shapes), so the two can be validated
against each other and against `interface.calibrate_null_thresholds`. The
difference is entirely in how the `(support, shift)` evaluation is executed:
rather than one call per support (parallelised across processes, as the CPU
module does), every support sharing the same size is batched into a single
set of batched torch linear-algebra calls, since for the small matrix sizes
involved here a GPU doing one call per item is dominated by per-call launch
overhead.

Supports are RESIDUE sets; covariances are (D, D) over feature columns with
D = N * m, residue-major, exactly as in `FluxWorkspace` -- see the
"RESIDUES VS FEATURE COLUMNS" note in `null_calibration_cpu.py`, whose
`_support_indices` / `_infer_n_feat` this module reuses.

The covariances and the shift-replica stacks are moved to the device ONCE
per calibration run (`_DeviceState`) and every block a batch needs is
gathered there, so a calibration round transfers only index vectors in and
one (supports x shifts) result block out.

This module is a validation harness, not a replacement for the production
path -- see the companion plan document and `null_calibration_cpu.py`'s
module docstring for what is deliberately out of scope (only the reversible
Koopman-basis branch is reproduced; it is not wired into
`decompose()`/`score()`/`calibrate_null_thresholds()`).
"""

from __future__ import annotations

import time

import numpy as np
import torch

from .interface import (
    _bootstrap_ci_rel_width,
    _bootstrap_ci_rel_width_two_way,
    _grow_ball_support,
    _support_descriptors,
)
from .null_calibration_cpu import _infer_n_feat, _support_indices


def _resolve_device(device):
    """Pick the actual torch device to run on.

    Implemented independently of interface.py's own `_resolve_backend` (not
    imported or called here), but in the same spirit: a silent fallback to
    CPU on a requested GPU job would be worse than a slow one you know
    about, so a fallback is always printed.
    """
    if device == "cuda" and not torch.cuda.is_available():
        print(
            "null_calibration_gpu: CUDA requested but not available; "
            "falling back to CPU."
        )
        return "cpu"
    return device


def _inv_sqrt_batched(C, ridge):
    """Batched ridge-regularised inverse square root of a symmetric PSD
    matrix, matching `interface.py`'s `_inv_sqrt` applied independently to
    each matrix in the leading batch dimensions of `C` (..., p, p).
    """
    C = 0.5 * (C + C.transpose(-1, -2))
    p = C.shape[-1]
    eye = torch.eye(p, dtype=C.dtype, device=C.device)
    trace = torch.diagonal(C, dim1=-2, dim2=-1).sum(-1)
    scale = torch.clamp(trace / max(p, 1), min=1e-30)
    C = C + ridge * scale[..., None, None] * eye
    w, V = torch.linalg.eigh(C)
    w_floor = torch.clamp(w.amax(dim=-1, keepdim=True), min=1e-30) * 1e-14
    w = torch.clamp(w, min=w_floor)
    return (V * (1.0 / torch.sqrt(w))[..., None, :]) @ V.transpose(-1, -2)


def _koopman_basis_batched(C0, C1, k, ridge):
    """Batched reversible-branch Koopman basis: one batched eigh call for
    every matrix in the leading batch dimension of `C0`/`C1`, matching
    `_koopman_basis_np` in `null_calibration_cpu.py` item by item.
    """
    L = _inv_sqrt_batched(C0, ridge)
    A = L @ (0.5 * (C1 + C1.transpose(-1, -2))) @ L
    A = 0.5 * (A + A.transpose(-1, -2))
    w, U = torch.linalg.eigh(A)
    idx_desc = torch.argsort(w, dim=-1, descending=True)[..., :k]
    idx_expand = idx_desc.unsqueeze(-2).expand(*U.shape[:-1], k)
    U_top = torch.gather(U, dim=-1, index=idx_expand)
    return L @ U_top


class _DeviceState:
    """C00, C01, C11 (D, D) and the shift stacks Cs00, Cs01 (J, D, D), placed
    on the device once and reused for every batch of a calibration run.
    """

    def __init__(
        self, C00, C01, C11, Cs00, Cs01, *, N, device, dtype=torch.float64
    ):
        self.device = _resolve_device(device)
        self.dtype = dtype
        self.C00_np = np.asarray(C00, dtype=np.float64)
        self.N = int(N)
        self.m = _infer_n_feat(self.C00_np.shape[0], self.N)

        def put(a):
            return torch.as_tensor(
                np.asarray(a, dtype=np.float64),
                dtype=dtype,
                device=self.device,
            )

        self.C00 = put(self.C00_np)
        self.C01 = put(C01)
        self.C11 = put(C11)
        self.Cs00 = put(
            np.stack([np.asarray(c, dtype=np.float64) for c in Cs00])
        )
        self.Cs01 = put(
            np.stack([np.asarray(c, dtype=np.float64) for c in Cs01])
        )
        self.n_shift = self.Cs00.shape[0]


def _gather(mat, rows, cols):
    """Stack mat[rows[g]][:, cols[g]] over g; mat (D, D) or (J, D, D)."""
    if mat.dim() == 2:
        return torch.stack(
            [
                mat.index_select(0, r).index_select(1, c)
                for r, c in zip(rows, cols)
            ]
        )
    return torch.stack(
        [mat.index_select(1, r).index_select(2, c) for r, c in zip(rows, cols)]
    )


def _leading_null_rho_group_gpu(
    state, group_supports, unit_dim, env_dim, ridge
):
    """One batched pass for a group of supports that all share the same
    support size (and therefore the same S/E column counts and basis
    shapes): batches the basis step across supports, then batches the
    leading-correlation step across every (support, shift) pair.

    Returns a numpy array of shape (len(group_supports), n_shift).
    """
    N, m, dev = state.N, state.m, state.device
    G = len(group_supports)
    J = state.n_shift

    idx = [_support_indices(s, N, m) for s in group_supports]
    S_list = [
        torch.as_tensor(S, dtype=torch.long, device=dev) for _, S, _ in idx
    ]
    E_list = [
        torch.as_tensor(E, dtype=torch.long, device=dev) for _, _, E in idx
    ]
    n_s, n_e = S_list[0].numel(), E_list[0].numel()
    ku, ke = min(unit_dim, n_s), min(env_dim, n_e)

    C00_SS = _gather(state.C00, S_list, S_list)
    C01_SS = _gather(state.C01, S_list, S_list)
    C11_SS = _gather(state.C11, S_list, S_list)
    C00_EE = _gather(state.C00, E_list, E_list)
    C01_EE = _gather(state.C01, E_list, E_list)

    Vu = _koopman_basis_batched(C00_SS, C01_SS, ku, ridge)  # (G, n_s, ku)
    Ve = _koopman_basis_batched(C00_EE, C01_EE, ke, ridge)  # (G, n_e, ke)
    del C01_EE

    Chh = Vu.transpose(-1, -2) @ C00_SS @ Vu
    Cyy = Vu.transpose(-1, -2) @ C11_SS @ Vu
    Chy = Vu.transpose(-1, -2) @ C01_SS @ Vu
    Czz = Ve.transpose(-1, -2) @ C00_EE @ Ve
    del C00_EE

    Ih = _inv_sqrt_batched(Chh, ridge)
    Ih = Ih @ Ih
    Ryy = Cyy - Chy.transpose(-1, -2) @ Ih @ Chy

    Cs00_ES = _gather(state.Cs00, E_list, S_list)  # (G, J, n_e, n_s)
    Cs01_ES = _gather(state.Cs01, E_list, S_list)  # (G, J, n_e, n_s)

    GJ = G * J
    Cs00_flat = Cs00_ES.reshape(GJ, n_e, n_s)
    Cs01_flat = Cs01_ES.reshape(GJ, n_e, n_s)

    Vu_rep = Vu.repeat_interleave(J, dim=0)
    Ve_rep = Ve.repeat_interleave(J, dim=0)
    Ih_rep = Ih.repeat_interleave(J, dim=0)
    Ryy_rep = Ryy.repeat_interleave(J, dim=0)
    Chy_rep = Chy.repeat_interleave(J, dim=0)
    Czz_rep = Czz.repeat_interleave(J, dim=0)

    Chz_n = Vu_rep.transpose(-1, -2) @ Cs00_flat.transpose(-1, -2) @ Ve_rep
    Czy_n = Ve_rep.transpose(-1, -2) @ Cs01_flat @ Vu_rep
    Rzz_n = Czz_rep - Chz_n.transpose(-1, -2) @ Ih_rep @ Chz_n

    inv_sqrt_Ryy = _inv_sqrt_batched(Ryy_rep, ridge)
    inv_sqrt_Rzz_n = _inv_sqrt_batched(Rzz_n, ridge)
    inner = (Czy_n - Chz_n.transpose(-1, -2) @ Ih_rep @ Chy_rep).transpose(
        -1, -2
    )
    M_n = inv_sqrt_Ryy @ inner @ inv_sqrt_Rzz_n

    sv = torch.linalg.svdvals(M_n)
    rho = torch.clamp(sv[:, 0], 0.0, 1.0)
    return rho.reshape(G, J).to("cpu", dtype=torch.float64).numpy()


def _rows_gpu(state, supports, unit_dim, env_dim, ridge):
    """Group supports by size (which fixes every downstream shape) and run
    one batched pass per group.
    """
    groups = {}
    for i, support in enumerate(supports):
        residues, _, _ = _support_indices(support, state.N, state.m)
        groups.setdefault(residues.size, []).append((i, support))
    result = np.empty((len(supports), state.n_shift), dtype=np.float64)
    for items in groups.values():
        block = _leading_null_rho_group_gpu(
            state,
            [s for _, s in items],
            int(unit_dim),
            int(env_dim),
            float(ridge),
        )
        for pos, (i, _) in enumerate(items):
            result[i] = block[pos]
    return result


def leading_null_rho_batch_gpu(
    C00,
    C01,
    C11,
    Cs00,
    Cs01,
    supports,
    *,
    n_residues,
    unit_dim,
    env_dim,
    ridge=1e-2,
    device="cuda",
    _state=None,
):
    """Return a numpy array of shape (len(supports), len(Cs00)).

    Same contract as leading_null_rho_batch_cpu.

    Batches rather than loops: supports are grouped by their support size,
    and each group is evaluated with one batched eigh call for the basis
    step and one batched svd call for the leading-correlation step across
    every (support, shift) pair in that group -- see
    `_leading_null_rho_group_gpu`.

    Runs the batched (D, D) linear algebra in float64, matching the
    precision interface.py's own GPU path uses for the same conditioning
    reason (see _koopman_basis's torch branch).  `_state` (internal) reuses
    an existing `_DeviceState` instead of uploading the covariances again.
    """
    state = (
        _state
        if _state is not None
        else _DeviceState(
            C00, C01, C11, Cs00, Cs01, N=n_residues, device=device
        )
    )
    return _rows_gpu(state, supports, unit_dim, env_dim, ridge)


def calibrate_null_thresholds_gpu(
    C00,
    C01,
    C11,
    adjacency,
    sizes,
    Cs00,
    Cs01,
    *,
    unit_dim,
    env_dim,
    ridge=1e-2,
    target_rel_width=0.02,
    batch_supports=10,
    min_supports=50,
    max_pool_size=5000,
    seed=0,
    device="cuda",
    on_log=None,
):
    """GPU reimplementation of calibrate_null_thresholds, structurally
    identical to calibrate_null_thresholds_cpu except that each round is one
    batched device pass.  Same sampling, random stream, stopping rule and
    return shape -- including "config", per-size "elapsed_seconds" and
    "per_support".  The covariances are uploaded once for the whole run;
    the upload is included in the total "elapsed_seconds".
    """
    log = on_log if on_log is not None else (lambda *_: None)
    t_start = time.perf_counter()
    rng = np.random.default_rng(seed)
    adjacency = np.asarray(adjacency, dtype=bool)
    N = adjacency.shape[0]
    n_shift_given = len(Cs00)
    if int(min_supports) * n_shift_given > int(max_pool_size):
        raise ValueError(
            f"min_supports={min_supports} x {n_shift_given} shifts exceeds "
            f"max_pool_size={max_pool_size}: no size could ever be declared "
            f"converged"
        )
    state = _DeviceState(C00, C01, C11, Cs00, Cs01, N=N, device=device)

    def sample_support(k, tries=40):
        s = _grow_ball_support(N, k, adjacency, rng)
        for _ in range(tries - 1):
            if len(s) == k:
                break
            s = _grow_ball_support(N, k, adjacency, rng)
        return s

    def sync():
        if state.device == "cuda":
            torch.cuda.synchronize()

    theta, diagnostics = {}, {}
    for k in sizes:
        k = int(k)
        t_size = time.perf_counter()
        pool, densities, eff_dims, per_support = [], [], [], []
        support_rows = []
        n_supports = 0
        rel_width, converged, stopped_by = float("inf"), False, None

        while True:
            batch = [sample_support(k) for _ in range(batch_supports)]
            rows = _rows_gpu(state, batch, unit_dim, env_dim, ridge)

            for support, row in zip(batch, rows):
                residues, S, _ = _support_indices(support, N, state.m)
                density, eff_dim = _support_descriptors(
                    residues, S, adjacency, state.C00_np
                )
                densities.append(density)
                eff_dims.append(eff_dim)
                n_supports += 1
                pool.extend(row.tolist())
                support_rows.append(row)
                per_support.append(
                    {
                        "support": [int(i) for i in residues],
                        "leading_rho": [float(v) for v in row],
                        "theta_S": float(np.percentile(row, 95)),
                        "contact_density": float(density),
                        "effective_dim": float(eff_dim),
                    }
                )

            # The support-level bootstrap is only informative once there are
            # enough supports, so it is not run before min_supports.
            if n_supports >= min_supports:
                _, rel_width = _bootstrap_ci_rel_width_two_way(
                    support_rows, rng=rng
                )
            if n_supports >= min_supports and rel_width < target_rel_width:
                converged, stopped_by = True, "converged"
                break
            if len(pool) >= max_pool_size:
                stopped_by = "max_pool_size"
                log(
                    f"calibrate_null_thresholds_gpu: size {k} hit "
                    f"max_pool_size ({max_pool_size}) before converging "
                    f"(rel CI width {rel_width:.4f} >= {target_rel_width})"
                )
                break

        sync()
        _, rel_width_naive = _bootstrap_ci_rel_width(pool, rng=rng)
        theta[k] = float(np.percentile(pool, 95))
        diagnostics[k] = {
            "n_pairs": len(pool),
            "n_supports": n_supports,
            "rel_ci_width": float(rel_width),
            "rel_ci_width_naive": float(rel_width_naive),
            "converged": bool(converged),
            "stopped_by": stopped_by,
            "elapsed_seconds": time.perf_counter() - t_size,
            "contact_density": densities,
            "effective_dim": eff_dims,
            "per_support": per_support,
        }
        log(
            f"calibrate_null_thresholds_gpu: size {k} theta={theta[k]:.4f} "
            f"n_pairs={len(pool)} n_supports={n_supports} "
            f"rel_ci_width={rel_width:.4f} (naive {rel_width_naive:.4f}) "
            f"converged={converged} "
            f"({diagnostics[k]['elapsed_seconds']:.1f}s)"
        )

    config = {
        "unit_dim": int(unit_dim),
        "env_dim": int(env_dim),
        "ridge": float(ridge),
        "n_shifts": int(state.n_shift),
        "target_rel_width": float(target_rel_width),
        "batch_supports": int(batch_supports),
        "min_supports": int(min_supports),
        "max_pool_size": int(max_pool_size),
        "seed": int(seed),
        "bootstrap": "two_way",
        "device": str(state.device),
    }
    return {
        "theta": theta,
        "diagnostics": diagnostics,
        "config": config,
        "elapsed_seconds": time.perf_counter() - t_start,
    }
