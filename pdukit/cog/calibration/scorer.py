"""Calibration of FluxScorer: size-conditioned thresholds, ball baseline.

See the module notes in flux/scorer.py (CALIBRATION).
"""

from __future__ import annotations

import time

import numpy as np

from pdukit.cog.calibration.bootstrap import (
    _bootstrap_ci_rel_width_two_way,
    _variance_components,
)
from pdukit.cog.calibration.controls import _grow_ball_support
from pdukit.cog.calibration.null import workspace_fingerprint
from pdukit.cog.scoring.scorer import (
    STATISTIC,
    VERSION,
    FluxScorer,
    _combine,
    soft_count,
)

# ---------------------------------------------------------------------------
# calibration: theta_in, theta_out and the ball baseline, per size
# ---------------------------------------------------------------------------


def _ball(N, k, adjacency, rng, tries=40):
    s = _grow_ball_support(N, k, adjacency, rng)
    for _ in range(tries - 1):
        if len(s) == k:
            break
        s = _grow_ball_support(N, k, adjacency, rng)
    return sorted(int(i) for i in s)


def calibrate_cog2(
    ws,
    adjacency,
    sizes,
    *,
    unit_dim=8,
    env_dim=20,
    backend="flux",
    schur=None,
    combine="max",
    eta=None,
    percentile=95.0,
    cond_unit="full",
    cond_env="full",
    target_rel_width=0.03,
    batch_supports=10,
    min_supports=40,
    max_supports=400,
    min_shifts=8,
    seed=0,
    on_log=None,
    batched=None,
    env_basis="exact",
    engine_device=None,
    scorer=None,
):
    """Per support size k: theta_in[k], theta_out[k] (the `percentile` of the
    leading shift-null correlation, pooled over compact ball supports and
    shifts) and the baseline distribution of nu over the same balls.

    Stopping: after `min_supports`, stop when the two-way (supports x shifts)
    bootstrap 95% CI of BOTH thetas is narrower than `target_rel_width`
    (relative), or at `max_supports`.  Returns a dict with theta_in,
    theta_out, baseline {k: median, q25, q75, frac_open, values},
    diagnostics and config (with the workspace fingerprint).
    """
    log = on_log if on_log is not None else (lambda *_: None)
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    adjacency = np.asarray(adjacency, dtype=bool)
    N = adjacency.shape[0]
    if len(ws.Cs00) < max(8, int(min_shifts)):
        ws.extend_null_shifts(max(8, int(min_shifts)) - len(ws.Cs00))
    if scorer is None:
        scorer = FluxScorer(
            ws,
            unit_dim=unit_dim,
            env_dim=env_dim,
            backend=backend,
            schur=schur,
            combine=combine,
            eta=eta,
            cond_unit=cond_unit,
            cond_env=cond_env,
            batched=batched,
            env_basis=env_basis,
            engine_device=engine_device,
        )
    theta_in, theta_out, baseline, diagnostics = {}, {}, {}, {}
    for k in sorted({int(s) for s in sizes}):
        if not 1 <= k < N:
            continue
        tk = time.perf_counter()
        rows_in, rows_out, real_in, real_out, sups = [], [], [], [], []
        w_in = w_out = float("inf")
        stopped = "max_supports"
        while len(sups) < max_supports:
            batch = [
                s_
                for s_ in (
                    _ball(N, k, adjacency, rng) for _ in range(batch_supports)
                )
                if len(s_) == k
            ]
            try:
                parts_list = scorer.parts_batch(batch, with_null=True)
            except (ValueError, np.linalg.LinAlgError):
                parts_list = []
                for s_ in batch:
                    try:
                        parts_list.append(scorer.parts(s_, with_null=True))
                    except (ValueError, np.linalg.LinAlgError):
                        parts_list.append(None)
            for sup, p in zip(batch, parts_list):
                if p is None:
                    continue
                sups.append(sup)
                rows_in.append(p["null_in"])
                rows_out.append(p["null_out"])
                real_in.append(p["rho_in"])
                real_out.append(p["rho_out"])
            if not sups:
                break
            if len(sups) >= min_supports:
                _, w_in = _bootstrap_ci_rel_width_two_way(
                    rows_in, percentile=percentile, n_boot=500, rng=rng
                )
                _, w_out = _bootstrap_ci_rel_width_two_way(
                    rows_out, percentile=percentile, n_boot=500, rng=rng
                )
                if max(w_in, w_out) < target_rel_width:
                    stopped = "converged"
                    break
        if not sups:
            log(f"size {k}: no ball support of that size")
            continue
        th_i = float(np.percentile(np.asarray(rows_in), percentile))
        th_o = float(np.percentile(np.asarray(rows_out), percentile))
        nu_i = np.array([soft_count(r, th_i, eta) for r in real_in])
        nu_o = np.array([soft_count(r, th_o, eta) for r in real_out])
        nu = np.array([_combine(a, b, combine) for a, b in zip(nu_i, nu_o)])
        theta_in[k], theta_out[k] = th_i, th_o
        baseline[k] = {
            "median": float(np.median(nu)),
            "q25": float(np.percentile(nu, 25)),
            "q75": float(np.percentile(nu, 75)),
            "median_nu_in": float(np.median(nu_i)),
            "median_nu_out": float(np.median(nu_o)),
            "frac_open": float(np.mean(nu > 0)),
            "values": [round(float(v), 5) for v in nu],
        }
        diagnostics[k] = {
            "n_supports": len(sups),
            "stopped_by": stopped,
            "rel_ci_width_in": float(w_in),
            "rel_ci_width_out": float(w_out),
            "variance_components_in": _variance_components(rows_in),
            "variance_components_out": _variance_components(rows_out),
            "elapsed_seconds": time.perf_counter() - tk,
            "supports": sups,
        }
        log(
            f"size {k:3d}  theta_in {th_i:.4f}  theta_out {th_o:.4f}  "
            f"nu0 {baseline[k]['median']:.2f} "
            f"[{baseline[k]['q25']:.2f}, {baseline[k]['q75']:.2f}]  "
            f"n={len(sups)} {stopped}  CI {w_in:.3f}/{w_out:.3f}  "
            f"{diagnostics[k]['elapsed_seconds']:.1f}s"
        )
    config = {
        "statistic": STATISTIC,
        "version": VERSION,
        "unit_dim": int(unit_dim),
        "env_dim": int(env_dim),
        "backend": backend,
        "combine": combine,
        "eta": eta,
        "cond_unit": cond_unit,
        "cond_env": cond_env,
        "env_basis": scorer.env_basis,
        "batched": scorer.engine is not None,
        "percentile": float(percentile),
        "n_shifts": int(len(ws.Cs00)),
        "target_rel_width": float(target_rel_width),
        "min_supports": int(min_supports),
        "max_supports": int(max_supports),
        "seed": int(seed),
        "fingerprint": workspace_fingerprint(ws),
    }
    return {
        "theta_in": theta_in,
        "theta_out": theta_out,
        "baseline": baseline,
        "diagnostics": diagnostics,
        "config": config,
        "elapsed_seconds": time.perf_counter() - t0,
    }
