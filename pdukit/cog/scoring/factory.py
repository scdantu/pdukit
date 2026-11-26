"""Scoring helpers shared by the COG pipeline stages.

Workspace construction, scorer set-up, unit status and reporting, and the
random-region diagnostics used by the discovery and verification stages.
"""

from __future__ import annotations

import numpy as np

from pdukit.cog.runinfo import available_memory, gpu_free_memory
from pdukit.cog.scoring.backend import _resolve_backend
from pdukit.cog.scoring.pair import PairFluxWorkspace
from pdukit.cog.scoring.scorer import FluxScorer
from pdukit.cog.scoring.workspace import FluxWorkspace
from pdukit.errors import ResourceError
from pdukit.runlog import log


def cond_kwargs(a):
    """How each block's own present is subtracted (see FluxDirection)."""
    ce = a.cond_env
    if ce not in (None, "full"):
        ce = int(ce)
    return {"cond_unit": a.cond_unit, "cond_env": ce}


def channels_vs_truth(units_report, truth, min_jaccard=0.8):
    """Measured nu_in / nu_out against the planted gear counts, for units
    that match a planted unit well.  Face residues add real channels the
    gear list does not count, so runs with faces read high by design.
    """
    T = [set(u) for u in truth["units"]]
    cpl = truth["couplings"]
    rows = []
    for e in units_report:
        if "residues" not in e:
            continue
        u = set(e.get("unit_residues", e["residues"]))
        j = [len(u & t) / len(u | t) for t in T]
        k = int(np.argmax(j))
        if j[k] < min_jaccard:
            continue
        ti = sum(r for s_, d_, r in cpl if d_ == k)
        to = sum(r for s_, d_, r in cpl if s_ == k)
        rows.append(
            {
                "unit": e["unit"],
                "true_unit": k,
                "jaccard": round(j[k], 3),
                "planted_in": ti,
                "planted_out": to,
                "nu_in": e["nu_in"],
                "nu_out": e["nu_out"],
            }
        )
    err = [r["nu_in"] - r["planted_in"] for r in rows] + [
        r["nu_out"] - r["planted_out"] for r in rows
    ]
    return {
        "per_unit": rows,
        "mean_error": float(np.mean(err)) if err else float("nan"),
        "mean_abs_error": float(np.mean(np.abs(err))) if err else float("nan"),
        "exact_after_rounding": (
            float(np.mean(np.round(err) == 0)) if err else float("nan")
        ),
        "n_faces": len(truth.get("face_residues", [])),
    }


def choose_window(rows, tol=0.15):
    """Smallest scanned high-pass window whose null threshold
    max(theta_in, theta_out) is within a fraction `tol` of the lowest one in
    the scan.  Longer windows keep slower (less well sampled) motion and
    raise the null; they are taken only if they cost nothing, i.e. never in
    preference to a shorter window with a similar threshold.
    Returns (window_frames, lowest threshold).
    """
    score = {
        int(r["window_frames"]): max(r["theta_in"], r["theta_out"])
        for r in rows
    }
    best = min(score.values())
    near = [w for w, v in score.items() if v <= (1.0 + tol) * best]
    return min(near), float(best)


def ball_support(N, k, adjacency, rng):
    """Return a random ball of ``k`` residues grown on the graph."""
    from pdukit.cog.calibration.scorer import _ball

    return _ball(N, k, adjacency, rng)


def engine_scorer(a, ws, schur=None):
    """Return an uncalibrated scorer whose GPU engine can be shared.

    The engine (if any) is then shared by the calibration and the
    calibrated scorer: one copy of the shifts on the device, not two.
    """
    b = use_batched(a, ws)
    return FluxScorer(
        ws,
        unit_dim=a.unit_dim,
        env_dim=a.env_dim,
        **cond_kwargs(a),
        backend=a.backend,
        schur=schur,
        combine=a.combine,
        batched=b,
        env_basis=a.env_basis if b else "exact",
    )


def use_batched(a, ws):
    """Decide whether to use the batched engine.

    --batched auto: when the workspace is on CUDA and the settings are
    supported.
    """
    if a.batched == "off":
        return False
    supported = (
        a.cond_unit == "full"
        and a.cond_env == "full"
        and a.backend == "flux"
        and not a.pair_features
    )
    if a.batched == "on":
        return supported
    from pdukit.cog.scoring.scorer import _null_device

    return supported and _null_device(ws) == "cuda"


def build_workspace(a, feats, segs, lag, n_null, seed, pair=None):
    """Build the lagged-covariance workspace of the given replicas."""
    mode = a.shift_mode
    if mode == "auto":
        mode = "cross" if len(segs) >= 2 else "within"
    kw = dict(
        lag=lag,
        segments=segs,
        n_null=n_null,
        seed=seed,
        ridge=a.ridge,
        device=a.device,
        shift_mode=mode,
        reversible=True,
        truncation="koopman",
    )
    if pair is not None:
        pf, pairs = pair
        return PairFluxWorkspace(
            pf, pairs, feats.shape[1], residue_feats=feats, **kw
        )
    return FluxWorkspace(feats, **kw)


def workspace_bytes(n_frames, D, n_null):
    """Return the estimated peak RAM of one workspace, in bytes.

    The workspace stacks every residue's K features into one vector of
    length D = N x K and keeps D x D covariances in double precision: C00,
    C01, C11, three accumulators while they are built, and two per null
    shift; plus the trajectory and its working copies (~12 bytes per frame
    per D).  So it grows with (N K)^2 x shifts:

        (6 + 2 n_null) D^2 x 8  +  12 F D  bytes

    Measured peaks are 0.9-1.25 times this for D = 240-600.
    """
    D = int(D)
    return (6 + 2 * int(n_null)) * D * D * 8 + int(n_frames) * D * 12


def gpu_bytes(D, n_null):
    """Return the estimated GPU memory of the batched engine, in bytes.

    C00, C01, C11 and two matrices per null shift, D x D float64.
    """
    return (3 + 2 * int(n_null)) * int(D) ** 2 * 8


def _gb(b):
    return f"{b / 1e9:.1f} GB"


def check_memory(a, feats, n_null, what, extra=0):
    """Log the memory a workspace needs; raise ResourceError if it cannot fit.

    ``feats`` (frames, N, K) are the features it will be built on,
    ``n_null`` its number of shifts, ``what`` names it in the messages and
    ``extra`` adds bytes needed at the same time (e.g. a filtered copy).
    The message says which settings make it fit.  ``no-memory-check = 1``
    skips this.
    """
    F, N = int(feats.shape[0]), int(feats.shape[1])
    K = int(feats.shape[2]) if feats.ndim == 3 else 1
    D = N * K
    need = workspace_bytes(F, D, n_null) + int(extra)
    free = available_memory()
    on_gpu = (
        _resolve_backend(a.device)[2] == "cuda"
        and getattr(a, "batched", "auto") != "off"
    )
    gneed = gpu_bytes(D, n_null) if on_gpu else 0
    gfree = gpu_free_memory() if on_gpu else None
    text = f"memory for {what}: ~{_gb(need)} (D = {N} x {K} = {D}, "
    text += f"{n_null} shifts), "
    text += f"{_gb(free)} free" if free else "free memory unknown"
    if on_gpu:
        text += f"; GPU ~{_gb(gneed)}, " + (
            f"{_gb(gfree)} free" if gfree else "free unknown"
        )
    log(text)
    short = []
    if free and need > free:
        short.append(("RAM", free, lambda k, n: (
            workspace_bytes(F, N * k, n) + extra)))
    if gfree and gneed > gfree:
        short.append(("GPU memory", gfree, lambda k, n: gpu_bytes(N * k, n)))
    if not short or getattr(a, "no_memory_check", False):
        return need
    _, room, _ = min(short, key=lambda s: s[1])

    def fits(k, n):
        return all(cost(k, n) <= 0.85 * r for _, r, cost in short)

    k_fit = max((k for k in range(1, K + 1) if fits(k, n_null)), default=0)
    n_fit = max((n for n in range(8, n_null + 1) if fits(K, n)), default=0)
    key = "rse-components" if int(a.rse_components) > 0 else (
        "rse-max-components")
    shifts = "null-cal-n-null" if "scan" in what else "n-null"
    ram_cost = short[0][2] if short[0][0] == "RAM" else None
    fixes = []
    if k_fit >= 2:
        fixes.append(f"{key}={k_fit}" + (
            f" (~{_gb(ram_cost(k_fit, n_null))})" if ram_cost else ""))
    if n_fit:
        fixes.append(f"{shifts}={n_fit}" + (
            f" (~{_gb(ram_cost(K, n_fit))})" if ram_cost else ""))
    if "scan" in what:
        fixes.append("no-null-calibration=1 (with a fixed highpass-window)")
    if not ram_cost:
        fixes.append("device=cpu (the GPU is what is short)")
    names = " and ".join(name for name, _, _ in short)
    msg = (
        f"not enough {names} for {what}: needs ~{_gb(need)}"
        + (f" (GPU ~{_gb(gneed)})" if on_gpu else "")
        + f", {_gb(room)} free.  Memory grows with (residues x K)^2 x "
        "shifts; set "
        + (" or ".join(fixes) if fixes else "nothing smaller fits: use a "
           "larger machine")
        + ".  no-memory-check=1 runs anyway."
    )
    log("!!" + msg)
    raise ResourceError(msg)
