"""Scoring helpers shared by the COG pipeline stages.

Workspace construction, scorer set-up, unit status and reporting, and the
random-region diagnostics used by the discovery and verification stages.
"""

from __future__ import annotations

import numpy as np

from pdukit.cog.scoring.pair import PairFluxWorkspace
from pdukit.cog.scoring.scorer import FluxScorer
from pdukit.cog.scoring.workspace import FluxWorkspace


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
