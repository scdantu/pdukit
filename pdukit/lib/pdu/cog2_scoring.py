"""Scoring helpers shared by the COG pipeline stages.

Workspace construction, scorer set-up, unit status and reporting, and the
random-region diagnostics used by the discovery and verification stages.
"""

from __future__ import annotations

import numpy as np

from .cog2 import Cog2Scorer, PairFluxWorkspace, cohesion_check
from .interface import FluxWorkspace


def cond_kwargs(a):
    """How each block's own present is subtracted (see cog2._Direction)."""
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


def _rho8(x, n=8):
    x = [round(float(v), 4) for v in list(x)[:n]]
    return x + [None] * (n - len(x))


def unit_rho_rows(phase, results, units):
    """One row per unit: channel strengths (rho) next to the thresholds."""
    rows = []
    for i, (r, u) in enumerate(zip(results, units)):
        rows.append(
            {
                "phase": phase,
                "unit": i,
                "size": len(u),
                "nu_in": round(float(r["nu_in"]), 3),
                "nu_out": round(float(r["nu_out"]), 3),
                "cost": round(float(r["cost"]), 3),
                "theta_in": round(float(r["theta_in"]), 4),
                "theta_out": round(float(r["theta_out"]), 4),
                **{
                    f"rho_in_{k + 1}": v
                    for k, v in enumerate(_rho8(r["rho_in"]))
                },
                **{
                    f"rho_out_{k + 1}": v
                    for k, v in enumerate(_rho8(r["rho_out"]))
                },
            }
        )
    return rows


def random_region_rho(scorer, adjacency, sizes, n=30, seed=0):
    """Channel strengths of random regions (same kind of regions as the
    baseline): median and 90th percentile of rho_k, k = 1..8, per size.
    """
    rng = np.random.default_rng(seed)
    N = np.asarray(adjacency).shape[0]
    out = {}
    for k in sizes:
        if not 2 <= k < N:
            continue
        sups = [
            s_
            for s_ in (ball_support(N, k, adjacency, rng) for _ in range(n))
            if len(s_) == k
        ]
        if not sups:
            continue
        res = scorer.score_batch(sups)
        for d in ("in", "out"):
            M = np.array(
                [[*(r[f"rho_{d}"][:8]), *([np.nan] * 8)][:8] for r in res],
                dtype=float,
            )
            out.setdefault(str(k), {})[d] = {
                "median": [round(float(v), 4) for v in np.nanmedian(M, 0)],
                "p90": [
                    round(float(v), 4) for v in np.nanpercentile(M, 90, axis=0)
                ],
                "theta": round(float(res[0][f"theta_{d}"]), 4),
            }
        out[str(k)]["nu_median"] = round(
            float(np.median([r["nu"] for r in res])), 3
        )
    return out


def ball_support(N, k, adjacency, rng):
    """Return a random ball of ``k`` residues grown on the graph."""
    from .cog2 import _ball

    return _ball(N, k, adjacency, rng)


def engine_scorer(a, ws, schur=None):
    """Return an uncalibrated scorer whose GPU engine can be shared.

    The engine (if any) is then shared by the calibration and the
    calibrated scorer: one copy of the shifts on the device, not two.
    """
    b = use_batched(a, ws)
    return Cog2Scorer(
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
    from .cog2 import _null_device

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


def unit_status(cost, p_size, p_closed=0.1):
    """Only groups at least as closed as a typical same-size compact region
    are units.  cost > 0 (more channels than that) -> 'unresolved': not a
    unit, residues awaiting re-assignment.  cost <= 0 -> 'closed' if
    p_size <= p_closed, else 'weak'.
    """
    if cost is None or not np.isfinite(cost) or cost > 0:
        return "unresolved"
    return "closed" if p_size <= p_closed else "weak"


def unit_report(scorer, units, *, cohesion=True):
    """Return a status and cohesion report for every unit."""
    out = []
    for i, u in enumerate(units):
        try:
            r = scorer(u)
        except (ValueError, np.linalg.LinAlgError) as exc:
            out.append({"unit": i, "size": len(u), "error": str(exc)})
            continue
        e = {
            "unit": i,
            "size": len(u),
            "residues": list(map(int, u)),
            **{
                k: r[k]
                for k in (
                    "nu",
                    "nu_in",
                    "nu_out",
                    "theta_in",
                    "theta_out",
                    "k_star_in",
                    "k_star_out",
                    "baseline_nu",
                    "p_size",
                    "cost",
                    "sigma",
                )
            },
            "rho_in": [round(x, 5) for x in r["rho_in"][:6]],
            "rho_out": [round(x, 5) for x in r["rho_out"][:6]],
            "status": unit_status(r["cost"], r["p_size"]),
        }
        if cohesion:
            try:
                c = cohesion_check(scorer, u, result=r)
                e["cohesion"] = (
                    {
                        k: c[k]
                        for k in (
                            "verdict",
                            "nu_cut",
                            "nu_boundary",
                            "cohesion_margin",
                            "nu_a_to_b",
                            "nu_b_to_a",
                            "half_a",
                            "half_b",
                        )
                    }
                    if "nu_cut" in c
                    else {"verdict": c["verdict"]}
                )
            except (ValueError, np.linalg.LinAlgError) as exc:
                e["cohesion"] = {"verdict": "error", "error": str(exc)}
        out.append(e)
    return out
