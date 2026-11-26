"""Checks and diagnostics run around the discovery of COG units.

Each class is one stage: the high-pass window scan, the lag scan, the
channel shadow, the per-block / per-replica / amount-of-data checks, the
held-out check and the COG v1.0 comparison.  Stages that work on the state
of a pipeline take the run's `pipeline.state.RunState` and write their
results into its ``results`` and ``timings`` dictionaries.
"""

from __future__ import annotations

import time

import numpy as np

from pdukit.cog.calibration.controls import control_supports
from pdukit.cog.features import (
    center_replicas,
    highpass_replicas,
    take_replicas,
)
from pdukit.cog.scoring.factory import (
    ball_support,
    build_workspace,
    check_memory,
    choose_window,
    cond_kwargs,
    use_batched,
)
from pdukit.cog.scoring.scorer import FluxScorer
from pdukit.runlog import log


class WindowScan:
    """Scan of the high-pass window before discovery."""

    def __init__(self, args):
        """Keep the parsed command-line options."""
        self.args = args

    def run(self, feats_raw, segments, adjacency):
        """Scan the highpass window BEFORE discovery.

        For each window: the shift null threshold (theta_in, theta_out) and
        the ball baseline at a couple of sizes, on a copy of the unfiltered
        features.

        WHY.  The window decides which motions count as sampled; too long and
        the null threshold rises (lysozyme GA: 0.048 at 10 ns, 0.10 at 50 ns,
        0.14 at 100 ns), too short and real slow coupling is thrown away.  This
        puts the curve in every run's results and, with --highpass-window auto,
        picks the LONGEST window whose thresholds stay at or below
        --null-cal-target.

        The scan filters the reduced features (fast); a run with
        --highpass-before-features filters the raw distances instead, so its
        final thresholds can differ slightly -- they are recalibrated anyway.
        """
        a = self.args
        rng = np.random.default_rng(a.seed + 4242)
        n_rep = len(segments)
        keep = list(range(n_rep))
        if a.null_cal_max_replicas and n_rep > a.null_cal_max_replicas:
            keep = sorted(
                set(
                    np.linspace(0, n_rep - 1, a.null_cal_max_replicas).astype(
                        int
                    )
                )
            )
        f, s_ = take_replicas(feats_raw, segments, keep)
        Lmin = min(b - a_ for a_, b in s_)
        if a.null_cal_windows and a.null_cal_windows != "auto":
            wins = [int(x) for x in a.null_cal_windows.split(",") if x]
        else:
            lo = max(10 * a.lag, 10)
            hi = max(lo + 1, Lmin // 8)
            wins = sorted(
                set(
                    int(round(x))
                    for x in np.geomspace(lo, hi, a.null_cal_n_windows)
                )
            )
        sizes = [int(x) for x in str(a.null_cal_sizes).split(",") if x]
        N = feats_raw.shape[1]
        rows = []
        check_memory(
            a, f, a.null_cal_n_null, "the high-pass window scan",
            extra=f.nbytes,
        )
        for w in wins:
            t = time.perf_counter()
            X = f.copy()
            highpass_replicas(X, s_, w)
            center_replicas(X, s_)
            ws = build_workspace(
                a, X, s_, a.lag, a.null_cal_n_null, a.seed + 77, None
            )
            sc = FluxScorer(
                ws,
                unit_dim=a.unit_dim,
                env_dim=a.env_dim,
                combine=a.combine,
                **cond_kwargs(a),
                env_basis=a.env_basis if use_batched(a, ws) else "exact",
                batched=use_batched(a, ws),
            )
            ball_adj = adjacency
            if a.baseline == "random":
                ball_adj = ~np.eye(adjacency.shape[0], dtype=bool)
            elif a.baseline == "graph" and a.search_graph != "contact":
                from pdukit.cog.graph import predictive_graph

                ball_adj = predictive_graph(
                    ws, k=a.graph_k, null_shifts=4, log=lambda *_: None
                )["adjacency"]
            row = {"window_frames": int(w)}
            th_i, th_o, base = [], [], []
            for k in sizes:
                if not 1 <= k < N:
                    continue
                sups = [
                    s2
                    for s2 in (
                        ball_support(N, k, ball_adj, rng)
                        for _ in range(a.null_cal_supports)
                    )
                    if len(s2) == k
                ]
                P = sc.parts_batch(sups, with_null=True)
                ni = np.concatenate([p_["null_in"] for p_ in P])
                no = np.concatenate([p_["null_out"] for p_ in P])
                ti = float(np.percentile(ni, 95))
                to = float(np.percentile(no, 95))
                from pdukit.cog.scoring.scorer import soft_count

                nu = [
                    max(
                        soft_count(p_["rho_in"], ti),
                        soft_count(p_["rho_out"], to),
                    )
                    for p_ in P
                ]
                row[f"theta_in_{k}"] = ti
                row[f"theta_out_{k}"] = to
                row[f"baseline_nu_{k}"] = float(np.median(nu))
                th_i.append(ti)
                th_o.append(to)
                base.append(float(np.median(nu)))
            row["theta_in"] = float(np.max(th_i))
            row["theta_out"] = float(np.max(th_o))
            row["baseline_nu"] = float(np.median(base))
            row["seconds"] = time.perf_counter() - t
            rows.append(row)
            log(
                f"null_calibration: window {w} frames "
                f"({w * a.frame_ps / 1000:.3g}"
                f" ns): theta_in {row['theta_in']:.3f} theta_out "
                f"{row['theta_out']:.3f} baseline nu {row['baseline_nu']:.2f} "
                f"({row['seconds']:.0f}s)"
            )
            del ws, sc, X
        chosen, theta_min = choose_window(rows, a.null_cal_tol)
        return {
            "rows": rows,
            "target": a.null_cal_target,
            "tolerance": a.null_cal_tol,
            "theta_min": theta_min,
            "replicas_used": keep,
            "chosen_window": int(chosen),
            "any_below_target": bool(theta_min <= a.null_cal_target),
        }


class LagScan:
    """Re-score the units at other lags."""

    def __init__(self, args):
        """Keep the parsed command-line options."""
        self.args = args

    def run(self, feats, segs, units, adjacency, pair, highpass=0):
        """Re-score the units at each lag in --verify-lags.

        A lag comparable to the highpass window is not a measurement: the
        filter has removed the motions such a lag would see (synaptotagmin run
        7264296 rescored at 100 frames with a 100-frame window and every unit
        'faded').  Lags above window / 5 are skipped and logged.
        """
        a = self.args
        rows = []
        for lag in a.verify_lags:
            if highpass and lag > highpass / 5:
                log(
                    f"lag {lag}: skipped -- above a fifth of the highpass "
                    f"window ({highpass} frames); motions this slow were "
                    "removed"
                )
                rows.append({"lag": int(lag), "skipped": "beyond_highpass"})
                continue
            t = time.perf_counter()
            try:
                ws = build_workspace(
                    a,
                    feats,
                    segs,
                    lag,
                    a.verify_null,
                    a.seed + 1000 + lag,
                    pair,
                )
            except ValueError as exc:
                log(f"lag {lag}: cannot build workspace ({exc})")
                continue
            sc = FluxScorer(
                ws,
                unit_dim=a.unit_dim,
                env_dim=a.env_dim,
                **cond_kwargs(a),
                combine=a.combine,
                batched=use_batched(a, ws),
                env_basis=a.env_basis if use_batched(a, ws) else "exact",
            )
            for i, u in enumerate(units):
                try:
                    r = sc(u)
                except (ValueError, np.linalg.LinAlgError):
                    continue
                cs = control_supports(
                    feats,
                    len(u),
                    adjacency=adjacency,
                    n_random=0,
                    n_segment=0,
                    n_ball=a.n_controls,
                    seed=a.seed + lag + i,
                    exclude=units,
                    max_overlap=0.3,
                )
                vals = []
                for S in cs.get("ball", []):
                    try:
                        vals.append(sc(S)["nu"])
                    except (ValueError, np.linalg.LinAlgError):
                        pass
                med = float(np.median(vals)) if vals else float("nan")
                rows.append(
                    {
                        "lag": int(lag),
                        "unit": i,
                        "size": len(u),
                        "nu": r["nu"],
                        "nu_in": r["nu_in"],
                        "nu_out": r["nu_out"],
                        "control_ball_median_nu": med,
                        "gap": med - r["nu"] if vals else float("nan"),
                        "n_controls": len(vals),
                    }
                )
            log(
                f"lag {lag}: {len(units)} units re-scored "
                f"({time.perf_counter() - t:.1f}s)"
            )
            del ws, sc
        return rows


