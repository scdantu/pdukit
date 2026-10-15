"""Checks and diagnostics run around the discovery of COG units.

Each class is one stage: the high-pass window scan, the lag scan, the
channel shadow, the per-block / per-replica / amount-of-data checks, the
held-out check and the COG v1.0 comparison.  Stages that work on the state
of a pipeline take the running `Cog2Run` and write their results into its
``results`` and ``timings`` dictionaries.
"""

from __future__ import annotations

import time

import numpy as np

from .cog2 import Cog2Scorer, calibrate_cog2
from .cog2_data import (
    block_frames,
    center_replicas,
    highpass_replicas,
    make_blocks,
    take_blocks,
    take_replicas,
)
from .cog2_scoring import (
    ball_support,
    build_workspace,
    choose_window,
    cond_kwargs,
    engine_scorer,
    random_region_rho,
    unit_report,
    unit_rho_rows,
    use_batched,
)
from .interface import control_supports, decompose
from .null_calibration_run import log


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
        for w in wins:
            t = time.perf_counter()
            X = f.copy()
            highpass_replicas(X, s_, w)
            center_replicas(X, s_)
            ws = build_workspace(
                a, X, s_, a.lag, a.null_cal_n_null, a.seed + 77, None
            )
            sc = Cog2Scorer(
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
                from .cog2_graph import predictive_graph

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
                from .cog2 import soft_count

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
            sc = Cog2Scorer(
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


class ChannelShadow:
    """Report-only growth on the channel cost, next to cohesion growth.

    A channel-closed group that cohesion does not find and whose members do
    not predict each other pairwise better than random is COLLECTIVE: closed
    as a whole (few channels), but its members predict each other only
    jointly (e.g. a loop that follows two helices only when both move).  It
    is recorded and classified; the units are not changed.
    """

    def __init__(self, run):
        """Attach to the running pipeline."""
        self.run = run

    def execute(self):
        """Grow from the same seeds on the channel cost and classify."""
        run = self.run
        a = run.args
        units = run.units
        coh = run.coh
        t = time.perf_counter()
        log(
            "channel shadow: growing from the same seeds on the channel "
            "cost (report only)"
        )
        dsh = decompose(
            run.f_disc,
            run.search_adj,
            run.ws,
            unit_dim=a.unit_dim,
            env_dim=a.env_dim,
            n_seeds=a.dec_seeds,
            max_size=a.max_size,
            min_size=a.min_size,
            max_unit=a.max_unit,
            boundary_tol=a.boundary_tol,
            merge_jaccard=a.merge_jaccard,
            size_penalty=a.size_penalty,
            max_iter=a.dec_iters,
            n_controls=a.n_controls,
            seed=a.seed,
            on_log=lambda *_: None,
            scorer=run.scorer,
            cost_fn=Cog2Scorer.cost_fn,
            keep_best=not a.no_keep_best,
            batch_scorer=(
                run.scorer.score_batch
                if run.scorer.engine is not None
                else None
            ),
            seed_residues=run.seed_residues(),
            merge_mode=a.merge_mode,
            merge_gate=run.add_gate,
            merge_margin=a.stab_margin,
        )
        shadow = []
        rep_sh = unit_report(run.scorer, dsh["units"], cohesion=False)
        for g_, e_ in zip(dsh["units"], rep_sh):
            gs = set(int(x) for x in g_)
            jac = [len(gs & set(u)) / len(gs | set(u)) for u in units]
            jb = int(np.argmax(jac)) if jac else -1
            c_ = coh.cost_of(g_)[0]
            sg_ = coh.significant(g_)
            st_ = e_.get("status", "unresolved")
            if jac and jac[jb] >= 0.5:
                cls = "also found by cohesion"
            elif st_ != "closed":
                cls = "not closed by channels"
            elif not sg_:
                cls = "collective (closed by channels, not pairwise cohesive)"
            else:
                cls = "cohesive and closed, missed by the cohesion search"
            shadow.append(
                {
                    "residues": sorted(gs),
                    "size": len(gs),
                    "channel_cost": e_.get("cost"),
                    "p_size": e_.get("p_size"),
                    "closure": st_,
                    **{k: round(float(v), 4) for k, v in c_.items()},
                    "q95_random": round(coh.q95(len(gs)), 4),
                    "pairwise_cohesive": bool(sg_),
                    "best_cohesion_unit": jb,
                    "jaccard": round(float(jac[jb]), 3) if jac else 0.0,
                    "class": cls,
                }
            )
        counts = {}
        for e_ in shadow:
            counts[e_["class"]] = counts.get(e_["class"], 0) + 1
        run.results["channel_shadow"] = {
            "groups": shadow,
            "counts": counts,
            "seconds": round(time.perf_counter() - t, 1),
        }
        run.timings["channel_shadow_seconds"] = time.perf_counter() - t
        log(
            f"channel shadow: {len(shadow)} groups grown on channels: {counts}"
        )
        for e_ in shadow:
            if e_["class"].startswith(("collective", "cohesive and closed")):
                log(
                    f"  {e_['class']}: size {e_['size']}, channel cost "
                    f"{e_['channel_cost']:+.2f}, cohesion {e_['cohesion']:.3f}"
                    f" (q95 {e_['q95_random']:.3f}); residues "
                    f"{e_['residues']}"
                )
        run.writer.channel_shadow(shadow)


class BlockCheck:
    """Score the final units on every equal-length block of the data."""

    def __init__(self, run):
        """Attach to the running pipeline."""
        self.run = run

    def execute(self):
        """Score the units block by block with each block's calibration."""
        run = self.run
        a = run.args
        units = run.units
        f_disc, s_disc = run.f_disc, run.s_disc
        base_adj = run.base_adj
        disc = run.disc
        t = time.perf_counter()
        bf = block_frames(a, sum(b - a_ for a_, b in s_disc))
        blocks = make_blocks(s_disc, bf)
        sizes_b = sorted({len(u) for u in units})
        per_b = []
        for j, blk in enumerate(blocks):
            f_b, s_b = take_blocks(f_disc, [blk])  # one segment: shifts
            ws_b = build_workspace(
                a,
                f_b,
                s_b,
                a.lag,
                a.n_null,  # stay in
                a.seed + 500 + j,
                None,
            )  # block
            sr_b = engine_scorer(a, ws_b)
            cal_b = calibrate_cog2(
                ws_b,
                base_adj,
                sizes_b,
                unit_dim=a.unit_dim,
                **cond_kwargs(a),
                env_dim=a.env_dim,
                backend="flux",
                combine=a.combine,
                target_rel_width=a.target_rel_width,
                batch_supports=a.batch_supports,
                min_supports=a.min_supports,
                max_supports=a.max_supports,
                seed=a.seed + 600 + j,
                scorer=sr_b,
            )
            sc_b = Cog2Scorer(
                ws_b,
                unit_dim=a.unit_dim,
                env_dim=a.env_dim,
                **cond_kwargs(a),
                calibration=cal_b,
                combine=a.combine,
                engine=sr_b.engine,
                env_basis=sr_b.env_basis,
            )
            res_b = sc_b.score_batch([sorted(u) for u in units])
            per_b.append(
                {
                    "block": j,
                    "replica": int(disc[blk[0]])
                    if blk[0] < len(disc)
                    else int(blk[0]),
                    "frames": [int(blk[1]), int(blk[2])],
                    "costs": [round(float(r_["cost"]), 3) for r_ in res_b],
                    "nu": [round(float(r_["nu"]), 3) for r_ in res_b],
                }
            )
            del ws_b, sc_b, sr_b
        summ_b = []
        for i in range(len(units)):
            cs = [p_["costs"][i] for p_ in per_b]
            by_rep = {}
            for p_ in per_b:
                by_rep.setdefault(p_["replica"], []).append(p_["costs"][i] < 0)
            summ_b.append(
                {
                    "unit": i,
                    "closed_in": int(sum(c < 0 for c in cs)),
                    "of": len(cs),
                    "replicas_closed_in_most_blocks": int(
                        sum(np.mean(v) > 0.5 for v in by_rep.values())
                    ),
                    "replicas": len(by_rep),
                    "median_cost": round(float(np.median(cs)), 3),
                }
            )
        run.results["per_block"] = {
            "block_frames": bf,
            "block_us": (
                round(bf * a.frame_ps / 1e6, 3) if a.frame_ps else None
            ),
            "blocks": per_b,
            "summary": summ_b,
        }
        run.timings["per_block_seconds"] = time.perf_counter() - t
        log(
            f"per-block check: {len(blocks)} blocks of {bf} frames; units "
            "closed in (blocks): "
            + ", ".join(
                f"u{x['unit']} {x['closed_in']}/{x['of']}" for x in summ_b
            )
        )


class ReplicaCheck:
    """Score the final units on each replica on its own."""

    def __init__(self, run):
        """Attach to the running pipeline."""
        self.run = run

    def execute(self):
        """Re-calibrate on every replica and score the units there."""
        run = self.run
        a = run.args
        units = run.units
        t = time.perf_counter()
        per = []
        sizes_r = sorted({len(u) for u in units})
        for r_ in range(run.n_rep):
            f_r, s_r = take_replicas(run.feats, run.segments, [r_])
            p_r = (
                None
                if run.pair_all is None
                else (
                    take_replicas(run.pair_all[0], run.segments, [r_])[0],
                    run.pair_all[1],
                )
            )
            ws_r = build_workspace(
                a, f_r, s_r, a.lag, a.n_null, a.seed + 10 + r_, p_r
            )
            sr0 = engine_scorer(a, ws_r)
            cal_r = calibrate_cog2(
                ws_r,
                run.base_adj,
                sizes_r,
                unit_dim=a.unit_dim,
                **cond_kwargs(a),
                env_dim=a.env_dim,
                backend="flux",
                combine=a.combine,
                target_rel_width=a.target_rel_width,
                batch_supports=a.batch_supports,
                min_supports=a.min_supports,
                max_supports=a.max_supports,
                seed=a.seed + 10 + r_,
                scorer=sr0,
            )
            sc_r = Cog2Scorer(
                ws_r,
                unit_dim=a.unit_dim,
                env_dim=a.env_dim,
                **cond_kwargs(a),
                calibration=cal_r,
                combine=a.combine,
                engine=sr0.engine,
                env_basis=sr0.env_basis,
            )
            rep_r = unit_report(sc_r, units, cohesion=False)
            per.append(
                {
                    "replica": r_,
                    "in_discovery": r_ in run.disc,
                    "units": [
                        {
                            k: e.get(k)
                            for k in (
                                "unit",
                                "size",
                                "nu",
                                "nu_in",
                                "nu_out",
                                "cost",
                                "p_size",
                            )
                        }
                        for e in rep_r
                    ],
                }
            )
            log(
                f"replica {r_}: unit costs "
                f"{[round(e.get('cost', float('nan')), 2) for e in rep_r]}"
            )
            del ws_r, sc_r
        # per unit: does it stay more closed than balls in EVERY replica?
        summ = []
        for i in range(len(units)):
            costs = [p_["units"][i].get("cost") for p_ in per]
            costs = [c for c in costs if c is not None]
            summ.append(
                {
                    "unit": i,
                    "costs": costs,
                    "closed_in": int(sum(c < 0 for c in costs)),
                    "of": len(costs),
                    "spread": (
                        float(np.ptp(costs)) if costs else float("nan")
                    ),
                }
            )
        run.results["per_replica"] = {"replicas": per, "summary": summ}
        run.timings["per_replica_seconds"] = time.perf_counter() - t
        log(
            "per-replica: units more closed than balls in every replica: "
            f"{sum(s_['closed_in'] == s_['of'] for s_ in summ)}/{len(summ)}"
        )


class RhoScan:
    """Channel strengths and counts against the amount of data."""

    def __init__(self, run):
        """Attach to the running pipeline."""
        self.run = run

    def execute(self):
        """Re-score units and random regions on growing subsets of data."""
        run = self.run
        a = run.args
        units = run.units
        n_res = run.n_res
        f_disc, s_disc = run.f_disc, run.s_disc
        base_adj = run.base_adj
        t = time.perf_counter()
        fracs = [float(x) for x in str(a.rho_scan).split(",") if x]
        rng_s = np.random.default_rng(a.seed + 99)
        bf_s = block_frames(a, sum(b - a_ for a_, b in s_disc))
        blocks_s = make_blocks(s_disc, bf_s)
        order_s = list(rng_s.permutation(len(blocks_s)))
        sizes_s = sorted({len(u) for u in units} | {5, 10, 20})
        sizes_s = [k for k in sizes_s if 2 <= k < n_res]
        scan = []
        mode_keep = a.shift_mode
        a.shift_mode = "within"  # shifts stay inside each block
        for fr_ in fracs:
            # amount: in us of data if the frame interval is known (values
            # > 1 or a.rho_scan_us), else a fraction of all blocks
            if a.frame_ps and a.rho_scan_us:
                nb = max(1, int(round(fr_ * 1e6 / a.frame_ps / bf_s)))
            else:
                nb = max(1, int(round(fr_ * len(blocks_s))))
            nb = min(nb, len(blocks_s))
            f_s, s_s = take_blocks(
                f_disc, [blocks_s[k] for k in sorted(order_s[:nb])]
            )
            ws_s = build_workspace(
                a, f_s, s_s, a.lag, a.rho_scan_null, a.seed + 200, None
            )
            sr_s = engine_scorer(a, ws_s)
            cal_s = calibrate_cog2(
                ws_s,
                base_adj,
                sizes_s,
                unit_dim=a.unit_dim,
                **cond_kwargs(a),
                env_dim=a.env_dim,
                backend="flux",
                combine=a.combine,
                target_rel_width=a.target_rel_width,
                batch_supports=a.batch_supports,
                min_supports=30,
                max_supports=60,
                seed=a.seed + 300,
                scorer=sr_s,
            )
            sc_s = Cog2Scorer(
                ws_s,
                unit_dim=a.unit_dim,
                env_dim=a.env_dim,
                **cond_kwargs(a),
                calibration=cal_s,
                combine=a.combine,
                engine=sr_s.engine,
                env_basis=sr_s.env_basis,
            )
            ures = sc_s.score_batch([sorted(u) for u in units])
            nfr = int(sum(b - b0 for b0, b in s_s))
            theta_10 = (
                round(float(cal_s["theta_in"].get(10, np.nan)), 4)
                if 10 in cal_s["theta_in"]
                else None
            )
            row = {
                "amount": fr_,
                "n_blocks": nb,
                "block_frames": bf_s,
                "n_frames": nfr,
                "data_us": (
                    round(nfr * a.frame_ps / 1e6, 3) if a.frame_ps else None
                ),
                "theta_in_10": theta_10,
                "baseline_nu": {
                    str(k): round(float(v["median"]), 3)
                    for k, v in cal_s["baseline"].items()
                },
                "units": unit_rho_rows("scan", ures, units),
                "random": random_region_rho(
                    sc_s, base_adj, [5, 10, 20], seed=a.seed + 400
                ),
            }
            scan.append(row)
            log(
                f"rho scan {fr_:g} ({nb} blocks, {nfr} frames): theta(10) "
                f"{row['theta_in_10']}, random-region nu (size 10) "
                f"{row['baseline_nu'].get('10')}, unit nu "
                f"{[round(float(r_['nu']), 2) for r_ in ures]}"
            )
            del ws_s, sc_s, sr_s
            run.results["rho_scan"] = scan
            run.save()
        a.shift_mode = mode_keep
        run.timings["rho_scan_seconds"] = time.perf_counter() - t


class HoldoutCheck:
    """Score the discovered units on replicas they were not found on."""

    def __init__(self, run):
        """Attach to the running pipeline."""
        self.run = run

    def execute(self):
        """Calibrate on the held-out data and score the units there."""
        run = self.run
        a = run.args
        units = run.units
        results = run.results
        f_h, s_h = take_replicas(run.feats, run.segments, run.hold)
        p_h = (
            None
            if run.pair_all is None
            else (
                take_replicas(run.pair_all[0], run.segments, run.hold)[0],
                run.pair_all[1],
            )
        )
        t = time.perf_counter()
        ws_h = build_workspace(a, f_h, s_h, a.lag, a.n_null, a.seed + 1, p_h)
        schur_h = None
        if a.backend == "schur":
            from .schur import SchurWorkspace

            schur_h = SchurWorkspace(ws_h)
        sizes_h = sorted({len(u) for u in units})
        sh0 = engine_scorer(a, ws_h, schur_h)
        cal_h = calibrate_cog2(
            ws_h,
            run.base_adj,
            sizes_h,
            unit_dim=a.unit_dim,
            **cond_kwargs(a),
            env_dim=a.env_dim,
            backend=a.backend,
            schur=schur_h,
            combine=a.combine,
            target_rel_width=a.target_rel_width,
            batch_supports=a.batch_supports,
            min_supports=a.min_supports,
            max_supports=a.max_supports,
            seed=a.seed + 1,
            on_log=log,
            scorer=sh0,
        )
        sc_h = Cog2Scorer(
            ws_h,
            unit_dim=a.unit_dim,
            env_dim=a.env_dim,
            **cond_kwargs(a),
            calibration=cal_h,
            backend=a.backend,
            schur=schur_h,
            combine=a.combine,
            engine=sh0.engine,
            env_basis=sh0.env_basis,
        )
        rep = unit_report(sc_h, units, cohesion=True)
        for e in rep:
            e["replicates"] = bool(
                e.get("cost", 1.0) < 0.0 and e.get("p_size", 1.0) <= 0.1
            )
        results["units_holdout"] = rep
        results["holdout_calibration"] = {
            "theta_in": cal_h["theta_in"],
            "theta_out": cal_h["theta_out"],
            "baseline_median": {
                k: v["median"] for k, v in cal_h["baseline"].items()
            },
            "config": cal_h["config"],
        }
        run.timings["holdout_seconds"] = time.perf_counter() - t
        n_ok = sum(e["replicates"] for e in rep)
        log(
            f"held-out: {n_ok}/{len(rep)} units more closed than "
            f"compact balls (cost < 0 and p_size <= 0.1) "
            f"({run.timings['holdout_seconds']:.1f}s)"
        )
        del ws_h, sc_h
        run.save()
        return f_h, s_h, p_h


class Cog1Comparison:
    """COG v1.0 on the same discovery data, for comparison."""

    def __init__(self, run):
        """Attach to the running pipeline."""
        self.run = run

    def execute(self):
        """Run the v1.0 decomposition and score it against the truth."""
        run = self.run
        a = run.args
        results = run.results
        t = time.perf_counter()
        ws1 = build_workspace(
            a, run.f_disc, run.s_disc, a.lag, a.cog1_null, a.seed, run.p_disc
        )
        d1 = decompose(
            run.f_disc,
            run.adj,
            ws1,
            unit_dim=a.unit_dim,
            env_dim=a.env_dim,
            n_seeds=a.dec_seeds,
            max_size=a.max_size,
            min_size=a.min_size,
            max_unit=a.max_unit,
            boundary_tol=a.boundary_tol,
            merge_jaccard=a.merge_jaccard,
            size_penalty=a.size_penalty,
            max_iter=a.dec_iters,
            n_controls=a.n_controls,
            seed=a.seed,
            on_log=None,
            keep_best=not a.no_keep_best,
        )
        results["cog1"] = {
            "units": d1["units"],
            "scores": d1["scores"],
            "boundary_residues": d1["boundary_residues"],
            "gap_vs_best_control": d1["gap_vs_best_control"],
        }
        if run.data["truth"] is not None:
            from .benchmark import face_recovery, match_units

            results["cog1"]["vs_truth"] = match_units(
                d1["units"], run.data["truth"]["units"]
            )
            results["cog1"]["faces"] = face_recovery(
                d1["boundary_residues"], run.data["truth"]
            )
            log(f"COG v1.0 vs truth: {results['cog1']['vs_truth']}")
        run.timings["cog1_seconds"] = time.perf_counter() - t
