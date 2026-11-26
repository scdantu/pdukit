"""Stages 1-4: features, replica split, window scan, preprocessing."""

from __future__ import annotations

import time

from pdukit.cog.calibration.scans import WindowScan
from pdukit.cog.features import (
    FeatureSource,
    center_replicas,
    highpass_replicas,
    load_data,
    skip_start,
    take_replicas,
    tica_spectrum_summary,
)
from pdukit.cog.scoring.pair import pair_distance_features
from pdukit.runlog import log


class LoadFeaturesStage:
    """Stage 1: planted system or MD replicas -> features."""

    code = "DAT"

    def run(self, state):
        """Advance ``state`` by this stage."""
        args, results = state.args, state.results
        state.source = (
            None if args.planted else FeatureSource(args, state.timings)
        )
        data = load_data(args, state.timings, state.source)
        state.data = data
        state.feats = data["feats"]
        state.segments = data["segments"]
        state.adj = data["adjacency"]
        state.spectra = data.get("tica_spectra")
        state.rse_report = data.get("rse_report")
        state.n_res = state.feats.shape[1]
        results["data"] = {
            "n_frames": int(state.feats.shape[0]),
            "n_residues": int(state.n_res),
            "n_feat": int(state.feats.shape[2]),
            "segments": [list(s) for s in state.segments],
            **data["meta"],
        }
        if data["truth"] is not None:
            results["truth"] = {
                k: v for k, v in data["truth"].items() if k != "K"
            }
        state.pair_all = None
        if args.pair_features:
            coords, _, _, dmed = state.source.coords()
            pf, pairs = pair_distance_features(
                coords, cutoff=args.pair_cutoff, dmed=dmed
            )
            state.pair_all = (pf, pairs)
            log(f"pair-owned features: {pairs.shape[0]} contact pairs")
            results["data"]["n_pairs_features"] = int(pairs.shape[0])
        state.save()


class SplitReplicasStage:
    """Stage 2: discovery and held-out replicas, replica overlap.

    The replica overlap is measured on the UNcentred features: it is
    the state differences it is there to see; centring follows it.
    """

    code = "REP"

    def run(self, state):
        """Advance ``state`` by this stage."""
        args, results, timings = state.args, state.results, state.timings
        state.n_rep = n_rep = len(state.segments)
        n_hold = args.n_holdout
        if n_hold is not None and n_hold >= n_rep:
            log(f"--n-holdout {n_hold} >= {n_rep} replicas: holding out 1")
            n_hold = 1
        if args.discovery_replicas is not None:
            disc = [int(x) for x in args.discovery_replicas.split(",") if x]
        elif n_hold is not None:
            disc = list(range(n_rep - n_hold))
        else:
            disc = list(range(n_rep - 1)) if n_rep >= 3 else list(range(n_rep))
        if args.holdout_replicas is not None:
            hold = (
                []
                if args.holdout_replicas.strip().lower() in ("", "none")
                else [int(x) for x in args.holdout_replicas.split(",") if x]
            )
        elif n_hold is not None:
            hold = list(range(n_rep - n_hold, n_rep))
        else:
            hold = [n_rep - 1] if n_rep >= 3 else []
        if set(disc) & set(hold):
            raise SystemExit("discovery and held-out replicas overlap")
        state.disc, state.hold = disc, hold
        results["split"] = {"discovery": disc, "holdout": hold}
        if n_rep >= 2 and not args.no_replica_overlap:
            from pdukit.cog.features import replica_overlap

            t = time.perf_counter()
            results["replica_overlap"] = replica_overlap(
                state.feats, state.segments, lag=args.lag
            )
            timings["replica_overlap_seconds"] = time.perf_counter() - t
            ro = results["replica_overlap"]
            for pr in ro["pairs"]:
                log(
                    f"replicas {pr['replicas']}  overlap "
                    f"{pr['overlap_per_tic']} (slowest {ro['n_tics']} tICs, "
                    f"1 = same states)  shift {pr['mean_shift_sd']} sd"
                )
            for i_, row in enumerate(ro["tau_int_frames"]):
                log(
                    f"replica {i_}  tau_int {row} frames  independent "
                    f"samples {ro['independent_samples_per_replica'][i_]}"
                )
            if ro["worst_overlap"] < 0.5:
                log(
                    "!! replicas sample different states on a slow "
                    "coordinate: held-out failures may be state dependence; "
                    "prefer --holdout-replicas none --per-replica-check"
                )
            state.save()


class ScanWindowStage:
    """Stage 2b: choose the high-pass window (``highpass-window = auto``).

    The scan builds a workspace for each of several windows, so it costs
    about as much time and memory as several runs of the null.  It is
    needed only to CHOOSE the window; with a fixed window (or none) it is
    skipped unless ``null-cal-check = 1`` asks for it as a check.
    """

    code = "WIN"

    def run(self, state):
        """Advance ``state`` by this stage."""
        args, results = state.args, state.results
        hp = args.highpass_window
        wanted = hp == "auto" or getattr(args, "null_cal_check", False)
        if wanted and not args.no_null_calibration and not args.planted:
            t = time.perf_counter()
            nc = WindowScan(args).run(state.feats, state.segments, state.adj)
            state.timings["null_calibration_seconds"] = time.perf_counter() - t
            results["null_calibration"] = nc
            if hp == "auto":
                hp = nc["chosen_window"]
                log(
                    f"highpass window {hp} frames "
                    f"({hp * args.frame_ps / 1000:.3g} ns)  theta_min "
                    f"{nc['theta_min']:.3f} (tol {args.null_cal_tol:.0%})"
                )
                if not nc["any_below_target"]:
                    log(
                        f"!! no window reached target "
                        f"{args.null_cal_target}; channel counts conservative"
                    )
                big = [x for x in (args.verify_lags or []) if x > hp / 5]
                if big:
                    log(f"!! lags {big} exceed window/5: skipped in lag scan")
            else:
                near = min(
                    nc["rows"],
                    key=lambda r: abs(r["window_frames"] - int(hp or 0)),
                )
                if (
                    max(near["theta_in"], near["theta_out"])
                    > args.null_cal_target
                ):
                    log(
                        f"!! window {hp}: theta {near['theta_in']:.3f} / "
                        f"{near['theta_out']:.3f} above target "
                        f"{args.null_cal_target}; try --highpass-window "
                        f"{nc['chosen_window']} or auto"
                    )
            state.save()
        elif hp == "auto":
            raise SystemExit(
                "--highpass-window auto needs the null "
                "calibration (MD data, without "
                "--no-null-calibration)"
            )
        elif not args.planted:
            log(
                "high-pass window scan skipped: "
                + (f"window fixed at {hp} frames" if int(hp or 0)
                   else "high-pass off")
                + " (highpass-window = auto chooses one; null-cal-check = 1 "
                "runs the scan as a check)"
            )
        state.hp = int(hp or 0)


class PreprocessStage:
    """Trim, high-pass and centre the replicas; slice discovery data."""

    code = "PRE"

    def run(self, state):
        """Advance ``state`` by this stage."""
        args, results = state.args, state.results
        hp = state.hp
        planted = args.planted
        if hp and args.highpass_before_features and not planted:
            # the unfiltered set is replaced next: free it first (memory)
            state.feats = None
            if isinstance(state.data, dict):
                state.data["feats"] = None
            fd = state.source.get(window=hp)
            state.feats, state.spectra = fd["feats"], fd.get("tica_spectra")
            state.rse_report = fd.get("rse_report")
        if args.skip_frames:
            old = state.segments
            state.feats, state.segments = skip_start(
                state.feats, old, args.skip_frames
            )
            if state.pair_all is not None:
                state.pair_all = (
                    skip_start(state.pair_all[0], old, args.skip_frames)[0],
                    state.pair_all[1],
                )
            log(
                f"skip {args.skip_frames} frames per replica; "
                f"segments {state.segments}"
            )
        feats, segments = state.feats, state.segments
        # Residue features built with --highpass-before-features were
        # filtered at the distance level; everything else is filtered here.
        if hp and args.highpass_before_features and not planted:
            log(f"highpass {hp} frames applied to raw distances")
        if hp and (not args.highpass_before_features or planted):
            t = time.perf_counter()
            highpass_replicas(feats, segments, hp)
            log(
                f"highpass {hp} frames per replica "
                f"({time.perf_counter() - t:.1f}s)"
            )
        if hp and state.pair_all is not None:
            highpass_replicas(state.pair_all[0], segments, hp)
        if state.spectra is not None:
            results["tica_spectrum"] = tica_spectrum_summary(
                state.spectra,
                args.lag,
                args.frame_ps,
                int(state.feats.shape[2]),
                report=state.rse_report,
            )
            ts = results["tica_spectrum"]
            log(
                "tICA median lag autocorr: "
                + " ".join(
                    f"{k + 1}:{v:.2f}"
                    for k, v in enumerate(ts["median_eigenvalue"][:16])
                )
                + f"  kept {ts['kept']}, strong beyond "
                f"{ts['strong_beyond_kept']}"
            )
            cov = ts.get("coverage")
            if cov:
                log(
                    f"tICA coverage (sum lambda^2) with {cov['n_components']}"
                    f" modes: median residue {cov['kept_median']:.2f}, 10th "
                    f"pct {cov['kept_q10']:.2f}; "
                    f"{cov['fraction_residues_reaching_target']:.0%} of "
                    f"residues reach {cov['target']:.0%} (median residue "
                    f"needs {cov['modes_for_target_median']})"
                )
        results["preprocessing"] = {
            "skip_frames": int(args.skip_frames),
            "highpass_window": int(hp),
            "highpass_before_features": bool(args.highpass_before_features),
            "fit": bool(args.fit),
            "segments_used": [list(x) for x in segments],
        }
        if not args.no_center_replicas:
            center_replicas(feats, segments)
            if state.pair_all is not None:
                center_replicas(state.pair_all[0], segments)
            log("replicas centred (--no-center-replicas to keep offsets)")
        results["centered_replicas"] = not args.no_center_replicas
        log(f"discovery {state.disc}  held-out {state.hold or 'none'}")
        state.f_disc, state.s_disc = take_replicas(feats, segments, state.disc)
        state.p_disc = (
            None
            if state.pair_all is None
            else (
                take_replicas(state.pair_all[0], segments, state.disc)[0],
                state.pair_all[1],
            )
        )
