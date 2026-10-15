"""The COG end-to-end pipeline as one class with a method per stage.

`Cog2Run` owns the state of a run (options, data, workspace, scorers, units)
and writes ``results.json`` after every stage, so a job that runs out of time
keeps what it finished.  The stages, in order:

    1. load_features        planted system or MD replicas -> features
    2. split_replicas       discovery and held-out replicas
    2b. scan_window         null-calibration scan of the high-pass window
    3. preprocess           trim, high-pass and centre the replicas
       make_workspace       lagged covariances and shift-null pairs
       make_graph           predictive (or contact) search graph
       calibrate            thresholds and the size-matched baseline
       make_gate            addition gate and direction protection
       make_cohesion_scorer predictive-cohesion scorer
    4. discover             seeds -> growth -> merge -> reassignment
    4b. find_faces          cores and faces (optional)
    5. report_units         status and cohesion of the discovered units
       resolve_round        second round for unresolved residues
       stabilise            Monte Carlo stable minimum and write-back
       score_pdudef         linear-estimator PDU definition (optional)
    6. verification         per-block, per-replica, rho scan, held-out, lags
    8. compare_cog1         COG v1.0 on the same data (optional)
       write_outputs        CSV tables
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from . import cog2
from .cog2 import (
    STATISTIC,
    VERSION,
    Cog2Scorer,
    calibrate_cog2,
    pair_distance_features,
    split_incoherent,
)
from .cog2_checks import (
    BlockCheck,
    ChannelShadow,
    Cog1Comparison,
    HoldoutCheck,
    LagScan,
    ReplicaCheck,
    RhoScan,
    WindowScan,
)
from .cog2_data import (
    FeatureSource,
    center_replicas,
    coarse_sizes,
    highpass_replicas,
    load_data,
    skip_start,
    take_replicas,
    tica_spectrum_summary,
)
from .cog2_ledger import ResidueLedger
from .cog2_outputs import OutputWriter
from .cog2_scoring import (
    build_workspace,
    channels_vs_truth,
    cond_kwargs,
    engine_scorer,
    random_region_rho,
    unit_report,
)
from .interface import decompose
from .null_calibration_run import _environment, _jsonable, _parse_sizes, log


class Cog2Run:
    """One COG run: state plus the stage methods that advance it."""

    def __init__(self, args):
        """Prepare the output directory and the results record."""
        self.args = args
        cog2.RHO_MIN = float(args.rho_min or 0.0)
        self.out = Path(args.outdir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.timings = {}
        repo_root = Path(__file__).resolve().parents[3]
        self.results = {
            "version": VERSION,
            "statistic": STATISTIC,
            "args": {
                k: v for k, v in vars(args).items() if not k.startswith("_")
            },
            "environment": _environment(repo_root),
            "timings": self.timings,
        }
        ppf = getattr(args, "_ppf", None)
        if ppf:
            (self.out / "pdu_out.ppf").write_text(ppf)
        self.writer = OutputWriter(self.out)
        self.ledger = ResidueLedger(self)
        # data
        self.source = None
        self.data = None
        self.feats = None
        self.segments = None
        self.adj = None
        self.spectra = None
        self.n_res = 0
        self.pair_all = None
        # replica split and preprocessing
        self.n_rep = 0
        self.disc = []
        self.hold = []
        self.hp = 0
        self.f_disc = self.s_disc = self.p_disc = None
        # workspace, graph and calibration
        self.ws = None
        self.schur = None
        self.search_adj = None
        self.base_adj = None
        self.rho_all = None
        self.theta_pair = None
        self.engine_probe = None
        self.cal = None
        self.add_gate = None
        self.protect = None
        self.coh = None
        self.grow_coh = False
        # discovery
        self.scorer = None
        self.scorer_eff = None
        self.dec = None
        self.units = []
        self.face_list = []
        # data used by the lag scan (held-out data if there is any)
        self.lag_data = None

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def save(self):
        """Rewrite ``results.json`` with everything recorded so far."""
        with (self.out / "results.json").open("w") as fh:
            json.dump(_jsonable(self.results), fh, indent=1, allow_nan=True)

    @property
    def cal_max(self):
        """Largest size that was calibrated."""
        return max(int(k) for k in self.results["calibration"]["theta_in"])

    def seed_residues(self):
        """Residues that seed the growth (None = spread over the graph)."""
        args = self.args
        if args.seeding == "spread":
            return None
        return (
            np.random.default_rng(args.seed)
            .choice(self.n_res, min(args.dec_seeds, self.n_res), replace=False)
            .tolist()
        )

    def _truth(self):
        """Planted truth, or None for MD data."""
        return self.data["truth"]

    def run(self):
        """Run every stage; return the process exit code."""
        args = self.args
        log(f"{VERSION} run -> {self.out}")
        self.save()
        self.load_features()
        self.split_replicas()
        self.scan_window()
        self.preprocess()
        self.make_workspace()
        self.make_graph()
        self.calibrate()
        if args.calibrate_only:
            log("--calibrate-only: stopping after calibration")
            return 0
        self.make_gate()
        self.make_cohesion_scorer()
        self.discover()
        self.find_faces()
        self.report_units()
        self.resolve_round()
        self.finish_ledger()
        if args.stability:
            self.stabilise()
        if args.pdudef:
            self.score_pdudef()
        self.verification()
        if args.compare_cog1:
            self.compare_cog1()
        self.write_outputs()
        return 0

    # ------------------------------------------------------------------
    # 1-2. data and replica split
    # ------------------------------------------------------------------

    def load_features(self):
        """Stage 1: planted system or MD replicas -> features."""
        args, results = self.args, self.results
        self.source = (
            None if args.planted else FeatureSource(args, self.timings)
        )
        data = load_data(args, self.timings, self.source)
        self.data = data
        self.feats = data["feats"]
        self.segments = data["segments"]
        self.adj = data["adjacency"]
        self.spectra = data.get("tica_spectra")
        self.n_res = self.feats.shape[1]
        results["data"] = {
            "n_frames": int(self.feats.shape[0]),
            "n_residues": int(self.n_res),
            "n_feat": int(self.feats.shape[2]),
            "segments": [list(s) for s in self.segments],
            **data["meta"],
        }
        if data["truth"] is not None:
            results["truth"] = {
                k: v for k, v in data["truth"].items() if k != "K"
            }
        self.pair_all = None
        if args.pair_features:
            coords, _, _, dmed = self.source.coords()
            pf, pairs = pair_distance_features(
                coords, cutoff=args.pair_cutoff, dmed=dmed
            )
            self.pair_all = (pf, pairs)
            log(f"pair-owned features: {pairs.shape[0]} contact pairs")
            results["data"]["n_pairs_features"] = int(pairs.shape[0])
        self.save()

    def split_replicas(self):
        """Stage 2: discovery and held-out replicas, replica overlap.

        The replica overlap is measured on the UNcentred features: it is
        the state differences it is there to see; centring follows it.
        """
        args, results, timings = self.args, self.results, self.timings
        self.n_rep = n_rep = len(self.segments)
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
        self.disc, self.hold = disc, hold
        results["split"] = {"discovery": disc, "holdout": hold}
        if n_rep >= 2 and not args.no_replica_overlap:
            from .cog2_data import replica_overlap

            t = time.perf_counter()
            results["replica_overlap"] = replica_overlap(
                self.feats, self.segments, lag=args.lag
            )
            timings["replica_overlap_seconds"] = time.perf_counter() - t
            ro = results["replica_overlap"]
            for pr in ro["pairs"]:
                log(
                    f"replicas {pr['replicas']}: histogram overlap on the "
                    f"{ro['n_tics']} slowest tICs {pr['overlap_per_tic']} "
                    f"(1 = same states), mean shift "
                    f"{pr['mean_shift_sd']} sd"
                )
            for i_, row in enumerate(ro["tau_int_frames"]):
                log(
                    f"replica {i_}: autocorrelation time of the slowest "
                    f"tICs {row} frames -> independent samples "
                    f"{ro['independent_samples_per_replica'][i_]}"
                )
            if ro["worst_overlap"] < 0.5:
                log(
                    "!! some replicas visit clearly different states on a "
                    "slow coordinate: read held-out failures as possible "
                    "state dependence, and prefer --holdout-replicas none "
                    "with --per-replica-check"
                )
            self.save()

    def scan_window(self):
        """Stage 2b: check that the high-pass window suits this lag."""
        args, results = self.args, self.results
        hp = args.highpass_window
        if not args.no_null_calibration and not args.planted:
            t = time.perf_counter()
            nc = WindowScan(args).run(self.feats, self.segments, self.adj)
            self.timings["null_calibration_seconds"] = time.perf_counter() - t
            results["null_calibration"] = nc
            if hp == "auto":
                hp = nc["chosen_window"]
                log(
                    f"null_calibration: using highpass window {hp} frames "
                    f"({hp * args.frame_ps / 1000:.3g} ns): smallest window "
                    f"with theta within {args.null_cal_tol:.0%} of the "
                    f"lowest ({nc['theta_min']:.3f})"
                )
                if not nc["any_below_target"]:
                    log(
                        f"!! null_calibration: NO window reached the target "
                        f"{args.null_cal_target}; channel counts will be "
                        "conservative"
                    )
                big = [x for x in (args.verify_lags or []) if x > hp / 5]
                if big:
                    log(
                        f"!! lags {big} exceed window/5 and will be skipped "
                        "in the lag scan"
                    )
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
                        f"!! null_calibration: at a window near {hp} frames "
                        f"the null threshold is {near['theta_in']:.3f} / "
                        f"{near['theta_out']:.3f}, above the target "
                        f"{args.null_cal_target}; consider "
                        f"--highpass-window {nc['chosen_window']} (or auto)"
                    )
            self.save()
        elif hp == "auto":
            raise SystemExit(
                "--highpass-window auto needs the null "
                "calibration (MD data, without "
                "--no-null-calibration)"
            )
        self.hp = int(hp or 0)

    def preprocess(self):
        """Trim, high-pass and centre the replicas; slice discovery data."""
        args, results = self.args, self.results
        hp = self.hp
        planted = args.planted
        if hp and args.highpass_before_features and not planted:
            fd = self.source.get(window=hp)
            self.feats, self.spectra = fd["feats"], fd.get("tica_spectra")
        if args.skip_frames:
            old = self.segments
            self.feats, self.segments = skip_start(
                self.feats, old, args.skip_frames
            )
            if self.pair_all is not None:
                self.pair_all = (
                    skip_start(self.pair_all[0], old, args.skip_frames)[0],
                    self.pair_all[1],
                )
            log(
                f"dropped the first {args.skip_frames} frames of every "
                f"replica; segments now {self.segments}"
            )
        feats, segments = self.feats, self.segments
        # Residue features built with --highpass-before-features were
        # filtered at the distance level; everything else is filtered here.
        if hp and args.highpass_before_features and not planted:
            log(
                f"residue features: {hp}-frame running mean "
                f"removed from the raw distances before the reduction"
            )
        if hp and (not args.highpass_before_features or planted):
            t = time.perf_counter()
            highpass_replicas(feats, segments, hp)
            log(
                f"removed a {hp}-frame running mean within every "
                f"replica ({time.perf_counter() - t:.1f}s): motions slower "
                f"than that are treated as unsampled"
            )
        if hp and self.pair_all is not None:
            highpass_replicas(self.pair_all[0], segments, hp)
        if self.spectra is not None:
            results["tica_spectrum"] = tica_spectrum_summary(
                self.spectra, args.lag, args.frame_ps, args.rse_components
            )
            ts = results["tica_spectrum"]
            log(
                "per-residue tICA (lag autocorrelation, median over "
                "residues) by component: "
                + " ".join(
                    f"{k + 1}:{v:.2f}"
                    for k, v in enumerate(ts["median_eigenvalue"][:16])
                )
                + f"  (kept {args.rse_components}; components beyond that "
                f"with median > 0.5: {ts['strong_beyond_kept']})"
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
            if self.pair_all is not None:
                center_replicas(self.pair_all[0], segments)
            log(
                "each replica centred on its own mean "
                "(--no-center-replicas to keep the between-replica "
                "offsets)"
            )
        results["centered_replicas"] = not args.no_center_replicas
        log(
            f"discovery replicas {self.disc}, held-out replicas "
            f"{self.hold or 'none'}"
        )
        self.f_disc, self.s_disc = take_replicas(feats, segments, self.disc)
        self.p_disc = (
            None
            if self.pair_all is None
            else (
                take_replicas(self.pair_all[0], segments, self.disc)[0],
                self.pair_all[1],
            )
        )

    # ------------------------------------------------------------------
    # 3. workspace, graph, calibration
    # ------------------------------------------------------------------

    def make_workspace(self):
        """Lagged covariances and shift-null pairs on the discovery data."""
        args = self.args
        t = time.perf_counter()
        self.ws = build_workspace(
            args,
            self.f_disc,
            self.s_disc,
            args.lag,
            args.n_null,
            args.seed,
            self.p_disc,
        )
        ws = self.ws
        self.timings["workspace_seconds"] = time.perf_counter() - t
        log(
            f"discovery workspace D={ws.D}, {ws.n} lagged pairs, "
            f"{len(ws.Cs00)} shifts ({ws.shift_mode}), backend "
            f"{ws.backend} ({self.timings['workspace_seconds']:.1f}s)"
        )

    def make_graph(self):
        """Build the neighbourhood graph: dynamical (default) or contacts."""
        args, results = self.args, self.results
        ws, adj, n_res = self.ws, self.adj, self.n_res
        dyn_adj = adj
        pg = None
        if args.search_graph != "contact":
            from .cog2_graph import predictive_graph

            t = time.perf_counter()
            pg = predictive_graph(
                ws,
                k=args.graph_k,
                null_shifts=args.graph_null_shifts,
                contact=adj,
                log=log,
            )
            self.timings["predictive_graph_seconds"] = time.perf_counter() - t
            dyn_adj = (
                pg["adjacency"] | np.asarray(adj, bool)
                if args.search_graph == "union"
                else pg["adjacency"]
            )
            results["predictive_graph"] = {
                k_: v_
                for k_, v_ in pg.items()
                if k_ not in ("adjacency", "rho")
            }
            results["predictive_graph"]["rho"] = np.round(
                pg["rho"], 3
            ).tolist()
            results["predictive_graph"]["edges"] = [
                [int(i), int(j)]
                for i, j in zip(*np.where(np.triu(dyn_adj, 1)))
            ]
            self.save()
        self.search_adj = dyn_adj
        self.rho_all = self.theta_pair = None
        if args.search_graph != "contact":
            self.rho_all, self.theta_pair = pg["rho"], pg["theta"]
        elif args.ledger or args.faces:
            from .cog2_graph import predictive_graph

            pg_quiet = predictive_graph(
                ws,
                k=args.graph_k,
                null_shifts=8,
                contact=adj,
                log=lambda *_: None,
            )
            self.rho_all, self.theta_pair = (
                pg_quiet["rho"],
                pg_quiet["theta"],
            )
        # comparison regions for the baseline / per-replica / lag controls
        if args.baseline == "random":
            self.base_adj = ~np.eye(n_res, dtype=bool)  # ball on a complete
        elif args.baseline == "contact":  # graph = a random
            self.base_adj = np.asarray(adj, bool)  # set
        else:
            self.base_adj = dyn_adj
        results["baseline_regions"] = args.baseline
        results["null_device"] = cog2._null_device(ws) or "cpu (numpy loop)"
        log(f"shift null computed on: {results['null_device']}")
        self.schur = None
        if args.backend == "schur":
            from .schur import SchurWorkspace

            self.schur = SchurWorkspace(ws)
            log(f"Schur precompute {self.schur.precompute_seconds:.1f}s")

    def calibrate(self):
        """Thresholds and the size-matched baseline on discovery data."""
        args, results, timings = self.args, self.results, self.timings
        n_res = self.n_res
        sizes = (
            _parse_sizes(args.sizes)
            if args.sizes
            else list(range(2, min(args.max_unit, n_res - 1) + 1))
        )
        if args.calibrate_coarse:
            full = sizes
            sizes = coarse_sizes(min(full), max(full), args.calibrate_coarse)
            log(
                f"calibrating {len(sizes)} sizes {sizes} and interpolating "
                f"the other {len(full) - len(sizes)} (--calibrate-coarse)"
            )
        t = time.perf_counter()
        probe = engine_scorer(args, self.ws, self.schur)
        self.engine_probe = probe
        log(
            "candidate scoring: "
            + (
                f"batched on the GPU engine ({probe.env_basis} environment "
                f"basis)"
                if probe.engine is not None
                else "one at a time"
            )
        )
        cal = calibrate_cog2(
            self.ws,
            self.base_adj,
            sizes,
            unit_dim=args.unit_dim,
            **cond_kwargs(args),
            env_dim=args.env_dim,
            backend=args.backend,
            schur=self.schur,
            combine=args.combine,
            target_rel_width=args.target_rel_width,
            batch_supports=args.batch_supports,
            min_supports=args.min_supports,
            max_supports=args.max_supports,
            seed=args.seed,
            on_log=log,
            scorer=probe,
        )
        self.cal = cal
        timings["calibration_seconds"] = time.perf_counter() - t
        with (self.out / "calibration.json").open("w") as fh:
            json.dump(_jsonable(cal), fh, indent=1, allow_nan=True)
        results["calibration"] = {
            "theta_in": cal["theta_in"],
            "theta_out": cal["theta_out"],
            "baseline_median": {
                k: v["median"] for k, v in cal["baseline"].items()
            },
            "baseline_iqr": {
                k: [v["q25"], v["q75"]] for k, v in cal["baseline"].items()
            },
            "n_supports": {
                k: d["n_supports"] for k, d in cal["diagnostics"].items()
            },
            "stopped_by": {
                k: d["stopped_by"] for k, d in cal["diagnostics"].items()
            },
            "config": cal["config"],
            "interpolated_between": (sizes if args.calibrate_coarse else None),
        }
        if cal["theta_in"]:
            ti = np.array(list(cal["theta_in"].values()))
            to = np.array(list(cal["theta_out"].values()))
            results["calibration"]["theta_spread"] = {
                "in": float(np.ptp(ti) / np.mean(ti)),
                "out": float(np.ptp(to) / np.mean(to)),
            }
            log(
                f"theta varies across calibrated sizes by "
                f"{100 * np.ptp(ti) / np.mean(ti):.0f}% (in) and "
                f"{100 * np.ptp(to) / np.mean(to):.0f}% (out)"
            )
        self.save()

    # ------------------------------------------------------------------
    # gate, protection and the cohesion scorer
    # ------------------------------------------------------------------

    def make_gate(self):
        """Addition gate and direction protection from the coupling rho."""
        args = self.args
        self.add_gate = None
        if args.add_gate == "ratio" and self.rho_all is not None:
            from .cog2_gate import make_gate

            self.add_gate = make_gate(self.rho_all)
        log(
            f"addition gate: "
            f"{args.add_gate if self.add_gate is not None else 'none'}; "
            f"merge mode: {args.merge_mode}"
        )
        self.protect = None
        if args.stab_protect == "direction" and self.rho_all is not None:
            from .cog2_gate import make_protect

            self.protect = make_protect(
                self.rho_all, alpha=args.stab_protect_alpha
            )

    def make_cohesion_scorer(self):
        """Cohesion scorer (q95 baseline, noise margin) when rho exists.

        It is the objective with ``--grow-objective cohesion``; otherwise it
        is the end-of-run check.
        """
        args, results = self.args, self.results
        wants_cohesion = args.grow_objective == "cohesion" or (
            args.stability and args.stab_objective == "cohesion"
        )
        if wants_cohesion and self.rho_all is None:
            raise SystemExit(
                "cohesion objective needs the predictive graph "
                "(--search-graph predictive)"
            )
        self.coh = None
        if self.rho_all is not None:
            from .cog2_cohesion import build_cohesion_scorer

            t = time.perf_counter()
            margin = (
                args.cohesion_margin
                if str(args.cohesion_margin) == "auto"
                else float(args.cohesion_margin)
            )
            self.coh, report = build_cohesion_scorer(
                self.ws,
                self.rho_all,
                self.base_adj,
                max_size=max(args.max_unit, args.max_size),
                margin=margin,
                n_boot=args.cohesion_boot,
                min_blocks=args.cohesion_blocks,
                n_baseline=args.cohesion_baseline_n,
                min_size=args.min_size,
                seed=args.seed + 31,
                log=log,
            )
            results["cohesion_calibration"] = {
                k_: (
                    {str(kk): vv for kk, vv in v_.items()}
                    if isinstance(v_, dict)
                    else v_
                )
                for k_, v_ in report.items()
            }
            self.timings["cohesion_calibration_seconds"] = (
                time.perf_counter() - t
            )
            self.save()
        self.grow_coh = (
            self.coh is not None and args.grow_objective == "cohesion"
        )
        log(f"growth objective: {args.grow_objective}")

    # ------------------------------------------------------------------
    # 4. discovery
    # ------------------------------------------------------------------

    def discover(self):
        """Stage 4: seeds -> growth -> merge -> reassignment, then split."""
        args, results, timings = self.args, self.results, self.timings
        coh, grow = self.coh, self.grow_coh
        probe = self.engine_probe
        self.scorer = scorer = Cog2Scorer(
            self.ws,
            unit_dim=args.unit_dim,
            env_dim=args.env_dim,
            **cond_kwargs(args),
            calibration=self.cal,
            backend=args.backend,
            schur=self.schur,
            combine=args.combine,
            engine=probe.engine,
            env_basis=probe.env_basis,
        )
        t = time.perf_counter()
        dec = decompose(
            self.f_disc,
            self.search_adj,
            self.ws,
            unit_dim=args.unit_dim,
            env_dim=args.env_dim,
            n_seeds=args.dec_seeds,
            max_size=args.max_size,
            min_size=args.min_size,
            max_unit=args.max_unit,
            boundary_tol=args.boundary_tol,
            merge_jaccard=args.merge_jaccard,
            size_penalty=args.size_penalty,
            max_iter=args.dec_iters,
            n_controls=args.n_controls,
            seed=args.seed,
            on_log=log,
            scorer=(coh if grow else scorer),
            cost_fn=(coh.cost_fn if grow else Cog2Scorer.cost_fn),
            keep_best=not args.no_keep_best,
            batch_scorer=(
                coh.score_batch
                if grow
                else scorer.score_batch
                if scorer.engine is not None
                else None
            ),
            knee_tol=(coh.margin if grow else 0.15),
            seed_residues=self.seed_residues(),
            merge_mode=args.merge_mode,
            merge_gate=self.add_gate,
            merge_margin=(coh.margin if grow else args.stab_margin),
        )
        timings["discovery_seconds"] = time.perf_counter() - t
        self.dec = dec
        units = dec["units"]
        try:
            results["random_region_rho"] = random_region_rho(
                scorer, self.base_adj, [5, 10, 20], seed=args.seed + 5
            )
            rr = results["random_region_rho"]
            log(
                "random regions, channel strength rho_1..rho_8 (median, "
                "in) vs threshold: "
                + "; ".join(
                    f"size {k}: "
                    + " ".join(f"{x:.3f}" for x in v["in"]["median"] if x == x)
                    + f" | theta {v['in']['theta']:.3f}"
                    for k, v in rr.items()
                )
            )
        except Exception as exc:  # reporting only
            log(f"random-region rho not recorded: {exc}")
        self.ledger.snap("growth", units, scorer)
        if not args.no_cohesion_split:
            units, events = split_incoherent(
                scorer, units, min_size=args.min_size, on_log=log
            )
            dec["units_before_split"] = dec["units"]
            dec["units"] = units
            dec["cohesion_splits"] = events
        results["discovery"] = {
            k: dec[k]
            for k in ("units_before_split", "cohesion_splits")
            if k in dec
        }
        results["discovery"].update(
            {
                k: dec[k]
                for k in (
                    "units",
                    "scores",
                    "gap_vs_best_control",
                    "gap_vs_random",
                    "boundary_residues",
                    "boundary_detail",
                    "orphans",
                    "coverage",
                    "n_units",
                    "trace",
                    "score_cache",
                    "flat_curve_seeds",
                    "n_seeds",
                    "membership",
                    "config",
                )
            }
        )
        log(
            f"discovery: {len(units)} units, sizes "
            f"{[len(u) for u in units]} "
            f"({timings['discovery_seconds']:.1f}s)"
        )
        truth = self._truth()
        if truth is not None:
            from .benchmark import face_recovery, match_units

            results["discovery"]["vs_truth"] = match_units(
                units, truth["units"]
            )
            if "units_before_split" in dec:
                results["discovery"]["vs_truth_before_split"] = match_units(
                    dec["units_before_split"], truth["units"]
                )
            results["discovery"]["faces"] = face_recovery(
                dec["boundary_residues"], truth
            )
            log(f"vs truth: {results['discovery']['vs_truth']}")
        self.save()
        self.ledger.snap("cohesion_split", units, scorer)
        self.units = units
        self.scorer_eff = scorer
        self.face_list = []

    def find_faces(self):
        """Stage 4b: cores and faces (``--faces``)."""
        args, results, timings = self.args, self.results, self.timings
        if not (args.faces and self.rho_all is not None):
            return
        from .cog2_faces import MaskedScorer, find_faces

        t = time.perf_counter()
        fr = find_faces(
            self.scorer,
            self.units,
            self.rho_all,
            self.theta_pair,
            delta_min=args.face_delta,
            balance_min=args.face_balance,
            min_size=args.min_size,
            log=log,
        )
        timings["faces_seconds"] = time.perf_counter() - t
        keep_idx = [i for i, u in enumerate(fr["cores"]) if len(u)]
        for f in fr["faces"]:
            f["units"] = [
                keep_idx.index(b) for b in f["units"] if b in keep_idx
            ]
            f["home"] = (
                keep_idx.index(f["home"]) if f["home"] in keep_idx else -1
            )
        fr["cores"] = [fr["cores"][i] for i in keep_idx]
        face_list = fr["faces"]
        self.face_list = face_list
        self.units = [list(u) for u in fr["cores"]]
        self.scorer_eff = MaskedScorer(self.scorer, fr["face_set"])
        results["faces"] = {k: v for k, v in fr.items() if k != "cores"}
        with_faces = [
            sorted(
                set(u) | {f["residue"] for f in face_list if i in f["units"]}
            )
            for i, u in enumerate(fr["cores"])
        ]
        results["faces"]["units_with_faces"] = with_faces
        log(
            f"cores and faces: {len(face_list)} face residues "
            f"{[f['residue'] for f in face_list]}; cores scored with faces "
            f"removed from both sides ({timings['faces_seconds']:.0f}s)"
        )
        truth = self._truth()
        if truth is not None:
            from .benchmark import match_units

            true_faces = set(truth.get("face_residues", []))
            got = set(fr["face_set"])
            results["faces"]["vs_truth"] = match_units(
                with_faces, truth["units"]
            )
            results["faces"]["true_faces"] = len(true_faces)
            results["faces"]["faces_found"] = len(true_faces & got)
            results["faces"]["false_faces"] = len(got - true_faces)
            log(
                f"faces vs truth: found {len(true_faces & got)}/"
                f"{len(true_faces)}, false {len(got - true_faces)}; units "
                f"(core + faces) vs truth {results['faces']['vs_truth']}"
            )
        self.save()
        self.ledger.snap("faces", self.units, self.scorer_eff, face_list)

    # ------------------------------------------------------------------
    # 5. status, cohesion and round 2
    # ------------------------------------------------------------------

    def report_units(self):
        """Stage 5: closure status and cohesion check of every unit."""
        args, results, timings = self.args, self.results, self.timings
        units, face_list, coh = self.units, self.face_list, self.coh
        t = time.perf_counter()
        results["units_discovery"] = unit_report(
            self.scorer_eff, units, cohesion=True
        )
        if face_list:
            for e in results["units_discovery"]:
                i = e["unit"]
                e["faces"] = [
                    f["residue"] for f in face_list if i in f["units"]
                ]
                e["unit_residues"] = sorted(
                    set(e.get("residues", [])) | set(e["faces"])
                )
        timings["cohesion_seconds"] = time.perf_counter() - t
        truth = self._truth()
        if truth is not None:
            results["discovery"]["channels_vs_truth"] = channels_vs_truth(
                results["units_discovery"], truth
            )
            cv = results["discovery"]["channels_vs_truth"]
            log(
                f"channels vs planted gears: mean error "
                f"{cv['mean_error']:+.2f}, mean |error| "
                f"{cv['mean_abs_error']:.2f}, exact after rounding "
                f"{cv['exact_after_rounding']:.0%} (faces: {cv['n_faces']})"
            )
        for e in results["units_discovery"]:
            if "cohesion" in e:
                log(
                    f"unit {e['unit']} (size {e['size']}): nu "
                    f"{e['nu']:.2f} (in {e['nu_in']:.2f}, out "
                    f"{e['nu_out']:.2f}) cost {e['cost']:+.2f} p_size "
                    f"{e['p_size']:.3f}  cohesion "
                    f"{e['cohesion']['verdict']}  -> {e.get('status')}"
                )
        st = [
            e.get("status", "unresolved") for e in results["units_discovery"]
        ]
        results["discovery"]["status_counts"] = {
            k: st.count(k) for k in ("closed", "weak", "unresolved")
        }
        results["discovery"]["unresolved_residues"] = sorted(
            int(x)
            for e in results["units_discovery"]
            if e.get("status") == "unresolved"
            for x in e.get("residues", [])
        )
        log(
            f"units: {st.count('closed')} closed, {st.count('weak')} weak; "
            f"{st.count('unresolved')} groups with cost > 0 are NOT units "
            f"({len(results['discovery']['unresolved_residues'])} residues "
            "unresolved)"
        )
        if self.grow_coh:
            # with cohesion growth, a group is a unit if it is more cohesive
            # than 95% of random sets of its size; the others and every
            # residue in no group go to round 2
            sig = [coh.significant(u) for u in units]
            results["discovery"]["cohesion"] = [
                {
                    "unit": i,
                    "size": len(u),
                    **{
                        k: round(float(v), 4)
                        for k, v in coh.cost_of(u)[0].items()
                    },
                    "q95_random": round(coh.q95(len(u)), 4),
                    "significant": sg,
                }
                for i, (u, sg) in enumerate(zip(units, sig))
            ]
            in_units = set(int(x) for u in units for x in u)
            results["discovery"]["unresolved_residues"] = sorted(
                set(int(x) for u, sg in zip(units, sig) if not sg for x in u)
                | (set(range(self.n_res)) - in_units)
            )
            log(
                f"cohesion of the discovered groups: "
                f"{sum(sig)} of {len(units)} more cohesive"
                f" than 95% of random regions of their size; "
                f"{len(results['discovery']['unresolved_residues'])} "
                "residues to round 2 (in a non-cohesive group or in no "
                "group)"
            )
        if self.grow_coh and args.channel_shadow in ("on", "auto"):
            ChannelShadow(self).execute()
        self.save()
        self.ledger.snap("status", units, self.scorer_eff, face_list)

    def _round2(self):
        """Run `resolve_unresolved` on the current units and pool."""
        from .cog2_resolve import resolve_unresolved

        args, coh, grow = self.args, self.coh, self.grow_coh
        return resolve_unresolved(
            coh if grow else self.scorer_eff,
            self.units,
            self.search_adj,
            assigned=[f["residue"] for f in self.face_list],
            max_unit=min(args.max_unit, self.cal_max),
            min_size=args.min_size,
            margin=(coh.margin if grow else args.resolve_margin),
            is_unit=(coh.significant if grow else None),
            n_seeds=args.resolve_seeds,
            log=log,
            gate=self.add_gate,
        )

    def resolve_round(self):
        """Stage 5b: second round for the unresolved residues."""
        args, results, timings = self.args, self.results, self.timings
        if not (args.resolve and results["discovery"]["unresolved_residues"]):
            return
        t = time.perf_counter()
        res = self._round2()
        timings["resolve_seconds"] = time.perf_counter() - t
        results["resolution"] = {k: v for k, v in res.items() if k != "units"}
        truth = self._truth()
        if truth is not None:
            from .benchmark import match_units

            results["resolution"]["vs_truth"] = match_units(
                res["units"], truth["units"]
            )
            log(f"after round 2 vs truth: {results['resolution']['vs_truth']}")
        self.units = res["units"]
        results["units_resolved"] = unit_report(
            self.scorer_eff, self.units, cohesion=True
        )
        st = [e.get("status") for e in results["units_resolved"]]
        log(
            f"after round 2: {st.count('closed')} closed, "
            f"{st.count('weak')} weak, {st.count('unresolved')} unresolved "
            f"groups; {len(res['secondary'])} residues secondary, "
            f"{len(res['isolated'])} isolated "
            f"({timings['resolve_seconds']:.0f}s)"
        )
        self.save()

    def finish_ledger(self):
        """Record the final phase, coverage and the secondary labels."""
        self.ledger.snap("final", self.units, self.scorer_eff, self.face_list)
        if self.ledger.rows:
            self.ledger.update_coverage("at the end")
            self.save()
        self.ledger.label_secondary("final", self.units)
        self.save()

    # ------------------------------------------------------------------
    # 5c. Monte Carlo stable minimum, write-back and its tables
    # ------------------------------------------------------------------

    def _units_from_file(self):
        """Units (and statuses) of an earlier run given by --units-from."""
        args = self.args
        with open(args.units_from) as fh:
            prior = json.load(fh)
        key = (
            "units_resolved"
            if "units_resolved" in prior
            else "units_discovery"
        )
        units = [list(map(int, e["residues"])) for e in prior[key]]
        status = [e.get("status", "") for e in prior[key]]
        return units, status, f"{args.units_from} ({key})"

    def _stab_report(self, units):
        """Report the units as ``units_stable`` and log their status."""
        self.results["units_stable"] = unit_report(
            self.scorer_eff, units, cohesion=True
        )
        status = [e.get("status") for e in self.results["units_stable"]]
        self.writer.units_stable(self.results["units_stable"])
        return status

    def stabilise(self):
        """Stage 5c: stable minimum per unit, write-back, round 2, tables."""
        args, results, timings = self.args, self.results, self.timings
        t = time.perf_counter()
        self._stab_descent()
        self._stab_write_back()
        if args.resolve and results["stability"]["write_back"]["pool"]:
            self._stab_round2()
        if self.rho_all is not None:
            self._stab_cohesion_table()
            self._stab_directions()
        truth = self._truth()
        if truth is not None:
            from .benchmark import match_units

            results["stability"]["vs_truth"] = match_units(
                self.units, truth["units"]
            )
            log(
                f"after MC write-back vs truth: "
                f"{results['stability']['vs_truth']}"
            )
        self.ledger.snap("stable", self.units, self.scorer_eff, self.face_list)
        if self.ledger.rows:
            self.ledger.update_coverage("after write-back")
        self.ledger.label_secondary("stable", self.units)
        self.writer.stability_trace(results["stability"]["units"])
        timings["stability_seconds"] = time.perf_counter() - t
        self.save()

    def _stab_descent(self):
        """Monte Carlo descent from every unit to its stable minimum."""
        from .cog2_stability import run_stability

        args, results = self.args, self.results
        st_units, st_src = self.units, "this run"
        if args.units_from:
            st_units, _, st_src = self._units_from_file()
        log(f"stability: {len(st_units)} units from {st_src}")
        mc_scorer, mc_margin = self.scorer_eff, args.stab_margin
        if args.stab_objective == "cohesion":
            if self.rho_all is None:
                raise SystemExit(
                    "--stab-objective cohesion needs the "
                    "predictive graph (rho)"
                )
            mc_scorer = self.coh
            mc_margin = self.coh.margin
        log(
            f"stability objective: {args.stab_objective} (margin "
            f"{'noise table' if callable(mc_margin) else mc_margin})"
        )
        self.mc_scorer = mc_scorer
        stab = run_stability(
            mc_scorer,
            st_units,
            self.base_adj,
            n_cycles=args.stab_cycles,
            patience=args.stab_patience,
            kmax=args.stab_kmax,
            margin=mc_margin,
            min_size=args.min_size,
            max_unit=min(args.max_unit, self.cal_max),
            max_steps=args.stab_steps,
            n_null=args.stab_null,
            seed=args.seed + 11,
            log=log,
            gate=self.add_gate,
            protect=self.protect,
        )
        results["stability"] = stab
        stab["units_source"] = st_src
        stab["objective"] = args.stab_objective
        mc_faces = [
            {
                "unit": e["unit"],
                "residue": int(r),
                "dJ_if_removed": d,
                "direction": e["direction"].get(int(r)),
            }
            for e in stab["units"]
            for r, d in (e.get("faces") or {}).items()
        ]
        stab["faces"] = mc_faces
        log(
            f"stability: {len(mc_faces)} faces (members kept by dynamics "
            f"whose removal would lower J): "
            f"{[f['residue'] for f in mc_faces]}"
        )
        truth = self._truth()
        if truth is not None:
            tf = set(truth.get("face_residues", []))
            got = {f["residue"] for f in mc_faces}
            stab["faces_vs_truth"] = {
                "true_faces": len(tf),
                "found": len(tf & got),
                "false": len(got - tf),
            }
            log(f"MC faces vs truth: {stab['faces_vs_truth']}")

    def _stab_write_back(self):
        """Make the minima the units (one place per residue)."""
        from .cog2_stability import _Memo, reconcile

        args, results = self.args, self.results
        stab = results["stability"]
        stable, wb = reconcile(
            _Memo(self.mc_scorer),
            [e["minimum"] for e in stab["units"]],
            self.n_res,
            min_size=args.min_size,
            log=log,
        )
        stab["write_back"] = wb
        self.units = stable
        st = self._stab_report(self.units)
        log(
            f"after MC write-back: {len(self.units)} units "
            f"({st.count('closed')} closed, {st.count('weak')} weak, "
            f"{st.count('unresolved')} unresolved); "
            f"{len(wb['pool'])} residues in no unit"
        )

    def _stab_round2(self):
        """Round 2 for the residues MC removed (and any in no unit)."""
        results = self.results
        res2 = self._round2()
        results["stability"]["round2"] = {
            k: v for k, v in res2.items() if k != "units"
        }
        self.units = res2["units"]
        st = self._stab_report(self.units)
        log(
            f"after MC + round 2: {len(self.units)} units "
            f"({st.count('closed')} closed, {st.count('weak')} weak, "
            f"{st.count('unresolved')} unresolved); "
            f"{len(res2['secondary'])} secondary, "
            f"{len(res2['isolated'])} isolated"
        )

    def _stab_cohesion_table(self):
        """Predictive cohesion of the final units (the PDU definition).

        p is against random sets and against random sets put through the
        same MC descent; closure is the channel cost (``units_stable``).
        """
        from .cog2_cohesion import CohesionScorer, unit_table
        from .cog2_stability import _Memo, descended_null

        args, results, units = self.args, self.results, self.units
        table = unit_table(self.rho_all, units, seed=args.seed)
        scorer = (
            self.coh
            if self.coh is not None
            else CohesionScorer(self.rho_all, margin=0.002)
        )
        memo, nulls = _Memo(scorer), {}
        channel = {e["unit"]: e for e in results["units_stable"]}
        for row in table:
            k = row["size"]
            if args.stab_null and k not in nulls:
                nulls[k] = descended_null(
                    memo,
                    self.base_adj,
                    k,
                    self.n_res,
                    args.stab_null,
                    margin=scorer.margin,
                    min_size=args.min_size,
                    max_unit=min(args.max_unit, self.cal_max),
                    max_steps=args.stab_steps,
                    rng=np.random.default_rng(args.seed + 23 + k),
                    gate=self.add_gate,
                    protect=self.protect,
                )
            # costs (q95(size) - cohesion): lower = more cohesive than
            # random regions of its size
            null = list(nulls.get(k, []))
            ucost = scorer.cost_of(units[row["unit"]])[1]  # -cohesion
            row["q95_random_regions"] = (
                round(scorer.q95(k), 4) if scorer.baseline else None
            )
            row["p_descended"] = (
                round((1 + sum(x <= ucost for x in null)) / (len(null) + 1), 4)
                if null
                else None
            )
            row["descended_null_costs"] = [round(x, 4) for x in null]
            e = channel.get(row["unit"], {})
            row["channel_cost"] = e.get("cost")
            row["closure"] = e.get("status")
            p_use = row["p_descended"] if null else row["p_random"]
            cohesive = p_use is not None and p_use <= 0.1
            closed = e.get("status") == "closed"
            row["category"] = (
                "PDU"
                if cohesive and closed
                else "cohesive, wide interface"
                if cohesive
                else "closed, not cohesive"
                if closed
                else "not a unit"
            )
        results["cohesion"] = table
        cats = {}
        for row in table:
            cats[row["category"]] = cats.get(row["category"], 0) + 1
        results["cohesion_categories"] = cats
        log(f"predictive cohesion of the final units: {cats}")
        for row in table:
            log(
                f"  unit {row['unit']} size {row['size']}: AUC driver "
                f"{row['auc_driver']:.3f} listener "
                f"{row['auc_listener']:.3f} (random median "
                f"{row['random_median']:.3f}; p random "
                f"{row['p_random']}, p descended {row['p_descended']});"
                f" median link rho {row.get('rho_inside_median')} "
                f"(inside / boundary {row.get('inside_over_boundary')});"
                f" channel cost {row['channel_cost']} -> "
                f"{row['category']}"
            )
        self.writer.unit_cohesion(table)

    def _stab_directions(self):
        """Driver / listener label of every member of the final units."""
        from .cog2_gate import direction_table

        args, results = self.args, self.results
        table = direction_table(
            self.rho_all, self.units, alpha=args.stab_protect_alpha
        )
        # MC faces are indexed by the pre-write-back unit; mark by residue
        face_res = {
            f["residue"] for f in results["stability"].get("faces", [])
        }
        for row in table:
            row["mc_face"] = int(row["residue"] in face_res)
        results["stability"]["directions"] = table
        counts = {}
        for row in table:
            counts[row["label"]] = counts.get(row["label"], 0) + 1
        results["stability"]["direction_counts"] = counts
        log(f"member directions in the final units: {counts}")
        self.writer.unit_directions(table)

    # ------------------------------------------------------------------
    # 5d. the PDU definition (scoring only)
    # ------------------------------------------------------------------

    def score_pdudef(self):
        """Stage 5d: score units with the linear-estimator PDU definition."""
        from .cog2_pdudef import run_pdudef

        args, results = self.args, self.results
        t = time.perf_counter()
        units, src, status = self.units, "this run", None
        if args.units_from:
            units, status, src = self._units_from_file()
        else:
            prior = (
                results.get("units_resolved")
                or results.get("units_discovery")
                or []
            )
            if len(prior) == len(units):
                status = [e.get("status", "") for e in prior]
        log(f"PDU definition: scoring {len(units)} units from {src}")
        results["pdu_definition"] = run_pdudef(
            self.ws,
            units,
            self.base_adj,
            self.f_disc,
            n_random=args.pdudef_random,
            n_shifts=args.pdudef_shifts,
            n_frames=args.pdudef_frames,
            seed=args.seed + 13,
            statuses=status,
            log=log,
        )
        results["pdu_definition"]["units_source"] = src
        self.timings["pdudef_seconds"] = time.perf_counter() - t
        self.save()

    # ------------------------------------------------------------------
    # 6-8. verification, comparison and outputs
    # ------------------------------------------------------------------

    def verification(self):
        """Stage 6: block, replica, rho-scan, held-out and lag checks."""
        args, results = self.args, self.results
        if args.block_check:
            BlockCheck(self).execute()
        if args.per_replica_check:
            ReplicaCheck(self).execute()
        if args.rho_scan:
            RhoScan(self).execute()
        if self.hold:
            self.lag_data = HoldoutCheck(self).execute()
        else:
            self.lag_data = (self.f_disc, self.s_disc, self.p_disc)
        if args.verify_lags:
            t = time.perf_counter()
            feats, segs, pair = self.lag_data
            results["lag_scan"] = LagScan(args).run(
                feats, segs, self.units, self.base_adj, pair, highpass=self.hp
            )
            results["lag_scan_data"] = "holdout" if self.hold else "discovery"
            self.timings["lag_scan_seconds"] = time.perf_counter() - t
            self.save()

    def compare_cog1(self):
        """Stage 8: COG v1.0 on the same discovery data."""
        Cog1Comparison(self).execute()

    def write_outputs(self):
        """Write the CSV tables and the final ``results.json``."""
        self.writer.final_tables(self.results, self.ledger)
        self.results["timings"] = self.timings
        self.save()
        log(f"done; outputs in {self.out}")
