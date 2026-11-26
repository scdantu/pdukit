"""Stages 5-7: workspace, search graph and calibration."""

from __future__ import annotations

import time

import numpy as np

from pdukit.cog.calibration.scorer import calibrate_cog2
from pdukit.cog.features import coarse_sizes
from pdukit.cog.runinfo import _parse_sizes
from pdukit.cog.scoring import scorer as flux_scorer
from pdukit.cog.scoring.backend import gpu_note
from pdukit.cog.scoring.factory import (
    build_workspace,
    check_memory,
    cond_kwargs,
    engine_scorer,
)
from pdukit.runlog import log


class BuildWorkspaceStage:
    """Lagged covariances and shift-null pairs on the discovery data."""

    code = "WSP"

    def run(self, state):
        """Advance ``state`` by this stage."""
        args = state.args
        t = time.perf_counter()
        if state.p_disc is None:
            check_memory(args, state.f_disc, args.n_null, "the workspace")
        state.ws = build_workspace(
            args,
            state.f_disc,
            state.s_disc,
            args.lag,
            args.n_null,
            args.seed,
            state.p_disc,
        )
        ws = state.ws
        state.timings["workspace_seconds"] = time.perf_counter() - t
        log(
            f"D={ws.D}  pairs {ws.n}  shifts {len(ws.Cs00)} "
            f"({ws.shift_mode})  backend {ws.backend} on {ws.device}  "
            f"{state.timings['workspace_seconds']:.1f}s"
        )
        note = gpu_note(args.device, ws.device)
        if note:
            log(note)


class BuildGraphStage:
    """Build the neighbourhood graph: dynamical (default) or contacts."""

    code = "GRA"

    def run(self, state):
        """Advance ``state`` by this stage."""
        args, results = state.args, state.results
        ws, adj, n_res = state.ws, state.adj, state.n_res
        dyn_adj = adj
        pg = None
        if args.search_graph != "contact":
            from pdukit.cog.graph import predictive_graph

            t = time.perf_counter()
            pg = predictive_graph(
                ws,
                k=args.graph_k,
                null_shifts=args.graph_null_shifts,
                contact=adj,
                log=log,
            )
            state.timings["predictive_graph_seconds"] = time.perf_counter() - t
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
            results["predictive_graph"]["rho"] = pg["rho"].tolist()
            results["predictive_graph"]["edges"] = [
                [int(i), int(j)]
                for i, j in zip(*np.where(np.triu(dyn_adj, 1)))
            ]
            state.save()
        state.search_adj = dyn_adj
        state.rho_all = state.theta_pair = None
        if args.search_graph != "contact":
            state.rho_all, state.theta_pair = pg["rho"], pg["theta"]
        elif args.ledger or args.faces or args.grow_objective == "incoming":
            from pdukit.cog.graph import predictive_graph

            pg_quiet = predictive_graph(
                ws,
                k=args.graph_k,
                null_shifts=8,
                contact=adj,
                log=lambda *_: None,
            )
            state.rho_all, state.theta_pair = (
                pg_quiet["rho"],
                pg_quiet["theta"],
            )
        # comparison regions for the baseline / per-replica / lag controls
        if args.baseline == "random":
            state.base_adj = ~np.eye(n_res, dtype=bool)  # ball on a complete
        elif args.baseline == "contact":  # graph = a random
            state.base_adj = np.asarray(adj, bool)  # set
        else:
            state.base_adj = dyn_adj
        results["baseline_regions"] = args.baseline
        results["null_device"] = (
            flux_scorer._null_device(ws) or "cpu (numpy loop)"
        )
        log(f"shift null on {results['null_device']}")
        state.schur = None
        if args.backend == "schur":
            from pdukit.cog.scoring.workspace_schur import SchurWorkspace

            state.schur = SchurWorkspace(ws)
            log(f"Schur precompute {state.schur.precompute_seconds:.1f}s")


class CalibrateStage:
    """Thresholds and the size-matched baseline on discovery data."""

    code = "CAL"

    def run(self, state):
        """Advance ``state`` by this stage."""
        args, results, timings = state.args, state.results, state.timings
        n_res = state.n_res
        sizes = (
            _parse_sizes(args.sizes)
            if args.sizes
            else list(range(2, min(args.max_unit, n_res - 1) + 1))
        )
        if args.calibrate_coarse:
            full = sizes
            sizes = coarse_sizes(min(full), max(full), args.calibrate_coarse)
            log(f"sizes {sizes}, interpolating {len(full) - len(sizes)} more")
        t = time.perf_counter()
        probe = engine_scorer(args, state.ws, state.schur)
        state.engine_probe = probe
        log(
            "scoring: "
            + (
                f"batched GPU engine ({probe.env_basis} env basis)"
                if probe.engine is not None
                else "one at a time"
            )
        )
        cal = calibrate_cog2(
            state.ws,
            state.base_adj,
            sizes,
            unit_dim=args.unit_dim,
            **cond_kwargs(args),
            env_dim=args.env_dim,
            backend=args.backend,
            schur=state.schur,
            combine=args.combine,
            target_rel_width=args.target_rel_width,
            batch_supports=args.batch_supports,
            min_supports=args.min_supports,
            max_supports=args.max_supports,
            seed=args.seed,
            on_log=log,
            scorer=probe,
        )
        state.cal = cal
        timings["calibration_seconds"] = time.perf_counter() - t
        state.writer.write_calibration(cal)
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
                f"theta spread over sizes: in "
                f"{100 * np.ptp(ti) / np.mean(ti):.0f}%  out "
                f"{100 * np.ptp(to) / np.mean(to):.0f}%"
            )
        state.save()
