"""Stages 8-9: discovery of the units, and global-mode recovery."""

from __future__ import annotations

import json
import time

import numpy as np

from pdukit.cog.scoring.factory import cond_kwargs
from pdukit.cog.scoring.scorer import FluxScorer
from pdukit.runlog import log, set_phase


class DiscoverUnitsStage:
    """Strict incoming-only growth, saved before legacy post-processing."""

    code = "GRO"

    def run(self, state):
        """Advance ``state`` by this stage."""
        from pdukit.cog.units.grow import grow_incoming
        from pdukit.cog.units.seeds import _farthest_point_seeds, seed_groups

        args, probe = state.args, state.engine_probe
        scorer = FluxScorer(
            state.ws,
            unit_dim=args.unit_dim,
            env_dim=args.env_dim,
            **cond_kwargs(args),
            calibration=state.cal,
            backend=args.backend,
            schur=state.schur,
            combine=args.combine,
            engine=probe.engine,
            env_basis=probe.env_basis,
        )
        if args.incoming_starts:
            with open(args.incoming_starts) as stream:
                starts = json.load(stream)
        else:
            seeds = self._seed_residues(state)
            if seeds is None:
                seeds = _farthest_point_seeds(
                    state.search_adj,
                    self._n_seeds(state),
                    np.random.default_rng(args.seed),
                )
            starts = [[int(seed)] for seed in seeds]
            size = self._start_size(state)
            if size > 1:
                starts = seed_groups(
                    [s[0] for s in starts], state.rho_all, state.search_adj,
                    size,
                )
                log(
                    f"growth starts: each seed plus its {size - 1} strongest "
                    f"predictive partner(s); {len(starts)} distinct starts"
                )
            state.results["growth_starts"] = {
                "seed_group": int(args.seed_group),
                "size": int(size),
                "starts": starts,
            }
        start = time.perf_counter()
        result = grow_incoming(
            scorer,
            state.search_adj,
            starts,
            max_size=min(args.max_size, args.max_unit, state.n_res - 1),
            min_size=args.min_size,
            min_gain=args.incoming_min_gain,
            start_min_gain=args.start_min_gain,
            on_log=log,
        )
        state.timings["incoming_growth_seconds"] = time.perf_counter() - start
        state.results["mode"] = "incoming-only growth"
        state.results["incoming_growth"] = result
        state.results["units_discovery"] = result["units"]
        state.results["postprocessing"] = {
            "executed": False,
            "mode": args.incoming_postprocess,
            "reason": "growth complete; selected postprocessing has not run",
        }
        state.units = [unit["residues"] for unit in result["units"]]
        # Monitor collectivity without changing the incoming-only growth rule.
        from pdukit.numerics.cohesion import cohesion

        if state.rho_all is None:
            raise ValueError(
                "incoming cohesion monitoring requires pairwise rho"
            )
        for run in result["runs"]:
            for row in run["history"]:
                if len(row["residues"]) >= 2:
                    row.update(cohesion(state.rho_all, row["residues"]))
        for unit in result["units"]:
            unit.update(cohesion(state.rho_all, unit["residues"]))
        state.writer.write_incoming_tables(result)
        # Save growth before post-processing: partial jobs stay inspectable.
        state.save()
        if args.incoming_postprocess == "split":
            self.split_incoming(state, scorer, result)
            set_phase("GRO")
        state.save()
        log(
            f"done: {len(result['runs'])} starts, {len(result['units'])} "
            f"candidates (>= {args.min_size} residues)"
        )

    def split_incoming(self, state, scorer, growth):
        """Find explicit cohesion boundaries after incoming-only growth."""
        from pdukit.cog.units.cohesion import block_bootstrap_rho
        from pdukit.cog.units.split import merge_overlapping, split_candidates

        args = state.args
        start = time.perf_counter()
        set_phase("MRG")
        merge = merge_overlapping(growth["units"], args.merge_overlap, scorer)
        state.results["incoming_merge"] = merge
        state.writer.write_merge(merge)
        parents = [f["residues"] for f in merge["families"]]

        def fmt(v):
            return "n/a" if v is None else f"{v:.3f}"

        for i, c, ov in merge["joins"]:
            log(f"candidate {i} -> family {c}  overlap {ov:.2f}")
        for f in merge["families"]:
            log(
                f"family {f['family']}  members {f['members']}  sizes "
                f"{f['member_sizes']}  union {f['union_size']}  core "
                f"{len(f['core'])}  majority {len(f['majority'])}  -> "
                f"parent {f['size']} ({f['decision']})"
            )
            log.debug(
                f"family {f['family']} nu_in members "
                f"[{', '.join(fmt(v) for v in f['member_nu_in'])}]  union "
                f"{fmt(f['union_nu_in'])}"
            )
        log(
            f"merge overlap>={args.merge_overlap}: "
            f"{merge['n_candidates']} supports -> {merge['n_families']} "
            f"families; parents "
            + ", ".join(
                f"{f['size']} ({f['decision']})" for f in merge["families"]
            )
        )
        set_phase("SPL")
        boot, bootstrap_info = [], {}
        if args.split_test in {"both", "bootstrap"} and growth["units"]:
            boot, bootstrap_info = block_bootstrap_rho(
                state.ws,
                n_boot=args.cohesion_boot,
                min_blocks=args.cohesion_blocks,
                seed=args.seed + 31,
                log=log,
            )
            if len(boot) < 2:
                raise ValueError(
                    "cannot estimate selected-cut bootstrap uncertainty; "
                    "use --split-test graph-null explicitly for "
                    "graph-only analysis"
                )
        result = split_candidates(
            state.rho_all,
            parents,
            scorer,
            method=args.split_method,
            min_size=max(2, args.min_size),
            markov_time=args.split_markov_time,
            test=args.split_test if parents else "none",
            n_null=args.split_null,
            alpha=args.split_alpha,
            bootstrap=boot,
            max_depth=args.split_max_depth,
            seed=args.seed + 51,
            on_log=log,
        )
        result["bootstrap"] = bootstrap_info
        state.results["incoming_split"] = result
        state.results["units_split"] = result["units"]
        state.results["postprocessing"] = {
            "executed": True,
            "mode": "split",
            "method": args.split_method,
            "growth_cohesion_veto": False,
            "mc": False,
            "merging": {
                "executed": True,
                "min_overlap": args.merge_overlap,
                "n_grown": merge["n_candidates"],
                "n_parents": merge["n_families"],
            },
        }
        state.units = [unit["residues"] for unit in result["units"]]
        state.timings["incoming_split_seconds"] = time.perf_counter() - start
        state.writer.write_split(result)
        state.save()
        # cores -> ports and bridges (residues left out of every core)
        from pdukit.cog.units.ports import classify_residues

        set_phase("POR")

        start = time.perf_counter()
        if not boot and state.units:
            # port tests need the bootstrap noise even when the split did
            # not use it
            boot, _ = block_bootstrap_rho(
                state.ws,
                n_boot=args.cohesion_boot,
                min_blocks=args.cohesion_blocks,
                seed=args.seed + 31,
                log=log,
            )
        if boot:
            # saved so that port / cohesion checks can be redone offline
            state.writer.write_bootstrap(boot)
        classes = classify_residues(
            state.rho_all,
            state.units,
            bootstrap=boot,
            z=args.port_z,
            scorer=scorer,
            on_log=log,
        )
        state.results["incoming_ports"] = classes
        state.results["postprocessing"]["ports"] = {
            "executed": True,
            "z": args.port_z,
            "counts": classes["counts"],
            "mode_units": args.mode_units,
        }
        state.writer.write_classes(classes)
        state.timings["incoming_ports_seconds"] = time.perf_counter() - start
        self._membership_records(state, classes, scorer)
        if args.mode_units == "extended":
            state.units = [u["extended"] for u in classes["units"]]
            log("mode recovery on extended units (core + ports)")
        state.save()
        log(
            f"split ({args.split_method}): {len(parents)} parents -> "
            f"{len(state.units)} terminal supports, "
            f"{sum(node['split'] for node in result['nodes'])} cuts"
        )

    def _membership_records(self, state, classes, scorer):
        """Record residue affinities and how block-like the units are."""
        from pdukit.cog.units.affinity import residue_affinity
        from pdukit.numerics.cohesion import blockiness

        args = state.args
        cores = [u["core"] for u in classes["units"]]
        if not cores:
            return
        blk = blockiness(
            state.rho_all, cores, n_random=200,
            rng=np.random.default_rng(args.seed + 71),
        )
        state.results["blockiness"] = blk
        log(
            f"blockiness of {blk['n_units']} cores: within/between coupling "
            f"{blk['ratio']:.2f}, auc {blk['auc']:.3f} (random same-size sets "
            f"{blk['null_auc_mean']:.3f} +- {blk['null_auc_sd']:.3f}, "
            f"z {blk['z']:.1f}, percentile {blk['percentile']:.0f})"
        )
        if not args.residue_affinity:
            state.save()
            return
        start = time.perf_counter()
        rows, summary = residue_affinity(state.rho_all, cores, scorer=scorer)
        state.writer.write_affinity(rows, summary)
        tag = "nu" if rows and "delta_nu_in" in rows[0] else "cohesion"
        eff = [s[f"{tag}_effective_units"] for s in summary
               if f"{tag}_effective_units" in s]
        state.results["residue_affinity"] = {
            "definition": "change when the residue is in rather than out of "
            "each unit core; affinity = -delta_nu_in and delta_cohesion; "
            "effective units = exp(entropy) of the positive affinities",
            "files": ["residue_affinity.csv", "residue_affinity_summary.csv"],
            "summary": summary,
        }
        state.timings["residue_affinity_seconds"] = (
            time.perf_counter() - start
        )
        log(
            f"residue affinity to {len(cores)} units: median effective "
            f"units {np.median(eff):.1f} ({tag}) "
            f"({state.timings['residue_affinity_seconds']:.0f}s)"
        )
        state.save()

    def _n_seeds(self, state):
        """Return dec-seeds, capped at seed-fraction x residues."""
        n = min(int(state.args.dec_seeds), state.n_res)
        frac = float(state.args.seed_fraction)
        if frac > 0:
            n = min(n, max(1, int(np.ceil(frac * state.n_res))))
        return n

    def _start_size(self, state):
        """Residues per growth start (``seed-group``; 0 = fill unit-dim)."""
        group = int(state.args.seed_group)
        if group >= 1:
            return group
        per_residue = int(state.feats.shape[2])
        return int(-(-int(state.args.unit_dim) // max(per_residue, 1)))

    def _seed_residues(self, state):
        """Residues that seed the growth (None = spread over the graph)."""
        args = state.args
        if args.seeding == "spread":
            return None
        return (
            np.random.default_rng(args.seed)
            .choice(
                state.n_res, self._n_seeds(state), replace=False
            )
            .tolist()
        )


class ReportModesStage:
    """Report local-basis combinations with channels off/on/rank limited."""

    code = "MOD"

    def run(self, state):
        """Advance ``state`` by this stage."""
        from pdukit.cog.modes import recover_global_modes

        state.results["units_final"] = [
            dict(unit=i, residues=list(map(int, support)), size=len(support))
            for i, support in enumerate(state.units)
        ]
        if state.args.mode_recovery == "off":
            state.results["global_mode_recovery"] = {"status": "disabled"}
            state.save()
            return
        args = state.args
        subsets = None
        if args.recovery_subsets:
            with open(args.recovery_subsets) as stream:
                subsets = json.load(stream)
        start = time.perf_counter()
        result = recover_global_modes(
            state.ws,
            state.units,
            n_modes=args.recovery_modes,
            local_dim=args.recovery_local_dim or args.unit_dim,
            ranks=args.recovery_channel_ranks,
            policy=args.recovery_combinations,
            max_combinations=args.recovery_max_combinations,
            subsets=subsets,
            frame_ps=args.frame_ps,
        )
        state.results["global_mode_recovery"] = result
        state.timings["global_mode_recovery_seconds"] = (
            time.perf_counter() - start
        )
        state.writer.write_mode_recovery(result)
        state.save()
        log(
            f"global-mode recovery: {len(result['combinations'])} PDU "
            f"combinations, {len(result['modes'])} reference modes; "
            "channels off/on/rank limited"
        )
