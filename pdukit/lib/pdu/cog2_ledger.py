"""Per-residue ledger: the state of every residue after each phase.

The ledger is recording only; it never changes the units.  It also holds the
secondary labelling of residues that belong to no unit.
"""

from __future__ import annotations

import time

from .cog2_faces import leave_one_out, ledger_rows, ledger_summary
from .cog2_gate import ratio
from .cog2_scoring import unit_rho_rows, unit_status
from .null_calibration_run import log


class ResidueLedger:
    """Rows of per-residue records, plus channel-strength logs per phase."""

    def __init__(self, run):
        """Attach to the running pipeline (which owns the results)."""
        self.run = run
        self.rows = []
        self.rho_log = []

    def snap(self, phase, units, scorer, faces=None):
        """Record every residue's state after a phase (recording only)."""
        run = self.run
        args = run.args
        if not args.ledger or run.rho_all is None:
            return
        t0 = time.perf_counter()
        unit_lists = [sorted(int(x) for x in g) for g in units if len(g)]
        scored = scorer.score_batch(unit_lists) if unit_lists else []
        statuses = [unit_status(r["cost"], r["p_size"]) for r in scored]
        self.rho_log.extend(unit_rho_rows(phase, scored, unit_lists))
        loo = (
            leave_one_out(
                scorer, unit_lists, min_size=args.min_size, key="cost"
            )
            if args.ledger_loo
            else None
        )
        self.rows.extend(
            ledger_rows(
                phase,
                run.n_res,
                unit_lists,
                statuses,
                run.rho_all,
                run.theta_pair,
                loo=loo,
                faces=faces,
            )
        )
        run.timings["ledger_seconds"] = (
            run.timings.get("ledger_seconds", 0.0) + time.perf_counter() - t0
        )

    def update_coverage(self, when):
        """Summarise the ledger as core / face / unresolved / unassigned."""
        results = self.run.results
        n_res = self.run.n_res
        results["ledger_summary"] = ledger_summary(self.rows, n_res)
        labels = [x["final_label"] for x in results["ledger_summary"]]
        results["coverage_labels"] = {
            k: labels.count(k)
            for k in ("core", "face", "unresolved", "unassigned")
        }
        results["coverage_labels"]["core_or_face_fraction"] = round(
            (labels.count("core") + labels.count("face")) / max(n_res, 1), 3
        )
        log(f"residue labels {when}: {results['coverage_labels']}")

    def label_secondary(self, phase, units):
        """Label every residue that is in no unit as secondary or isolated.

        The units a residue contributes to are those it predicts better than
        the rest of the protein (the addition gate), with the coupling ratio,
        its share and the cost change if it joined.  "isolated" only if it
        contributes to no unit.  Updates the ledger rows of `phase`, the
        coverage labels and ``results["secondary"]``
        (secondary_assignments.csv).
        """
        run = self.run
        rho_all = run.rho_all
        if rho_all is None:
            return
        results = run.results
        n_res = run.n_res
        scorer_eff = run.scorer_eff
        in_units = set(int(x) for u in units for x in u)
        free = [r for r in range(n_res) if r not in in_units]
        rows, contrib = [], {}
        for r in free:
            ratios = [
                (ratio(rho_all, r, u), k)
                for k, u in enumerate(units)
                if len(u)
            ]
            ok = sorted([(v, k) for v, k in ratios if v > 1.0], reverse=True)
            if ok:
                base = scorer_eff.score_batch(
                    [sorted(units[k]) for _, k in ok]
                )
                join = scorer_eff.score_batch(
                    [sorted(set(units[k]) | {r}) for _, k in ok]
                )
                dc = [
                    float(j["cost"]) - float(b["cost"])
                    for j, b in zip(join, base)
                ]
            else:
                dc = []
            tot = sum(v for v, _ in ok) or 1.0
            contrib[r] = [k for _, k in ok]
            rows.append(
                {
                    "residue": r,
                    "index": r,
                    "label": "secondary" if ok else "isolated",
                    "units": " ".join(str(k) for _, k in ok),
                    "ratios": " ".join(f"{v:.2f}" for v, _ in ok),
                    "shares": " ".join(f"{v / tot:.2f}" for v, _ in ok),
                    "cost_change_if_joined": " ".join(f"{d:+.3f}" for d in dc),
                    "best_unit": ok[0][1] if ok else -1,
                }
            )
        results["secondary"] = rows
        for e in self.rows:
            if e.get("phase") == phase and e["index"] in contrib:
                e["label"] = "secondary" if contrib[e["index"]] else "isolated"
                e["face_units"] = " ".join(map(str, contrib[e["index"]]))
        if self.rows:
            results["ledger_summary"] = ledger_summary(self.rows, n_res)
            labels = [x["final_label"] for x in results["ledger_summary"]]
            cl = {
                k: labels.count(k)
                for k in (
                    "core",
                    "face",
                    "secondary",
                    "unresolved",
                    "isolated",
                    "unassigned",
                )
            }
            cl["core_or_face_fraction"] = round(
                (cl["core"] + cl["face"]) / max(n_res, 1), 3
            )
            cl["assigned_fraction"] = round(
                (cl["core"] + cl["face"] + cl["secondary"]) / max(n_res, 1), 3
            )
            results["coverage_labels"] = cl
        n_sec = sum(x["label"] == "secondary" for x in rows)
        multi = sum(len(x["units"].split()) > 1 for x in rows)
        log(
            f"secondary residues ({phase}): {n_sec} contribute to at least "
            f"one unit ({multi} to two or more), "
            f"{len(rows) - n_sec} isolated; labels "
            f"{results.get('coverage_labels')}"
        )
