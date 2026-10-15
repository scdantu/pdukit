"""CSV outputs of a COG run.

One method per file, so the pipeline stages never format text themselves.
"""

from __future__ import annotations

from pathlib import Path

from .null_calibration_run import _write_csv


def _join(values):
    """Space-separated string of values."""
    return " ".join(str(x) for x in values)


class OutputWriter:
    """Writes the CSV files of a run into its output directory."""

    def __init__(self, outdir):
        """Remember the output directory."""
        self.out = Path(outdir)

    def units_stable(self, report):
        """Write ``units_stable.csv`` (one row per stable unit)."""
        with open(self.out / "units_stable.csv", "w") as fh:
            fh.write("unit,size,cost,p_size,status,residues\n")
            for e in report:
                fh.write(
                    f"{e['unit']},{e['size']},{e['cost']:.4f},"
                    f"{e['p_size']:.4f},{e.get('status', '')},"
                    + _join(e["residues"])
                    + "\n"
                )

    def unit_cohesion(self, table):
        """Write ``unit_cohesion.csv`` from the cohesion table."""
        keys = [
            "unit",
            "size",
            "auc_driver",
            "auc_listener",
            "cohesion",
            "q95_random_regions",
            "random_median",
            "p_random",
            "p_descended",
            "channel_cost",
            "closure",
            "rho_inside_median",
            "rho_inside_mean",
            "rho_boundary_mean",
            "inside_over_boundary",
            "explained_share_median",
            "category",
        ]
        with open(self.out / "unit_cohesion.csv", "w") as fh:
            fh.write(",".join(keys) + "\n")
            for row in table:
                fh.write(",".join(str(row.get(k)) for k in keys) + "\n")

    def unit_directions(self, table):
        """Write ``unit_directions.csv`` (driver / listener labels)."""
        keys = (
            "unit",
            "residue",
            "label",
            "p_driver",
            "p_listener",
            "ratio_driver",
            "ratio_listener",
            "mc_face",
        )
        with open(self.out / "unit_directions.csv", "w") as fh:
            fh.write(",".join(keys) + "\n")
            for row in table:
                fh.write(",".join(str(row[k]) for k in keys) + "\n")

    def stability_trace(self, unit_entries):
        """Write ``stability_trace.csv`` (every Monte Carlo cycle)."""
        with open(self.out / "stability_trace.csv", "w") as fh:
            fh.write(
                "unit,cycle,k,cost_kicked,cost_end,dJ_vs_minimum,"
                "descent_moves,size_end,outcome,minimum_cost\n"
            )
            for e in unit_entries:
                for t_ in e["trace"]:
                    fh.write(
                        ",".join(
                            str(x)
                            for x in (
                                e["unit"],
                                t_["cycle"],
                                t_["k"],
                                t_["cost_kicked"],
                                t_["cost_end"],
                                t_["dJ_vs_minimum"],
                                t_["descent_moves"],
                                t_["size_end"],
                                t_["outcome"],
                                t_["minimum_cost"],
                            )
                        )
                        + "\n"
                    )

    def channel_shadow(self, shadow):
        """Write ``channel_shadow.csv`` (report-only channel groups)."""
        keys = (
            "class",
            "size",
            "channel_cost",
            "p_size",
            "closure",
            "auc_driver",
            "auc_listener",
            "cohesion",
            "q95_random",
            "best_cohesion_unit",
            "jaccard",
        )
        with open(self.out / "channel_shadow.csv", "w") as fh:
            fh.write(
                "class,size,channel_cost,p_size,closure,auc_driver,"
                "auc_listener,cohesion,q95_random,best_cohesion_unit,"
                "jaccard,residues\n"
            )
            for e in shadow:
                fh.write(
                    ",".join(str(e.get(k)) for k in keys)
                    + ","
                    + _join(e["residues"])
                    + "\n"
                )

    def final_tables(self, results, ledger):
        """Write units, ledger, channel-strength and secondary tables."""
        rows = []
        hold_by = {e["unit"]: e for e in results.get("units_holdout", [])}
        for e in results["units_discovery"]:
            h = hold_by.get(e["unit"], {})
            rows.append(
                {
                    "unit": e["unit"],
                    "status": e.get("status"),
                    "size": e["size"],
                    "residues": _join(e.get("residues", [])),
                    "nu": e.get("nu"),
                    "nu_in": e.get("nu_in"),
                    "nu_out": e.get("nu_out"),
                    "cost": e.get("cost"),
                    "p_size": e.get("p_size"),
                    "cohesion": e.get("cohesion", {}).get("verdict"),
                    "holdout_nu": h.get("nu"),
                    "holdout_cost": h.get("cost"),
                    "holdout_p_size": h.get("p_size"),
                    "holdout_cohesion": h.get("cohesion", {}).get("verdict"),
                    "replicates": h.get("replicates"),
                }
            )
        _write_csv(self.out / "units.csv", rows)
        if ledger.rows:
            _write_csv(self.out / "residue_ledger.csv", ledger.rows)
        if ledger.rho_log:
            results["unit_rho_log"] = ledger.rho_log
            _write_csv(self.out / "unit_rho.csv", ledger.rho_log)
        if results.get("units_resolved"):
            _write_csv(
                self.out / "units_resolved.csv",
                [
                    {
                        "unit": e["unit"],
                        "status": e.get("status"),
                        "size": e["size"],
                        "residues": _join(e.get("residues", [])),
                        "nu": e.get("nu"),
                        "nu_in": e.get("nu_in"),
                        "nu_out": e.get("nu_out"),
                        "cost": e.get("cost"),
                        "p_size": e.get("p_size"),
                        "cohesion": e.get("cohesion", {}).get("verdict"),
                    }
                    for e in results["units_resolved"]
                ],
            )
            secondary = results["resolution"]["secondary"]
            if secondary and not results.get("secondary"):
                _write_csv(self.out / "secondary_assignments.csv", secondary)
        if results.get("secondary"):
            _write_csv(
                self.out / "secondary_assignments.csv", results["secondary"]
            )
