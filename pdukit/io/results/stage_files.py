"""The files the stages write into a run directory, one function per stage.

Each function takes the run directory and that stage's result and writes
the files named below.  ``results.json``, ``calibration.json``,
``pdu_out.ppf`` and ``bootstrap_rho.npz`` are written by `ResultsWriter`.

    write_incoming_tables   incoming_growth.csv, incoming_candidates.csv,
                            incoming_proposals.csv
    write_merge             incoming_merge.json, incoming_merge_frequency.csv,
                            incoming_merged_parents.csv
    write_split             incoming_split.json, incoming_split_boundaries.csv,
                            incoming_split_units.csv
    write_classes           incoming_ports.json, incoming_residue_classes.csv,
                            incoming_units_extended.csv
    write_mode_recovery     global_mode_recovery.json / .md / .csv,
                            global_mode_combinations.csv,
                            global_mode_best_combinations.csv,
                            global_modes_reference.csv
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


def write_incoming_tables(outdir, result):
    """Write candidate membership, accepted paths, and rejected proposals."""
    outdir = Path(outdir)
    metrics = [
        "nu_in",
        "nu_out",
        "theta_in",
        "theta_out",
        "keff_in",
        "keff_out",
        "rho_in",
        "rho_out",
        "cohesion",
        "auc_driver",
        "auc_listener",
    ]

    def write(name, rows, fields):
        with (outdir / name).open("w", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=fields, extrasaction="ignore"
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        k: (
                            " ".join(map(str, v)) if isinstance(v, list) else v
                        )
                        for k, v in row.items()
                        if k in fields
                    }
                )

    write(
        "incoming_candidates.csv",
        result["units"],
        ["unit", "size", "residues", "source_run", "stop_reason"] + metrics,
    )
    write(
        "incoming_growth.csv",
        (
            {"run": run["run"], **step}
            for run in result["runs"]
            for step in run["history"]
        ),
        ["run", "step", "added_residue", "residues"] + metrics,
    )
    write(
        "incoming_proposals.csv",
        (
            {"run": run["run"], **step}
            for run in result["runs"]
            for step in run["proposals"]
        ),
        [
            "run",
            "step",
            "parent_residues",
            "candidate_residue",
            "parent_nu_in",
            "gain_in",
            "accepted",
        ]
        + metrics,
    )


def write_merge(outdir, merge):
    """Write family parents and per-residue membership frequencies."""
    outdir = Path(outdir)
    (outdir / "incoming_merge.json").write_text(
        json.dumps(
            {
                **merge,
                "families": [
                    {
                        **f,
                        "residue_frequency": {
                            str(k): v
                            for k, v in f["residue_frequency"].items()
                        },
                    }
                    for f in merge["families"]
                ],
            },
            indent=1,
        )
    )
    with (outdir / "incoming_merged_parents.csv").open(
        "w", newline=""
    ) as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            [
                "family",
                "decision",
                "size",
                "n_members",
                "members",
                "member_sizes",
                "member_nu_in",
                "best_member",
                "best_member_nu_in",
                "union_size",
                "union_nu_in",
                "core_size",
                "majority_size",
                "residues",
                "core",
                "majority",
                "union",
            ]
        )
        for f in merge["families"]:

            def fmt(v):
                return "" if v is None else f"{v:.4f}"

            writer.writerow(
                [
                    f["family"],
                    f["decision"],
                    f["size"],
                    len(f["members"]),
                    " ".join(map(str, f["members"])),
                    " ".join(map(str, f["member_sizes"])),
                    " ".join(fmt(v) for v in f["member_nu_in"]),
                    f["best_member"],
                    fmt(f["best_member_nu_in"]),
                    f["union_size"],
                    fmt(f["union_nu_in"]),
                    len(f["core"]),
                    len(f["majority"]),
                    " ".join(map(str, f["residues"])),
                    " ".join(map(str, f["core"])),
                    " ".join(map(str, f["majority"])),
                    " ".join(map(str, f["union"])),
                ]
            )
    with (outdir / "incoming_merge_frequency.csv").open(
        "w", newline=""
    ) as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["family", "residue", "frequency"])
        for f in merge["families"]:
            for r, v in sorted(f["residue_frequency"].items()):
                writer.writerow([f["family"], r, f"{v:.4f}"])


def write_split(outdir, result):
    """Write the hierarchy, final supports, and directional cut diagnostics."""
    outdir = Path(outdir)
    (outdir / "incoming_split.json").write_text(json.dumps(result, indent=1))
    for name, rows, fields in (
        (
            "incoming_split_units.csv",
            result["units"],
            [
                "unit",
                "size",
                "residues",
                "node_ids",
                "root_ids",
                "cohesion",
                "auc_listener",
                "auc_driver",
                "nu_in",
                "nu_out",
                "keff_in",
                "keff_out",
            ],
        ),
        (
            "incoming_split_boundaries.csv",
            result["boundaries"],
            [
                "node",
                "side",
                "residue",
                "incoming_within_vs_sibling_auc",
                "outgoing_within_vs_sibling_auc",
            ],
        ),
    ):
        with (outdir / name).open("w", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=fields,
                extrasaction="ignore",
                lineterminator="\n",
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        k: " ".join(map(str, v)) if isinstance(v, list) else v
                        for k, v in row.items()
                        if k in fields
                    }
                )


def write_classes(outdir, result):
    """Write incoming_ports.json and the per-residue class table to outdir."""
    outdir = Path(outdir)
    (outdir / "incoming_ports.json").write_text(json.dumps(result, indent=1))
    fields = [
        "residue",
        "cls",
        "unit",
        "tested_unit",
        "drives",
        "driven",
        "delta_cohesion",
        "delta_cohesion_sd",
        "guard_delta",
        "guard_sd",
        "failed",
        "round",
        "best_units",
        "best_drives",
        "best_driven",
    ]
    with (outdir / "incoming_residue_classes.csv").open("w", newline="") as fh:
        w = csv.DictWriter(
            fh, fieldnames=fields, extrasaction="ignore", lineterminator="\n"
        )
        w.writeheader()
        for row in result["residues"]:
            w.writerow(row)
    ufields = [
        "unit",
        "core_size",
        "n_ports",
        "extended_size",
        "cohesion_core",
        "cohesion_extended",
        "cohesion_change",
        "cohesion_change_sd",
        "nu_in_core",
        "nu_in_extended",
        "nu_out_core",
        "nu_out_extended",
        "strength_in_core",
        "strength_in_extended",
        "strength_out_core",
        "strength_out_extended",
        "keff_in_core",
        "keff_in_extended",
        "keff_out_core",
        "keff_out_extended",
        "core",
        "ports",
        "extended",
    ]
    with (outdir / "incoming_units_extended.csv").open("w", newline="") as fh:
        w = csv.DictWriter(
            fh, fieldnames=ufields, extrasaction="ignore", lineterminator="\n"
        )
        w.writeheader()
        for u in result["units"]:
            w.writerow(
                {
                    k: " ".join(map(str, v)) if isinstance(v, list) else v
                    for k, v in u.items()
                }
            )


def write_mode_recovery(outdir, report):
    """Save per-mode and per-combination reports and a reader guide."""
    outdir = Path(outdir)
    (outdir / "global_mode_recovery.json").write_text(
        json.dumps(report, indent=1)
    )
    for filename, key in (
        ("global_mode_recovery.csv", "rows"),
        ("global_mode_combinations.csv", "combinations"),
        ("global_mode_best_combinations.csv", "best_combinations"),
        ("global_modes_reference.csv", "modes"),
    ):
        rows = report.get(key, [])
        if not rows:
            continue
        fields = list(dict.fromkeys(k for row in rows for k in row))
        with (outdir / filename).open("w", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=fields, lineterminator="\n"
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        k: " ".join(map(str, v)) if isinstance(v, list) else v
                        for k, v in row.items()
                    }
                )
    text = [
        "# Global-mode recovery",
        "",
        "Mode and unit IDs are zero-based. Coverage measures the fraction "
        "of a global mode's variance represented by the selected local "
        "bases; it does not change when channels are switched. Mode overlap "
        "and prediction R² measure the assembled dynamics and can change "
        "with channels. Prediction R² may be negative. Reference and "
        "reconstructed relaxation times are reported separately.",
        "",
        "Channels off removes cross-PDU blocks of the fitted propagator. "
        "Rank-k retains k incoming directions per target PDU against all "
        "other selected PDUs jointly. On retains every fitted cross block. "
        "These ranks are ablation budgets, not theta-calibrated detected "
        "channel counts.",
        "",
        "Results use the same sampled ensemble as detection and are "
        "in-sample. Overlapping supports and redundant coordinates are "
        "flagged in global_mode_combinations.csv. Ablating channels in "
        "overlapping coordinate descriptions does not constitute a physical "
        "causal decoupling experiment.",
        "",
    ]
    if report.get("enumeration", {}).get("truncated"):
        text += [
            "Combination enumeration reached the configured cap; this is "
            "not an exhaustive combination search.",
            "",
        ]
    full = next(
        (
            c
            for c in report["combinations"]
            if c.get("units") == list(range(report.get("n_units", 0)))
        ),
        None,
    )
    if full:
        text += [
            "## All-PDU combination",
            "",
            "| Mode | Channels | Coverage | Mode overlap² | Prediction R² | "
            "Timescale error |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for row in report["rows"]:
            if row["combination"] == full["combination"]:
                error = row["timescale_relative_error"]
                text.append(
                    f"| {row['mode']} | {row['channels']} | "
                    f"{row['coverage_fraction']:.1%} | "
                    f"{row['mode_overlap_squared']:.1%} | "
                    f"{row['prediction_r2']:.3f} | {error:.1%} |"
                    if error is not None
                    else f"| {row['mode']} | {row['channels']} | "
                    f"{row['coverage_fraction']:.1%} | "
                    f"{row['mode_overlap_squared']:.1%} | "
                    f"{row['prediction_r2']:.3f} | undefined |"
                )
    if report.get("best_combinations"):
        text += [
            "",
            "## Best evaluated combinations by in-sample prediction R²",
            "",
            "These are the best among the evaluated combinations for each "
            "mode and channel setting; they are not independently validated "
            "optimal combinations.",
            "",
            "| Mode | Channels | PDU IDs | Coverage | Mode overlap² | "
            "Prediction R² |",
            "|---|---|---|---:|---:|---:|",
        ]
        for row in report["best_combinations"]:
            ids = " ".join(map(str, row["units"]))
            text.append(
                f"| {row['mode']} | {row['channels']} | {ids} | "
                f"{row['coverage_fraction']:.1%} | "
                f"{row['mode_overlap_squared']:.1%} | "
                f"{row['prediction_r2']:.3f} |"
            )
    (outdir / "global_mode_recovery.md").write_text("\n".join(text) + "\n")
