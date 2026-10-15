#!/usr/bin/env python3
"""One table from many cog2_run outputs.

    python -m pdukit.lib.pdu.cog2_summarise RESULTS_ROOT [--csv out.csv]

Finds every results.json under RESULTS_ROOT and prints, per run: units
found, (planted runs) recovery against the truth for COG 2.1 before and
after the cohesion split and for COG v1.0, face-residue recovery, and the
held-out replication count.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _row(path):
    R = json.loads(Path(path).read_text())
    d = R.get("discovery", {})
    row = {
        "run": str(Path(path).parent.name),
        "n_units": d.get("n_units", len(d.get("units", []))),
        "sizes": " ".join(str(len(u)) for u in d.get("units", [])),
    }
    for key, src in (
        ("cog2", d.get("vs_truth")),
        ("cog2_presplit", d.get("vs_truth_before_split")),
        ("cog1", R.get("cog1", {}).get("vs_truth")),
    ):
        if src:
            row[f"{key}_recovered"] = f"{src['n_recovered']}/{src['n_true']}"
            row[f"{key}_recall"] = round(src["mean_recall"], 3)
            row[f"{key}_precision"] = round(src["mean_precision"], 3)
            row[f"{key}_merged"] = src["n_merged"]
    for key, src in (
        ("cog2_faces", d.get("faces")),
        ("cog1_faces", R.get("cog1", {}).get("faces")),
    ):
        if src and src.get("n_faces"):
            row[key] = f"{src['found']}/{src['n_faces']}"
    cv = d.get("channels_vs_truth")
    if cv:
        row["chan_err"] = round(cv["mean_error"], 2)
        row["chan_abs_err"] = round(cv["mean_abs_error"], 2)
        row["chan_exact"] = round(cv["exact_after_rounding"], 2)
    hold = R.get("units_holdout")
    if hold:
        row["holdout_replicates"] = (
            f"{sum(bool(e.get('replicates')) for e in hold)}/{len(hold)}"
        )
    coh = [
        e.get("cohesion", {}).get("verdict")
        for e in R.get("units_discovery", [])
    ]
    if coh:
        row["cohesive"] = f"{coh.count('cohesive')}/{len(coh)}"
    row["splits"] = len(d.get("cohesion_splits", []))
    return row


def main(argv=None):
    """Tabulate the results of several runs from the command line."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("root")
    ap.add_argument("--csv", default=None)
    a = ap.parse_args(argv)
    rows = []
    for p in sorted(Path(a.root).rglob("results.json")):
        try:
            rows.append(_row(p))
        except Exception as exc:  # noqa: BLE001
            print(f"skip {p}: {exc}", file=sys.stderr)
    if not rows:
        print("no results.json found")
        return 1
    cols = []
    for r in rows:
        cols += [k for k in r if k not in cols]
    width = {
        c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in cols
    }
    print("  ".join(c.ljust(width[c]) for c in cols))
    for r in rows:
        print("  ".join(str(r.get(c, "")).ljust(width[c]) for c in cols))
    if a.csv:
        import csv

        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
