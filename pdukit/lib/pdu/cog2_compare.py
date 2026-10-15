#!/usr/bin/env python3
"""Compare the units found by several COG runs.

    python -m pdukit.lib.pdu.cog2_compare RUN_DIR RUN_DIR [RUN_DIR ...]

For example the three leave-one-out lysozyme runs (SPLIT=holdout HOLDOUT=0,
1, 2), or a pooled run against them.  For every pair of runs: each unit's
best Jaccard match in the other run, the mean of those, and how many units
are matched at >= 0.67.  Then a residue co-membership consensus over all
runs: the fraction of runs in which each residue pair shares a unit,
written to consensus.csv in the first run directory (N x N).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _units(run):
    R = json.loads((Path(run) / "results.json").read_text())
    return [set(map(int, u)) for u in R["discovery"]["units"]], R["data"][
        "n_residues"
    ]


def main(argv=None):
    """Compare the units of several runs and print the overlap table."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--threshold", type=float, default=0.67)
    a = ap.parse_args(argv)
    runs = [(Path(r).name, *_units(r)) for r in a.runs]
    N = runs[0][2]
    print(f"{'run A':40s} {'run B':40s} {'mean best J':>11s} {'matched':>8s}")
    for i in range(len(runs)):
        for j in range(len(runs)):
            if i == j:
                continue
            A, B = runs[i][1], runs[j][1]
            best = [
                max((len(u & v) / len(u | v) for v in B), default=0.0)
                for u in A
            ]
            print(
                f"{runs[i][0][:40]:40s} {runs[j][0][:40]:40s} "
                f"{np.mean(best):11.2f} "
                f"{sum(b >= a.threshold for b in best):>4d}/{len(A)}"
            )
    co = np.zeros((N, N))
    for _, U, _ in runs:
        m = np.zeros((N, N), dtype=bool)
        for u in U:
            idx = np.array(sorted(u))
            m[np.ix_(idx, idx)] = True
        co += m
    co /= len(runs)
    out = Path(a.runs[0]) / "consensus.csv"
    np.savetxt(out, co, delimiter=",", fmt="%.3f")
    stable = np.mean(
        co[np.triu_indices(N, 1)][co[np.triu_indices(N, 1)] > 0] == 1.0
    )
    print(
        f"\nresidue pairs that share a unit in ANY run share it in ALL "
        f"runs: {stable:.0%}   (consensus matrix: {out})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
