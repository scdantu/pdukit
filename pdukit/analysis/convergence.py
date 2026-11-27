"""How much trajectory is needed?  Compare cheaper runs with a full one.

    pdu converge FULL_RUN RUN [RUN ...] -o tables/

Every run is compared with the reference (the first folder) at three
levels, from the most continuous to the most discrete:

  1. the predictive coupling rho (correlation of the off-diagonal entries),
  2. the null thresholds theta_in / theta_out (median relative change over
     the calibrated sizes),
  3. the units: adjusted Rand index of the core partition (residues in no
     core count as singletons), symmetric mean best Jaccard of the cores,
     number of cores that beat the random-set null, % unassigned.

A cheaper run shares data with the full run, so its agreement is inflated.
The yardstick is therefore the full run's own variability: runs on
independent data (disjoint replica halves) or with another seed.  Every
pair of runs is also written to ``convergence_pairs.csv``; a cheaper run is
good enough when it agrees with the reference at least as well as those
noise-floor pairs agree with each other.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pdukit.analysis.synthetic import SyntheticScorer
from pdukit.io.results.reader import ResultsReader
from pdukit.io.tables import CsvWriter
from pdukit.numerics.matching import adjusted_rand, jaccard, labels_from_units


class RunSummary:
    """What one run found, read once: cores, rho, thetas, settings, cost."""

    def __init__(self, run, n_random=1000):
        """Read the run folder ``run``."""
        self.path = Path(run)
        self.name = self.path.name
        self.results = ResultsReader.load(run)
        scorer = SyntheticScorer(run, n_random=n_random)
        self.cores = [set(c) for c in scorer.cores]
        self.n = int(self.results["data"]["n_residues"])
        self.rho = self.results.rho()
        self.real_cores = int(sum(scorer.real_flags)) if self.cores else 0
        self.unassigned_pct = scorer.unassigned_pct()
        cal = self.results.get("calibration") or {}
        self.theta = {
            side: {int(k): float(v) for k, v in cal.get(side, {}).items()}
            for side in ("theta_in", "theta_out")
        }
        self.args = self.results.get("args", {})
        data = self.results.get("data", {})
        self.n_frames = int(data.get("n_frames") or 0)
        self.n_replicas = len(data.get("segments") or [])
        self.seconds = float(sum(
            v for v in self.results.get("timings", {}).values()
            if isinstance(v, (int, float))
        ))

    def settings(self):
        """Return the settings that distinguish convergence runs."""
        a = self.args
        return {
            "stride": a.get("stride"),
            "frames_per_replica": a.get("frames"),
            "lag_frames": a.get("lag"),
            "frame_ps": a.get("frame_ps"),
            "seed": a.get("seed"),
            "replicas": self.n_replicas,
            "frames_used": self.n_frames,
        }


def _theta_change(a, b, side):
    common = sorted(set(a.theta[side]) & set(b.theta[side]))
    if not common:
        return None
    rel = [b.theta[side][k] / a.theta[side][k] - 1 for k in common]
    return round(float(np.median(rel)), 4)


def _sym_best_jaccard(A, B):
    if not A or not B:
        return 0.0 if (A or B) else 1.0
    ab = [max(jaccard(u, v) for v in B) for u in A]
    ba = [max(jaccard(v, u) for u in A) for v in B]
    return round(float(np.mean(ab + ba)), 3)


def compare(a, b):
    """Return the agreement of run ``b`` with run ``a`` (a dict)."""
    if a.n != b.n:
        raise ValueError(f"{a.name} and {b.name}: different residue counts")
    iu = np.triu_indices(a.n, 1)
    x = np.concatenate([a.rho[iu], a.rho.T[iu]])
    y = np.concatenate([b.rho[iu], b.rho.T[iu]])
    return {
        "rho_corr": round(float(np.corrcoef(x, y)[0, 1]), 4),
        "theta_in_change": _theta_change(a, b, "theta_in"),
        "theta_out_change": _theta_change(a, b, "theta_out"),
        "ari": round(float(adjusted_rand(
            labels_from_units(a.cores, a.n), labels_from_units(b.cores, b.n)
        )), 4),
        "mean_best_jaccard": _sym_best_jaccard(a.cores, b.cores),
    }


class ConvergenceComparer:
    """Compare runs with a reference run and with each other."""

    def __init__(self, reference, runs, n_random=1000):
        """Read the reference folder and the other run folders."""
        self.ref = RunSummary(reference, n_random)
        self.runs = [RunSummary(r, n_random) for r in runs]

    def rows(self):
        """Return one row per run (the reference first) against the ref."""
        out = []
        for r in [self.ref, *self.runs]:
            row = {"run": r.name, **r.settings(), "cores": len(r.cores),
                   "real_cores": r.real_cores,
                   "unassigned_pct": r.unassigned_pct,
                   "seconds": round(r.seconds, 1)}
            row.update(compare(self.ref, r))
            out.append(row)
        return out

    def pair_rows(self):
        """Return the agreement of every pair of runs (noise floor)."""
        allr = [self.ref, *self.runs]
        return [
            {"run_a": a.name, "run_b": b.name, **compare(a, b)}
            for i, a in enumerate(allr) for b in allr[i + 1:]
        ]

    def write(self, outdir):
        """Write convergence.csv and convergence_pairs.csv; return paths."""
        out = Path(outdir)
        return [
            CsvWriter(out / "convergence.csv").write(
                self.rows(), note=f"reference: {self.ref.name}"),
            CsvWriter(out / "convergence_pairs.csv").write(self.pair_rows()),
        ]


def _finished(run):
    """Return True when ``run`` holds the results.json of a finished run."""
    path = Path(run) / "results.json"
    if not path.is_file():
        return False
    try:
        r = ResultsReader.load(run)
    except (OSError, ValueError):
        return False
    return "predictive_graph" in r and "units_final" in r


def main(argv=None):
    """Compare runs with a reference run (``pdu converge``)."""
    ap = argparse.ArgumentParser(
        prog="pdu converge", description=__doc__.split("\n\n")[0]
    )
    ap.add_argument("reference", help="run folder on all the data")
    ap.add_argument("runs", nargs="+", help="run folders to compare with it")
    ap.add_argument("-o", "--outdir", required=True, help="table folder")
    ap.add_argument("--n-random", type=int, default=1000,
                    help="random sets per core for the real-core null")
    a = ap.parse_args(argv)
    runs = [r for r in a.runs if _finished(r)]
    skipped = [r for r in a.runs if r not in runs]
    if skipped:
        print("skipped (no finished results.json):", " ".join(
            Path(r).name for r in skipped))
    if not _finished(a.reference) or not runs:
        print("need a reference run and at least one other run with "
              "results.json")
        return 1
    comp = ConvergenceComparer(a.reference, runs, n_random=a.n_random)
    for path in comp.write(a.outdir):
        print("wrote", path)
    for row in comp.rows():
        print(f"{row['run'][:38]:38s} rho {row['rho_corr']:.3f}  "
              f"ARI {row['ari']:.3f}  J {row['mean_best_jaccard']:.2f}  "
              f"cores {row['cores']:2d} real {row['real_cores']:2d}  "
              f"{row['seconds']:.0f}s")
    return 0
