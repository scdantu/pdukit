"""Score ``pdu bench`` runs on synthetic designs against the planted answer.

    pdu score-bench RUN_DIR [RUN_DIR ...] -o TABLE_DIR

For every run: how many planted units were recovered (best Jaccard of a
found core >= 0.67), how many found cores are "real" and how many of those
are false units.

real core   cohesion(core) - 1.645 x its bootstrap sd is above the 95th
            percentile of cohesion of 1000 random residue sets of the same
            size (one-sided 95 %).  Optionally (``--compact``) the null is
            spatially connected regions grown on the contact graph instead.
false unit  a real core that matches no planted unit (Jaccard < 0.67).

Tables written to TABLE_DIR:

    synthetic_runs.csv         one row per run
    synthetic_named_units.csv  per run and named planted unit (tau design):
                               best Jaccard of any found core
    synthetic_test1.csv, synthetic_test4.csv, synthetic_tau.csv,
    synthetic_tau_counts.csv, synthetic_membership.csv
                               the views the manuscript figure reads
                               (figures/scripts/figS_synthetic.py)

Design, seed, lag and growth seeds are read from each run's results, not
from the folder name.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from functools import cached_property
from pathlib import Path

import numpy as np

from pdukit.io.results.reader import ResultsReader
from pdukit.io.tables import CsvWriter
from pdukit.numerics.cohesion import (
    above_null,
    boot_sd,
    cohesion,
    compact_regions,
    null_quantile,
    random_sets,
)
from pdukit.numerics.matching import best_jaccard, match_units

RECOVERY_JACCARD = 0.67


@dataclass
class RunScore:
    """One run scored against its planted answer (one table row)."""

    run: str
    design: str
    gen_seed: int
    lag_frames: int
    lag_ns: float | None
    growth_seeds: int
    obs_noise: float | None
    n_planted: int
    n_found: int
    recovered: int
    mean_best_jaccard: float | None
    real_cores: int
    false_units: int
    unassigned_pct: float
    core_sizes: str


class SyntheticScorer:
    """Score one synthetic run (a ``pdu bench`` output folder)."""

    def __init__(self, run, n_random=1000, seed=0, null="random",
                 truth=None):
        """Read the run; ``null`` is 'random' or 'compact' (see module).

        ``truth`` (dict, or a truth.json path) gives the planted answer of
        an MD run on synthetic coordinates (`pdukit.bench.coords`), whose
        results.json holds none.
        """
        self.results = ResultsReader.load(run)
        self.name = Path(self.results.rundir).name
        self.n_random = int(n_random)
        self.seed = int(seed)
        self.null = null
        ports = self.results.ports()
        self.has_ports = ports is not None
        self.ports = ports or {"units": [], "counts": {}, "residues": []}
        self.cores = [list(map(int, u["core"])) for u in self.ports["units"]]
        if isinstance(truth, (str, Path)):
            import json

            truth = json.loads(Path(truth).read_text())
        self.truth = truth or self.results.get("truth") or {}
        planted = self.results.get("data", {}).get("planted") or {}
        self.design = str(planted.get("design", "planted"))
        self.dt_ns = planted.get("dt_ns", 0.1 if "design" in planted else None)
        self.settings = self.results.get("args", {})

    @property
    def planted_units(self):
        """Return the planted units (empty for designs without units)."""
        return self.truth.get("units") or []

    @cached_property
    def real_flags(self):
        """Return, per found core, whether it beats the null (see module)."""
        if not self.cores:
            return []
        R = self.results.rho()
        boot = self.results.bootstrap_rho()
        rng = np.random.default_rng(self.seed)
        adj = self._adjacency() if self.null == "compact" else None
        flags = []
        for S in self.cores:
            if adj is None:
                sets = random_sets(len(R), len(S), self.n_random, rng)
            else:
                sets = compact_regions(adj, len(S), 300, rng, max_tries=600)
            q = null_quantile(R, sets)
            sd = boot_sd(boot, S) if boot is not None else 0.0
            flags.append(above_null(cohesion(R, S)["cohesion"], sd, q))
        return flags

    def unassigned_pct(self):
        """Return the % of residues that are bridges or loose."""
        counts = self.ports.get("counts", {})
        n = len(self.ports.get("residues", [])) or self._n_residues()
        lost = counts.get("bridge", 0) + counts.get("loose", 0)
        if not self.has_ports:  # no unit stage output: nothing assigned
            lost = n
        return round(100 * lost / max(n, 1), 1)

    def score(self):
        """Return the `RunScore` of this run."""
        T = self.planted_units
        m = match_units(self.cores, T, threshold=RECOVERY_JACCARD)
        false = sum(
            1
            for S, real in zip(self.cores, self.real_flags)
            if real and best_jaccard(S, T) < RECOVERY_JACCARD
        )
        lag = int(self.settings.get("lag", 0) or 0)
        return RunScore(
            run=self.name,
            design=self.design,
            gen_seed=int(self.settings.get("bench_seed", 0) or 0),
            lag_frames=lag,
            lag_ns=round(lag * self.dt_ns, 3) if self.dt_ns else None,
            growth_seeds=int(self.settings.get("dec_seeds", 0) or 0),
            obs_noise=(
                self.settings.get("design_obs_noise")
                if self.design != "planted"
                else None
            ),
            n_planted=len(T),
            n_found=len(self.cores),
            recovered=m["n_recovered"],
            mean_best_jaccard=round(m["mean_recall"], 3) if T else None,
            real_cores=sum(self.real_flags),
            false_units=false,
            unassigned_pct=self.unassigned_pct(),
            core_sizes=" ".join(str(len(s)) for s in self.cores),
        )

    def named_rows(self):
        """Return best Jaccard of any found core for each named unit."""
        base = asdict(self.score())
        return [
            {
                "run": self.name,
                "gen_seed": base["gen_seed"],
                "lag_frames": base["lag_frames"],
                "lag_ns": base["lag_ns"],
                "unit": name,
                "size": len(res),
                "best_jaccard": round(best_jaccard(res, self.cores), 3),
            }
            for name, res in (self.truth.get("named") or {}).items()
        ]

    def membership_rows(self):
        """Return residue-level planted unit, found core(s) and role."""
        planted = {i: a for a, u in enumerate(self.planted_units) for i in u}
        found = {}
        for k, S in enumerate(self.cores):
            for i in S:
                found.setdefault(i, []).append(str(k))
        role = {r["residue"]: r["cls"] for r in self.ports["residues"]}
        return [
            {
                "residue": i,
                "planted_unit": planted.get(i, ""),
                "found_core": " ".join(found.get(i, [])),
                "role": role.get(i, ""),
            }
            for i in range(self._n_residues())
        ]

    def _n_residues(self):
        return int(self.results.get("data", {}).get("n_residues", 0))

    def _adjacency(self):
        N = self._n_residues()
        adj = np.zeros((N, N), dtype=bool)
        for i, j in self.truth.get("contacts", []):
            adj[i, j] = adj[j, i] = True
        if not adj.any():
            raise ValueError(
                f"{self.name}: no contacts in the planted answer; the "
                "compact null needs a run made with bench-design"
            )
        return adj


class SuiteScorer:
    """Score many synthetic runs and build the tables of the test suite."""

    def __init__(self, runs, **options):
        """Score every folder in ``runs`` that holds a results.json."""
        dirs = [Path(r) for r in runs if (Path(r) / "results.json").is_file()]
        self.scorers = [SyntheticScorer(d, **options) for d in sorted(dirs)]
        self.scores = [s.score() for s in self.scorers]

    def run_table(self):
        """Return one row per run."""
        return [asdict(s) for s in self.scores]

    def named_table(self):
        """Return one row per run and named planted unit."""
        return [row for s in self.scorers for row in s.named_rows()]

    def figure_tables(self):
        """Return the tables the manuscript figure reads, by file name.

        test1      units / mixed / medium at lag 1 frame, 32 growth seeds
        test4      units / nocomm at lag 1 frame, by number of growth seeds
        tau        tau design: best Jaccard per named unit, by lag
        tau_counts tau design: units found and recovered, by lag
        membership residue-level view of units seed 0 (lag 1, 32 seeds)
        """
        cols = [
            "design",
            "gen_seed",
            "n_planted",
            "n_found",
            "recovered",
            "mean_best_jaccard",
            "real_cores",
            "false_units",
            "unassigned_pct",
            "core_sizes",
        ]

        def pick(s, extra=()):
            d = asdict(s)
            keys = cols[:2] + list(extra) + cols[2:]
            return {k: d[k] for k in keys}

        test1 = [
            pick(s)
            for s in self.scores
            if s.design in ("units", "mixed", "medium")
            and s.lag_frames == 1
            and s.growth_seeds == 32
        ]
        test4 = [
            pick(s, ["growth_seeds"])
            for s in self.scores
            if s.design in ("units", "nocomm") and s.lag_frames == 1
        ]
        tau_scorers = [
            (sc, s)
            for sc, s in zip(self.scorers, self.scores)
            if s.design == "tau"
        ]
        tau = [row for sc, _ in tau_scorers for row in sc.named_rows()]
        counts = []
        for sc, s in tau_scorers:
            fine = match_units(sc.cores, sc.truth.get("fine_units", []))
            coarse = match_units(sc.cores, sc.truth.get("coarse_units", []))
            counts.append(
                {
                    "gen_seed": s.gen_seed,
                    "lag_frames": s.lag_frames,
                    "lag_ns": s.lag_ns,
                    "n_found": s.n_found,
                    "fine_recovered": fine["n_recovered"],
                    "coarse_recovered": coarse["n_recovered"],
                    "unassigned_pct": s.unassigned_pct,
                }
            )
        member = next(
            (
                sc.membership_rows()
                for sc, s in zip(self.scorers, self.scores)
                if s.design == "units"
                and s.gen_seed == 0
                and s.lag_frames == 1
                and s.growth_seeds == 32
            ),
            [],
        )
        return {
            "synthetic_test1.csv": test1,
            "synthetic_test4.csv": test4,
            "synthetic_tau.csv": tau,
            "synthetic_tau_counts.csv": counts,
            "synthetic_membership.csv": member,
        }

    def write(self, outdir):
        """Write every table to ``outdir``; return the paths written."""
        out = Path(outdir)
        tables = {
            "synthetic_runs.csv": self.run_table(),
            "synthetic_named_units.csv": self.named_table(),
            **self.figure_tables(),
        }
        written = [CsvWriter(out / name).write(rows) for name, rows in
                   tables.items()]
        return [p for p in written if p is not None]


def main(argv=None):
    """Score synthetic runs from the command line (``pdu score-bench``)."""
    ap = argparse.ArgumentParser(
        prog="pdu score-bench", description=__doc__.split("\n\n")[0]
    )
    ap.add_argument("runs", nargs="+", help="pdu bench output folders")
    ap.add_argument("-o", "--outdir", required=True, help="table folder")
    ap.add_argument("--n-random", type=int, default=1000,
                    help="random sets per core for the null (default 1000)")
    ap.add_argument("--compact", action="store_true",
                    help="null = compact regions on the contact graph")
    ap.add_argument("--truth", metavar="FILE",
                    help="truth.json of synthetic coordinates (runs made "
                    "with pdu run on pdukit.bench.coords data)")
    a = ap.parse_args(argv)
    suite = SuiteScorer(
        a.runs, n_random=a.n_random, null="compact" if a.compact else "random",
        truth=a.truth,
    )
    if not suite.scores:
        print("no results.json in the given folders")
        return 1
    for path in suite.write(a.outdir):
        print("wrote", path)
    return 0
