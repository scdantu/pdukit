"""Follow residues and units across lag times: which stay, which change.

    pdu track RUN_100ps RUN_200ps ... -o tables/

A PDU is treated as a graph whose nodes (residues) may persist across time
scales or change.  The runs are ordered by their lag time (lag x frame-ps)
and four tables are written:

  track_residues.csv   per residue: its class (core, port, shared, loose,
                       bridge) at every lag, how many lags it is core, and
                       the mean Jaccard of its core unit-mates between
                       neighbouring lags (an "anchor" stays with the same
                       mates; a "switcher" does not)
  track_affinity.csv   per residue and lag: best unit, second unit and the
                       effective number of units, from
                       residue_affinity_summary.csv (runs made with 1.0.5a4
                       or later; empty otherwise)
  track_lineage.csv    between neighbouring lags: each unit's best match,
                       and its fate: persists (one-to-one, Jaccard >= 0.5),
                       splits (its residues go to two or more units), merges
                       (it shares a later unit with another unit), dissolves
                       (no later unit takes >= 30% of it); new units appear
  track_core_graph.csv residue pairs that share a core in at least
                       ``--min-fraction`` of the lags (default 0.5), with
                       the connected components of that graph in
                       track_core_components.csv: the lag-robust cores
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from pdukit.io.results.reader import ResultsReader
from pdukit.io.tables import CsvWriter
from pdukit.numerics.matching import jaccard


class LagRun:
    """One run: lag time, residue classes and unit cores."""

    def __init__(self, run):
        """Read results.json, incoming_ports.json and the affinity table."""
        self.path = Path(run)
        self.name = self.path.name
        r = ResultsReader.load(run)
        a = r.get("args", {})
        self.lag_ps = float(a.get("lag") or 0) * float(a.get("frame_ps") or 0)
        self.n = int(r["data"]["n_residues"])
        ports = r.ports() or {"units": [], "residues": []}
        self.cores = [set(map(int, u["core"])) for u in ports["units"]]
        self.cls = {int(x["residue"]): x["cls"] for x in ports["residues"]}
        self.affinity = {}
        path = self.path / "residue_affinity_summary.csv"
        if path.is_file():
            with path.open() as fh:
                for row in csv.DictReader(
                    line for line in fh if not line.startswith("#")
                ):
                    self.affinity[int(row["residue"])] = row

    def mates(self, r):
        """Residues sharing a core with r (r excluded)."""
        out = set()
        for S in self.cores:
            if r in S:
                out |= S
        return out - {r}


def _label(lag_ps):
    return f"{lag_ps / 1000:g} ns" if lag_ps >= 1000 else f"{lag_ps:g} ps"


class LagTracker:
    """Compare runs on the same protein at different lag times."""

    def __init__(self, runs, min_fraction=0.5):
        """Read the runs and order them by lag time."""
        self.runs = sorted((LagRun(r) for r in runs), key=lambda x: x.lag_ps)
        if len({r.n for r in self.runs}) != 1:
            raise ValueError("runs have different numbers of residues")
        self.n = self.runs[0].n
        self.min_fraction = float(min_fraction)

    def residue_rows(self):
        """Per residue: class at every lag, core count, mate stability."""
        rows = []
        for i in range(self.n):
            row = {"residue": i}
            for r in self.runs:
                row[f"class_{_label(r.lag_ps)}"] = r.cls.get(i, "")
            row["lags_core"] = sum(
                r.cls.get(i) in ("core", "shared") for r in self.runs
            )
            js = []
            for a, b in zip(self.runs, self.runs[1:]):
                ma, mb = a.mates(i), b.mates(i)
                if ma and mb:
                    js.append(jaccard(ma, mb))
            row["mate_jaccard_mean"] = (
                round(float(np.mean(js)), 3) if js else None
            )
            row["lag_pairs_compared"] = len(js)
            rows.append(row)
        return rows

    def affinity_rows(self):
        """Per residue and lag: the affinity summary, if the run has one."""
        rows = []
        for r in self.runs:
            for i, a in sorted(r.affinity.items()):
                rows.append({
                    "residue": i,
                    "lag": _label(r.lag_ps),
                    "lag_ps": r.lag_ps,
                    "class": r.cls.get(i, ""),
                    **{k: v for k, v in a.items() if k != "residue"},
                })
        return rows

    def lineage_rows(self, persist=0.5, keep=0.3):
        """Fate of every unit between neighbouring lags."""
        rows = []
        for a, b in zip(self.runs, self.runs[1:]):
            takers = {}
            for k, S in enumerate(a.cores):
                # later units that take >= keep of S
                take = [
                    (m, len(S & T) / len(S)) for m, T in enumerate(b.cores)
                    if len(S & T) / len(S) >= keep
                ]
                takers[k] = take
            later_from = {}
            for k, take in takers.items():
                for m, _ in take:
                    later_from.setdefault(m, []).append(k)
            for k, S in enumerate(a.cores):
                best = max(
                    ((m, jaccard(S, T)) for m, T in enumerate(b.cores)),
                    key=lambda t: t[1], default=(None, 0.0),
                )
                take = takers[k]
                if not take:
                    fate = "dissolves"
                elif len(take) >= 2:
                    fate = "splits"
                elif len(later_from.get(take[0][0], [])) >= 2:
                    fate = "merges"
                elif best[1] >= persist:
                    fate = "persists"
                else:
                    fate = "changes"
                rows.append({
                    "from_lag": _label(a.lag_ps),
                    "to_lag": _label(b.lag_ps),
                    "unit": k,
                    "size": len(S),
                    "best_match": best[0],
                    "best_jaccard": round(best[1], 3),
                    "taken_by": " ".join(
                        f"{m}:{f:.2f}" for m, f in take
                    ),
                    "fate": fate,
                })
            for m, T in enumerate(b.cores):
                if m not in later_from:
                    rows.append({
                        "from_lag": _label(a.lag_ps),
                        "to_lag": _label(b.lag_ps),
                        "unit": None,
                        "size": len(T),
                        "best_match": m,
                        "best_jaccard": None,
                        "taken_by": "",
                        "fate": "appears",
                    })
        return rows

    def core_graph(self):
        """Pairs sharing a core in >= min_fraction of the lags; components."""
        L = len(self.runs)
        co = np.zeros((self.n, self.n))
        for r in self.runs:
            m = np.zeros((self.n, self.n), bool)
            for S in r.cores:
                idx = np.array(sorted(S))
                m[np.ix_(idx, idx)] = True
            co += m
        co /= max(L, 1)
        np.fill_diagonal(co, 0)
        edges = [
            {"residue_a": i, "residue_b": j, "fraction": round(co[i, j], 3)}
            for i in range(self.n) for j in range(i + 1, self.n)
            if co[i, j] >= self.min_fraction
        ]
        parent = list(range(self.n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for e in edges:
            parent[find(e["residue_a"])] = find(e["residue_b"])
        groups = {}
        for e in edges:
            for x in (e["residue_a"], e["residue_b"]):
                groups.setdefault(find(x), set()).add(x)
        comps = sorted((sorted(g) for g in groups.values()), key=len,
                       reverse=True)
        comp_rows = [
            {"component": c, "size": len(g),
             "residues": " ".join(map(str, g))}
            for c, g in enumerate(comps)
        ]
        return edges, comp_rows

    def write(self, outdir):
        """Write the five tables; return their paths (None if empty)."""
        out = Path(outdir)
        edges, comps = self.core_graph()
        lags = ", ".join(_label(r.lag_ps) for r in self.runs)
        return [
            CsvWriter(out / "track_residues.csv").write(
                self.residue_rows(), note=f"lags: {lags}"),
            CsvWriter(out / "track_affinity.csv").write(self.affinity_rows()),
            CsvWriter(out / "track_lineage.csv").write(self.lineage_rows()),
            CsvWriter(out / "track_core_graph.csv").write(
                edges, note=f"pairs sharing a core in >= "
                f"{self.min_fraction:g} of the lags"),
            CsvWriter(out / "track_core_components.csv").write(comps),
        ]


def main(argv=None):
    """Follow residues and units across lag times (``pdu track``)."""
    ap = argparse.ArgumentParser(
        prog="pdu track", description=__doc__.split("\n\n")[0]
    )
    ap.add_argument("runs", nargs="+", help="run folders, one per lag")
    ap.add_argument("-o", "--outdir", required=True, help="table folder")
    ap.add_argument("--min-fraction", type=float, default=0.5,
                    help="core graph: share a core in at least this "
                    "fraction of the lags (default 0.5)")
    a = ap.parse_args(argv)
    runs = [r for r in a.runs if (Path(r) / "incoming_ports.json").is_file()]
    if len(runs) < 2:
        print("need at least two finished runs (incoming_ports.json)")
        return 1
    tr = LagTracker(runs, min_fraction=a.min_fraction)
    for p in tr.write(a.outdir):
        if p:
            print("wrote", p)
    fates = {}
    for row in tr.lineage_rows():
        fates[row["fate"]] = fates.get(row["fate"], 0) + 1
    _, comps = tr.core_graph()
    print("lags:", ", ".join(_label(r.lag_ps) for r in tr.runs))
    print("unit fates between neighbouring lags:", fates)
    print(f"lag-robust cores (>= {a.min_fraction:g} of lags):",
          [c["size"] for c in comps])
    return 0
