"""Collect pdu results of many mdCATH domains into one table.

    python collect_results.py RESULTS_ROOT --selected mdcath_selected.tsv

Reads ``RESULTS_ROOT/<domain>/pdu/results.json`` and writes a CSV with, per
domain: CATH class and architecture, residues, units found, mean unit size,
and the fraction of residues in a unit.  A class summary is printed.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def units_of(res: dict) -> list:
    """Return the final unit list of a results.json, if any."""
    d = res.get("discovery") or {}
    return d.get("units") or res.get("units_final") or []


def read_selected(path: Path | None) -> dict:
    """Return {domain: row} from the selection table."""
    if path is None:
        return {}
    with path.open() as fh:
        return {r["domain"]: r for r in csv.DictReader(fh, delimiter="\t")}


def main(argv=None) -> int:
    """Write the table and print a per-class summary."""
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("root", type=Path)
    p.add_argument("--selected", type=Path)
    p.add_argument("--csv", type=Path, default=Path("mdcath_pdu_summary.csv"))
    args = p.parse_args(argv)
    sel = read_selected(args.selected)
    rows, missing = [], []
    for d in sorted(sel) if sel else sorted(
            x.name for x in args.root.iterdir() if x.is_dir()):
        f = args.root / d / "pdu" / "results.json"
        if not f.exists():
            missing.append(d)
            continue
        res = json.loads(f.read_text())
        units = units_of(res)
        n_res = (res.get("data") or {}).get("n_residues")
        covered = len({r for u in units for r in u})
        sizes = [len(u) for u in units]
        s = sel.get(d, {})
        rows.append({
            "domain": d, "class": s.get("class", ""),
            "architecture": s.get("architecture", ""),
            "n_residues": n_res, "n_units": len(units),
            "mean_unit_size": round(statistics.mean(sizes), 1) if sizes
            else 0,
            "covered_fraction": round(covered / n_res, 3) if n_res else "",
        })
    fields = list(rows[0]) if rows else ["domain"]
    with args.csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    by = defaultdict(list)
    for r in rows:
        by[r["class"]].append(r)
    print(f"{len(rows)} domains with results, {len(missing)} missing")
    for c, rs in sorted(by.items()):
        cov = [r["covered_fraction"] for r in rs
               if r["covered_fraction"] != ""]
        med_cov = statistics.median(cov) if cov else "n/a"
        med_units = statistics.median(r["n_units"] for r in rs)
        print(f"class {c or '?'}: n={len(rs)}  median units {med_units}  "
              f"median coverage {med_cov}")
    if missing:
        print("missing:", " ".join(missing[:20]),
              "..." if len(missing) > 20 else "")
    return 0


if __name__ == "__main__":
    sys.exit(main())
