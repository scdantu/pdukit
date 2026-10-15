"""Fast tests of COG 2.1 ideas on small planted systems.

Runs a matrix  design x config x seed  of small planted problems (minutes on
CPU) and prints one table: strict recovery (best Jaccard >= 0.67), mean
Jaccard, merges, units found, unresolved groups, seconds.

Designs (6 units x 6 residues, 2 replicas x 3000 frames by default):
  base       gears at gain 0.35, one face residue per gear
  scattered  same dynamics, residue positions permuted: units are NOT
             spatially contiguous -> only a dynamics-only search can win
  weak       gain 0.20 (harder: couplings near the noise)
  strong     gain 0.60 (geared units tempt merges)
  nofaces    no shared residues

Configs:
  contact       old: contact graph, spread seeds, compact-ball baseline
  dyn           dynamics only: predictive graph (k = 5), 16 random seeds,
                baseline regions grown on the predictive graph
  dyn_randbase  dyn with uniformly random baseline regions
  dyn_faces     dyn + cores and faces (--faces)
  dyn_r2        dyn + round 2 for unresolved residues (--resolve)
Add your own with --extra NAME="--flag value ...".

Usage:
  python -m pdukit.lib.pdu.cog2_fasttest --out DIR [--designs base,scattered]
         [--configs contact,dyn5] [--seeds 0,1] [--jobs 4] [--device cpu]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

COMMON = [
    "--planted",
    "--n-units",
    "6",
    "--unit-size",
    "6",
    "--frames-per-replica",
    "3000",
    "--n-replicas",
    "2",
    "--sizes",
    "2-14",
    "--max-unit",
    "14",
    "--max-size",
    "10",
    "--dec-seeds",
    "8",
    "--dec-iters",
    "4",
    "--n-null",
    "32",
    "--min-supports",
    "20",
    "--max-supports",
    "40",
    "--verify-lags",
    "1",
    "--decoy-pairs",
    "3-4",
]
DESIGNS = {
    "base": [],
    "scattered": ["--scatter"],
    "weak": ["--coupling-gain", "0.20"],
    "strong": ["--coupling-gain", "0.60"],
    "nofaces": ["--face-residues", "0"],
}
CONFIGS = {
    "contact": [
        "--search-graph",
        "contact",
        "--seeding",
        "spread",
        "--baseline",
        "contact",
    ],
    "dyn": [
        "--search-graph",
        "predictive",
        "--graph-k",
        "5",
        "--seeding",
        "random",
        "--dec-seeds",
        "16",
        "--baseline",
        "graph",
    ],
    "dyn_randbase": [
        "--search-graph",
        "predictive",
        "--graph-k",
        "5",
        "--seeding",
        "random",
        "--dec-seeds",
        "16",
        "--baseline",
        "random",
    ],
    "dyn_faces": [
        "--search-graph",
        "predictive",
        "--graph-k",
        "5",
        "--seeding",
        "random",
        "--dec-seeds",
        "16",
        "--baseline",
        "graph",
        "--faces",
    ],
    "dyn_r2": [
        "--search-graph",
        "predictive",
        "--graph-k",
        "5",
        "--seeding",
        "random",
        "--dec-seeds",
        "16",
        "--baseline",
        "graph",
        "--resolve",
    ],
    # before 13 Oct: union merges, no addition gate
    "dyn_r2_old": [
        "--search-graph",
        "predictive",
        "--graph-k",
        "5",
        "--seeding",
        "random",
        "--dec-seeds",
        "16",
        "--baseline",
        "graph",
        "--resolve",
        "--merge-mode",
        "union",
        "--add-gate",
        "none",
    ],
    # 13 Oct defaults + MC stable minimum and write-back
    "dyn_r2_mc": [
        "--search-graph",
        "predictive",
        "--graph-k",
        "5",
        "--seeding",
        "random",
        "--dec-seeds",
        "16",
        "--baseline",
        "graph",
        "--resolve",
        "--stability",
        "--stab-patience",
        "20",
        "--stab-null",
        "0",
    ],
    # 13 Oct MC: J alone decides removals (no direction protection)
    "dyn_r2_mc_noprot": [
        "--search-graph",
        "predictive",
        "--graph-k",
        "5",
        "--seeding",
        "random",
        "--dec-seeds",
        "16",
        "--baseline",
        "graph",
        "--resolve",
        "--stability",
        "--stab-patience",
        "20",
        "--stab-null",
        "0",
        "--stab-protect",
        "none",
    ],
    # 14 Oct (4ff32ae): MC minimises the channel cost J, protected
    "dyn_r2_mc_chan": [
        "--search-graph",
        "predictive",
        "--graph-k",
        "5",
        "--seeding",
        "random",
        "--dec-seeds",
        "16",
        "--baseline",
        "graph",
        "--resolve",
        "--stability",
        "--stab-patience",
        "20",
        "--stab-null",
        "0",
        "--stab-objective",
        "channels",
    ],
}


def run_one(out, design, config, seed, device, extra):
    """Run one design/config/seed through the pipeline in a subprocess."""
    d = Path(out) / f"{design}__{config}__s{seed}"
    d.mkdir(parents=True, exist_ok=True)
    argv = (
        [sys.executable, "-m", "pdukit.lib.pdu.cog2_run"]
        + COMMON
        + DESIGNS[design]
        + extra
        + [
            "--bench-seed",
            str(seed),
            "--seed",
            str(seed),
            "--device",
            device,
            "--outdir",
            str(d),
        ]
    )
    t = time.perf_counter()
    with open(d / "log.txt", "w") as f:
        rc = subprocess.call(argv, stdout=f, stderr=subprocess.STDOUT)
    sec = time.perf_counter() - t
    row = {
        "design": design,
        "config": config,
        "seed": seed,
        "rc": rc,
        "seconds": round(sec, 1),
    }
    try:
        R = json.load(open(d / "results.json"))
    except (OSError, ValueError):
        return row
    vt = R.get("discovery", {}).get("vs_truth") or {}
    st = R.get("discovery", {}).get("status_counts") or {}
    row.update(
        {
            "recovered": vt.get("n_recovered"),
            "n_true": vt.get("n_true"),
            "mean_jaccard": round(vt.get("mean_recall", float("nan")), 3),
            "merged": vt.get("n_merged"),
            "found": vt.get("n_found"),
            "unresolved_groups": st.get("unresolved"),
        }
    )
    pg = R.get("predictive_graph") or {}
    if pg:
        row["edges_are_contacts"] = round(
            pg.get("edges_that_are_contacts", float("nan")), 2
        )
    fc = R.get("faces") or {}
    if fc.get("vs_truth"):
        row["recovered_with_faces"] = fc["vs_truth"]["n_recovered"]
        row["jaccard_with_faces"] = round(fc["vs_truth"]["mean_recall"], 3)
        row["faces_found"] = fc.get("faces_found")
        row["true_faces"] = fc.get("true_faces")
        row["false_faces"] = fc.get("false_faces")
    cv = R.get("discovery", {}).get("channels_vs_truth") or {}
    if cv:
        row["channel_abs_error"] = round(
            cv.get("mean_abs_error", float("nan")), 2
        )
    ud = [e for e in R.get("units_discovery", []) if "cost" in e]
    if ud:
        cs = [float(e["cost"]) for e in ud]
        row["unit_cost_mean"] = round(sum(cs) / len(cs), 3)
        row["units_cost_le0"] = f"{sum(c <= 0 for c in cs)}/{len(cs)}"
    cl = R.get("coverage_labels") or {}
    if cl:
        row["core_or_face"] = cl.get("core_or_face_fraction")
        row["labels"] = "/".join(
            str(cl.get(k, 0))
            for k in ("core", "face", "unresolved", "unassigned")
        )
    sv = (R.get("stability") or {}).get("vs_truth")
    if sv:
        row["recovered_after_mc"] = sv["n_recovered"]
        row["jaccard_after_mc"] = round(sv["mean_recall"], 3)
    fv = (R.get("stability") or {}).get("faces_vs_truth")
    if fv:
        row["mc_faces_found"] = f"{fv['found']}/{fv['true_faces']}"
        row["mc_false_faces"] = fv["false"]
    cc = R.get("cohesion_categories")
    if cc:
        row["categories"] = "/".join(
            f"{k.replace(' ', '_')}:{v}" for k, v in sorted(cc.items())
        )
    ch = R.get("cohesion") or []
    if ch:
        row["cohesion_median"] = round(
            float(sorted(r_["cohesion"] for r_ in ch)[len(ch) // 2]), 3
        )
    dc = (R.get("stability") or {}).get("direction_counts")
    if dc:
        row["directions"] = "/".join(f"{k}:{v}" for k, v in sorted(dc.items()))
    cl = R.get("coverage_labels") or {}
    if cl:
        row["coverage"] = cl.get("core_or_face_fraction")
    rv = (R.get("resolution") or {}).get("vs_truth")
    if rv:
        row["recovered_after_r2"] = rv["n_recovered"]
        row["jaccard_after_r2"] = round(rv["mean_recall"], 3)
    return row


def main(argv=None):
    """Run the fast test grid from the command line."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--designs", default="base,scattered")
    ap.add_argument("--configs", default="contact,dyn,dyn_r2")
    ap.add_argument("--seeds", default="0,1,2,3")
    ap.add_argument(
        "--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 4)
    )
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--extra",
        action="append",
        default=[],
        help='NAME="--flag value ..." adds a config',
    )
    a = ap.parse_args(argv)
    for e in a.extra:
        name, flags = e.split("=", 1)
        CONFIGS[name] = flags.split()
    designs = [x for x in re.split(r"[,+]", a.designs) if x]
    configs = [x for x in re.split(r"[,+]", a.configs) if x]
    seeds = [int(x) for x in re.split(r"[,+]", a.seeds) if x]
    jobs = [(d, c, s) for d in designs for c in configs for s in seeds]
    print(
        f"{len(jobs)} runs, {a.jobs} at a time, device {a.device}", flush=True
    )
    rows = []
    with ThreadPoolExecutor(a.jobs) as ex:
        futs = [
            ex.submit(run_one, a.out, d, c, s, a.device, CONFIGS[c])
            for d, c, s in jobs
        ]
        for f in futs:
            r = f.result()
            rows.append(r)
            print(
                "  " + "  ".join(f"{k}={v}" for k, v in r.items()), flush=True
            )
    Path(a.out).mkdir(parents=True, exist_ok=True)
    keys = sorted(
        {k for r in rows for k in r},
        key=lambda k: (
            ["design", "config", "seed"].index(k)
            if k in ("design", "config", "seed")
            else 9,
            k,
        ),
    )
    with open(Path(a.out) / "fasttest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print("\nsummary (recovered / true, mean Jaccard; summed over seeds)")
    print(f"{'design':10s} " + " ".join(f"{c:>18s}" for c in configs))
    for d in designs:
        cells = []
        for c in configs:
            g = [
                r
                for r in rows
                if r["design"] == d
                and r["config"] == c
                and r.get("recovered") is not None
            ]
            if not g:
                cells.append(f"{'failed':>18s}")
                continue
            rec = sum(r["recovered"] for r in g)
            tot = sum(r["n_true"] for r in g)
            jac = sum(r["mean_jaccard"] for r in g) / len(g)
            cells.append(f"{rec:>3d}/{tot:<3d} J={jac:.2f}".rjust(18))
        print(f"{d:10s} " + " ".join(cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
