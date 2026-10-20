"""Choose mdCATH domains that represent the CATH classes.

Inputs (both are plain files; download them once on a machine with network
access):

* ``mdCATH_domains.txt`` from the Hugging Face dataset
  ``compsciencelab/mdCATH`` (one domain id per line);
* ``cath-domain-list.txt`` from CATH (``download.cathdb.info``, folder
  ``cath/releases/latest-release/cath-classification-data``).  Columns used:
  domain id, Class, Architecture, Topology, Homologous superfamily, and the
  domain length in residues (column 11).

Selection:

1. keep domains that are in mdCATH and have a length in the chosen range;
2. split the quota between classes (default 1 Mainly Alpha 30, 2 Mainly Beta
   30, 3 Alpha Beta 30, 4 Few Secondary Structures 10);
3. inside a class pick greedily, in a seeded random order, preferring a
   domain from an Architecture not used yet, then a Topology not used yet,
   then a Homologous superfamily not used yet; never two domains from the same
   homologous superfamily, so the set is not dominated by one family.

The result is a tab-separated file that the Slurm array script reads.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

CLASS_NAMES = {
    "1": "Mainly Alpha",
    "2": "Mainly Beta",
    "3": "Alpha Beta",
    "4": "Few Secondary Structures",
}


def read_domain_list(path: Path) -> list[str]:
    """Return the domain ids listed in an mdCATH domain file."""
    ids = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            ids.append(line.split()[0])
    return ids


def read_cath(path: Path) -> dict[str, dict]:
    """Return {domain: {C, A, T, H, length}} from cath-domain-list.txt."""
    out = {}
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        f = line.split()
        if len(f) < 11:
            continue
        out[f[0]] = {
            "C": f[1],
            "A": f"{f[1]}.{f[2]}",
            "T": f"{f[1]}.{f[2]}.{f[3]}",
            "H": f"{f[1]}.{f[2]}.{f[3]}.{f[4]}",
            "length": int(f[10]),
        }
    return out


def pick(cands: list[str], info: dict, quota: int, rng: random.Random):
    """Pick ``quota`` domains from ``cands``, spread over A, T and H."""
    pool = list(cands)
    rng.shuffle(pool)
    chosen, seen_a, seen_t, seen_h = [], set(), set(), set()
    while pool and len(chosen) < quota:
        best = None
        for d in pool:
            c = info[d]
            if c["H"] in seen_h:
                continue
            score = (c["A"] not in seen_a) * 2 + (c["T"] not in seen_t)
            if best is None or score > best[0]:
                best = (score, d)
                if score == 3:
                    break
        if best is None:
            break
        d = best[1]
        chosen.append(d)
        c = info[d]
        seen_a.add(c["A"])
        seen_t.add(c["T"])
        seen_h.add(c["H"])
        pool = [x for x in pool if info[x]["H"] != c["H"]]
    return chosen


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--mdcath-list", type=Path, required=True)
    p.add_argument("--cath-list", type=Path, required=True)
    p.add_argument("-o", "--out", type=Path, default=Path("selected.tsv"))
    p.add_argument("--quota", default="1:30,2:30,3:30,4:10",
                   help="class:count pairs")
    p.add_argument("--min-len", type=int, default=80)
    p.add_argument("--max-len", type=int, default=250)
    p.add_argument("--seed", type=int, default=0)
    return p


def main(argv=None) -> int:
    """Write the selection table and print a summary."""
    args = build_parser().parse_args(argv)
    quota = {k: int(v) for k, v in
             (x.split(":") for x in args.quota.split(","))}
    in_md = set(read_domain_list(args.mdcath_list))
    cath = read_cath(args.cath_list)
    info = {d: cath[d] for d in in_md if d in cath
            and args.min_len <= cath[d]["length"] <= args.max_len}
    missing = len(in_md) - sum(d in cath for d in in_md)
    print(f"mdCATH domains {len(in_md)}; with CATH record "
          f"{len(in_md) - missing}; in length range {len(info)}")
    by_class = defaultdict(list)
    for d, c in info.items():
        by_class[c["C"]].append(d)
    rng = random.Random(args.seed)
    rows = []
    for cls, n in sorted(quota.items()):
        got = pick(sorted(by_class.get(cls, [])), info, n, rng)
        print(f"class {cls} ({CLASS_NAMES.get(cls, '?')}): "
              f"{len(got)}/{n} chosen from {len(by_class.get(cls, []))}")
        rows += [(d, info[d]) for d in got]
    with args.out.open("w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["domain", "class", "architecture", "topology",
                    "homology", "length"])
        for d, c in rows:
            w.writerow([d, c["C"], c["A"], c["T"], c["H"], c["length"]])
    print(f"wrote {len(rows)} domains to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
