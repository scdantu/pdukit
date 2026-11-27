"""Write synthetic coordinates with planted units (PDB + XTC + truth.json).

    python tools/make_planted_md.py data/planted --seed 0
    pdu run -f data/planted/rep*.xtc -s data/planted/top.pdb -p my.ppf -o out
    pdu score-bench out -o tables --truth data/planted/truth.json

See `pdukit.bench.coords` for the system.  Needs MDAnalysis.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pdukit.bench.coords import PlantedMD, write_planted_md  # noqa: E402


def main(argv=None):
    """Write the files; print where."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("outdir")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--replicas", type=int, default=4)
    ap.add_argument("--frames", type=int, default=10000)
    ap.add_argument("--units", type=int, default=6)
    a = ap.parse_args(argv)
    cfg = PlantedMD(replicas=a.replicas, frames=a.frames, n_units=a.units)
    paths, top, truth = write_planted_md(a.outdir, cfg, seed=a.seed)
    print(f"wrote {top}, {len(paths)} XTC files and {truth}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
