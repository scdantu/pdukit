"""Write a small synthetic protein as PDB + XTC files for the tutorial.

The "protein" is a 40-residue CA-only helix whose residues move in four
coupled blocks of ten (the planted units); see `pdukit.bench.demo`.

    python tools/make_demo_trajectories.py demo_data
    pdu run -f demo_data/rep*.xtc -s demo_data/top.pdb \\
        -p pdukit/params/tutorial.ppf -o out/demo

Needs MDAnalysis (``pip install "pdukit[xtc]"``).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run from a checkout

from pdukit.bench.demo import write_demo  # noqa: E402


def main(argv=None):
    """Write ``top.pdb`` and ``rep1.xtc`` ... into the output folder."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("outdir")
    ap.add_argument("--replicas", type=int, default=3)
    ap.add_argument("--frames", type=int, default=1500)
    ap.add_argument("--dt-ps", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    try:
        write_demo(args.outdir, args.replicas, args.frames, args.dt_ps,
                   args.seed)
    except ImportError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"wrote top.pdb and {args.replicas} XTC files to {args.outdir}")
    print("planted units: residues 1-10, 11-20, 21-30, 31-40 (1 drives 2)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
