#!/usr/bin/env python3
"""Prepare one DyNoPy beta-lactamase system for `pdu run`.

Systems: KPC-2, PDC-3, SHV-1, TEM-1.  The DyNoPy trajectories are
adaptive-sampling data: many short runs written back to back, so one xtc
holds hundreds of segments whose time axis resets at every seam.  `pdu run`
treats each file as one continuous replica, so a pair of frames straddling a
seam would be a false transition.  This script cuts every segment into its
own xtc and writes a CA-only topology:

    <out>/topology.pdb   CA atoms, first frame
    <out>/seg_00001.xtc  ...  one file per segment (CA only)
    <out>/segments.json  frames per segment, source files, how cuts were found

A cut is made at every file boundary and wherever the time stamp does not
increase.  Some files have a wrong dt (SHV-1 reports 1,000,000 ps) but the
resets are exact.  If a file has no resets and its length is a multiple of
--seg-len, it is cut every --seg-len frames.

    python split_segments.py --root DyNoPy_Data --system TEM1 -o out/TEM1
    python split_segments.py --root DyNoPy_Data --system TEM1 --dry-run
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

PATTERNS = {  # case-insensitive match on the path relative to --root
    "KPC2": r"kpc[-_ ]?2",
    "PDC3": r"pdc[-_ ]?3",
    "SHV1": r"shv[-_ ]?1",
    "TEM1": r"tem[-_ ]?1",
}
TOP_EXT = (".prmtop", ".parm7", ".top", ".pdb", ".gro")


def find_files(root: Path, system: str):
    """Return (topology candidates, sorted xtc files) for a system."""
    pat = re.compile(PATTERNS.get(system.upper(), re.escape(system)), re.I)
    allf = [
        p
        for p in root.rglob("*")
        if p.is_file() and pat.search(str(p.relative_to(root)))
    ]
    xtc = sorted(
        (p for p in allf if p.suffix == ".xtc"),
        key=lambda p: [
            int(t) if t.isdigit() else t for t in re.split(r"(\d+)", str(p))
        ],
    )
    tops = [p for p in allf if p.suffix in TOP_EXT]
    tops.sort(key=lambda p: (TOP_EXT.index(p.suffix), len(str(p))))
    return tops, xtc


def cut_points(times: np.ndarray, seg_len: int | None):
    """Return the indices where a new segment starts (always includes 0)."""
    cuts = [0] + [int(i) for i in np.nonzero(np.diff(times) <= 0)[0] + 1]
    how = "time reset"
    if (
        len(cuts) == 1
        and seg_len
        and len(times) > seg_len
        and len(times) % seg_len == 0
    ):
        cuts = list(range(0, len(times), seg_len))
        how = f"fixed length {seg_len}"
    return cuts, how


def main() -> int:
    """Command line entry point."""
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--root", type=Path, required=True, help="DyNoPy_Data directory"
    )
    ap.add_argument("--system", required=True, help="KPC2, PDC3, SHV1 or TEM1")
    ap.add_argument("-o", "--out", type=Path)
    ap.add_argument(
        "--topology",
        type=Path,
        help="override the topology found under --root",
    )
    ap.add_argument(
        "--seg-len", type=int, help="fallback segment length in frames"
    )
    ap.add_argument(
        "--min-frames",
        type=int,
        default=10,
        help="drop segments shorter than this",
    )
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    tops, xtcs = find_files(a.root, a.system)
    top = a.topology or (tops[0] if tops else None)
    print(f"[{a.system}] topology candidates: {[str(t) for t in tops[:5]]}")
    print(f"[{a.system}] using topology: {top}")
    names = [str(x.relative_to(a.root)) for x in xtcs[:6]]
    more = " ..." if len(xtcs) > 6 else ""
    print(f"[{a.system}] {len(xtcs)} xtc file(s): {names}{more}")
    if top is None or not xtcs:
        print(
            "nothing usable found: check --root and the system name",
            file=sys.stderr,
        )
        return 1
    if a.dry_run:
        return 0
    if a.out is None:
        ap.error("-o/--out is required unless --dry-run")

    import warnings

    warnings.filterwarnings("ignore")
    import MDAnalysis as mda  # noqa: N813
    from MDAnalysis.coordinates.XTC import XTCWriter

    a.out.mkdir(parents=True, exist_ok=True)
    wrote_top = False
    seg_lengths, sources, hows = [], [], []
    n = 0
    for xf in xtcs:
        u = mda.Universe(str(top), str(xf))
        ca = u.select_atoms("protein and name CA")
        if ca.n_atoms == 0:
            print(f"no protein CA atoms in {top}", file=sys.stderr)
            return 1
        times = np.array([ts.time for ts in u.trajectory])
        cuts, how = cut_points(times, a.seg_len)
        bounds = list(zip(cuts, cuts[1:] + [len(times)]))
        hows.append(f"{xf.name}: {len(bounds)} segments by {how}")
        for b0, b1 in bounds:
            if b1 - b0 < a.min_frames:
                continue
            n += 1
            with XTCWriter(
                str(a.out / f"seg_{n:05d}.xtc"), n_atoms=ca.n_atoms
            ) as w:
                for fi in range(b0, b1):
                    u.trajectory[fi]
                    w.write(ca)
            if not wrote_top:
                u.trajectory[b0]
                ca.write(str(a.out / "topology.pdb"))
                wrote_top = True
            seg_lengths.append(b1 - b0)
            sources.append(xf.name)
        print(hows[-1], flush=True)

    L = np.array(seg_lengths)
    info = {
        "system": a.system,
        "topology": str(top),
        "n_segments": int(n),
        "n_frames": int(L.sum()),
        "frames_per_segment_min": int(L.min()),
        "frames_per_segment_median": float(np.median(L)),
        "frames_per_segment_max": int(L.max()),
        "cuts": hows,
        "segment_sources": sources,
        "segment_lengths": seg_lengths,
    }
    (a.out / "segments.json").write_text(json.dumps(info, indent=1))
    print(
        f"[{a.system}] {n} segments, {int(L.sum())} frames "
        f"(per segment {L.min()}-{L.max()}, median {np.median(L):.0f})"
    )
    print(
        "expected from the eLife methods / earlier checks: "
        "TEM1 404x120, KPC2 364x100, "
        "SHV1 593x60, PDC3 30x1000"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
