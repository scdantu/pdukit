"""Reading local trajectories for `pdu run`.

Until 1.0.2a this module held the mdCATH pilot (Gate A measurement:
run_domain, power_scan and a CLI).  Nothing called it any more, so only the
trajectory loader and its two helpers remain; the pilot is in the tag
archive/pre-1.0.2a.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

_LAST_TICK = {}


def _progress(done, total, prefix, t0, verbose=True):
    """Progress counter in the style already used elsewhere in pdukit (a
    carriage-return line, no tqdm dependency).

    On a terminal it redraws one line.  When stdout is a file -- a
    batch-scheduler job log, say -- carriage returns are useless, so it falls
    back to one plain line per quarter, which keeps logs readable instead of
    producing a single enormous line.
    """
    if not verbose or not total:
        return
    frac = done / total
    w = len(str(total))
    if sys.stdout.isatty():
        bar_w = 24
        filled = int(round(bar_w * frac))
        sys.stdout.write(
            f"\r    {prefix} [{'#' * filled}{'-' * (bar_w - filled)}] "
            f"{done:<{w}}/{total}, {frac:.0%}  {time.time() - t0:5.1f}s"
        )
        sys.stdout.flush()
        if done >= total:
            sys.stdout.write("\n")
            sys.stdout.flush()
    else:
        step = max(1, total // 4)
        if done >= total or done % step == 0:
            if _LAST_TICK.get(prefix) != done:
                _LAST_TICK[prefix] = done
                print(
                    f"    {prefix} {done:<{w}}/{total}, {frac:.0%}  "
                    f"{time.time() - t0:5.1f}s",
                    flush=True,
                )


def _log(msg, verbose=True):
    if verbose:
        print(msg, flush=True)


def load_xtc_segments(
    topology,
    replica_paths,
    stride=1,
    max_frames=None,
    atom_selection="CA",
    verbose=True,
):
    """Load local GROMACS replicas into (coords, segments, meta).

    Each replica becomes its own segment, so no lagged pair is ever formed
    across a replica seam -- the same discipline `koopman.tica_multi` applies
    to mdCATH replicates, and it matters more here: these are independent runs
    from different seeds, not one continuous trajectory.

    A replica may be given as SEVERAL files (a list, or one string with the
    parts separated by commas): they are read in the order given as one
    continuous trajectory (MDAnalysis ChainReader, continuous=True, so a
    frame repeated at a restart boundary is read once).

    Needs MDAnalysis (`pip install MDAnalysis`, or the repo's `.[xtc]` extra).
    """
    try:
        import MDAnalysis
    except ImportError as exc:  # noqa: BLE001
        raise ImportError(
            "MDAnalysis is required to read local trajectories: "
            "pip install MDAnalysis"
        ) from exc
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning)

    sel = {"CA": "protein and name CA", "all": "protein"}.get(
        atom_selection, atom_selection
    )
    blocks, segments, cursor, meta = [], [], 0, []
    t0 = time.time()
    _log(f"  loading {len(replica_paths)} replica(s) via MDAnalysis", verbose)
    for ri, path in enumerate(replica_paths, 1):
        parts = (
            list(path)
            if isinstance(path, (list, tuple))
            else [p_ for p_ in str(path).split(",") if p_]
        )
        if len(parts) > 1:
            u = MDAnalysis.Universe(topology, parts, continuous=True)
        else:
            u = MDAnalysis.Universe(topology, parts[0])
        atoms = u.select_atoms(sel)
        n_total = len(u.trajectory)
        frames = range(0, n_total, max(1, int(stride)))
        if max_frames:
            frames = list(frames)[: int(max_frames)]
        frames = list(frames)
        X = np.empty((len(frames), atoms.n_atoms, 3), dtype=np.float32)
        for k, fi in enumerate(frames):
            u.trajectory[fi]
            X[k] = atoms.positions
            if k % 2000 == 0:
                _progress(
                    cursor + k,
                    cursor + len(frames),
                    f"rep {ri}/{len(replica_paths)}",
                    t0,
                    verbose,
                )
        _progress(1, 1, f"rep {ri}/{len(replica_paths)}", t0, verbose)
        blocks.append(X)  # keep float32: 114k frames is 450 MB in f64
        segments.append((cursor, cursor + len(frames)))
        cursor += len(frames)
        dt_ps = float(u.trajectory.dt) * max(1, int(stride))
        meta.append(
            {
                "replicate": str(ri - 1),
                "n_frames": len(frames),
                "dt_ps": dt_ps,
                "source": ",".join(parts),
                "n_parts": len(parts),
            }
        )
        _log(
            f"    {Path(parts[0]).name}"
            + (f" (+{len(parts) - 1} parts)" if len(parts) > 1 else "")
            + f": {len(frames)} frames, dt {dt_ps:.1f} ps, "
            f"{atoms.n_atoms} atoms",
            verbose,
        )
    shapes = {b.shape[1] for b in blocks}
    if len(shapes) != 1:
        raise ValueError(f"replicas disagree on atom count {shapes}")
    total = sum(b.shape[0] for b in blocks)
    _log(
        f"    {total} frames total "
        f"({total * meta[0]['dt_ps'] / 1000:.0f} ns aggregate)",
        verbose,
    )
    return np.concatenate(blocks, axis=0), segments, meta
