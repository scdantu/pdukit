"""Check a BioEmu2 rollout folder for trajectory-like behaviour.

    python check_rollout.py OUTDIR

Reads ``OUTDIR/topology.pdb`` and ``OUTDIR/rep_*.xtc`` (no GPU needed).
Two tests, each against a null in which the replica labels of frame k+1 are
randomly permuted (what a scrambled output order would give):

1. lag-1 autocorrelation of the leading principal components of the
   C-alpha coordinates, pooled over replicas.  If each frame depends on the
   previous frame of its own replica the autocorrelation is above the null.
2. mean RMSD between a frame and the next frame of the same replica.

It also reports the lag-1 autocorrelation of every principal component so
that you can see how much memory the 1 ns step keeps.  Compare the same
numbers from MD strided to 1 ns.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def load_ca(outdir: Path) -> np.ndarray:
    """Return C-alpha coordinates (replica, frame, atom, 3) in nm."""
    import mdtraj as md

    top = md.load_topology(str(outdir / "topology.pdb"))
    ca = top.select("name CA")
    reps = [md.load_xtc(str(f), top=top).xyz[:, ca, :]
            for f in sorted(outdir.glob("rep_*.xtc"))]
    n = min(len(r) for r in reps)
    return np.stack([r[:n] for r in reps])


def fit_to(ref: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Return x (frames, atoms, 3) rigidly fitted onto ref (atoms, 3)."""
    ref = ref - ref.mean(0)
    out = np.empty_like(x)
    for f in range(len(x)):
        a = x[f] - x[f].mean(0)
        u, _, vt = np.linalg.svd(a.T @ ref)
        d = np.sign(np.linalg.det(u @ vt))
        out[f] = a @ (u @ np.diag([1, 1, d]) @ vt)
    return out


def main(argv=None) -> int:
    """Run the tests and print a JSON report."""
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("outdir", type=Path)
    p.add_argument("--n-pc", type=int, default=5)
    p.add_argument("--n-null", type=int, default=500)
    args = p.parse_args(argv)
    xyz = load_ca(args.outdir)
    n_rep, n_fr = xyz.shape[:2]
    flat = xyz.reshape(n_rep * n_fr, -1, 3)
    ref = flat[0]
    fitted = fit_to(ref, flat)
    for _ in range(2):                       # refit onto the mean once
        fitted = fit_to(fitted.mean(0), fitted)
    x = fitted.reshape(n_rep * n_fr, -1)
    x = x - x.mean(0)
    _, sv, vt = np.linalg.svd(x, full_matrices=False)
    npc = min(args.n_pc, len(sv))
    pcs = (x @ vt[:npc].T).reshape(n_rep, n_fr, npc)
    rng = np.random.default_rng(0)

    def lag1(z, perm=None):
        a = z[:, :-1, :]
        b = z[:, 1:, :]
        if perm is not None:
            b = np.stack([b[perm[k], k] for k in range(b.shape[1])], axis=1)
        a = a.reshape(-1, z.shape[2])
        b = b.reshape(-1, z.shape[2])
        return (a * b).sum(0) / np.sqrt((a * a).sum(0) * (b * b).sum(0))

    obs = lag1(pcs)
    nulls = np.empty((args.n_null, npc))
    for m in range(args.n_null):
        perm = [rng.permutation(n_rep) for _ in range(n_fr - 1)]
        nulls[m] = lag1(pcs, perm)
    p_pc = [(1 + (nulls[:, c] >= obs[c]).sum()) / (1 + args.n_null)
            for c in range(npc)]
    d_same = np.linalg.norm(fitted.reshape(n_rep, n_fr, -1)[:, 1:]
                            - fitted.reshape(n_rep, n_fr, -1)[:, :-1],
                            axis=2).mean()
    report = {
        "n_replicas": n_rep, "frames_per_replica": n_fr,
        "pc_variance_fraction": (sv[:npc] ** 2 / (sv ** 2).sum()).round(3)
        .tolist(),
        "lag1_autocorrelation_pc": obs.round(3).tolist(),
        "null_mean_pc": nulls.mean(0).round(3).tolist(),
        "p_pc": [round(float(v), 4) for v in p_pc],
        "mean_step_distance_nm_euclid": float(d_same),
        "pairing_ok": bool(min(p_pc) < 0.05),
    }
    print(json.dumps(report, indent=2))
    return 0 if report["pairing_ok"] else 2


if __name__ == "__main__":
    sys.exit(main())
