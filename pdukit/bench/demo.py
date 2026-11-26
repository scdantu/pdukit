"""A small synthetic protein with a known answer, for demos and tests.

A 40-residue CA-only helix whose residues move in four coupled blocks of ten
(the planted units).  Each block follows its own slow random process; block 0
drives block 1 with a delay.  `write_demo` writes ``top.pdb`` and one XTC per
replica; ``Ensemble.demo()`` wraps it.  Needs MDAnalysis (``pdukit[xtc]``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

N_RES = 40
UNIT_SIZE = 10
AMP = 0.12
NOISE = 0.1


def helix(n):
    """Return ideal alpha-helix CA positions in angstrom."""
    i = np.arange(n)
    ang = np.deg2rad(100.0) * i
    return np.c_[2.3 * np.cos(ang), 2.3 * np.sin(ang), 1.5 * i]


def write_pdb(path, xyz):
    """Write a CA-only PDB, residues ALA 1..n, chain A."""
    lines = []
    for i, (x, y, z) in enumerate(xyz):
        lines.append(
            f"ATOM  {i + 1:5d}  CA  ALA A{i + 1:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C"
        )
    Path(path).write_text("\n".join(lines) + "\nEND\n")


def latent(n_frames, n_units, rng, tau=60.0, delay=8, gain=0.8):
    """Slow random processes per unit; unit 0 drives unit 1 after ``delay``."""
    a = np.exp(-1.0 / tau)
    z = np.zeros((n_frames, n_units))
    noise = rng.normal(size=(n_frames, n_units)) * np.sqrt(1 - a * a)
    for t in range(1, n_frames):
        z[t] = a * z[t - 1] + noise[t]
    z[delay:, 1] += gain * z[:-delay, 0]
    return z


def make_replica(base, dirs, n_frames, rng):
    """Return ``(n_frames, n_res, 3)`` coordinates for one replica.

    Each block breathes: its residues move in and out from the block centre by
    the amplitude of the block's latent process (so distances inside the block
    change together), plus independent noise.  ``dirs`` adds a small private
    wobble per residue.
    """
    n_units = base.shape[0] // UNIT_SIZE
    z = latent(n_frames, n_units, rng)
    unit_of = np.arange(base.shape[0]) // UNIT_SIZE
    centre = np.stack([base[unit_of == k].mean(0) for k in range(n_units)])[
        unit_of
    ]
    scale = 1.0 + AMP * z[:, unit_of][:, :, None]
    xyz = (
        centre[None]
        + (base - centre)[None] * scale
        + 0.3 * dirs[None] * z[:, unit_of][:, :, None]
    )
    return xyz + rng.normal(scale=NOISE, size=xyz.shape)


def write_demo(outdir, replicas=3, frames=1500, dt_ps=20.0, seed=0):
    """Write ``top.pdb`` and ``rep1.xtc`` ... into ``outdir``.

    Returns ``(trajectory paths, topology path)``.
    """
    try:
        import MDAnalysis
    except ImportError as exc:
        raise ImportError(
            'the demo needs MDAnalysis: pip install "pdukit[xtc]"'
        ) from exc
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    base = helix(N_RES)
    write_pdb(out / "top.pdb", base)
    dirs = rng.normal(size=(N_RES, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    uni = MDAnalysis.Universe(str(out / "top.pdb"))
    paths = []
    for r in range(replicas):
        xyz = make_replica(base, dirs, frames, rng)
        path = out / f"rep{r + 1}.xtc"
        with MDAnalysis.Writer(str(path), N_RES) as w:
            for t in range(frames):
                uni.atoms.positions = xyz[t].astype(np.float32)
                uni.trajectory.ts.time = t * dt_ps
                w.write(uni.atoms)
        paths.append(path)
    return paths, out / "top.pdb"
