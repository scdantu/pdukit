"""Synthetic coordinates with planted units: PDB + XTC with a known answer.

Unlike the named designs (`pdukit.bench.designs`), which hand COG ready-made
features, these are coordinates, so the whole MD path runs on them: the
high-pass filter, the per-residue tICA (K) and everything after.  Used to
test settings that act on coordinates against the truth.

The protein is ``n_units`` compact blobs of ``unit_size`` CA atoms placed on a
3-D lattice, ``spacing`` apart (20 A: blobs at least ~14 A apart, so with
rse-cutoff 12 each residue's environment is its own unit and units are
coupled only through the planted dynamics; at 9 A the environments mix
units and COG found 0 of 6 units).  Each unit has ``modes`` slow latent
processes with the given lifetimes (in frames); every residue follows a
fixed random mix of its unit's processes (displacement along its own
direction), plus fast private noise.  ``couplings`` (source, target) make
the target's processes lag the source's:
x_target(t) += gain * x_source(t - delay).  Write with `write_planted_md`;
the answer is saved as ``truth.json``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from pdukit.bench.demo import write_pdb
from pdukit.numerics.latent import lifetime_to_rho


@dataclass
class PlantedMD:
    """Settings of a synthetic coordinate system (frames are the unit)."""

    n_units: int = 6
    unit_size: int = 10
    modes: int = 3
    lifetimes: tuple = (50.0, 100.0, 300.0)   # frames
    couplings: tuple = ((0, 1), (1, 2), (3, 4))
    gain: float = 0.6
    delay: int = 10
    amplitude: float = 1.0                     # angstrom per unit latent
    noise: float = 0.3                         # angstrom, white
    replicas: int = 4
    frames: int = 10000
    spacing: float = 20.0                      # angstrom between blob centres
    radius: float = 4.0                        # angstrom, blob radius
    extra: dict = field(default_factory=dict)

    def truth(self):
        """Planted units (0-based residues), couplings, settings."""
        units = [list(range(k * self.unit_size, (k + 1) * self.unit_size))
                 for k in range(self.n_units)]
        return {"units": units, "couplings": [list(c) for c in self.couplings],
                "config": asdict(self)}


def _layout(cfg, rng):
    side = int(np.ceil(cfg.n_units ** (1 / 3)))
    grid = np.array([(i, j, k) for i in range(side) for j in range(side)
                     for k in range(side)][: cfg.n_units], float)
    centres = grid * cfg.spacing
    pos = []
    for c in centres:
        v = rng.normal(size=(cfg.unit_size, 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        r = cfg.radius * rng.uniform(0.5, 1.0, size=(cfg.unit_size, 1))
        pos.append(c + v * r)
    return np.concatenate(pos)


def _latents(cfg, rng):
    """(frames, units, modes) slow processes with the planted couplings."""
    T, U, M = cfg.frames, cfg.n_units, cfg.modes
    rho = np.array([lifetime_to_rho(t, 1.0) for t in cfg.lifetimes[:M]])
    z = np.zeros((T, U, M))
    e = rng.normal(size=(T, U, M)) * np.sqrt(1 - rho ** 2)
    for t in range(1, T):
        z[t] = rho * z[t - 1] + e[t]
    for s, g in cfg.couplings:
        z[cfg.delay:, g] += cfg.gain * z[: -cfg.delay, s]
    return z / z.std(axis=0, keepdims=True)


def make_replica(cfg, base, mix, dirs, rng):
    """Return (frames, residues, 3) coordinates for one replica.

    Residue r moves along its own direction dirs[r] by
    amplitude * sum_m mix[r, m] * z[t, unit(r), m], plus white noise.
    """
    z = _latents(cfg, rng)
    unit_of = np.arange(base.shape[0]) // cfg.unit_size
    zr = z[:, unit_of, :]                                  # (T, R, M)
    shift = np.einsum("trm,rm->tr", zr, mix)               # (T, R)
    xyz = base[None] + cfg.amplitude * shift[:, :, None] * dirs[None]
    return xyz + rng.normal(scale=cfg.noise, size=xyz.shape)


def build(cfg, seed=0):
    """Return (base positions, list of replica coordinates, truth)."""
    rng = np.random.default_rng(seed)
    base = _layout(cfg, rng)
    R = base.shape[0]
    mix = rng.normal(size=(R, cfg.modes))
    mix /= np.linalg.norm(mix, axis=1, keepdims=True)
    dirs = rng.normal(size=(R, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    reps = [make_replica(cfg, base, mix, dirs, rng)
            for _ in range(cfg.replicas)]
    return base, reps, {**cfg.truth(), "seed": int(seed)}


def write_planted_md(outdir, cfg=None, seed=0, dt_ps=20.0):
    """Write top.pdb, rep1.xtc ... and truth.json; return the paths."""
    import MDAnalysis

    cfg = cfg or PlantedMD()
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    base, reps, truth = build(cfg, seed)
    write_pdb(out / "top.pdb", base)
    uni = MDAnalysis.Universe(str(out / "top.pdb"))
    paths = []
    for r, xyz in enumerate(reps):
        path = out / f"rep{r + 1}.xtc"
        with MDAnalysis.Writer(str(path), base.shape[0]) as w:
            for t in range(xyz.shape[0]):
                uni.atoms.positions = xyz[t].astype(np.float32)
                uni.trajectory.ts.time = t * dt_ps
                w.write(uni.atoms)
        paths.append(path)
    (out / "truth.json").write_text(json.dumps(truth, indent=1))
    return paths, out / "top.pdb", out / "truth.json"
