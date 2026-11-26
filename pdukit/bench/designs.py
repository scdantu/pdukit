"""Named synthetic systems with a known answer (the manuscript test suite).

Each design is a small "protein" built from linear latent dynamics (see
`pdukit.numerics.latent`): residues read the hidden coordinates of their
unit, units are joined by directed couplings, and the answer (which
residues form which unit) is stored with the data.  COG runs on them
unchanged through ``pdu bench --set bench-design=<name>``.

One frame is 0.1 ns in the 100-residue designs: a mode with lifetime t ns
has per-frame autocorrelation exp(-0.1 / t).

100 residues, 9 units of 4-19 residues (``units`` family):

    units     units joined by 11 directed couplings (Test 1, positive)
    mixed     same dynamics, every residue reads ALL units (Test 1, negative)
    nocomm    units with every coupling removed (Test 4)
    medium    uniform medium: one latent per residue on a 5 x 5 x 4 lattice,
              equal nearest-neighbour coupling, no groups (Test 1, negative)
    tau       processes from 0.1 to 10 ns: two parents with sub-units, a
              persistent unit, a fast-only and a slow-only unit (Tests 2-3)

36 residues, 6 units of 6 (``small-`` family, the first suite):

    small-units      planted system, ring of couplings
    small-nocomm     the same without couplings
    small-mixed      the same dynamics, every residue reads all units
    small-medium     uniform medium on a 4 x 3 x 3 lattice
    small-hierarchy  slow units with fast sub-units + a persistent unit
                     (30 residues)

Every ``build(seed)`` draws from one generator seeded with ``seed`` in a
fixed order, so the same design and seed give the same system.
"""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar

import numpy as np

from pdukit.bench.planted import planted_system
from pdukit.errors import ConfigError
from pdukit.numerics.geometry import (
    ball_layout,
    connect_components,
    contact_adjacency,
    lattice_layout,
)
from pdukit.numerics.latent import (
    add_coupling,
    block_diagonal,
    lifetime_to_rho,
    normal_block,
    orthonormal_columns,
    read_out,
    simulate_var,
    spectral_radius,
    stabilise,
    unit_vector,
)

DT_NS = 0.1
UNIT_SIZES = (4, 6, 8, 10, 11, 12, 14, 16, 19)  # sums to 100
COUPLINGS = (  # (source, target, rank)
    (0, 1, 1),
    (1, 2, 1),
    (2, 3, 2),
    (3, 4, 1),
    (4, 5, 1),
    (5, 6, 2),
    (6, 7, 1),
    (7, 8, 1),
    (8, 0, 1),
    (2, 6, 1),
    (4, 1, 1),
)
TAU_GROUPS = (  # residue group, size, latent blocks it reads
    ("P1a", 10, ("P1slow", "P1a")),
    ("P1b", 10, ("P1slow", "P1b")),
    ("P2a", 8, ("P2slow", "P2a")),
    ("P2b", 8, ("P2slow", "P2b")),
    ("PERS", 14, ("PERS",)),
    ("FAST", 10, ("FAST",)),
    ("SLOW", 14, ("SLOW",)),
    ("M1", 14, ("M1",)),
    ("M2", 12, ("M2",)),
)
TAU_LIFETIMES_NS = {  # latent block: lifetime of each of its modes (ns)
    "P1slow": (8.0, 10.0, 12.0),
    "P1a": (0.3, 0.3, 0.3),
    "P1b": (0.3, 0.3, 0.3),
    "P2slow": (2.5, 3.0, 3.5),
    "P2a": (1.0, 1.0, 1.0),
    "P2b": (1.0, 1.0, 1.0),
    "PERS": tuple(float(x) for x in np.geomspace(0.1, 10, 6)),
    "FAST": (0.1, 0.15, 0.2),
    "SLOW": (8.0, 10.0, 12.0),
    "M1": (0.8, 1.0, 1.2),
    "M2": (0.8, 1.0, 1.2),
}
TAU_COUPLINGS = (  # (source block, target block), rank 1
    ("P1slow", "P2slow"),
    ("P2slow", "PERS"),
    ("PERS", "SLOW"),
    ("SLOW", "M1"),
    ("M1", "M2"),
    ("M2", "P1slow"),
    ("FAST", "M1"),
)
TAU_PARENTS = {"P1": ("P1a", "P1b"), "P2": ("P2a", "P2b")}


class DesignName(str, Enum):
    """The synthetic systems ``pdu bench`` can build (``bench-design``)."""

    PLANTED = "planted"
    UNITS = "units"
    MIXED = "mixed"
    NOCOMM = "nocomm"
    MEDIUM = "medium"
    TAU = "tau"
    SMALL_UNITS = "small-units"
    SMALL_MIXED = "small-mixed"
    SMALL_NOCOMM = "small-nocomm"
    SMALL_MEDIUM = "small-medium"
    SMALL_HIERARCHY = "small-hierarchy"

    def __str__(self):
        """Return the plain word."""
        return self.value


@dataclass
class SyntheticSystem:
    """A generated system: features, geometry and the planted answer.

    feats      (replicas * frames, residues, features), float32
    segments   [(start, end)] frame range of each replica
    coords     (residues, 3) one static structure
    adjacency  (residues, residues) bool contact graph (connected)
    truth      the answer: ``units`` (lists of residues) and design extras
    config     how it was made (design name, seed, parameters)
    """

    feats: np.ndarray
    segments: list
    coords: np.ndarray
    adjacency: np.ndarray
    truth: dict
    config: dict

    TRUTH_DEFAULTS: ClassVar[dict] = {
        "units": [],
        "home_units": [],
        "couplings": [],
        "face_residues": [],
        "face_owners": {},
        "decoy_pairs": [],
        "latent_dim": 0,
        "unit_of_latent": [],
    }

    @classmethod
    def from_replicas(cls, feats, coords, adjacency, truth, config):
        """Stack per-replica feature arrays and fill the truth defaults."""
        n = feats[0].shape[0]
        return cls(
            feats=np.concatenate(feats, 0).astype(np.float32),
            segments=[(r * n, (r + 1) * n) for r in range(len(feats))],
            coords=coords,
            adjacency=adjacency,
            truth={**cls.TRUTH_DEFAULTS, **truth},
            config=config,
        )

    def __post_init__(self):
        """Record the contact graph in the answer (as residue pairs).

        Scoring compares found cores with compact regions grown on these
        contacts, so the run's results carry them.
        """
        if "contacts" not in self.truth:
            pairs = np.argwhere(np.triu(self.adjacency, 1))
            self.truth = {**self.truth, "contacts": pairs.tolist()}

    def as_dict(self):
        """Return the dict layout of `planted_system` (what COG reads)."""
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class Design(ABC):
    """A recipe for a synthetic system; ``build(seed)`` makes one."""

    name: DesignName
    obs_noise: float = 0.25
    n_frames: int = 5000
    n_replicas: int = 2
    m: int = 4

    @abstractmethod
    def build(self, seed):
        """Generate the system for one random seed."""

    def _config(self, seed, **extra):
        return {
            "design": str(self.name),
            "seed": int(seed),
            **{
                f.name: _plain(getattr(self, f.name))
                for f in _fields(self)
                if f.name != "name"
            },
            **extra,
        }


@dataclass(frozen=True)
class UnitsDesign(Design):
    """Units of varied size joined by directed couplings (100 residues).

    Each unit has ``latent_dim`` modes with lifetimes drawn uniformly
    between ``lifetime_ns``; couplings are (source, target, rank) with
    strength ``gain``.  ``mix_all`` makes every residue read every unit
    (no units exist; the answer is empty).
    """

    unit_sizes: tuple = UNIT_SIZES
    couplings: tuple = COUPLINGS
    mix_all: bool = False
    gain: float = 0.15
    latent_dim: int = 5
    lifetime_ns: tuple = (0.4, 1.2)
    dt_ns: float = DT_NS
    burn_in: int = 2000

    def build(self, seed):
        """Generate the system for one random seed."""
        rng = np.random.default_rng(seed)
        lo, hi = (lifetime_to_rho(t, self.dt_ns) for t in self.lifetime_ns)
        blocks = [
            normal_block(rng.uniform(lo, hi, self.latent_dim), rng)
            for _ in self.unit_sizes
        ]
        K, L, off = block_diagonal(blocks)
        for s, t, r in self.couplings:
            U = orthonormal_columns(self.latent_dim, r, rng)
            V = orthonormal_columns(self.latent_dim, r, rng)
            add_coupling(K, off, s, t, U, V, self.gain)
        K, rescaled = stabilise(K, 0.999, 0.998)
        groups = _consecutive_groups(self.unit_sizes)
        residue_blocks = [[a] for a, g in enumerate(groups) for _ in g]
        trajs = simulate_var(
            K, L, self.n_frames, self.n_replicas, rng, self.burn_in
        )
        feats = read_out(
            trajs,
            residue_blocks,
            off,
            self.m,
            self.obs_noise,
            rng,
            mix_all=self.mix_all,
        )
        coords, adj = ball_layout(groups, rng)
        truth = {"design": str(self.name)}
        if self.mix_all:
            truth.update(units=[], reference_units=groups)
        else:
            truth.update(units=groups)
        if self.couplings:
            truth["couplings"] = [list(c) for c in self.couplings]
        config = self._config(
            seed, spectral_radius=spectral_radius(K), rescaled=rescaled
        )
        return SyntheticSystem.from_replicas(feats, coords, adj, truth, config)


@dataclass(frozen=True)
class MediumDesign(Design):
    """A uniform medium: no units, only equal nearest-neighbour coupling.

    One latent of dimension ``latent_dim`` per residue on a jittered
    lattice; each couples with strength ``gain`` / max degree to every
    lattice neighbour within 8 A.
    """

    dims: tuple = (5, 5, 4)
    spacing: float = 5.0
    latent_dim: int = 2
    self_rho: float = lifetime_to_rho(0.6, DT_NS)
    gain: float = 0.35
    burn_in: int = 2000

    def build(self, seed):
        """Generate the system for one random seed."""
        rng = np.random.default_rng(seed)
        coords, adj = lattice_layout(self.dims, self.spacing, rng)
        N, d = len(coords), self.latent_dim
        A = self.self_rho * np.eye(N) + self.gain * adj / adj.sum(1).max()
        K, rescaled = stabilise(np.kron(A, np.eye(d)), 0.98, 0.97)
        trajs = simulate_var(
            K, np.ones(N * d), self.n_frames, self.n_replicas, rng,
            self.burn_in,
        )
        feats = read_out(
            trajs,
            [[i] for i in range(N)],
            np.arange(0, N * d + 1, d),
            self.m,
            self.obs_noise,
            rng,
        )
        truth = {"units": [], "design": str(self.name)}
        config = self._config(
            seed, spectral_radius=spectral_radius(K), rescaled=rescaled
        )
        return SyntheticSystem.from_replicas(feats, coords, adj, truth, config)


@dataclass(frozen=True)
class TimescaleDesign(Design):
    """Units whose processes span 0.1-10 ns, for the lag-time scan.

    Groups read latent blocks (``groups``); block lifetimes in ns are in
    ``lifetimes_ns``; rank-1 couplings join blocks (``couplings``).  A
    parent (``parents``) is the union of sub-units that share a slow block.
    The answer has ``fine_units`` (every group), ``coarse_units`` (parents
    and the rest) and ``named`` (each group and each parent by name).
    """

    n_frames: int = 20000
    groups: tuple = TAU_GROUPS
    lifetimes_ns: dict = field(default_factory=lambda: dict(TAU_LIFETIMES_NS))
    couplings: tuple = TAU_COUPLINGS
    parents: dict = field(default_factory=lambda: dict(TAU_PARENTS))
    gain: float = 0.02
    dt_ns: float = DT_NS
    burn_in: int = 5000

    def build(self, seed):
        """Generate the system for one random seed."""
        rng = np.random.default_rng(seed)
        names = list(self.lifetimes_ns)
        idx = {n: k for k, n in enumerate(names)}
        blocks = [
            normal_block(
                [lifetime_to_rho(t, self.dt_ns) for t in self.lifetimes_ns[n]],
                rng,
            )
            for n in names
        ]
        K, L, off = block_diagonal(blocks)
        sizes = [len(self.lifetimes_ns[n]) for n in names]
        for src, dst in self.couplings:
            s, t = idx[src], idx[dst]
            U = orthonormal_columns(sizes[t], 1, rng)
            V = orthonormal_columns(sizes[s], 1, rng)
            add_coupling(K, off, s, t, U, V, self.gain)
        K, rescaled = stabilise(K, 0.999, 0.998)
        groups = _consecutive_groups([size for _, size, _ in self.groups])
        residue_blocks, labels = [], []
        for name, size, reads in self.groups:
            residue_blocks += [[idx[b] for b in reads]] * size
            labels += [name] * size
        trajs = simulate_var(
            K, L, self.n_frames, self.n_replicas, rng, self.burn_in
        )
        feats = read_out(
            trajs, residue_blocks, off, self.m, self.obs_noise, rng
        )
        coords, adj = ball_layout(groups, rng)
        G = {name: g for (name, _, _), g in zip(self.groups, groups)}
        in_parent = {c for kids in self.parents.values() for c in kids}
        parents = {p: sum((G[c] for c in kids), []) for p, kids in
                   self.parents.items()}
        coarse = list(parents.values()) + [
            G[name] for name, _, _ in self.groups if name not in in_parent
        ]
        truth = {
            "units": coarse,
            "fine_units": list(G.values()),
            "coarse_units": coarse,
            "labels": labels,
            "named": {**G, **parents},
            "design": str(self.name),
            "lifetimes_ns": {k: list(v) for k, v in self.lifetimes_ns.items()},
        }
        config = {
            "design": str(self.name),
            "seed": int(seed),
            "dt_ns": self.dt_ns,
            "gain": self.gain,
            "n_frames": self.n_frames,
            "n_replicas": self.n_replicas,
            "obs_noise": self.obs_noise,
            "spectral_radius": spectral_radius(K),
            "rescaled": rescaled,
        }
        return SyntheticSystem.from_replicas(feats, coords, adj, truth, config)


@dataclass(frozen=True)
class SmallUnitsDesign(Design):
    """The first suite: 6 units of 6 residues (`planted_system`).

    ``couplings=None`` is the default ring of directed gears; ``()`` removes
    them.  ``mix_all`` keeps the same latent dynamics but lets every residue
    read every unit (re-simulated with its own generator, seed + 7919).
    """

    n_frames: int = 3000
    n_units: int = 6
    unit_size: int = 6
    latent_dim: int = 5
    coupling_gain: float = 0.35
    couplings: tuple | None = None
    mix_all: bool = False
    noise: float = 0.3
    burn_in: int = 500

    def build(self, seed):
        """Generate the system for one random seed."""
        b = planted_system(
            n_units=self.n_units,
            unit_size=self.unit_size,
            m=self.m,
            latent_dim=self.latent_dim,
            n_frames=self.n_frames,
            n_replicas=self.n_replicas,
            couplings=None if self.couplings is None else list(self.couplings),
            coupling_gain=self.coupling_gain,
            noise=self.noise,
            obs_noise=self.obs_noise,
            face_residues=0,
            decoy_pairs=(),
            seed=seed,
        )
        config = self._config(seed, spectral_radius=b["config"][
            "spectral_radius"])
        if not self.mix_all:
            truth = {**b["truth"], "design": str(self.name)}
            return SyntheticSystem(**{**b, "truth": truth, "config": config})
        K = b["truth"]["K"]
        DL = K.shape[0]
        rng = np.random.default_rng(seed + 7919)
        trajs = simulate_var(
            K, np.full(DL, self.noise), self.n_frames, self.n_replicas, rng,
            self.burn_in,
        )
        N = b["feats"].shape[1]
        feats = read_out(
            trajs, [[0]] * N, np.array([0, DL]), self.m, self.obs_noise, rng,
            mix_all=True,
        )
        truth = {
            "units": [],
            "reference_units": b["truth"]["units"],
            "design": str(self.name),
        }
        return SyntheticSystem.from_replicas(
            feats, b["coords"], b["adjacency"], truth, config
        )


@dataclass(frozen=True)
class HierarchyDesign(Design):
    """Slow units with fast sub-units, a persistent unit and a plain one.

    U0 = sub-units a, b and U1 = sub-units c, d: each sub-unit reads its
    parent's slow latent plus its own fast latent (``rho_fast`` per frame).
    U2 is persistent (modes spread from ``rho_fast`` to the slowest), U3 is
    a plain slow unit.  Rank-1 couplings between slow latents:
    U0 -> U1 -> U2 -> U3 -> U0.
    """

    n_frames: int = 6000
    rho_fast: float = 0.8
    rho_slow: tuple = (0.97, 0.98)
    gain: float = 0.02
    slow_dim: int = 3
    fast_dim: int = 3
    persist_dim: int = 6
    sub_size: int = 4
    persist_size: int = 8
    plain_size: int = 6
    burn_in: int = 3000

    def build(self, seed):
        """Generate the system for one random seed."""
        rng = np.random.default_rng(seed)
        slow = np.linspace(*self.rho_slow, self.slow_dim)
        spread = 1 - np.geomspace(
            1 - self.rho_fast, 1 - self.rho_slow[1], self.persist_dim
        )
        rhos = (
            [slow, slow]  # 0, 1: slow latents of U0, U1
            + [[self.rho_fast] * self.fast_dim] * 4  # 2-5: fast a, b, c, d
            + [spread, slow]  # 6: persistent U2, 7: slow U3
        )
        K, L, off = block_diagonal([normal_block(r, rng) for r in rhos])
        sizes = np.diff(off)
        slow_block = {0: 0, 1: 1, 2: 6, 3: 7}
        for s, t in [(0, 1), (1, 2), (2, 3), (3, 0)]:
            bs, bt = slow_block[s], slow_block[t]
            u = unit_vector(sizes[bt], rng)
            v = unit_vector(sizes[bs], rng)
            add_coupling(K, off, bs, bt, u, v, self.gain)
        K, rescaled = stabilise(K, 0.999, 0.998)
        trajs = simulate_var(
            K, L, self.n_frames, self.n_replicas, rng, self.burn_in
        )
        layout = [
            ("U0a", [0, 2], self.sub_size),
            ("U0b", [0, 3], self.sub_size),
            ("U1c", [1, 4], self.sub_size),
            ("U1d", [1, 5], self.sub_size),
            ("U2", [6], self.persist_size),
            ("U3", [7], self.plain_size),
        ]
        residue_blocks, labels = [], []
        for name, reads, size in layout:
            residue_blocks += [reads] * size
            labels += [name] * size
        feats = read_out(
            trajs, residue_blocks, off, self.m, self.obs_noise, rng,
            share_weight=False,
        )
        lab = np.array(labels)
        fine = [
            [int(i) for i in np.where(lab == name)[0]]
            for name, _, _ in layout
        ]
        coarse = [fine[0] + fine[1], fine[2] + fine[3], fine[4], fine[5]]
        centres = {
            "U0a": (0, 0, 0),
            "U0b": (6, 0, 0),
            "U1c": (20, 0, 0),
            "U1d": (26, 0, 0),
            "U2": (10, 15, 0),
            "U3": (30, 15, 0),
        }
        coords = np.array(
            [np.array(centres[x], float) + rng.normal(scale=2.0, size=3)
             for x in labels]
        )
        adj = connect_components(contact_adjacency(coords, 8.0), coords)
        truth = {
            "units": coarse,
            "fine_units": fine,
            "coarse_units": coarse,
            "labels": labels,
            "design": str(self.name),
        }
        config = self._config(
            seed, spectral_radius=spectral_radius(K), rescaled=rescaled
        )
        return SyntheticSystem.from_replicas(feats, coords, adj, truth, config)


PRESETS = {
    DesignName.UNITS: UnitsDesign(DesignName.UNITS),
    DesignName.MIXED: UnitsDesign(DesignName.MIXED, mix_all=True),
    DesignName.NOCOMM: UnitsDesign(DesignName.NOCOMM, couplings=()),
    DesignName.MEDIUM: MediumDesign(DesignName.MEDIUM),
    DesignName.TAU: TimescaleDesign(DesignName.TAU),
    DesignName.SMALL_UNITS: SmallUnitsDesign(DesignName.SMALL_UNITS),
    DesignName.SMALL_NOCOMM: SmallUnitsDesign(
        DesignName.SMALL_NOCOMM, couplings=()
    ),
    DesignName.SMALL_MIXED: SmallUnitsDesign(
        DesignName.SMALL_MIXED, mix_all=True
    ),
    DesignName.SMALL_MEDIUM: MediumDesign(
        DesignName.SMALL_MEDIUM,
        n_frames=3000,
        dims=(4, 3, 3),
        self_rho=0.9,
        burn_in=500,
    ),
    DesignName.SMALL_HIERARCHY: HierarchyDesign(DesignName.SMALL_HIERARCHY),
}


def make_design(name, **overrides):
    """Return the preset design ``name`` with some fields replaced.

    Overrides that are None, or that the design does not have (e.g.
    ``rho_fast`` for anything but small-hierarchy), are ignored.
    """
    try:
        key = DesignName(str(name))
    except ValueError:
        names = ", ".join(d.value for d in DesignName)
        raise ConfigError(
            f"unknown design {name!r}; choose one of: {names}"
        ) from None
    if key not in PRESETS:
        raise ConfigError(
            f"{key.value!r} is the classic planted system: build it with "
            "pdukit.bench.planted.planted_system, not make_design"
        )
    base = PRESETS[key]
    have = {f.name for f in _fields(base)}
    changes = {
        k: v for k, v in overrides.items() if v is not None and k in have
    }
    return dataclasses.replace(base, **changes)


def _fields(design):
    return dataclasses.fields(design)


def _plain(value):
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, tuple):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    return value


def _consecutive_groups(sizes):
    groups, start = [], 0
    for s in sizes:
        groups.append(list(range(start, start + s)))
        start += s
    return groups
