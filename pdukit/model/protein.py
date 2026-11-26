"""Protein, Residue, Unit: the residue-level result of a COG run.

    protein = Protein.from_topology("top.pdb")      # bare chain
    protein.assign(results)                          # COG fills in properties
    protein.residue(12).label                        # ResidueClass.PORT
    protein.units[2].ports                           # [8, 12, 13, ...]
    protein.to_frame()                               # one row per residue

Residue ``index`` is 0-based and is the position in the analysed chain (the
number every table of a run uses); ``resid`` is the number in the structure.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pdukit.model.enums import ResidueClass
from pdukit.numerics import confidence


@dataclass
class Residue:
    """One residue and what COG learned about it."""

    index: int
    resid: int = 0
    resname: str = ""
    chain: str = ""
    coords: tuple | None = None  # CA position (Angstrom)
    label: ResidueClass = ResidueClass.UNASSIGNED
    units: tuple = ()  # units whose core holds it
    port_of: tuple = ()  # unit it is a port of
    bridge_between: tuple = ()  # units it sits between
    drives: dict = field(default_factory=dict)  # unit -> strength
    driven_by: dict = field(default_factory=dict)  # unit -> strength
    rcs: float | None = None  # residue confidence 0-100
    band: str = ""  # very high / high / ...
    n_seeds: int = 0

    @property
    def in_unit(self):
        """Return True if the residue is a core or port of some unit."""
        return bool(self.units or self.port_of)


@dataclass
class Unit:
    """One protein dynamical unit."""

    id: int
    core: list
    ports: list = field(default_factory=list)
    ucs: float | None = None  # unit confidence 0-100
    band: str = ""
    n_seeds: int = 0
    cohesion: float | None = None
    nu_in: float | None = None
    nu_out: float | None = None

    @property
    def extended(self):
        """Return core plus ports, sorted."""
        return sorted(set(self.core) | set(self.ports))

    def __len__(self):
        """Return the core size."""
        return len(self.core)


def _split(text):
    return [t for t in str(text or "").split() if t != ""]


def _floats(text):
    return [float(t) for t in _split(text)]


class Protein:
    """A chain of `Residue` plus the `Unit` objects found on it."""

    def __init__(self, residues):
        """Hold ``residues`` (their ``index`` must be 0..n-1 in order)."""
        self.residues = list(residues)
        self.units = {}
        for k, r in enumerate(self.residues):
            if r.index != k:
                raise ValueError("residue indices must run 0..n-1 in order")

    # ---- construction --------------------------------------------------

    @classmethod
    def from_topology(cls, path, selection="name CA"):
        """Read the chain from a topology file (pdb, gro, tpr, prmtop...)."""
        import MDAnalysis  # lazy: heavy

        atoms = MDAnalysis.Universe(str(path)).select_atoms(selection)
        pos = atoms.positions if len(atoms) else []
        return cls(
            Residue(
                index=i,
                resid=int(a.resid),
                resname=str(a.resname),
                chain=str(
                    getattr(a, "segid", "") or getattr(a, "chainID", "") or ""
                ),
                coords=tuple(float(x) for x in pos[i]),
            )
            for i, a in enumerate(atoms)
        )

    @classmethod
    def blank(cls, n_residues):
        """Return a protein of ``n_residues`` with no structure information."""
        return cls(Residue(index=i, resid=i) for i in range(n_residues))

    @classmethod
    def from_results(cls, results, topology=None):
        """Build the protein of a run (``results`` = results.json dict)."""
        n = results.get("data", {}).get("n_residues")
        prot = (
            cls.from_topology(topology) if topology else cls.blank(int(n or 0))
        )
        return prot.assign(results)

    # ---- access --------------------------------------------------------

    def __len__(self):
        """Return the number of residues."""
        return len(self.residues)

    def __iter__(self):
        """Iterate over residues."""
        return iter(self.residues)

    def residue(self, index):
        """Return the residue with 0-based ``index``."""
        return self.residues[index]

    def by_label(self, label):
        """Return the residues with a given `ResidueClass`."""
        label = ResidueClass(label)
        return [r for r in self.residues if r.label == label]

    def counts(self):
        """Return ``{label: number of residues}``."""
        out = {}
        for r in self.residues:
            out[r.label.value] = out.get(r.label.value, 0) + 1
        return out

    # ---- fill in from a run -------------------------------------------

    def assign(self, results):
        """Fill unit, label, coupling and confidence from a run's results.

        ``results`` is the dict of ``results.json``.  Returns ``self``.
        """
        rows = (results.get("incoming_ports") or {}).get("residues") or []
        ports = {
            u["unit"]: u
            for u in (results.get("incoming_ports") or {}).get("units", [])
        }
        split = {u["unit"]: u for u in results.get("units_split") or []}
        scores = {s["unit"]: s for s in confidence.unit_scores(results)}
        self.units = {}
        for u in (
            results.get("units_final") or results.get("units_split") or []
        ):
            k = u["unit"]
            p, s, sc = ports.get(k, {}), split.get(k, {}), scores.get(k, {})
            self.units[k] = Unit(
                id=k,
                core=sorted(p.get("core", u["residues"])),
                ports=sorted(p.get("ports", [])),
                ucs=sc.get("ucs"),
                band=sc.get("band", ""),
                n_seeds=sc.get("n_seeds", 0),
                cohesion=s.get("cohesion", p.get("cohesion_core")),
                nu_in=s.get("nu_in", p.get("nu_in_core")),
                nu_out=s.get("nu_out", p.get("nu_out_core")),
            )
        for r in self.residues:
            r.label, r.units, r.port_of, r.bridge_between = (
                ResidueClass.UNASSIGNED,
                (),
                (),
                (),
            )
            r.drives, r.driven_by = {}, {}
        for row in rows:
            r = self.residues[int(row["residue"])]
            r.label = ResidueClass(row["cls"])
            owners = tuple(int(x) for x in _split(row.get("unit")))
            if r.label in (ResidueClass.CORE, ResidueClass.SHARED):
                r.units = owners
            elif r.label == ResidueClass.PORT:
                r.port_of = owners
            best = [int(x) for x in _split(row.get("best_units"))]
            if r.label == ResidueClass.BRIDGE:
                r.bridge_between = tuple(best)
            r.drives = dict(zip(best, _floats(row.get("best_drives"))))
            r.driven_by = dict(zip(best, _floats(row.get("best_driven"))))
        for i, d in confidence.residue_scores(results).items():
            r = self.residues[i]
            r.rcs, r.n_seeds = d["rcs"], d["n_seeds"]
            r.band = confidence.band(d["rcs"])
        return self

    # ---- output --------------------------------------------------------

    def to_records(self):
        """Return one plain dict per residue."""
        return [
            {
                "index": r.index,
                "resid": r.resid,
                "resname": r.resname,
                "chain": r.chain,
                "label": r.label.value,
                "units": " ".join(map(str, r.units)),
                "port_of": " ".join(map(str, r.port_of)),
                "bridge_between": " ".join(map(str, r.bridge_between)),
                "rcs": r.rcs,
                "band": r.band,
            }
            for r in self.residues
        ]

    def to_frame(self):
        """Return a pandas DataFrame, one row per residue."""
        import pandas as pd  # lazy

        return pd.DataFrame(self.to_records())

    def __repr__(self):
        """Return a short summary."""
        return (
            f"Protein({len(self)} residues, {len(self.units)} units, "
            f"{self.counts()})"
        )
