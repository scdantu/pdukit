"""Ensemble: the trajectories and topology of one protein.

    ens = Ensemble(["rep1.xtc", "rep2.xtc"], "top.pdb")
    ens.protein            # a bare Protein read from the topology
    ens = Ensemble.demo()  # a small synthetic example with a known answer

Nothing is read until it is needed: creating an Ensemble only checks that the
files exist.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from pdukit.errors import PduError


class Ensemble:
    """One protein's replicas: trajectory files plus one topology."""

    def __init__(self, trajectories, topology, name=None):
        """Hold the file names (``trajectories`` may be one path or many)."""
        if isinstance(trajectories, (str, Path)):
            trajectories = [trajectories]
        self.trajectories = [Path(p) for p in trajectories]
        self.topology = Path(topology)
        self.name = name or self.topology.stem
        if not self.trajectories:
            raise PduError("an ensemble needs at least one trajectory")
        for path in (*self.trajectories, self.topology):
            if not path.is_file():
                raise PduError(f"no such file: {path}")
        self._protein = None

    @property
    def n_replicas(self):
        """Return the number of trajectory files."""
        return len(self.trajectories)

    @property
    def protein(self):
        """Return the bare `Protein` of the topology (read once)."""
        if self._protein is None:
            from pdukit.model.protein import Protein

            self._protein = Protein.from_topology(self.topology)
        return self._protein

    def settings(self):
        """Return the settings that point a run at these files."""
        return {
            "xtc": [str(p) for p in self.trajectories],
            "xtc_topology": str(self.topology),
        }

    @classmethod
    def demo(cls, folder=None, replicas=3, frames=1500, seed=0):
        """Write and return the synthetic 4-unit example (needs MDAnalysis).

        ``folder`` defaults to a new temporary directory.
        """
        from pdukit.bench.demo import write_demo

        folder = folder or tempfile.mkdtemp(prefix="pdukit_demo_")
        paths, top = write_demo(folder, replicas, frames, seed=seed)
        return cls(paths, top, name="demo")

    def __repr__(self):
        """Return a short summary."""
        return (f"Ensemble({self.name!r}, {self.n_replicas} replicas, "
                f"topology={self.topology.name})")
