"""Result: what a COG run found, ready to look at.

    result = Cog(cfg).run(ens)
    result.protein            # residues with unit / label / confidence
    result.units             # {id: Unit}
    result.table("units")    # pandas DataFrame
    result.plot_units()      # figure file (shown in a notebook with .show())
    result.view()            # 3D view coloured by unit (py3Dmol)
    result.save_pml("u.pml") # PyMOL script
    result.save("keep/")     # copy the run folder somewhere permanent

Heavy libraries (pandas, matplotlib, py3Dmol) are imported only when used.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from pdukit.errors import PduError
from pdukit.io.results.reader import ResultsReader
from pdukit.model.protein import Protein


class Result:
    """The results of one run and the tools to inspect them."""

    def __init__(self, results, topology=None):
        """Wrap a `ResultsReader` (and the topology file, if known)."""
        self.results = results
        self.topology = Path(topology) if topology else None
        self._protein = None

    @classmethod
    def load(cls, path, topology=None):
        """Read a finished run from its folder (or its ``results.json``)."""
        return cls(ResultsReader.load(path), topology)

    @property
    def folder(self):
        """Return the run folder."""
        return self.results.rundir

    @property
    def protein(self):
        """Return the `Protein` with COG properties filled in."""
        if self._protein is None:
            self._protein = Protein.from_results(self.results, self.topology)
        return self._protein

    @property
    def units(self):
        """Return ``{unit id: Unit}``."""
        return self.protein.units

    # ---- tables ----------------------------------------------------------

    def tables(self):
        """Return the names of the available tables."""
        from pdukit.analysis import tables

        return list(tables.TABLES)

    def table(self, name="units"):
        """Return one table as a pandas DataFrame (see ``tables()``)."""
        import pandas as pd

        from pdukit.analysis import tables

        built = tables.build(self.results, [name])
        if name not in built:
            raise PduError(f"table {name!r} is empty for this run")
        columns, rows = built[name]
        return pd.DataFrame(rows, columns=columns)

    # ---- figures and structures ------------------------------------------

    def plot_units(self, out=None, title=None):
        """Draw the units and per-residue confidence; return the PNG path."""
        from pdukit.viz.units import units_figure

        out = Path(out) if out else self.folder / "units.png"
        units_figure(self.results, out, pdb=self.topology, title=title)
        return out

    def show(self, path=None):
        """Display a figure file in a notebook (default: ``plot_units``)."""
        from IPython.display import Image

        return Image(filename=str(path or self.plot_units()))

    def save_pml(self, out=None, **options):
        """Write a PyMOL script colouring the units; return its path."""
        from pdukit.viz.pymol import build_pml

        if self.topology is None:
            raise PduError("save_pml needs the topology (Result.load(..., "
                           "topology=...))")
        out = Path(out) if out else self.folder / "units.pml"
        out.write_text(build_pml(self.results, self.topology, **options))
        return out

    def view(self, width=600, height=450):
        """Return a py3Dmol view: cores in unit colours, ports as sticks."""
        import py3Dmol

        from pdukit.viz.style import unit_color

        if self.topology is None:
            raise PduError("view needs the topology")
        fmt = self.topology.suffix.lstrip(".").lower() or "pdb"
        v = py3Dmol.view(width=width, height=height)
        v.addModel(self.topology.read_text(), fmt)
        v.setStyle({"cartoon": {"color": "lightgrey"}})
        resid = {r.index: r.resid for r in self.protein}
        for unit in self.units.values():
            colour = unit_color(unit.id)
            v.addStyle({"resi": [resid[i] for i in unit.core]},
                       {"cartoon": {"color": colour},
                        "stick": {"color": colour, "radius": 0.15}})
            if unit.ports:
                v.addStyle({"resi": [resid[i] for i in unit.ports]},
                           {"stick": {"color": colour, "radius": 0.3}})
        v.zoomTo()
        return v

    # ---- keeping ---------------------------------------------------------

    def save(self, folder):
        """Copy the run folder to ``folder``; return the new `Result`."""
        folder = Path(folder)
        shutil.copytree(self.folder, folder, dirs_exist_ok=True)
        return Result.load(folder, self.topology)

    def __repr__(self):
        """Return a short summary."""
        n = len(self.units) if self.results.get("units_final") else 0
        return f"Result({self.folder}, {n} units)"
