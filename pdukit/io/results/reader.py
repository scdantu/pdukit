"""ResultsReader: the ``results.json`` of a run, read back."""

from __future__ import annotations

import json
from pathlib import Path

from pdukit.io.results import SCHEMA_VERSION


class ResultsReader(dict):
    """The contents of a run's ``results.json``, with where it came from.

    It is the plain dictionary the pipeline wrote, plus ``path`` (the
    file), ``rundir`` (the run directory) and the helpers the tools that
    read results share.
    """

    path: Path
    rundir: Path

    @classmethod
    def load(cls, path):
        """Read ``results.json`` from a run directory or a file path."""
        path = Path(path)
        if path.is_dir():
            path = path / "results.json"
        if not path.is_file():
            raise SystemExit(f"no results.json at {path}")
        with path.open() as fh:
            results = cls(json.load(fh))
        if results.schema_version > SCHEMA_VERSION:
            raise SystemExit(
                f"{path}: results layout {results.schema_version} is newer "
                f"than the one this pdukit reads ({SCHEMA_VERSION})"
            )
        results.path = path
        results.rundir = path.parent
        return results

    @property
    def schema_version(self):
        """Layout version of the file; 0 if it was written before versions."""
        return self.get("schema_version", 0)

    def units(self):
        """Return the residues of each unit the run found, as a list of lists.

        The last stage that ran decides: final units, else the split
        units, else the refined units, else those of an older layout.
        """
        for key in ("units_final", "units_split", "units_refined"):
            if key in self:
                return [u["residues"] for u in self[key]]
        return self.get("discovery", {}).get("units", [])

    def ports(self):
        """Return ``incoming_ports.json`` (unit cores, ports, residue roles).

        None when the run wrote no port file (for example no unit found).
        """
        path = self.rundir / "incoming_ports.json"
        if not path.is_file():
            return None
        with path.open() as fh:
            return json.load(fh)

    def bootstrap_rho(self):
        """Return the bootstrap copies of the pair coupling, (B, N, N).

        Read from ``bootstrap_rho.npz``; None when the run saved none.
        """
        path = self.rundir / "bootstrap_rho.npz"
        if not path.is_file():
            return None
        import numpy as np

        with np.load(path) as data:
            return data["rho"].astype(float)

    def rho(self):
        """Return the pair coupling rho[i, j] (j predicts i) as an array."""
        import numpy as np

        return np.array(self["predictive_graph"]["rho"], float)
