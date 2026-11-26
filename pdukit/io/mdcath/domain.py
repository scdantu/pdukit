"""Load one mdCATH domain/temperature/replicate into memory.

Also transparently loads the .npz mock produced by `mdcath.mock`, so the whole
pipeline is testable without h5py.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class MdcathDomain:
    """One domain, temperature and replicate loaded into memory."""

    domain: str
    temperature: str
    replicate: str
    coords: np.ndarray  # (n_frames, n_atoms, 3) Angstrom
    atom_names: list  # len n_atoms  ("" if unknown)
    resids: np.ndarray  # (n_atoms,) int
    z: np.ndarray | None  # atomic numbers or None
    ca_indices: np.ndarray  # indices into axis 1 of coords
    dt_ns: (
        float | None
    )  # frame spacing, USER SUPPLIED (see hdf5_layout docstring)
    source: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def n_frames(self):
        """Return the number of frames."""
        return self.coords.shape[0]

    @property
    def n_atoms(self):
        """Return the number of atoms."""
        return self.coords.shape[1]

    @property
    def n_residues(self):
        """Return the number of residues."""
        return (
            int(len(np.unique(self.resids))) if self.resids is not None else 0
        )

    def ca_coords(self):
        """Return the C-alpha coordinates of every frame."""
        return self.coords[:, self.ca_indices, :]

    def __repr__(self):
        """Return a one-line description."""
        return (
            f"MdcathDomain({self.domain} T={self.temperature} "
            f"r={self.replicate} "
            f"frames={self.n_frames} atoms={self.n_atoms} "
            f"CA={len(self.ca_indices)})"
        )


def _guess_ca_from_z(z, n_atoms):
    """Last-ditch fallback when no PDB text is available.

    THIS IS A GUESS AND WILL USUALLY BE WRONG.  It is here so the pipeline can
    still run for smoke-testing, never so that you should trust the output.
    """
    return np.arange(n_atoms)


def _finish(
    domain,
    temperature,
    replicate,
    coords,
    tbl,
    z,
    dt_ns,
    source,
    atom_selection,
    meta,
):
    n_atoms = coords.shape[1]
    if tbl is not None and len(tbl) == n_atoms:
        names = list(tbl.names)
        resids = np.asarray(tbl.resids, dtype=int)
        ca = tbl.ca_indices()
    else:
        if tbl is not None and len(tbl) != n_atoms:
            meta["warning_atom_count_mismatch"] = (len(tbl), n_atoms)
        names = [""] * n_atoms
        resids = np.arange(n_atoms)
        ca = _guess_ca_from_z(z, n_atoms)
        meta["ca_selection"] = (
            "GUESS (no usable pdbProteinAtoms) -- DO NOT TRUST"
        )

    if atom_selection.upper() == "CA":
        if len(ca) == 0:
            raise ValueError(
                "no CA atoms identified; use atom_selection='all' "
                "or fix hdf5_layout.parse_pdb_atoms"
            )
        coords = coords[:, ca, :]
        names = [names[i] for i in ca]
        resids = resids[ca]
        z = None if z is None else np.asarray(z)[ca]
        ca = np.arange(coords.shape[1])
    return MdcathDomain(
        domain=domain,
        temperature=str(temperature),
        replicate=str(replicate),
        coords=np.ascontiguousarray(coords, dtype=np.float64),
        atom_names=names,
        resids=resids,
        z=z,
        ca_indices=ca,
        dt_ns=dt_ns,
        source=source,
        meta=meta,
    )


