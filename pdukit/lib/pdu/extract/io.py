"""Load one mdCATH domain/temperature/replicate into memory.

Also transparently loads the .npz mock produced by `mdcath.mock`, so the whole
pipeline is testable without h5py.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from . import hdf5_layout as hl


@dataclass
class DomainTrajectory:
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
            f"DomainTrajectory({self.domain} T={self.temperature} "
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


def load_domain(
    path: str,
    domain: str | None = None,
    temperature="320",
    replicate="0",
    stride: int = 1,
    max_frames: int | None = None,
    dt_ns: float | None = None,
    atom_selection: str = "CA",
) -> DomainTrajectory:
    """Load a single trajectory.

    Parameters
    ----------
    atom_selection : "CA" | "all"
        "CA" keeps only alpha carbons in `coords` (recommended: the featuriser
        and the participation weights then live on residues, and CA count
        matches numResidues).  "all" keeps every protein atom.
    dt_ns : frame spacing in ns AFTER striding.  Required for rates in ns^-1.

    """
    if isinstance(path, str) and path.endswith(".npz"):
        return _load_npz(
            path,
            domain,
            temperature,
            replicate,
            stride,
            max_frames,
            dt_ns,
            atom_selection,
        )
    # `path` may also be a file-like object (e.g. a remote fsspec HTTP file
    # from mdcath.remote_io) -- h5py.File() accepts either transparently.
    return _load_h5(
        path,
        domain,
        temperature,
        replicate,
        stride,
        max_frames,
        dt_ns,
        atom_selection,
    )


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
    return DomainTrajectory(
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


def _load_h5(
    path,
    domain,
    temperature,
    replicate,
    stride,
    max_frames,
    dt_ns,
    atom_selection,
):
    h5py = hl._require_h5py()
    if domain is None:
        if not isinstance(path, str):
            raise ValueError(
                "domain must be given explicitly when path is a file-like "
                "object (e.g. a remote fsspec file) -- there is no filename "
                "to infer it from"
            )
        domain = _domain_from_filename(path)
    meta = {}
    with h5py.File(path, "r") as f:
        dg = hl.domain_group(f, domain)
        tg = hl.temperature_group(dg, temperature)
        rg = hl.replicate_group(tg, replicate)
        coords = hl.read_coords(rg, stride=stride, max_frames=max_frames)
        pdb = hl.read_pdb_text(dg)
        tbl = hl.parse_pdb_atoms(pdb) if pdb else None
        z = hl.read_z(dg)
        for k in (hl.ATTR_NUM_FRAMES, hl.ATTR_NUM_RESIDUES, hl.ATTR_NUM_ATOMS):
            if k in rg.attrs:
                meta[k] = hl._py(rg.attrs[k])
            elif k in dg.attrs:
                meta[k] = hl._py(dg.attrs[k])
    return _finish(
        domain,
        temperature,
        replicate,
        coords,
        tbl,
        z,
        dt_ns,
        path,
        atom_selection,
        meta,
    )


def _load_npz(
    path,
    domain,
    temperature,
    replicate,
    stride,
    max_frames,
    dt_ns,
    atom_selection,
):
    """Read the mock fixture.

    Keys mirror the H5 paths: '<dom>/<T>/<r>/coords'.
    """
    d = np.load(path, allow_pickle=True)
    if domain is None:
        domain = (
            str(d["__domain__"])
            if "__domain__" in d
            else _domain_from_filename(path)
        )
    key = f"{domain}/{temperature}/{replicate}/coords"
    if key not in d:
        raise KeyError(f"{key} not in {path}; keys={list(d.keys())[:20]}")
    coords = d[key]
    n = coords.shape[0]
    stop = n if max_frames is None else min(n, stride * max_frames)
    coords = coords[0:stop:stride]
    pdb_key = f"{domain}/{hl.DS_PDB}"
    tbl = hl.parse_pdb_atoms(str(d[pdb_key])) if pdb_key in d else None
    zk = f"{domain}/{hl.DS_Z}"
    z = d[zk] if zk in d else None
    meta = {"mock": True}
    if "__truth__" in d:
        meta["truth"] = d["__truth__"].item()
    return _finish(
        domain,
        temperature,
        replicate,
        coords,
        tbl,
        z,
        dt_ns,
        path,
        atom_selection,
        meta,
    )


def _domain_from_filename(path):
    base = os.path.basename(path)
    for pre in ("mdcath_dataset_",):
        if base.startswith(pre):
            return base[len(pre) :].split(".")[0]
    return base.split(".")[0]
