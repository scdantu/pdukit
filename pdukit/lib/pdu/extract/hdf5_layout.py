"""THE ONLY MODULE THAT KNOWS ABOUT mdCATH's INTERNAL HDF5 STRUCTURE.

=============================================================================
READ THIS BEFORE TRUSTING ANY NUMBER PRODUCED BY THIS PACKAGE
=============================================================================
The layout encoded below is a BEST-EFFORT GUESS assembled from the mdCATH
paper and the torchmd-net dataset loader.  It has NOT been checked against a
real file by the author of this module (no network access at write time).

Run::

    python -m pdukit.lib.pdu.extract.hdf5_layout
    /path/to/mdcath_dataset_1a3dA00.h5

which dumps the real tree and tells you which of the assumptions below hold.
If something differs, fix it HERE -- no other module hardcodes structure.

Layout checked on a real file (1ztdA00, 2026-10-21): domain group at the top
level; temperature groups "320", "348", "379", "413", "450"; replicate groups
"0".."4" with coords (n_frames, n_atoms, 3), forces, dssp, rmsd, rmsf,
gyrationRadius and box; 420-450 frames per replicate; domain-level datasets
pdb, chain and element; file attribute layout = "mdcath-only-protein-v1.0".
The frame spacing is still not stored in the file.

Assumed layout (before checking; see the note above for what was confirmed)
--------------
    /<domain_id>/                       e.g. "1a3dA00"
        attrs: numResidues, numProteinAtoms, numChains, ...
        z                  (numProteinAtoms,)   int  atomic numbers
        pdb                scalar string        PDB text of protein atoms
                           (stored as "pdb"; "pdbProteinAtoms" was a guess)
        chain, element     (numProteinAtoms,)   object
        <temperature>/     "320" "348" "379" "413" "450"  (string keys)
            <replicate>/   "0" "1" "2" "3" "4"            (string keys)
                attrs: numFrames, (maybe) numResidues, numProteinAtoms
                coords     (numFrames, numProteinAtoms, 3)  float, Angstrom
                forces     (numFrames, numProteinAtoms, 3)  float, kcal/mol/A
                dssp       (numFrames, numResidues)         bytes
                rmsd       (numFrames,)
                rmsf                (numResidues,)
                gyrationRadius      (numFrames,)

Known unknowns (each one is flagged in verify_layout):
  * whether temperature keys are "320" or 320 or "T320"
  * whether the domain id is the top-level group or the file is flat
  * coordinate units (Angstrom assumed) and whether frames are already aligned
    (we do NOT rely on alignment: all features are internal distances)
  * the frame spacing in ns.  mdCATH reports ~464 ns average trajectories;
    the per-frame dt is NOT stored in a way we can guarantee, so `dt_ns` is a
    USER-SUPPLIED parameter everywhere.  Rates are meaningless without it.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# Constants: change these if the real file disagrees.
# --------------------------------------------------------------------------
TEMPERATURES = ("320", "348", "379", "413", "450")
REPLICATES = ("0", "1", "2", "3", "4")

DS_COORDS = "coords"
DS_FORCES = "forces"
DS_DSSP = "dssp"
DS_RMSD = "rmsd"
DS_RMSF = "rmsf"
DS_RGYR = "gyrationRadius"

DS_Z = "z"
DS_PDB = "pdb"  # verified on 1ztdA00, 2026-10-21 (older guess: pdbProteinAtoms)
DS_PDB_ALIASES = ("pdb", "pdbProteinAtoms")

ATTR_NUM_FRAMES = "numFrames"
ATTR_NUM_RESIDUES = "numResidues"
ATTR_NUM_ATOMS = "numProteinAtoms"

HF_REPO_ID = "compsciencelab/mdCATH"
HF_FILE_TEMPLATE = "data/mdcath_dataset_{domain}.h5"


def _require_h5py():
    try:
        import h5py  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "h5py is required to read mdCATH files. "
            "pip install -r requirements-mdcath.txt"
        ) from exc
    import h5py

    return h5py


# --------------------------------------------------------------------------
# Accessors.  Every one takes an already-open h5py.File-like object.
# They are deliberately tiny so that a layout change is a 1-line fix.
# --------------------------------------------------------------------------
def domain_group(f, domain: str):
    """Return the group holding everything for `domain`.

    Assumption: the file has a single top-level group named after the domain.
    Fallback: if that key is absent but the file has exactly one top-level
    group, use it.  Fallback 2: treat the file root as the domain group.
    """
    if domain in f:
        return f[domain]
    keys = list(f.keys())
    grouplike = [k for k in keys if hasattr(f[k], "keys")]
    if len(grouplike) == 1:
        return f[grouplike[0]]
    if any(t in f for t in TEMPERATURES):
        return f
    raise KeyError(
        f"Cannot find domain group for {domain!r}. Top-level keys: {keys}. "
        "Fix mdcath/hdf5_layout.py:domain_group()."
    )


def temperature_group(dg, temperature):
    """Return the temperature group of a domain."""
    key = str(temperature)
    if key in dg:
        return dg[key]
    # tolerate int-ish or decorated keys
    for cand in (temperature, f"T{key}", f"{key}K", key + ".0"):
        if str(cand) in dg:
            return dg[str(cand)]
    raise KeyError(
        f"temperature {temperature!r} not found; available: {list(dg.keys())}"
    )


def replicate_group(tg, replicate):
    """Return the replicate group of a temperature group."""
    key = str(replicate)
    if key in tg:
        return tg[key]
    raise KeyError(
        f"replicate {replicate!r} not found; available: {list(tg.keys())}"
    )


def list_temperatures(dg):
    """List the temperatures available for a domain."""
    return [k for k in dg.keys() if hasattr(dg[k], "keys")]


def list_replicates(tg):
    """List the replicates available for a temperature."""
    return [k for k in tg.keys() if hasattr(tg[k], "keys")]


def read_coords(rg, stride: int = 1, max_frames: int | None = None):
    """Return coords as (n_frames, n_atoms, 3) float64, Angstrom (assumed)."""
    import numpy as np

    ds = rg[DS_COORDS]
    n = ds.shape[0]
    stop = n if max_frames is None else min(n, stride * max_frames)
    arr = ds[0:stop:stride]
    return np.asarray(arr, dtype=np.float64)


def read_pdb_text(dg) -> str | None:
    """Return the PDB text of the protein atoms, or None."""
    key = next((k for k in DS_PDB_ALIASES if k in dg), None)
    if key is None:
        return None
    raw = dg[key][()]
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace")
    if isinstance(raw, str):
        return raw
    try:  # array of bytes lines
        return "\n".join(
            x.decode() if isinstance(x, bytes) else str(x) for x in raw
        )
    except Exception:
        return None


def read_z(dg):
    """Return the atomic numbers of the protein atoms."""
    import numpy as np

    if DS_Z not in dg:
        return None
    return np.asarray(dg[DS_Z][()])


# --------------------------------------------------------------------------
# PDB parsing (we deliberately do NOT depend on mdtraj / biopython)
# --------------------------------------------------------------------------
@dataclass
class AtomTable:
    """Atom names, residue ids and element of the protein atoms."""

    names: list = field(default_factory=list)  # atom names, e.g. "CA"
    resnames: list = field(default_factory=list)
    resids: list = field(
        default_factory=list
    )  # integer residue sequence numbers
    chains: list = field(default_factory=list)
    elements: list = field(default_factory=list)
    x: list = field(
        default_factory=list
    )  # Angstrom, static reference structure
    y: list = field(default_factory=list)
    z: list = field(default_factory=list)

    def __len__(self):
        """Return the number of atoms."""
        return len(self.names)

    def ca_indices(self):
        """Return the atom indices of the C-alpha atoms."""
        import numpy as np

        return np.array(
            [i for i, n in enumerate(self.names) if n == "CA"], dtype=int
        )

    def coords(self):
        """(n_atoms, 3) float array, Angstrom.

        Empty/NaN rows if this table was parsed before coordinate parsing was
        added (x/y/z unset).
        """
        import numpy as np

        if not self.x:
            raise ValueError(
                "no coordinates parsed -- re-parse with the "
                "coordinate-aware parse_pdb_atoms()"
            )
        return np.column_stack([self.x, self.y, self.z]).astype(np.float64)

    def ca_coords(self):
        """(n_CA, 3) -- coordinates of just the CA atoms, in the same order
        as ca_indices() / as resids indexed by ca_indices().
        """
        return self.coords()[self.ca_indices()]


def parse_pdb_atoms(text: str) -> AtomTable:
    """Minimal fixed-column PDB ATOM/HETATM parser.

    Extracts x/y/z (columns 31-38, 39-46, 47-54, standard PDB fixed-width
    format) alongside the existing name/resname/resid/chain/element fields --
    needed for anything that compares PDU footprints geometrically (e.g.
    mdcath.gw_distance), as opposed to purely by residue-index weight vectors.
    """
    t = AtomTable()
    for line in text.splitlines():
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        t.names.append(line[12:16].strip())
        t.resnames.append(line[17:20].strip())
        try:
            t.resids.append(int(line[22:26]))
        except ValueError:
            t.resids.append(len(t.resids))
        t.chains.append(line[21:22].strip())
        t.elements.append(line[76:78].strip())
        try:
            t.x.append(float(line[30:38]))
            t.y.append(float(line[38:46]))
            t.z.append(float(line[46:54]))
        except ValueError:
            t.x.append(float("nan"))
            t.y.append(float("nan"))
            t.z.append(float("nan"))
    return t


# --------------------------------------------------------------------------
# verify_layout
# --------------------------------------------------------------------------
def verify_layout(
    path: str, domain: str | None = None, verbose: bool = True
) -> dict:
    """Dump the real structure of an mdCATH .h5 and check our assumptions.

    Returns a dict of findings.  Prints a human-readable report.
    """
    h5py = _require_h5py()
    report = {"path": path, "ok": True, "problems": [], "found": {}}

    def say(*a):
        if verbose:
            print(*a)

    with h5py.File(path, "r") as f:
        top = list(f.keys())
        report["found"]["top_level_keys"] = top
        say(f"top-level keys: {top}")

        if domain is None:
            domain = top[0] if top else ""
        try:
            dg = domain_group(f, domain)
        except KeyError as e:
            report["ok"] = False
            report["problems"].append(str(e))
            return report

        say(f"domain group: {dg.name}")
        say(f"  attrs: {dict(dg.attrs)}")
        report["found"]["domain_attrs"] = {
            k: _py(v) for k, v in dg.attrs.items()
        }
        report["found"]["domain_keys"] = list(dg.keys())
        say(f"  keys : {list(dg.keys())}")

        for want in (DS_Z, DS_PDB):
            if want not in dg:
                report["problems"].append(
                    f"missing domain-level dataset {want!r}"
                )

        temps = list_temperatures(dg)
        report["found"]["temperatures"] = temps
        say(f"  temperature-like groups: {temps}")
        if set(temps) != set(TEMPERATURES):
            report["problems"].append(
                f"temperature keys {temps} != assumed {list(TEMPERATURES)}"
            )
        if not temps:
            report["ok"] = False
            return report

        tg = dg[temps[0]]
        reps = list_replicates(tg)
        report["found"]["replicates"] = reps
        say(f"  replicates under {temps[0]}: {reps}")

        if reps:
            rg = tg[reps[0]]
            say(f"    replicate {reps[0]} attrs: {dict(rg.attrs)}")
            report["found"]["replicate_attrs"] = {
                k: _py(v) for k, v in rg.attrs.items()
            }
            shapes = {}
            for k in rg.keys():
                try:
                    shapes[k] = (tuple(rg[k].shape), str(rg[k].dtype))
                except Exception:
                    shapes[k] = "?"
            report["found"]["replicate_datasets"] = shapes
            for k, v in shapes.items():
                say(f"      {k:16s} {v}")
            for want in (DS_COORDS, DS_RMSF, DS_RMSD, DS_RGYR):
                if want not in rg:
                    report["problems"].append(
                        f"missing replicate dataset {want!r}"
                    )
            if DS_COORDS in rg and len(rg[DS_COORDS].shape) != 3:
                report["problems"].append(
                    f"coords rank {len(rg[DS_COORDS].shape)} != 3 (expected "
                    "F,N,3)"
                )

        pdb = read_pdb_text(dg)
        if pdb:
            tbl = parse_pdb_atoms(pdb)
            nca = len(tbl.ca_indices())
            report["found"]["n_atoms_in_pdb"] = len(tbl)
            report["found"]["n_CA"] = nca
            say(f"  pdbProteinAtoms parsed: {len(tbl)} atoms, {nca} CA")
            if nca == 0:
                report["problems"].append(
                    "no CA atoms parsed from pdbProteinAtoms"
                )
        else:
            report["problems"].append(
                "pdbProteinAtoms missing/unparseable -> CA "
                "selection will fall back to a GUESS"
            )

    report["ok"] = not report["problems"]
    if verbose:
        print()
        if report["ok"]:
            print(
                "LAYOUT OK: all assumptions in hdf5_layout.py hold for this "
                "file."
            )
        else:
            print("LAYOUT PROBLEMS (fix mdcath/hdf5_layout.py):")
            for p in report["problems"]:
                print("  - " + p)
    return report


def _py(v):
    try:
        import numpy as np

        if isinstance(v, np.generic):
            return v.item()
        if isinstance(v, np.ndarray):
            return v.tolist()
    except Exception:
        pass
    if isinstance(v, bytes):
        return v.decode(errors="replace")
    return v


def main(argv=None):
    """Check the file layout from the command line."""
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) < 1:
        print(__doc__)
        print(
            "usage: python -m pdukit.lib.pdu.cli extract verify-layout "
            "FILE.h5 [DOMAIN]"
        )
        sys.exit(1)
    verify_layout(argv[0], argv[1] if len(argv) > 1 else None)


if __name__ == "__main__":
    main()
