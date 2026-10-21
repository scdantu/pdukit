"""Convert one mdCATH domain (HDF5) into a topology and XTC replicas.

mdCATH stores every domain as an HDF5 file with five temperatures and five
replicas.  ``pdu run`` reads XTC files plus a topology, so this script writes
``topology.pdb`` and ``rep_<k>.xtc`` for the chosen temperature.

The HDF5 layout is read through ``pdukit.lib.pdu.extract.hdf5_layout``, the
one module that knows it.  That layout was written from the paper and has not
been checked against a real file, so on first use run::

    python mdcath_to_xtc.py --inspect mdcath_dataset_1a02F00.h5

and compare the printed tree with the assumptions listed in hdf5_layout.py.
The frame spacing of mdCATH is not read from the file: give it with
``--dt-ns`` (default 1.0; confirm it in the mdCATH paper) and use the same
value as ``frame-ps`` in the pdu parameter file.

Needs: h5py, mdtraj, numpy; huggingface_hub when the file is downloaded.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from pdukit.lib.pdu.extract import hdf5_layout as hl


def fetch(domain: str, cache: Path) -> Path:
    """Download the HDF5 file of ``domain`` into ``cache`` and return it."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=hl.HF_REPO_ID,
        repo_type="dataset",
        filename=hl.HF_FILE_TEMPLATE.format(domain=domain),
        local_dir=str(cache),
    )
    return Path(path)


def inspect(path: Path) -> None:
    """Print every key of the domain group and one replicate of one temp."""
    h5py = hl._require_h5py()
    with h5py.File(path, "r") as f:
        print("file attrs:", dict(f.attrs))
        for dom in f.keys():
            dg = f[dom]
            print(f"{dom}: {len(dg.keys())} keys: {list(dg.keys())}")
            for k, v in dg.items():
                if hasattr(v, "keys"):
                    continue
                shape = getattr(v, "shape", None)
                preview = ""
                if shape == ():
                    raw = v[()]
                    raw = raw.decode() if isinstance(raw, bytes) else raw
                    preview = repr(str(raw)[:70])
                print(f"  {k}: dataset {shape} {v.dtype} {preview}")
            temps = hl.list_temperatures(dg)
            print("  temperatures:", temps)
            for t in temps[:1]:
                reps = hl.list_replicates(dg[t])
                lens = {r: dg[t][r]["coords"].shape[0] for r in reps}
                print(f"  {t}: replicates and frame counts {lens}")


def convert(path: Path, domain: str, temperature: str, replicates: list[str],
            out: Path, dt_ns: float, heavy_only: bool) -> dict:
    """Write topology.pdb and one XTC per replicate; return a summary."""
    import mdtraj as md

    h5py = hl._require_h5py()
    out.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "r") as f:
        dg = hl.domain_group(f, domain)
        pdb_text = hl.read_pdb_text(dg)
        if pdb_text is None:
            raise SystemExit("no protein PDB text in the file; check "
                             "hdf5_layout.DS_PDB_ALIASES")
        (out / "full.pdb").write_text(pdb_text)
        top_full = md.load_pdb(str(out / "full.pdb")).topology
        keep = np.array([a.index for a in top_full.atoms
                         if not (heavy_only and a.element is not None
                                 and a.element.symbol == "H")])
        top = top_full.subset(keep)
        tg = hl.temperature_group(dg, temperature)
        avail = hl.list_replicates(tg)
        times = None
        frames = {}
        for r in replicates:
            if str(r) not in avail:
                print(f"replicate {r} missing (have {avail}); skipped")
                continue
            xyz = hl.read_coords(hl.replicate_group(tg, r))      # Angstrom
            if xyz.shape[1] != top_full.n_atoms:
                raise SystemExit(
                    f"coords have {xyz.shape[1]} atoms but the PDB has "
                    f"{top_full.n_atoms}; the atom order is not what the "
                    "converter assumes (see hdf5_layout.py)")
            xyz = xyz[:, keep, :] / 10.0                          # nm
            times = np.arange(xyz.shape[0]) * dt_ns * 1000.0      # ps
            traj = md.Trajectory(xyz.astype(np.float32), top, time=times)
            traj.save_xtc(str(out / f"rep_{r}.xtc"))
            frames[str(r)] = int(xyz.shape[0])
    sub = md.Trajectory(np.zeros((1, top.n_atoms, 3), np.float32), top)
    sub.save_pdb(str(out / "topology.pdb"))
    (out / "full.pdb").unlink()
    info = {"domain": domain, "temperature": str(temperature),
            "dt_ns": dt_ns, "frames_per_replicate": frames,
            "n_atoms": int(top.n_atoms),
            "n_residues": int(top.n_residues), "heavy_atoms_only": heavy_only,
            "source": str(path)}
    (out / "conversion.json").write_text(json.dumps(info, indent=2))
    return info


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("domain", nargs="?", help="CATH domain id, e.g. 1a02F00")
    p.add_argument("--h5", type=Path, help="local file (else downloaded)")
    p.add_argument("--cache", type=Path, default=Path("mddata/mdcath"))
    p.add_argument("-o", "--out", type=Path)
    p.add_argument("--temperature", default="320")
    p.add_argument("--replicates", default="0,1,2,3,4")
    p.add_argument("--dt-ns", type=float, default=1.0)
    p.add_argument("--keep-hydrogens", action="store_true")
    p.add_argument("--inspect", type=Path, metavar="H5",
                   help="print the structure of a file and stop")
    return p


def main(argv=None) -> int:
    """Convert (or inspect) one domain."""
    args = build_parser().parse_args(argv)
    if args.inspect:
        inspect(args.inspect)
        return 0
    if not args.domain or not args.out:
        raise SystemExit("give a domain and -o OUT (or --inspect FILE)")
    path = args.h5 or fetch(args.domain, args.cache)
    info = convert(path, args.domain, args.temperature,
                   args.replicates.split(","), args.out, args.dt_ns,
                   not args.keep_hydrogens)
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
