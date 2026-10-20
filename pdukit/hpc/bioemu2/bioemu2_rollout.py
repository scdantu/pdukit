"""Roll out BioEmu2 trajectories at a fixed lag time and write XTC replicas.

BioEmu2 has two modes: equilibrium sampling (unordered structures) and
time-lagged propagation (one structure in, its state a lag time later out).
A trajectory with frame spacing equal to the lag is made by sampling N start
structures and propagating them K times.  Every start structure becomes one
replica of K + 1 frames, so the output can be given to ``pdu run`` exactly
like MD replicas (one XTC per replica, one shared topology).

Frame spacing is the lag time (default 1 ns), so use ``lag = 1`` (frames) in
the pdu parameter file and compare with MD strided to the same spacing.

Two engines run the model:

``api``  (default) keeps the model loaded and calls the Python API shown in
         the BioEmu2 README (System, EmuSession, sample, propagate,
         save_frames).
``cli``  calls ``bioemu2 sample-equilibrium`` / ``bioemu2 propagate`` once per
         step; slower (the model is reloaded each time) but uses only the
         documented command line.

Assumption to check on first use: the i-th structure written by a propagation
is the endpoint of the i-th start structure.  ``check_pairing`` tests this on
the output (consecutive frames of one replica must be closer than frames of
different replicas) and the script stops if it fails.

Needs: bioemu2 (NVIDIA GPU, Linux), mdtraj, numpy, and gemmi or biopython to
convert the CIF topology to PDB.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def cif_to_pdb(cif: Path, pdb: Path) -> None:
    """Write ``pdb`` from the mmCIF file ``cif`` (gemmi, else biopython)."""
    try:
        import gemmi
    except ImportError:
        gemmi = None
    if gemmi is not None:
        st = gemmi.read_structure(str(cif))
        st.write_pdb(str(pdb))
        return
    try:
        from Bio.PDB import PDBIO, MMCIFParser
    except ImportError as err:
        raise SystemExit(
            "need gemmi or biopython to convert the CIF topology"
        ) from err
    io = PDBIO()
    io.set_structure(MMCIFParser(QUIET=True).get_structure("x", str(cif)))
    io.save(str(pdb))


def run_cli(cmd: list[str]) -> None:
    """Run a command, echo it, and stop on failure."""
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def rollout_api(args, steps_dir: Path) -> None:
    """Sample and propagate with the Python API; save every step."""
    from bioemu2 import EmuSession, System, load_model, save_frames

    kwargs = {"checkpoint": args.checkpoint} if args.checkpoint else {}
    model = load_model(**kwargs)
    if not args.pdb_id:
        raise SystemExit(
            "the api engine takes --pdb-id; use --engine cli for --pdb files"
        )
    system = System.from_pdb_id(args.pdb_id)
    session = EmuSession(system, model=model, mode="equilibrium")
    frames = session.sample(num_samples=args.n_replicas)
    save_frames(frames, steps_dir / "step_0000")
    for k in range(1, args.n_steps + 1):
        t0 = time.time()
        frames = session.propagate(frames, lag_time_ns=args.lag_ns)
        save_frames(frames, steps_dir / f"step_{k:04d}")
        if k % 10 == 0 or k == 1:
            print(f"step {k}/{args.n_steps}  {time.time() - t0:.1f} s",
                  flush=True)


def rollout_cli(args, steps_dir: Path) -> None:
    """Sample and propagate with the bioemu2 command line."""
    s0 = steps_dir / "step_0000"
    cmd = ["bioemu2", "sample-equilibrium", "--out", str(s0),
           "--num-samples", str(args.n_replicas)]
    cmd += (["--pdb-id", args.pdb_id] if args.pdb_id
            else ["--system", str(args.pdb)])
    if args.checkpoint:
        cmd += ["--checkpoint", args.checkpoint]
    run_cli(cmd)
    prev = s0
    for k in range(1, args.n_steps + 1):
        cur = steps_dir / f"step_{k:04d}"
        run_cli(["bioemu2", "propagate",
                 "--x0-cif", str(prev / "topology.cif"),
                 "--x0-xtc", str(prev / "samples.xtc"),
                 "--lag-time-ns", str(args.lag_ns),
                 "--out", str(cur)])
        prev = cur


def merge(steps_dir: Path, out: Path, n_steps: int, lag_ns: float):
    """Merge the step folders into one topology and one XTC per replica."""
    import mdtraj as md

    top_pdb = out / "topology.pdb"
    cif_to_pdb(steps_dir / "step_0000" / "topology.cif", top_pdb)
    top = md.load_topology(str(top_pdb))
    steps = []
    for k in range(n_steps + 1):
        d = steps_dir / f"step_{k:04d}"
        steps.append(md.load_xtc(str(d / "samples.xtc"), top=top).xyz)
    xyz = np.stack(steps, axis=1)          # (replica, step, atom, 3) nm
    n_rep = xyz.shape[0]
    times = np.arange(n_steps + 1) * lag_ns * 1000.0   # ps
    for r in range(n_rep):
        traj = md.Trajectory(xyz[r], top, time=times)
        traj.save_xtc(str(out / f"rep_{r:03d}.xtc"))
    return xyz


def ca_rmsd(a: np.ndarray, b: np.ndarray) -> float:
    """Return the rigid-fit RMSD of two (n, 3) coordinate sets."""
    a = a - a.mean(0)
    b = b - b.mean(0)
    u, _, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(u @ vt))
    r = u @ np.diag([1, 1, d]) @ vt
    return float(np.sqrt(((a @ r - b) ** 2).sum(1).mean()))


def check_pairing(xyz: np.ndarray, top_pdb: Path, n_pairs: int = 20) -> dict:
    """Test that replica i at step k+1 follows replica i at step k.

    Uses C-alpha atoms.  The median RMSD of true successors must be clearly
    smaller than that of successors of another replica.
    """
    import mdtraj as md

    top = md.load_topology(str(top_pdb))
    ca = top.select("name CA")
    n_rep, n_steps = xyz.shape[0], xyz.shape[1]
    if n_rep < 2 or n_steps < 2:
        return {"checked": False}
    rng = np.random.default_rng(0)
    same, other = [], []
    for _ in range(n_pairs):
        k = int(rng.integers(0, n_steps - 1))
        i, j = rng.choice(n_rep, size=2, replace=False)
        same.append(ca_rmsd(xyz[i, k][ca], xyz[i, k + 1][ca]))
        other.append(ca_rmsd(xyz[i, k][ca], xyz[j, k + 1][ca]))
    res = {"checked": True,
           "median_rmsd_same_replica_nm": float(np.median(same)),
           "median_rmsd_other_replica_nm": float(np.median(other))}
    res["ok"] = res["median_rmsd_same_replica_nm"] < 0.8 * (
        res["median_rmsd_other_replica_nm"])
    return res


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--pdb-id", help="PDB ID, e.g. 2LZM")
    g.add_argument("--pdb", type=Path, help="structure or system YAML file")
    p.add_argument("-o", "--out", type=Path, required=True)
    p.add_argument("--n-replicas", type=int, default=20,
                   help="independent start structures (= replicas)")
    p.add_argument("--n-steps", type=int, default=200,
                   help="propagation steps per replica (frames = steps + 1)")
    p.add_argument("--lag-ns", type=float, default=1.0)
    p.add_argument("--engine", choices=("api", "cli"), default="api")
    p.add_argument("--checkpoint", default=None,
                   help="e.g. bioemu2-preview (default: model default)")
    p.add_argument("--keep-steps", action="store_true",
                   help="keep the per-step folders after merging")
    return p


def main(argv=None) -> int:
    """Run the rollout, merge it and check the pairing."""
    args = build_parser().parse_args(argv)
    out = args.out
    steps_dir = out / "steps"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    (rollout_api if args.engine == "api" else rollout_cli)(args, steps_dir)
    xyz = merge(steps_dir, out, args.n_steps, args.lag_ns)
    pairing = check_pairing(xyz, out / "topology.pdb")
    meta = {"pdb_id": args.pdb_id, "pdb": str(args.pdb) if args.pdb else None,
            "lag_ns": args.lag_ns, "n_replicas": int(xyz.shape[0]),
            "n_frames_per_replica": int(xyz.shape[1]),
            "engine": args.engine, "checkpoint": args.checkpoint,
            "seconds": round(time.time() - t0, 1), "pairing_check": pairing}
    (out / "rollout.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    if pairing.get("checked") and not pairing["ok"]:
        print("PAIRING CHECK FAILED: output order does not follow the start "
              "structures; the replicas are not trajectories.",
              file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
