# BioEmu2 rollouts at a 1 ns lag

`bioemu2_rollout.py` makes trajectories from BioEmu2: N start structures are
sampled and each is propagated K times by the lag time (default 1 ns).  Each
start structure is one replica of K + 1 frames, written as `rep_000.xtc`,
... with a shared `topology.pdb`, so `pdu run` reads them like MD replicas.

    sbatch --export=ALL,PDB_ID=2LZM,N_REPLICAS=3,N_STEPS=20,RUN_PDU=0 \
           pdukit/hpc/bioemu2/submit_bioemu2_rollout.slurm      # speed test
    sbatch --export=ALL,PDB_ID=2LZM,N_REPLICAS=20,N_STEPS=200 \
           pdukit/hpc/bioemu2/submit_bioemu2_rollout.slurm      # real run

The MD reference is the same protein strided to 1 ns and run with the same
preset, `pdukit/params/lag1ns.ppf`:

    PARAMS=pdukit/params/lag1ns.ppf PDU_ARGS="--stride 50"   # T4L, 20 ps
    PARAMS=pdukit/params/lag1ns.ppf PDU_ARGS="--stride 10"   # Syt, 100 ps

## Not verified yet (checked only against a stub of the bioemu2 API)

1. The Python API names (`System.from_pdb_id`, `EmuSession`, `sample`,
   `propagate`, `save_frames`) follow the bioemu2 README.  If `propagate`
   does not accept its own output, use `ENGINE=cli`.
2. The i-th structure written by a propagation is assumed to be the endpoint
   of the i-th start structure.  The script checks this (consecutive frames of
   a replica must be closer than frames of different replicas) and exits with
   status 2 if it fails.
3. The speed of one step, and so how many replicas and steps are affordable.
4. Supported lags are 1 to 1000 ns (bioemu2 `docs/propagation.md`), so
   1 ns is the shortest step the model offers; motions faster than 1 ns
   cannot be resolved by a rollout.

BioEmu-2 is an all-atom diffusion model (about 500M parameters, MODEL_CARD).
Each propagation is one draw from a learned transition kernel
p(x at t+lag | x at t); a rollout is a Markov chain of such draws, not
integrated dynamics.  No kinetic validation is published, and sampling
frequencies are not calibrated Boltzmann probabilities.  The bioemu2 package
also has `simulate_trajectory()`, which chains steps and reuses the system
embedding; the script here loops over `propagate` instead.
