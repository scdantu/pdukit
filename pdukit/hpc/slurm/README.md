# Example Slurm scripts

These scripts run the incoming mode of COG on a GPU node (developed on Isambard-AI, 1 GPU per task). Adapt them
to your cluster.

1. Create a virtual environment and `pip install -e ".[xtc,gpu]"` from the repository root.
2. Create `logs/` in the directory you submit from (`mkdir -p logs`).
3. Set paths with environment variables, or edit the defaults at the top of each script:
   `WORKINGDIR` (default: submit directory), `MDDIR` (default: `$WORKINGDIR/mddata`),
   `OUTROOT` (default: `$WORKINGDIR/results`), `VENV` (default: `$HOME/venvs/pdu`),
   `CODEDIR` (default: the git checkout you submit from).
4. The lysozyme script expects trajectories named `CONV-PROD-R<id>-ATM-110_129-<VARIANT>*.xtc`
   plus `ATM-110_129-<VARIANT>.tpr` in `MDDIR`; adapt the file-name pattern for your data.
5. Submit with a parameter file (`PARAMS`; a `key = value` `.ppf` file, see the top-level
   README). To change a setting for one job, pass it as KEY=VALUE words in `PDU_ARGS` (they override
   the file); `MODE=calibrate` stops after calibration:

```bash
sbatch --array=0 --export=ALL,PARAMS=pdukit/params/t4l_incoming.ppf \
       pdukit/hpc/slurm/submit_incoming_lysozyme.slurm
sbatch --export=ALL,PARAMS=pdukit/params/syt_c2a_incoming.ppf,PDU_ARGS="lag=10" \
       pdukit/hpc/slurm/submit_incoming_synaptotagmin.slurm
```

The scripts for the older channels and cohesion modes (`submit_cog2_*.slurm`, `*_channels.ppf`,
`*_cohesion.ppf`, `lag1ns.ppf`) are in the git tag `archive/pre-incoming-only`.
