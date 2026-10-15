# Example Slurm scripts

These scripts run COG 2.1 on a GPU node (developed on Isambard-AI, 1 GPU per task). Adapt them
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
   README). To change a setting for one job, pass it as options with `PDU_ARGS` (they override
   the file); `MODE=calibrate` stops after calibration:

```bash
sbatch --array=0 --export=ALL,PARAMS=pdukit/hpc/slurm/params/t4l_channels.ppf \
       pdukit/hpc/slurm/submit_cog2_lysozyme.slurm
sbatch --export=ALL,PARAMS=pdukit/hpc/slurm/params/syt_c2a_channels.ppf,PDU_ARGS="--lag 10" \
       pdukit/hpc/slurm/submit_cog2_synaptotagmin.slurm
```

`submit_cog2_benchmark.slurm` and `submit_cog2_fasttest.slurm` run the planted-system tests.
The `*_cohesion.ppf` files are experimental (cohesion as the optimisation objective).
