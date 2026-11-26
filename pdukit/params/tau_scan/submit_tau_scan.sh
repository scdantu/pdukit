#!/bin/bash
# tau scan (100 ps, 500 ps, 1, 2, 5, 10 ns) for T4L GA and Syt1 C2A, cog1.1a+ports.
#   detect : full detection at each tau (growth, merge, split, ports)
#   fixed  : the cog1.1a units (tau 500 ps) + named sites scored at each tau, no search
# Run from the repository root on Isambard after `git pull`; logs/ must exist.
#   bash pdukit/params/tau_scan/submit_tau_scan.sh            # all 24 jobs
#   MODES=fixed bash pdukit/params/tau_scan/submit_tau_scan.sh # fixed sets only
set -euo pipefail
P=pdukit/params/tau_scan
MODES="${MODES:-detect fixed}"
TAUS="${TAUS:-0100ps 0500ps 1ns 2ns 5ns 10ns}"
mkdir -p logs
for mode in $MODES; do
  for tau in $TAUS; do
    sbatch --array=0 --export=ALL,PARAMS=$P/t4l_tau${tau}_${mode}.ppf pdukit/hpc/slurm/submit_incoming_lysozyme.slurm
    sbatch --export=ALL,PARAMS=$P/syt_c2a_tau${tau}_${mode}.ppf pdukit/hpc/slurm/submit_incoming_synaptotagmin.slurm
  done
done
# Mutation test (cog1.1a units scored on GA, IV, QV, L99A at tau 500 ps):
#   sbatch --array=0-3 --export=ALL,PARAMS=$P/t4l_tau0500ps_fixed.ppf pdukit/hpc/slurm/submit_incoming_lysozyme.slurm
