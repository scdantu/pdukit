#!/bin/bash
# T4L detection vs number of tICA modes per residue (rse-components 2, 4, 8, 12, 16) at tau 500 ps,
# WT (GA) and L99A, all replicas (~50 us each): 10 jobs.
# Run from the repository root on Isambard after `git pull`:
#   bash pdukit/params/modes_scan_50us/submit_modes_scan_50us.sh
#   KS="12 16" bash pdukit/params/modes_scan_50us/submit_modes_scan_50us.sh
set -euo pipefail
P=pdukit/params/modes_scan_50us
KS="${KS:-02 04 08 12 16}"
mkdir -p logs
for k in $KS; do
  k=$(printf "%02d" "$((10#$k))")
  sbatch --array=0,3 --export=ALL,PARAMS=$P/t4l_tau0500ps_K${k}.ppf pdukit/hpc/slurm/submit_incoming_lysozyme.slurm
done
