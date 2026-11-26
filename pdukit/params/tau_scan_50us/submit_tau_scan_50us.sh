#!/bin/bash
# T4L tau scan, WT (GA) and L99A, all replicas (~50 us each): 100 ps, 200 ps, 500 ps, 1, 2, 5, 10 ns.
# Run from the repository root on Isambard after `git pull`:
#   bash pdukit/params/tau_scan_50us/submit_tau_scan_50us.sh             # 14 jobs (7 tau x 2 proteins)
#   TAUS="0500ps 5ns" bash pdukit/params/tau_scan_50us/submit_tau_scan_50us.sh
# Optional replica-halves check (reproducibility), e.g. at 500 ps and 5 ns:
#   HALVES=1 TAUS="0500ps 5ns" bash pdukit/params/tau_scan_50us/submit_tau_scan_50us.sh
set -euo pipefail
P=pdukit/params/tau_scan_50us
TAUS="${TAUS:-0100ps 0200ps 0500ps 1ns 2ns 5ns 10ns}"
HALVES="${HALVES:-0}"
H1="R0001 R0002 R0003 R0004 R0005"
H2="R0006 R0007 R0008 R0009 R00010"
mkdir -p logs
for tau in $TAUS; do
  if [ "$HALVES" = 1 ]; then
    for half in H1 H2; do
      for v in GA L99A; do
        sbatch --export=ALL,VARIANT=$v,PARAMS=$P/t4l_tau${tau}.ppf,REPLICAS="${!half}",OUTROOT=${OUTROOT:-/projects/u6wf/scdantu/Projects/pdudynamics/results}/halves_$half \
          pdukit/hpc/slurm/submit_incoming_lysozyme.slurm
      done
    done
  else
    sbatch --array=0,3 --export=ALL,PARAMS=$P/t4l_tau${tau}.ppf pdukit/hpc/slurm/submit_incoming_lysozyme.slurm
  fi
done
