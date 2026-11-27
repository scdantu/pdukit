#!/bin/bash
# FINAL runs before Isambard access ends (30 Nov 2026): T4L (WT GA + L99A), Syt1 C2A, mdCATH.
# Run from the repository root on Isambard after `git pull` on this branch.
#   bash pdukit/params/final/submit_final.sh t4l            # 14 jobs: 7 tau x (GA, L99A)
#   bash pdukit/params/final/submit_final.sh syt            # 7 jobs
#   DOMAINS=<list> N=<n> bash pdukit/params/final/submit_final.sh mdcath   # array over N domains
#   TAUS="0500ps" bash pdukit/params/final/submit_final.sh t4l              # one tau only
# Stride 5 (100 ps frames) and rse-min-share 0.025 since 27 Nov: K ~15 at 100-200 ps (~30 GB);
# 100 and 200 ps still get a whole node (--exclusive, 24 h) to be safe.
set -euo pipefail
P=pdukit/params/final
WHAT="${1:?t4l | syt | mdcath}"
TAUS="${TAUS:-0100ps 0200ps 0500ps 1ns 2ns 5ns 10ns}"
mkdir -p logs
big() { case "$1" in 0100ps|0200ps) echo "--exclusive --time=24:00:00" ;; *) echo "" ;; esac; }
case "$WHAT" in
  t4l)
    for tau in $TAUS; do
      # shellcheck disable=SC2046
      sbatch $(big "$tau") --array=0,3 --export=ALL,PARAMS=$P/t4l_final_tau${tau}.ppf pdukit/hpc/slurm/submit_incoming_lysozyme.slurm
    done ;;
  syt)
    for tau in $TAUS; do
      # shellcheck disable=SC2046
      sbatch $(big "$tau") --export=ALL,PARAMS=$P/syt_final_tau${tau}.ppf pdukit/hpc/slurm/submit_incoming_synaptotagmin.slurm
    done ;;
  mdcath)
    : "${DOMAINS:?set DOMAINS=<domain list file>}"; : "${N:?set N=<number of domains in the list>}"
    sbatch --array=1-"$N" --export=ALL,PARAMS=$P/mdcath_final.ppf,DOMAINS="$DOMAINS" pdukit/hpc/slurm/submit_incoming_mdcath.slurm ;;
  *) echo "unknown: $WHAT (t4l | syt | mdcath)" >&2; exit 1 ;;
esac
