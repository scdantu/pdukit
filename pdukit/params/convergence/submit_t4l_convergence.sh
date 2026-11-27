#!/bin/bash
# How much trajectory is needed?  14 runs of T4L GA at tau = 1 ns, K = 8.
# Run from the repository root on Isambard:
#   bash pdukit/params/convergence/submit_t4l_convergence.sh          # all 14
#   ONLY="thin10 thin25" bash pdukit/params/convergence/submit_t4l_convergence.sh
# Then, when they are done:
#   pdu converge $OUT/full_seed0 $OUT/* -o $OUT/tables
#
# Arms (10 replicas of ~5 us, 20 ps frames; lag 1 ns = 50 frames):
#   noise floor : full_seed0 (reference), full_seed1, half_A (R1-R5), half_B (R6-R10)
#   thinning    : every 2nd ... 50th frame: lag and high-pass window divided by the
#                 same factor so they stay 1 ns and 20 ns; frame-ps scaled (block
#                 lengths use it)
#   truncation  : first 50 / 25 / 12.5 % of every replica
#   replicas    : 2 and 3 replicas
# A cheaper run is good enough when it agrees with full_seed0 as well as the
# noise-floor pairs agree with each other (convergence_pairs.csv).
set -euo pipefail
P=pdukit/params/convergence/t4l_conv_1ns.ppf
SLURM=pdukit/hpc/slurm/submit_incoming_lysozyme.slurm
WD="${WORKINGDIR:-/projects/u6wf/scdantu/Projects/pdudynamics}"
OUT="${OUT:-${OUTROOT:-$WD/results}/convergence_t4l_1ns}"
mkdir -p logs "$OUT"

# label | replicas (blank = all) | settings
RUNS=(
  "full_seed0||"
  "full_seed1||seed=1"
  "half_A|R0001 R0002 R0003 R0004 R0005|"
  "half_B|R0006 R0007 R0008 R0009 R00010|"
  "thin2||stride=2 lag=25 highpass-window=500 frame-ps=40"
  "thin5||stride=5 lag=10 highpass-window=200 frame-ps=100"
  "thin10||stride=10 lag=5 highpass-window=100 frame-ps=200"
  "thin25||stride=25 lag=2 highpass-window=40 frame-ps=500"
  "thin50||stride=50 lag=1 highpass-window=20 frame-ps=1000"
  "trunc50||frames=130000"
  "trunc25||frames=65000"
  "trunc12||frames=32500"
  "reps2|R0001 R0006|"
  "reps3|R0001 R0004 R0007|"
)
for line in "${RUNS[@]}"; do
  IFS='|' read -r label reps sets <<< "$line"
  if [ -n "${ONLY:-}" ] && [[ " $ONLY " != *" $label "* ]]; then continue; fi
  # values with spaces travel in the environment (--export=ALL), not in --export
  PARAMS="$P" OUTDIR="$OUT/$label" REPLICAS="${reps:-all}" PDU_ARGS="$sets" \
    sbatch --job-name="conv_$label" --array=0 --export=ALL "$SLURM"
done
echo "outputs in $OUT; compare with: pdu converge $OUT/full_seed0 $OUT/* -o $OUT/tables"
