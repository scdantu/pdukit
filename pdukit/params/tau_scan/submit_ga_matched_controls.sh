#!/bin/bash
# Matched-data controls for the mutation test. GA has 13 replicas x ~1.7 us;
# IV/QV/L99A have 3 replicas x 1 us (50001 frames each). Channel counts depend
# on the amount of data (noise threshold), so compare variants with GA
# subsets of the same size: 4 GA triplets, first 50001 frames per replica.
# The spread among GA triplets is the null for "a mutation changed the channels".
set -euo pipefail
P=pdukit/params/tau_scan/t4l_tau0500ps_fixed.ppf
mkdir -p logs
for trip in "R0001 R0002 R0003" "R0004 R0005 R0006" "R0007 R0008 R0009" "R00010 R00011 R00012"; do
  tag=$(echo "$trip" | tr -d ' R')
  sbatch --array=0 --export=ALL,PARAMS=$P,REPLICAS="$trip",PDU_ARGS="--frames 50001",OUTDIR=/projects/u6wf/scdantu/Projects/pdudynamics/results/lyso_GA3_${tag}_t4l_tau0500ps_fixed pdukit/hpc/slurm/submit_incoming_lysozyme.slurm
done
