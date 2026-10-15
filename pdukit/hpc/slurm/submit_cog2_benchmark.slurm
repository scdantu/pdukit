#!/bin/bash
#SBATCH --job-name=pdu_cog2_bench
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --array=0-20
#SBATCH --output=logs/pdu_cog2_bench_%A_%a.out
#SBATCH --error=logs/pdu_cog2_bench_%A_%a.err

# COG 2.1 vs COG v1.0 on planted systems with a known answer.
# ONE (design, seed) PER ARRAY TASK: task = 3 * design + seed, so tasks
# 0-2 are design 0 seeds 0-2, 3-5 design 1, ... (the first version ran
# three seeds in one task and lost the last ones to the time limit).
# (pdukit/lib/pdu/benchmark.py).  One planted design per array task; each
# runs the full COG 2.1 pipeline AND COG v1.0 on the same discovery data and
# scores both against the truth (results.json: discovery.vs_truth,
# cog1.vs_truth, discovery.faces).
#
#   0 base         6 units x 10 residues, 1 face residue per gear, decoy 3-4
#   1 discontinuous  as 0, every unit split into two sequence segments
#   2 weak         as 0, coupling gain 0.20 (gears near the noise)
#   3 strong       as 0, coupling gain 0.60
#   4 short        as 0, 3000 frames per replica (data-limited)
#   5 big          10 units x 12 residues, 2 face residues per gear
#   6 nofaces      as 0 with NO face residues: the clean test of the channel
#                  count (faces add real channels the gear list does not count)
#
#   sbatch submit_cog2_benchmark.slurm               # 7 designs x 3 seeds
#   sbatch --array=15-17 submit_cog2_benchmark.slurm # design 5 ('big') only
#   sbatch --array=0-17 --export=ALL,N_SEEDS=5 ...   # with --array=0-29
#
# Runs on the GPU (--device cuda): run 7260882 used the CPU and the 'big'
# design (120 residues) did not get past calibration in 4 h.

set -euo pipefail

export TMPDIR="/tmp/${USER:-user}_${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT

WORKINGDIR="${WORKINGDIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
OUTROOT="${OUTROOT:-$WORKINGDIR/results/cog2_benchmark_${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}}"
# The repository this job runs: by default the one you submitted from
# (sbatch copies the script elsewhere, so look at the submit directory),
# else $WORKINGDIR/pdu.  Override with --export=ALL,CODEDIR=...
_SUBMIT_REPO="$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel 2>/dev/null || true)"
CODEDIR="${CODEDIR:-${_SUBMIT_REPO:-$WORKINGDIR/pdu}}"
VENV="${VENV:-$HOME/venvs/pdu}"
N_SEEDS="${N_SEEDS:-3}"

TASK="${SLURM_ARRAY_TASK_ID:-0}"
DESIGN_ID=$(( TASK / N_SEEDS ))
BENCH_SEED=$(( TASK % N_SEEDS ))
SEARCH_GRAPH="${SEARCH_GRAPH:-predictive}"   # predictive | contact | union
GRAPH_K="${GRAPH_K:-5}"
SEEDING="${SEEDING:-random}"
BASELINE="${BASELINE:-graph}"
COMMON=( --planted --n-replicas 3 --frames-per-replica 10000
         --planted-m 4 --latent-dim 5 --decoy-pairs 3-4
         --unit-dim 6 --env-dim 16 --n-null 48 --verify-null 24
         --verify-lags 1,5,25 --dec-seeds 24 --dec-iters 6 --max-size 16
         --max-unit 30 --sizes 2-30 --min-supports 30 --max-supports 150
         --n-controls 8 --compare-cog1 --device cuda )
case "$DESIGN_ID" in
  0) NAME=base;          DESIGN=( --n-units 6 --unit-size 10 --face-residues 1 ) ;;
  1) NAME=discontinuous; DESIGN=( --n-units 6 --unit-size 10 --face-residues 1 --discontinuous ) ;;
  2) NAME=weak;          DESIGN=( --n-units 6 --unit-size 10 --face-residues 1 --coupling-gain 0.20 ) ;;
  3) NAME=strong;        DESIGN=( --n-units 6 --unit-size 10 --face-residues 1 --coupling-gain 0.60 ) ;;
  4) NAME=short;         DESIGN=( --n-units 6 --unit-size 10 --face-residues 1 --frames-per-replica 3000 ) ;;
  5) NAME=big;           DESIGN=( --n-units 10 --unit-size 12 --face-residues 2 --max-unit 40 --sizes 2-40 ) ;;
  6) NAME=nofaces;       DESIGN=( --n-units 6 --unit-size 10 --face-residues 0 ) ;;
  *) echo "unknown design $DESIGN_ID (task $TASK)"; exit 1 ;;
esac

mkdir -p "$WORKINGDIR/logs" "$OUTROOT"
module load cray-python 2>/dev/null || true
source "$VENV/bin/activate"
cd "$CODEDIR"
export PYTHONPATH="$CODEDIR${PYTHONPATH:+:$PYTHONPATH}"

echo "================================================================"
echo "PDU COG 2.1 BENCHMARK  $(date +%F)  design=$NAME  seed=$BENCH_SEED"
echo "Job: ${SLURM_JOB_ID:-local} task=$TASK host=$(hostname)"
echo "Code: $CODEDIR  branch=$(git -C "$CODEDIR" rev-parse --abbrev-ref HEAD) commit=$(git -C "$CODEDIR" rev-parse --short HEAD)"
echo "================================================================"

OUT="$OUTROOT/${NAME}_seed${BENCH_SEED}"
echo "--- $NAME seed $BENCH_SEED -> $OUT"
python3 -m pdukit.lib.pdu.cog2_run "${COMMON[@]}" "${DESIGN[@]}" \
    --search-graph "$SEARCH_GRAPH" --graph-k "$GRAPH_K" \
    --seeding "$SEEDING" --baseline "$BASELINE" \
    --bench-seed "$BENCH_SEED" --seed "$BENCH_SEED" --outdir "$OUT"

echo "DONE $(date +%T)  results under $OUTROOT"
