#!/bin/bash
#SBATCH --job-name=pdu_fasttest
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --output=logs/pdu_fasttest_%j.out
#SBATCH --error=logs/pdu_fasttest_%j.err
#
# Fast tests of COG 2.1 ideas on small planted systems (see
# pdukit/lib/pdu/cog2_fasttest.py).  Runs many small CPU jobs in parallel.
#   sbatch submit_cog2_fasttest.slurm
#   sbatch --export=ALL,DESIGNS=base+scattered,CONFIGS=contact+dyn+dyn_r2,SEEDS=0+1+2+3 submit_cog2_fasttest.slurm
#   (use + between items: sbatch --export splits on commas)
set -euo pipefail
WORKINGDIR="${WORKINGDIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
_SUBMIT_REPO="$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel 2>/dev/null || true)"
CODEDIR="${CODEDIR:-${_SUBMIT_REPO:-$WORKINGDIR/pdu}}"
VENV="${VENV:-$HOME/venvs/pdu}"
DESIGNS="${DESIGNS:-base,scattered,weak,strong,nofaces}"
CONFIGS="${CONFIGS:-contact,dyn,dyn_randbase,dyn_r2}"
SEEDS="${SEEDS:-0,1,2,3,4,5}"
JOBS="${JOBS:-24}"
DEVICE="${DEVICE:-cpu}"
OUT="${OUT:-$WORKINGDIR/results/fasttest_${SLURM_JOB_ID:-local}}"
mkdir -p "$WORKINGDIR/logs" "$OUT"
module load cray-python 2>/dev/null || true
source "$VENV/bin/activate"
export PYTHONPATH="$CODEDIR${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
echo "fasttest $(date +%F' '%T)  branch=$(git -C "$CODEDIR" rev-parse --abbrev-ref HEAD) commit=$(git -C "$CODEDIR" rev-parse --short HEAD)"
python3 -m pdukit.lib.pdu.cog2_fasttest --out "$OUT" --designs "$DESIGNS" \
    --configs "$CONFIGS" --seeds "$SEEDS" --jobs "$JOBS" --device "$DEVICE"
echo "DONE $(date +%T)  table: $OUT/fasttest.csv"
