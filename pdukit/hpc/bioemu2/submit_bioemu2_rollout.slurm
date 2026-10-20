#!/bin/bash
#SBATCH --job-name=bioemu2_roll
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --output=logs/bioemu2_roll_%j.out
#SBATCH --error=logs/bioemu2_roll_%j.err

# BioEmu2 rollout at a fixed lag (default 1 ns), then optionally pdu on it.
#
#   sbatch --export=ALL,PDB_ID=2LZM,N_REPLICAS=20,N_STEPS=200 \
#          pdukit/hpc/bioemu2/submit_bioemu2_rollout.slurm
#   sbatch --export=ALL,PDB_ID=2LZM,N_REPLICAS=3,N_STEPS=20,RUN_PDU=0 ...
#                                           # speed test: how long is a step?
#
# Output: $OUTDIR/{topology.pdb,rep_000.xtc,...,rollout.json}.  One frame is
# one lag time (LAG_NS), so use `lag = 1` in the pdu parameter file
# (pdukit/params/lag1ns.ppf).  The MD reference is the same protein strided to
# the same spacing, for example for T4L (20 ps frames):
#   --export=ALL,PARAMS=pdukit/params/lag1ns.ppf,PDU_ARGS="--stride 50" \
#   submit_cog2_lysozyme.slurm
#
# BIOEMU_VENV is a Python 3.12 environment with bioemu2 installed (see the
# bioemu2 README: pip install ".[cuda]") plus mdtraj and gemmi.  PDU_VENV is
# the environment of pdukit itself (it may be the same one).

set -euo pipefail

WORKINGDIR="${WORKINGDIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
_REPO="$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel 2>/dev/null || true)"
CODEDIR="${CODEDIR:-${_REPO:-$WORKINGDIR/pdukit}}"
BIOEMU_VENV="${BIOEMU_VENV:-$HOME/venvs/bioemu2}"
PDU_VENV="${PDU_VENV:-$HOME/venvs/pdu}"

PDB_ID="${PDB_ID:?set PDB_ID, e.g. 2LZM}"
N_REPLICAS="${N_REPLICAS:-20}"
N_STEPS="${N_STEPS:-200}"
LAG_NS="${LAG_NS:-1.0}"
ENGINE="${ENGINE:-api}"              # api | cli
CHECKPOINT="${CHECKPOINT:-}"         # e.g. bioemu2-preview
RUN_PDU="${RUN_PDU:-1}"
PARAMS="${PARAMS:-$CODEDIR/pdukit/params/lag1ns.ppf}"
PDU_ARGS="${PDU_ARGS:-}"
OUTDIR="${OUTDIR:-$WORKINGDIR/results/bioemu2_${PDB_ID}_lag${LAG_NS}ns_${SLURM_JOB_ID:-local}}"

mkdir -p "$WORKINGDIR/logs" "$OUTDIR"
echo "================================================================"
echo "BioEmu2 rollout  $(date +%F)  pdb=$PDB_ID replicas=$N_REPLICAS steps=$N_STEPS lag=${LAG_NS} ns engine=$ENGINE"
echo "Job: ${SLURM_JOB_ID:-local} host=$(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "Code: $CODEDIR  commit=$(git -C "$CODEDIR" rev-parse --short HEAD 2>/dev/null)"
echo "Output: $OUTDIR"
echo "================================================================"

# shellcheck disable=SC1091
source "$BIOEMU_VENV/bin/activate"
CKPT=()
[ -n "$CHECKPOINT" ] && CKPT=( --checkpoint "$CHECKPOINT" )
python "$CODEDIR/pdukit/hpc/bioemu2/bioemu2_rollout.py" \
    --pdb-id "$PDB_ID" -o "$OUTDIR" \
    --n-replicas "$N_REPLICAS" --n-steps "$N_STEPS" --lag-ns "$LAG_NS" \
    --engine "$ENGINE" ${CKPT[@]+"${CKPT[@]}"}
deactivate

[ "$RUN_PDU" = 1 ] || exit 0
mapfile -t XTC < <(ls "$OUTDIR"/rep_*.xtc | sort -V)
# shellcheck disable=SC1091
source "$PDU_VENV/bin/activate"
cd "$CODEDIR"
export PYTHONPATH="$CODEDIR${PYTHONPATH:+:$PYTHONPATH}"
# shellcheck disable=SC2086   # PDU_ARGS is meant to be split into words
python3 -m pdukit run -f "${XTC[@]}" -s "$OUTDIR/topology.pdb" \
    -p "$PARAMS" -o "$OUTDIR/pdu" --device cuda $PDU_ARGS
