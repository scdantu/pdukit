#!/bin/bash
#SBATCH --job-name=pdu_cog2_syt
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --output=logs/pdu_cog2_syt_%j.out
#SBATCH --error=logs/pdu_cog2_syt_%j.err

# PDU on synaptotagmin-1 C2A, calcium-free (Zenodo 6908073; the 184 us data
# set iVAMPnets used).  Every */hsynapto-protein-stride10.xtc under SYTDIR is
# one replica.
#
#   sbatch --export=ALL,PARAMS=pdukit/hpc/slurm/params/syt_c2a_channels.ppf \
#          submit_cog2_synaptotagmin.slurm
#   sbatch --export=ALL,MAX_REPLICAS=10,MODE=calibrate ... # quick test on the
#                                                          # first 10 replicas
#
# All settings are in the parameter file (PARAMS, default
# syt_c2a_channels.ppf); see `pdu run -h` for every option.  To change a
# setting for one job give it as extra options (they override the file):
#
#   --export=ALL,PDU_ARGS="--lag 10 --n-holdout 10"
#
# LAG and VERIFY_LAGS are in FRAMES.  ONE FRAME = 100 ps: the Zenodo record
# gives 2 us per trajectory and each file has 20,000 frames (the xtc header's
# 10 ps is wrong -- its times were not rescaled when the files were strided).
# SIZE.  92 x 20,000 frames = 1.84 M frames: start with MAX_REPLICAS=10.
# n-holdout = k in the .ppf holds out the last k replicas (sorted order).

set -euo pipefail

export TMPDIR="/tmp/${USER:-user}_${SLURM_JOB_ID:-local}"
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT

WORKINGDIR="${WORKINGDIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
SYTDIR="${SYTDIR:-$WORKINGDIR/mddata/syt-upload}"
TOP="${TOP:-$SYTDIR/hsynapto-protein.pdb}"     # protein-only; see README
XTC_NAME="${XTC_NAME:-hsynapto-protein-stride10.xtc}"
_SUBMIT_REPO="$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel 2>/dev/null || true)"
CODEDIR="${CODEDIR:-${_SUBMIT_REPO:-$WORKINGDIR/pdu}}"
VENV="${VENV:-$HOME/venvs/pdu}"

PARAMS="${PARAMS:-$CODEDIR/pdukit/hpc/slurm/params/syt_c2a_channels.ppf}"
case "$PARAMS" in /*) ;; *) PARAMS="${SLURM_SUBMIT_DIR:-$PWD}/$PARAMS" ;; esac
[ -f "$PARAMS" ] || { echo "parameter file not found: $PARAMS" >&2; exit 1; }
PRESET="$(basename "$PARAMS" .ppf)"

MODE="${MODE:-full}"                 # full | calibrate
MAX_REPLICAS="${MAX_REPLICAS:-0}"    # 0 = all
PDU_ARGS="${PDU_ARGS:-}"             # extra pdu run options (override the file)
FEATURE_CACHE="${FEATURE_CACHE:-$WORKINGDIR/cache/features}"
OUTDIR="${OUTDIR:-$WORKINGDIR/results/syt_C2A_${PRESET}_${SLURM_JOB_ID:-local}}"

mapfile -t XTC < <(ls "$SYTDIR"/*/"$XTC_NAME" | sort -V)
if [ "$MAX_REPLICAS" -gt 0 ]; then XTC=( "${XTC[@]:0:$MAX_REPLICAS}" ); fi
NREP=${#XTC[@]}
if [ "$NREP" -lt 3 ]; then echo "found only $NREP replicas under $SYTDIR"; exit 1; fi

mkdir -p "$WORKINGDIR/logs" "$OUTDIR"
printf '%s\n' "${XTC[@]}" > "$OUTDIR/replicas.txt"
module load cray-python 2>/dev/null || true
source "$VENV/bin/activate"
cd "$CODEDIR"
export PYTHONPATH="$CODEDIR${PYTHONPATH:+:$PYTHONPATH}"

echo "================================================================"
echo "PDU  synaptotagmin C2A  $(date +%F)  mode=$MODE params=$PRESET"
echo "Job: ${SLURM_JOB_ID:-local} host=$(hostname)"
echo "Code: $CODEDIR  branch=$(git -C "$CODEDIR" rev-parse --abbrev-ref HEAD) commit=$(git -C "$CODEDIR" rev-parse --short HEAD)"
echo "Topology: $TOP"
echo "Replicas: $NREP (list in $OUTDIR/replicas.txt)"
echo "Output: $OUTDIR"
echo "================================================================"
[ -s "$TOP" ] || { echo "MISSING topology $TOP"; exit 1; }

EXTRA=()
[ "$MODE" = calibrate ] && EXTRA+=( --calibrate-only )

# shellcheck disable=SC2086   # PDU_ARGS is meant to be split into words
python3 -m pdukit run \
    -f "${XTC[@]}" -s "$TOP" -p "$PARAMS" -o "$OUTDIR" \
    --device cuda --feature-cache "$FEATURE_CACHE" \
    ${EXTRA[@]+"${EXTRA[@]}"} $PDU_ARGS

echo "DONE $(date +%T)  results: $OUTDIR/results.json  units: $OUTDIR/units.csv"
