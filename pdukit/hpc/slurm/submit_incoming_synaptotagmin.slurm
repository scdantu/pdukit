#!/bin/bash
#SBATCH --job-name=pdu_incoming_syt
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --output=logs/pdu_incoming_syt_%j.out
#SBATCH --error=logs/pdu_incoming_syt_%j.err

# Submit from the updated repository; create logs/ before sbatch.
# Export WORKINGDIR=/projects/u6wf/scdantu/Projects/pdudynamics (code: /projects/u6wf/applications/pdu-extractor, venv: /projects/u6wf/applications/pypdu, or export CODEDIR / VENV).
# Default: all replicas, incoming growth, recursive spectral splits, mode recovery.
# Lysozyme task 0 = GA; --array=0-3 also runs IV, QV and L99A.

set -euo pipefail

export TMPDIR="/tmp/${USER:-user}_${SLURM_JOB_ID:-local}"
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT

WORKINGDIR="${WORKINGDIR:-/projects/u6wf/scdantu/Projects/pdudynamics}"
SYTDIR="${SYTDIR:-$WORKINGDIR/mddata/syt-upload}"
TOP="${TOP:-$SYTDIR/hsynapto-protein.pdb}"     # protein-only; see README
XTC_NAME="${XTC_NAME:-*.xtc}"
_SUBMIT_REPO="$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel 2>/dev/null || true)"
CODEDIR="${CODEDIR:-${_SUBMIT_REPO:-/projects/u6wf/applications/pdu-extractor}}"
VENV="${VENV:-/projects/u6wf/applications/pypdu}"

PARAMS="${PARAMS:-$CODEDIR/pdukit/params/syt_c2a_incoming.ppf}"
case "$PARAMS" in /*) ;; *) PARAMS="${SLURM_SUBMIT_DIR:-$PWD}/$PARAMS" ;; esac
[ -f "$PARAMS" ] || { echo "parameter file not found: $PARAMS" >&2; exit 1; }
PRESET="$(basename "$PARAMS" .ppf)"

MODE="${MODE:-full}"                 # full | calibrate
MAX_REPLICAS="${MAX_REPLICAS:-0}"    # 0 = all
PDU_ARGS="${PDU_ARGS:-}"             # extra settings, e.g. "lag=20 dec-seeds=64" (override the file)
FEATURE_CACHE="${FEATURE_CACHE:-$WORKINGDIR/cache/features_v2}"
OUTDIR="${OUTDIR:-$WORKINGDIR/results/syt_C2A_${PRESET}_${SLURM_JOB_ID:-local}}"

[ -d "$SYTDIR" ] || { echo "Missing MD directory: $SYTDIR" >&2; exit 1; }
# Every matching XTC below a replica subdirectory, including deeper folders.
XTC=()
while IFS= read -r file; do XTC+=( "$file" ); done < <(
    find -L "$SYTDIR" -mindepth 2 -type f -name "$XTC_NAME" | sort -V
)
if [ "$MAX_REPLICAS" -gt 0 ]; then XTC=( "${XTC[@]:0:$MAX_REPLICAS}" ); fi
NREP=${#XTC[@]}
[ "$NREP" -gt 0 ] || { echo "No $XTC_NAME files below $SYTDIR" >&2; exit 1; }
[ -s "$TOP" ] || { echo "Missing topology: $TOP" >&2; exit 1; }
if [ "${DRYRUN:-0}" = 1 ]; then
    printf 'Topology: %s\nTrajectories (%s):\n' "$TOP" "$NREP"
    printf '%s\n' "${XTC[@]}"
    exit 0
fi

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
[ "$MODE" = calibrate ] && EXTRA+=( calibrate-only=1 )

# shellcheck disable=SC2086   # PDU_ARGS is meant to be split into words
python3 -m pdukit run \
    -f "${XTC[@]}" -s "$TOP" -p "$PARAMS" -o "$OUTDIR" \
    --set device=cuda feature-cache="$FEATURE_CACHE" grow-objective=incoming \
    ${EXTRA[@]+"${EXTRA[@]}"} $PDU_ARGS

echo "DONE $(date +%T)  results: $OUTDIR/results.json  candidates: $OUTDIR/incoming_candidates.csv; growth: $OUTDIR/incoming_growth.csv"

if [ -f "$OUTDIR/incoming_split.json" ]; then
    echo "Split hierarchy: $OUTDIR/incoming_split.json; final supports: $OUTDIR/incoming_split_units.csv"
fi
if [ -f "$OUTDIR/global_mode_recovery.md" ]; then
    echo "Global-mode recovery: $OUTDIR/global_mode_recovery.md"
fi
