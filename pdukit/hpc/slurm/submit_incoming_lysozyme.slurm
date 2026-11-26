#!/bin/bash
#SBATCH --job-name=pdu_incoming_t4l
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --array=0
#SBATCH --output=logs/pdu_incoming_t4l_%A_%a.out
#SBATCH --error=logs/pdu_incoming_t4l_%A_%a.err

# Submit from the updated repository; create logs/ before sbatch.
# Export WORKINGDIR=/projects/u6wf/scdantu/Projects/pdudynamics (code: /projects/u6wf/applications/pdu-extractor, venv: /projects/u6wf/applications/pypdu, or export CODEDIR / VENV).
# Default: all replicas, incoming growth, recursive spectral splits, mode recovery.
# Lysozyme task 0 = GA; --array=0-3 also runs IV, QV and L99A.

set -euo pipefail

export TMPDIR="/tmp/${USER:-user}_${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT

WORKINGDIR="${WORKINGDIR:-/projects/u6wf/scdantu/Projects/pdudynamics}"
MDDIR="${MDDIR:-$WORKINGDIR/mddata}"
OUTROOT="${OUTROOT:-$WORKINGDIR/results}"
# The repository this job runs: by default the one you submitted from
# (sbatch copies the script elsewhere), else $WORKINGDIR/pdu.
_SUBMIT_REPO="$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel 2>/dev/null || true)"
CODEDIR="${CODEDIR:-${_SUBMIT_REPO:-/projects/u6wf/applications/pdu-extractor}}"
VENV="${VENV:-/projects/u6wf/applications/pypdu}"

PARAMS="${PARAMS:-$CODEDIR/pdukit/params/t4l_incoming.ppf}"
case "$PARAMS" in /*) ;; *) PARAMS="${SLURM_SUBMIT_DIR:-$PWD}/$PARAMS" ;; esac
[ -f "$PARAMS" ] || { echo "parameter file not found: $PARAMS" >&2; exit 1; }
PRESET="$(basename "$PARAMS" .ppf)"

VARIANTS=(GA IV QV L99A)
VARIANT="${VARIANT:-${VARIANTS[${SLURM_ARRAY_TASK_ID:-0}]}}"
MODE="${MODE:-full}"                 # full | calibrate
PDU_ARGS="${PDU_ARGS:-}"             # extra settings, e.g. "lag=20 dec-seeds=64" (override the file)
FEATURE_CACHE="${FEATURE_CACHE:-$WORKINGDIR/cache/features_v2}"
OUTDIR="${OUTDIR:-$OUTROOT/lyso_${VARIANT}_${PRESET}_${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}}"

TOP="$MDDIR/ATM-110_129-${VARIANT}.tpr"
# Exact trajectory pattern supplied for these runs. No suffix/restart glob.
REPLICAS="${REPLICAS:-all}"
DRYRUN="${DRYRUN:-0}"
[ -d "$MDDIR" ] || { echo "Missing MD directory: $MDDIR" >&2; exit 1; }
shopt -s nullglob
XTC=()
if [ "$REPLICAS" = all ]; then
    XTC=( "$MDDIR"/CONV-PROD-R*-ATM-110_129-"${VARIANT}".xtc )
else
    for r in $REPLICAS; do
        file="$MDDIR/CONV-PROD-${r}-ATM-110_129-${VARIANT}.xtc"
        [ -s "$file" ] || { echo "Missing trajectory: $file" >&2; exit 1; }
        XTC+=( "$file" )
    done
fi
shopt -u nullglob
NREP=${#XTC[@]}
[ "$NREP" -gt 0 ] || {
    echo "No trajectories match $MDDIR/CONV-PROD-R*-ATM-110_129-${VARIANT}.xtc" >&2
    exit 1
}
[ -s "$TOP" ] || { echo "Missing topology: $TOP" >&2; exit 1; }
# Discovery-only dry run needs neither a virtual environment nor a GPU.
if [ "$DRYRUN" = 1 ]; then
    printf 'Topology: %s\nTrajectories (%s):\n' "$TOP" "$NREP"
    printf '%s\n' "${XTC[@]}"
    exit 0
fi

mkdir -p "$WORKINGDIR/logs" "$OUTDIR"
module load cray-python 2>/dev/null || true
source "$VENV/bin/activate"
cd "$CODEDIR"
export PYTHONPATH="$CODEDIR${PYTHONPATH:+:$PYTHONPATH}"

echo "================================================================"
echo "PDU  $(date +%F)  variant=$VARIANT mode=$MODE params=$PRESET"
echo "Job: ${SLURM_JOB_ID:-local} task=${SLURM_ARRAY_TASK_ID:-0} host=$(hostname) time=$(date +%T)"
echo "Code: $CODEDIR  branch=$(git -C "$CODEDIR" rev-parse --abbrev-ref HEAD) commit=$(git -C "$CODEDIR" rev-parse --short HEAD)"
echo "Output: $OUTDIR"
python3 -c "import torch; print('CUDA:', torch.cuda.is_available(), 'device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')" || echo "!! torch not importable -- running on the CPU (numpy)"
echo "================================================================"

if [ ! -s "$TOP" ] || [ "$NREP" -lt 1 ]; then
    echo "MISSING: $TOP or no replicas -- variant $VARIANT skipped"; exit 0
fi
echo "Replicas ($NREP):"
i=0
for x in "${XTC[@]}"; do
    n=$(echo "$x" | tr ',' '\n' | wc -l)
    echo "  [$i] $(basename "${x%%,*}")  ($n part(s))"
    i=$((i + 1))
done
printf '%s\n' "${XTC[@]}" > "$OUTDIR/replicas.txt"
if [ "$DRYRUN" = 1 ]; then echo "DRYRUN: stopping"; exit 0; fi

EXTRA=()
[ "$MODE" = calibrate ] && EXTRA+=( calibrate-only=1 )

# shellcheck disable=SC2086   # PDU_ARGS is meant to be split into words
python3 -m pdukit run \
    -f "${XTC[@]}" -s "$TOP" -p "$PARAMS" -o "$OUTDIR" \
    --set device=cuda feature-cache="$FEATURE_CACHE" grow-objective=incoming \
    ${EXTRA[@]+"${EXTRA[@]}"} $PDU_ARGS

echo "variant $VARIANT DONE  $(date +%T)   results: $OUTDIR/results.json  candidates: $OUTDIR/incoming_candidates.csv; growth: $OUTDIR/incoming_growth.csv"

if [ -f "$OUTDIR/incoming_split.json" ]; then
    echo "Split hierarchy: $OUTDIR/incoming_split.json; final supports: $OUTDIR/incoming_split_units.csv"
fi
if [ -f "$OUTDIR/global_mode_recovery.md" ]; then
    echo "Global-mode recovery: $OUTDIR/global_mode_recovery.md"
fi
