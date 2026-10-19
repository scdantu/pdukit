#!/bin/bash
#SBATCH --job-name=pdu_cog2
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --array=0-3
#SBATCH --output=logs/pdu_cog2_%A_%a.out
#SBATCH --error=logs/pdu_cog2_%A_%a.err

# PDU on T4 lysozyme, one variant per array task (GA IV QV L99A).
#
#   sbatch --export=ALL,PARAMS=pdukit/params/t4l_channels.ppf \
#          submit_cog2_lysozyme.slurm                  # all four variants
#   sbatch --array=0 ... submit_cog2_lysozyme.slurm    # GA only
#
# All settings are in the parameter file (PARAMS, default t4l_channels.ppf);
# see `pdu run -h` for every option.  To change a setting for one job, give it
# as extra command-line options (they override the file):
#
#   --export=ALL,PDU_ARGS="--lag 10 --faces"
#   --export=ALL,MODE=calibrate   # calibration only: check the thresholds,
#                                 # the ball baseline and the run time first
#
# Pooled split (n-holdout = 0 in the .ppf): discover on all replicas, then
# score the units on each replica separately.  Use n-holdout = k to hold out
# the last k replicas instead.  A replica written in several parts (restarts)
# is read as ONE trajectory: the parts are ordered by their time stamps.
#
# The log directory must exist before submitting:  mkdir -p logs

set -euo pipefail

export TMPDIR="/tmp/${USER:-user}_${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT

WORKINGDIR="${WORKINGDIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
MDDIR="${MDDIR:-$WORKINGDIR/mddata}"
OUTROOT="${OUTROOT:-$WORKINGDIR/results}"
# The repository this job runs: by default the one you submitted from
# (sbatch copies the script elsewhere), else $WORKINGDIR/pdu.
_SUBMIT_REPO="$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel 2>/dev/null || true)"
CODEDIR="${CODEDIR:-${_SUBMIT_REPO:-$WORKINGDIR/pdu}}"
VENV="${VENV:-$HOME/venvs/pdu}"

PARAMS="${PARAMS:-$CODEDIR/pdukit/params/t4l_channels.ppf}"
case "$PARAMS" in /*) ;; *) PARAMS="${SLURM_SUBMIT_DIR:-$PWD}/$PARAMS" ;; esac
[ -f "$PARAMS" ] || { echo "parameter file not found: $PARAMS" >&2; exit 1; }
PRESET="$(basename "$PARAMS" .ppf)"

VARIANTS=(GA IV QV L99A)
VARIANT="${VARIANT:-${VARIANTS[${SLURM_ARRAY_TASK_ID:-0}]}}"
MODE="${MODE:-full}"                 # full | calibrate
PDU_ARGS="${PDU_ARGS:-}"             # extra pdu run options (override the file)
FEATURE_CACHE="${FEATURE_CACHE:-$WORKINGDIR/cache/features}"
OUTDIR="${OUTDIR:-$OUTROOT/lyso_${VARIANT}_${PRESET}_${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}}"

TOP="$MDDIR/ATM-110_129-${VARIANT}.tpr"
# Replicas: every CONV-PROD-<Rid>-ATM-110_129-<VARIANT>*.xtc in MDDIR, grouped
# by replica id.  REPLICAS="R0001 R0002 ..." picks a subset.  DRYRUN=1 prints
# the replicas and their parts and stops.
REPLICAS="${REPLICAS:-all}"
DRYRUN="${DRYRUN:-0}"
if [ "$REPLICAS" = all ]; then
    REPLICAS=$(ls "$MDDIR"/CONV-PROD-R*-ATM-110_129-"${VARIANT}"*.xtc 2>/dev/null \
               | sed -E 's/.*CONV-PROD-(R[0-9]+)-ATM.*/\1/' | sort -uV | tr '\n' ' ')
fi
XTC=()
for r in $REPLICAS; do
    parts=$(ls "$MDDIR"/CONV-PROD-"${r}"-ATM-110_129-"${VARIANT}"*.xtc 2>/dev/null \
            | sort -V | paste -sd, -)
    if [ -z "$parts" ]; then echo "MISSING: replica $r of $VARIANT"; exit 1; fi
    XTC+=( "$parts" )
done
NREP=${#XTC[@]}

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
[ "$MODE" = calibrate ] && EXTRA+=( --calibrate-only )

# shellcheck disable=SC2086   # PDU_ARGS is meant to be split into words
python3 -m pdukit run \
    -f "${XTC[@]}" -s "$TOP" -p "$PARAMS" -o "$OUTDIR" \
    --device cuda --feature-cache "$FEATURE_CACHE" \
    ${EXTRA[@]+"${EXTRA[@]}"} $PDU_ARGS

echo "variant $VARIANT DONE  $(date +%T)   results: $OUTDIR/results.json  units: $OUTDIR/units.csv"
