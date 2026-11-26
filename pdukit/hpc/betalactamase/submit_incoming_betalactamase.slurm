#!/bin/bash
#SBATCH --job-name=pdu_blac
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --array=0-3
#SBATCH --output=logs/pdu_blac_%A_%a.out
#SBATCH --error=logs/pdu_blac_%A_%a.err

# PDU (current pipeline: incoming growth, merge, split, ports, mode recovery) on the DyNoPy
# beta-lactamases, one system per array task:  0 KPC2   1 PDC3   2 SHV1   3 TEM1
#
#   mkdir -p logs
#   DRYRUN=1 sbatch --array=0-3 pdukit/hpc/betalactamase/submit_incoming_betalactamase.slurm   # files only
#   sbatch --array=1 pdukit/hpc/betalactamase/submit_incoming_betalactamase.slurm              # PDC3 first
#   sbatch pdukit/hpc/betalactamase/submit_incoming_betalactamase.slurm                        # all four
#
# The data are adaptive-sampling runs written back to back (KPC2 364 x 100 frames, TEM1 404 x 120,
# SHV1 593 x 60, PDC3 30 x 1000; 1 frame ~ 1 ns).  split_segments.py cuts every segment into its
# own xtc (CA only) so that no lagged pair straddles a seam; each segment is one replica.
# The preparation is skipped when $OUTROOT/<SYSTEM>/segments.json exists (FORCE_PREP=1 redoes it).
#
# Settings: pdukit/params/betalactamase_incoming.ppf (lag = 1 frame ~ 1 ns); override one job with
#   --export=ALL,PDU_ARGS="lag=2"     --export=ALL,MODE=calibrate
# The frames follow adaptive sampling, so state populations are biased; treat population-weighted
# quantities with care (transition statistics at short lag are unbiased).

set -euo pipefail

export TMPDIR="/tmp/${USER:-user}_${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT

WORKINGDIR="${WORKINGDIR:-/projects/u6wf/scdantu/Projects/pdudynamics}"
DYNOPY="${DYNOPY:-$WORKINGDIR/mddata/DyNoPy_Data}"
_SUBMIT_REPO="$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel 2>/dev/null || true)"
CODEDIR="${CODEDIR:-${_SUBMIT_REPO:-/projects/u6wf/applications/pdu-extractor}}"
VENV="${VENV:-/projects/u6wf/applications/pypdu}"
PARAMS="${PARAMS:-$CODEDIR/pdukit/params/betalactamase_incoming.ppf}"
case "$PARAMS" in /*) ;; *) PARAMS="${SLURM_SUBMIT_DIR:-$PWD}/$PARAMS" ;; esac
[ -f "$PARAMS" ] || { echo "parameter file not found: $PARAMS" >&2; exit 1; }
PRESET="$(basename "$PARAMS" .ppf)"

SYSTEMS=(KPC2 PDC3 SHV1 TEM1)
SYSTEM="${SYSTEM:-${SYSTEMS[${SLURM_ARRAY_TASK_ID:-0}]}}"
MODE="${MODE:-full}"                 # full | calibrate
PDU_ARGS="${PDU_ARGS:-}"             # extra settings, e.g. "lag=2" (override the file)
DRYRUN="${DRYRUN:-0}"
FORCE_PREP="${FORCE_PREP:-0}"
SEG_LEN="${SEG_LEN:-}"               # fallback segment length if a file has no time resets
OUTROOT="${OUTROOT:-$WORKINGDIR/mddata/betalactamase_segments}"
PREP="$OUTROOT/$SYSTEM"
OUTDIR="${OUTDIR:-$WORKINGDIR/results/blac_${SYSTEM}_${PRESET}_${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}}"

[ -d "$DYNOPY" ] || { echo "Missing data directory: $DYNOPY (export DYNOPY=...)" >&2; exit 1; }
mkdir -p "$WORKINGDIR/logs" "$OUTROOT" "$OUTDIR"
module load cray-python 2>/dev/null || true
# shellcheck disable=SC1091
source "$VENV/bin/activate"
cd "$CODEDIR"
export PYTHONPATH="$CODEDIR${PYTHONPATH:+:$PYTHONPATH}"

echo "================================================================"
echo "PDU beta-lactamase  $(date +%F)  system=$SYSTEM mode=$MODE params=$PRESET"
echo "Job: ${SLURM_JOB_ID:-local} task=${SLURM_ARRAY_TASK_ID:-0} host=$(hostname) time=$(date +%T)"
echo "Code: $CODEDIR  branch=$(git -C "$CODEDIR" rev-parse --abbrev-ref HEAD) commit=$(git -C "$CODEDIR" rev-parse --short HEAD)"
echo "Data: $DYNOPY   segments: $PREP   output: $OUTDIR"
python3 -c "import torch; print('CUDA:', torch.cuda.is_available())" || echo "!! torch not importable"
echo "================================================================"

if [ "$DRYRUN" = 1 ]; then
    python3 pdukit/hpc/betalactamase/split_segments.py --root "$DYNOPY" --system "$SYSTEM" --dry-run
    echo "DRYRUN: stopping"; exit 0
fi

if [ "$FORCE_PREP" = 1 ] || [ ! -s "$PREP/segments.json" ]; then
    rm -f "$PREP"/seg_*.xtc
    python3 pdukit/hpc/betalactamase/split_segments.py --root "$DYNOPY" --system "$SYSTEM" -o "$PREP" \
        ${SEG_LEN:+--seg-len "$SEG_LEN"}
fi
mapfile -t XTC < <(ls "$PREP"/seg_*.xtc | sort -V)
[ "${#XTC[@]}" -ge 3 ] || { echo "only ${#XTC[@]} segments in $PREP" >&2; exit 1; }
echo "segments: ${#XTC[@]}"

EXTRA=()
[ "$MODE" = calibrate ] && EXTRA+=( calibrate-only=1 )

# shellcheck disable=SC2086   # PDU_ARGS is meant to be split into words
python3 -m pdukit run \
    -f "${XTC[@]}" -s "$PREP/topology.pdb" -p "$PARAMS" -o "$OUTDIR" \
    --set device=cuda grow-objective=incoming \
    ${EXTRA[@]+"${EXTRA[@]}"} $PDU_ARGS

cp "$PREP/segments.json" "$OUTDIR/" 2>/dev/null || true
echo "$SYSTEM DONE  $(date +%T)   results: $OUTDIR/results.json"
