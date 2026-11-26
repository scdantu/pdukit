#!/bin/bash
#SBATCH --job-name=pdu_incoming_mdcath
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --array=1-1
#SBATCH --output=logs/pdu_incoming_mdcath_%A_%a.out
#SBATCH --error=logs/pdu_incoming_mdcath_%A_%a.err

# mdCATH domains with the CURRENT COG pipeline (incoming growth, merge, split, ports,
# global-mode recovery).  Run from a pdu-extractor checkout at or after c448ab7.
# Inputs are the converted files of each domain (rep_*.xtc, topology.pdb, conversion.json).
#
#   TEMP=320                     temperature folder results/mdcath_T${TEMP}
#   DOMAINS=<file>               one domain id per line (first column; header 'domain ...' skipped);
#                                array index n (1-based) = n-th domain
#   DOMAIN=1ztdA00               single domain, overrides DOMAINS / array index
#   PARAMS=<ppf>                 default params/mdcath_incoming.ppf
#   PDU_ARGS="..."               extra settings, e.g. "highpass-window=20" (override the file)
#   TAG=<text>                   appended to the output folder name (e.g. hp20)
#   DRYRUN=1                     print the resolved command and stop (no venv or GPU needed)
#
# Example:  DOMAIN=1ztdA00 DRYRUN=1 bash pdukit/hpc/slurm/submit_incoming_mdcath.slurm
#           sbatch --array=1-1 --export=ALL,DOMAIN=1ztdA00 pdukit/hpc/slurm/submit_incoming_mdcath.slurm

set -euo pipefail

WORKINGDIR="${WORKINGDIR:-/projects/u6wf/scdantu/Projects/pdudynamics}"
TEMP="${TEMP:-320}"
RESULTS="${RESULTS:-$WORKINGDIR/results/mdcath_T${TEMP}}"
_SUBMIT_REPO="$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel 2>/dev/null || true)"
CODEDIR="${CODEDIR:-${_SUBMIT_REPO:-/projects/u6wf/applications/pdu-extractor}}"
VENV="${VENV:-/projects/u6wf/applications/pypdu}"
DRYRUN="${DRYRUN:-0}"
TAG="${TAG:-}"
PDU_ARGS="${PDU_ARGS:-}"

PARAMS="${PARAMS:-$CODEDIR/pdukit/params/mdcath_incoming.ppf}"
case "$PARAMS" in /*) ;; *) PARAMS="${SLURM_SUBMIT_DIR:-$PWD}/$PARAMS" ;; esac
[ -f "$PARAMS" ] || { echo "parameter file not found: $PARAMS" >&2; exit 1; }

if [ -n "${DOMAIN:-}" ]; then
    :
elif [ -n "${DOMAINS:-}" ]; then
    [ -f "$DOMAINS" ] || { echo "domain list not found: $DOMAINS" >&2; exit 1; }
    # one domain per line, first column; '#' comments, blank lines and a header row
    # starting with 'domain' (e.g. mdcath_selected.tsv from select_domains.py) are skipped
    DOMAIN="$(grep -v '^[[:space:]]*\(#\|$\)' "$DOMAINS" | grep -v '^domain[[:space:]]' | sed -n "${SLURM_ARRAY_TASK_ID:-1}p" | awk '{print $1}')"
    [ -n "$DOMAIN" ] || { echo "no domain on line ${SLURM_ARRAY_TASK_ID:-1} of $DOMAINS" >&2; exit 1; }
else
    echo "set DOMAIN=<id> or DOMAINS=<file>" >&2; exit 1
fi

DDIR="$RESULTS/$DOMAIN"
TOP="$DDIR/topology.pdb"
shopt -s nullglob
XTC=( "$DDIR"/rep_*.xtc )
shopt -u nullglob
[ -s "$TOP" ] || { echo "Missing topology: $TOP (convert the domain first; see conversion.json)" >&2; exit 1; }
[ "${#XTC[@]}" -gt 0 ] || { echo "No trajectories $DDIR/rep_*.xtc" >&2; exit 1; }
for x in "${XTC[@]}"; do [ -s "$x" ] || { echo "Empty trajectory: $x" >&2; exit 1; }; done

PRESET="$(basename "$PARAMS" .ppf)"
JOB="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
OUTDIR="${OUTDIR:-$DDIR/pdu_incoming_${JOB}${TAG:+_$TAG}}"   # never the old pdu/ folder

CMD=( python3 -m pdukit run -f "${XTC[@]}" -s "$TOP" -p "$PARAMS" -o "$OUTDIR" --set device=cuda grow-objective=incoming )
# shellcheck disable=SC2206   # PDU_ARGS is meant to be split into words
CMD+=( $PDU_ARGS )

if [ "$DRYRUN" = 1 ]; then
    echo "domain:  $DOMAIN  (T=${TEMP} K, ${#XTC[@]} replicas)"
    echo "params:  $PARAMS"
    echo "output:  $OUTDIR"
    printf 'command: '; printf '%q ' "${CMD[@]}"; echo
    exit 0
fi

mkdir -p "$WORKINGDIR/logs" "$OUTDIR"
module load cray-python 2>/dev/null || true
source "$VENV/bin/activate"
cd "$CODEDIR"
export PYTHONPATH="$CODEDIR${PYTHONPATH:+:$PYTHONPATH}"

echo "================================================================"
echo "PDU mdCATH  $(date +%F)  domain=$DOMAIN T=$TEMP params=$PRESET"
echo "Job: ${SLURM_JOB_ID:-local} task=${SLURM_ARRAY_TASK_ID:-0} host=$(hostname) time=$(date +%T)"
echo "Code: $CODEDIR  branch=$(git -C "$CODEDIR" rev-parse --abbrev-ref HEAD) commit=$(git -C "$CODEDIR" rev-parse --short HEAD)"
echo "Params: $PARAMS   extra: ${PDU_ARGS:-none}"
echo "Output: $OUTDIR"
python3 -c "import torch; print('CUDA:', torch.cuda.is_available(), 'device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')" || echo "!! torch not importable -- running on the CPU (numpy)"
echo "================================================================"
echo "Replicas (${#XTC[@]}):"
printf '  %s\n' "${XTC[@]##*/}"
printf '%s\n' "${XTC[@]}" > "$OUTDIR/replicas.txt"
cp "$DDIR/conversion.json" "$OUTDIR/" 2>/dev/null || true

"${CMD[@]}"

echo "domain $DOMAIN DONE  $(date +%T)   results: $OUTDIR/results.json"
python3 analysis/mdcath/check_run.py "$OUTDIR" || echo "check_run reported a problem (see above)"
