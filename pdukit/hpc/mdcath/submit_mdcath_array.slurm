#!/bin/bash
#SBATCH --job-name=pdu_mdcath
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --output=logs/pdu_mdcath_%A_%a.out
#SBATCH --error=logs/pdu_mdcath_%A_%a.err

# PDU on mdCATH domains, one array task per domain.
#
# One-time preparation (a machine with network access):
#   huggingface-cli download compsciencelab/mdCATH mdCATH_domains.txt \
#       --repo-type dataset --local-dir mddata/mdcath
#   curl -O https://download.cathdb.info/cath/releases/latest-release/cath-classification-data/cath-domain-list.txt
#   python pdukit/hpc/mdcath/select_domains.py \
#       --mdcath-list mddata/mdcath/mdCATH_domains.txt \
#       --cath-list cath-domain-list.txt -o mdcath_selected.tsv
#
# Check ONE domain first (the HDF5 layout is unverified):
#   python pdukit/hpc/mdcath/mdcath_to_xtc.py 1a02F00 -o /tmp/chk
#   sbatch --array=1-1 submit_mdcath_array.slurm
#
# Then all 100:
#   sbatch --array=1-100 pdukit/hpc/mdcath/submit_mdcath_array.slurm
#
# Task k reads row k+1 of SELECTED (row 1 is the header).  Each domain is
# converted at TEMPERATURE (320 K by default, closest to native), the five
# replicas are pooled as replicas, and pdu runs with the shared preset
# pdukit/params/lag1ns.ppf.  mdCATH frames are DT_NS apart (default 1 ns,
# confirm in the paper); frame-ps in the preset must match.
#
# Results: $RESULTS/<domain>/pdu/results.json.  Collect with
#   python pdukit/hpc/mdcath/collect_results.py $RESULTS --selected FILE

set -euo pipefail

WORKINGDIR="${WORKINGDIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
_REPO="$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel 2>/dev/null || true)"
CODEDIR="${CODEDIR:-${_REPO:-$WORKINGDIR/pdukit}}"
VENV="${VENV:-$HOME/venvs/pdu}"
SELECTED="${SELECTED:-$WORKINGDIR/mdcath_selected.tsv}"
CACHE="${CACHE:-$WORKINGDIR/mddata/mdcath}"
TEMPERATURE="${TEMPERATURE:-320}"
DT_NS="${DT_NS:-1.0}"
PARAMS="${PARAMS:-$CODEDIR/pdukit/params/lag1ns.ppf}"
PDU_ARGS="${PDU_ARGS:-}"
KEEP_H5="${KEEP_H5:-0}"
RESULTS="${RESULTS:-$WORKINGDIR/results/mdcath_T${TEMPERATURE}}"

TASK="${SLURM_ARRAY_TASK_ID:?run as an array job}"
DOMAIN="$(awk -F'\t' -v n="$((TASK + 1))" 'NR==n {print $1}' "$SELECTED")"
[ -n "$DOMAIN" ] || { echo "no row $((TASK + 1)) in $SELECTED" >&2; exit 1; }

OUT="$RESULTS/$DOMAIN"
mkdir -p "$WORKINGDIR/logs" "$OUT"
echo "================================================================"
echo "PDU mdCATH  $(date +%F)  domain=$DOMAIN T=${TEMPERATURE} K  task=$TASK"
echo "Job: ${SLURM_JOB_ID:-local} host=$(hostname)  commit=$(git -C "$CODEDIR" rev-parse --short HEAD 2>/dev/null)"
echo "================================================================"

module load cray-python 2>/dev/null || true
# shellcheck disable=SC1091
source "$VENV/bin/activate"
cd "$CODEDIR"
export PYTHONPATH="$CODEDIR${PYTHONPATH:+:$PYTHONPATH}"

if [ ! -s "$OUT/topology.pdb" ]; then
    python pdukit/hpc/mdcath/mdcath_to_xtc.py "$DOMAIN" --cache "$CACHE" \
        --temperature "$TEMPERATURE" --dt-ns "$DT_NS" -o "$OUT"
fi
mapfile -t XTC < <(ls "$OUT"/rep_*.xtc | sort -V)
echo "replicas: ${#XTC[@]}"

# shellcheck disable=SC2086   # PDU_ARGS is meant to be split into words
python3 -m pdukit run -f "${XTC[@]}" -s "$OUT/topology.pdb" \
    -p "$PARAMS" -o "$OUT/pdu" --device cuda $PDU_ARGS

if [ "$KEEP_H5" != 1 ]; then
    find "$CACHE" -name "*${DOMAIN}*.h5" -delete 2>/dev/null || true
fi
