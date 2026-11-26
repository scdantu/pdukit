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
# Whole mdCATH (about 5400 domains): several domains per task and an offset, so
# the array stays under the site MaxArraySize; finished domains are skipped.
#   python pdukit/hpc/mdcath/select_domains.py --all \
#       --mdcath-list mddata/mdcath/mdCATH_domains.txt \
#       --cath-list cath-domain-list.txt -o mdcath_all.tsv
#   sbatch --array=1-540%40 --time=24:00:00 \
#       --export=ALL,SELECTED=$PWD/mdcath_all.tsv,PER_TASK=10 \
#       pdukit/hpc/mdcath/submit_mdcath_array.slurm
#
# Task k handles data rows (k-1)*PER_TASK+1+OFFSET ... k*PER_TASK+OFFSET of
# SELECTED (the header row is skipped; PER_TASK=1, OFFSET=0 is the old behaviour:
# task k reads row k+1).  Each domain is
# converted at TEMPERATURE (320 K by default, closest to native), the five
# replicas are pooled as replicas, and pdu runs the CURRENT pipeline (incoming
# growth, majority-core merge, recursive split on cohesion, guarded ports,
# global-mode recovery) with pdukit/params/mdcath_incoming.ppf.  mdCATH frames
# are DT_NS apart (1 ns, confirmed on 1ztdA00); frame-ps in the preset matches.
# (Until 21 Oct 2026 this script used the legacy COG 2.1 preset lag1ns.ppf.)
#
# Results: $RESULTS/<domain>/pdu_incoming_<job>/ (results.json, incoming_ports.json,
# bootstrap_rho.npz, ...).  Check one run, then summarise all:
#   python3 analysis/mdcath/check_run.py $RESULTS/<domain>/pdu_incoming_<job>
#   python3 analysis/mdcath/summarise_batch.py $RESULTS --classes mdcath_selected.tsv

set -euo pipefail

# Site settings (environment paths, ...) live outside the repository, in
# ~/.pdu_site.env (or the file named by PDU_SITE); see pdukit/hpc/README.md.
# shellcheck disable=SC1090
[ -f "${PDU_SITE:-$HOME/.pdu_site.env}" ] && . "${PDU_SITE:-$HOME/.pdu_site.env}"

WORKINGDIR="${WORKINGDIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
_REPO="$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel 2>/dev/null || true)"
# Code: the git repo you submit from; otherwise /projects/u6wf/applications/pdu-extractor
# (the Isambard checkout); or export CODEDIR.  Venv: /projects/u6wf/applications/pypdu.
CODEDIR="${CODEDIR:-${_REPO:-/projects/u6wf/applications/pdu-extractor}}"
[ -d "$CODEDIR/pdukit" ] || { echo "CODEDIR=$CODEDIR has no pdukit/ (submit from the repo or export CODEDIR=<pdu-extractor checkout>)" >&2; exit 1; }
PDU_VENV="${PDU_VENV:-${VENV:-/projects/u6wf/applications/pypdu}}"
SELECTED="${SELECTED:-$WORKINGDIR/mdcath_selected.tsv}"
CACHE="${CACHE:-$WORKINGDIR/mddata/mdcath}"
TEMPERATURE="${TEMPERATURE:-320}"
DT_NS="${DT_NS:-1.0}"
PARAMS="${PARAMS:-$CODEDIR/pdukit/params/mdcath_incoming.ppf}"
PDU_ARGS="${PDU_ARGS:-}"
KEEP_H5="${KEEP_H5:-0}"
PER_TASK="${PER_TASK:-1}"   # domains per array task
OFFSET="${OFFSET:-0}"       # data rows to skip (for arrays above MaxArraySize)
FORCE="${FORCE:-0}"         # 1 = rerun domains that already have a checked run
RESULTS="${RESULTS:-$WORKINGDIR/results/mdcath_T${TEMPERATURE}}"

TASK="${SLURM_ARRAY_TASK_ID:?run as an array job}"
FIRST=$(( (TASK - 1) * PER_TASK + 1 + OFFSET ))
LAST=$(( TASK * PER_TASK + OFFSET ))
# data rows FIRST..LAST (header skipped, blank lines and comments ignored)
mapfile -t DOMAINS < <(grep -v '^[[:space:]]*\(#\|$\)' "$SELECTED" | grep -v '^domain[[:space:]]' \
    | sed -n "${FIRST},${LAST}p" | awk -F'\t' '{print $1}')
[ "${#DOMAINS[@]}" -gt 0 ] || { echo "no data rows ${FIRST}-${LAST} in $SELECTED" >&2; exit 0; }

mkdir -p "$WORKINGDIR/logs" "$RESULTS"
echo "================================================================"
echo "PDU mdCATH  $(date +%F)  T=${TEMPERATURE} K  task=$TASK rows ${FIRST}-${LAST}: ${DOMAINS[*]}"
echo "Job: ${SLURM_JOB_ID:-local} host=$(hostname)  commit=$(git -C "$CODEDIR" rev-parse --short HEAD 2>/dev/null)"
echo "params: $PARAMS"
echo "================================================================"

module load cray-python 2>/dev/null || true
# shellcheck disable=SC1091
source "$PDU_VENV/bin/activate"
cd "$CODEDIR"
export PYTHONPATH="$CODEDIR${PYTHONPATH:+:$PYTHONPATH}"

run_domain() {
    local DOMAIN="$1" OUT="$RESULTS/$1" RUN XTC
    mkdir -p "$OUT"
    if [ "$FORCE" != 1 ] && ls "$OUT"/pdu_incoming_*/check_run.json >/dev/null 2>&1; then
        echo "[$DOMAIN] already checked, skipped (FORCE=1 to rerun)"; return 0
    fi
    echo "---- [$DOMAIN] $(date +%T)"
    if [ ! -s "$OUT/topology.pdb" ]; then
        python pdukit/hpc/mdcath/mdcath_to_xtc.py "$DOMAIN" --cache "$CACHE" \
            --temperature "$TEMPERATURE" --dt-ns "$DT_NS" -o "$OUT" || return 1
    fi
    mapfile -t XTC < <(ls "$OUT"/rep_*.xtc 2>/dev/null | sort -V)
    [ "${#XTC[@]}" -gt 0 ] || { echo "[$DOMAIN] no rep_*.xtc" >&2; return 1; }
    echo "[$DOMAIN] replicas: ${#XTC[@]}"
    RUN="$OUT/pdu_incoming_${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
    echo "[$DOMAIN] output: $RUN"
    # shellcheck disable=SC2086   # PDU_ARGS is meant to be split into words
    python3 -m pdukit run -f "${XTC[@]}" -s "$OUT/topology.pdb" \
        -p "$PARAMS" -o "$RUN" --set device=cuda grow-objective=incoming $PDU_ARGS || return 1
    cp "$OUT/conversion.json" "$RUN/" 2>/dev/null || true
    python3 analysis/mdcath/check_run.py "$RUN" --json || return 1
}

NFAIL=0
for D in "${DOMAINS[@]}"; do
    if run_domain "$D"; then STATUS=ok; else STATUS=FAILED; NFAIL=$((NFAIL + 1)); fi
    echo "[$D] $STATUS $(date +%T)"
    printf '%s\t%s\t%s\t%s\n' "$D" "$STATUS" "${SLURM_ARRAY_JOB_ID:-local}_$TASK" "$(date +%FT%T)" \
        >> "$RESULTS/_status_${SLURM_ARRAY_JOB_ID:-local}_$TASK.tsv"
    if [ "$KEEP_H5" != 1 ]; then
        find "$CACHE" -name "*${D}*.h5" -delete 2>/dev/null || true
    fi
done
echo "task $TASK done: ${#DOMAINS[@]} domains, $NFAIL failed"
