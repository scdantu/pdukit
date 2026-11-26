#!/bin/bash
# Run a synthetic job list with `pdu bench` (CPU; a few minutes per run).
#   bash pdukit/hpc/synthetic/run_suite.sh PRESET JOBS OUTROOT [NPROC]
#   bash pdukit/hpc/synthetic/run_suite.sh pdukit/params/synthetic100.ppf \
#        pdukit/hpc/synthetic/jobs100_tau.txt runs100 2
# Each job line: <run name> <setting=value ...>; '#' lines are comments.
# Output: OUTROOT/<run name>/ (results), OUTROOT/logs/<run name>.log.
# Score afterwards:  pdu score-bench OUTROOT/* -o tables/
set -euo pipefail
PRESET="$1"; JOBS="$2"; OUTROOT="$3"; NPROC="${4:-2}"
[ -f "$PRESET" ] || { echo "no preset $PRESET" >&2; exit 1; }
[ -f "$JOBS" ] || { echo "no job list $JOBS" >&2; exit 1; }
mkdir -p "$OUTROOT/logs"
grep -v '^\s*#' "$JOBS" | grep -v '^\s*$' | \
  xargs -P "$NPROC" -L 1 sh -c '
    name="$1"; shift
    python3 -m pdukit bench -p "'"$PRESET"'" -o "'"$OUTROOT"'/$name" \
      --set "$@" > "'"$OUTROOT"'/logs/$name.log" 2>&1 \
      && echo "done $name" || echo "FAILED $name (see '"$OUTROOT"'/logs/$name.log)"
  ' _
