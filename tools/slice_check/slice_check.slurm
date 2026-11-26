#!/bin/bash
#SBATCH --job-name=pdu_slice_check
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --output=logs/pdu_slice_check_%j.out
#SBATCH --error=logs/pdu_slice_check_%j.err

# Old code against new code on a SLICE of the real trajectories, on GPU and on CPU.
#
#   old = the code before the restructuring (tag archive/pre-restructure)
#   new = the restructured code (branch incoming-only)
#
# Runs, in this order (each writes its own folder under $OUTDIR):
#   old_cuda   new_cuda   old_cuda_b      GPU: old, new, and old a second time
#   old_cpu    new_cpu                    CPU: old and new
# old_cuda_b exists only to measure how much the GPU wanders when the SAME code
# is run twice; new_cuda is judged against that.  The verdict is printed at the
# end of the log and compare_runs.py can be re-run by hand on any pair.
#
# Submit from any directory (create logs/ there first):
#   mkdir -p logs
#   sbatch --export=ALL,OLD_REPO=/path/to/pdu-old,NEW_REPO=/path/to/pdu-new slice_check.slurm
# T4L instead of synaptotagmin:   --export=ALL,SYSTEM=t4l,VARIANT=GA,OLD_REPO=...,NEW_REPO=...
# Look at the inputs without running anything:  DRYRUN=1 bash slice_check.slurm
# Redo only some of the runs (e.g. after a time-out), keeping the others from the earlier job:
#   sbatch --time=03:00:00 --export=ALL,DEVICES=cpu,RUNS=new_cpu,OUTDIR=<earlier results folder>,SYSTEM=...,OLD_REPO=...,NEW_REPO=... slice_check.slurm
# (same SYSTEM, VARIANT, FRAMES, NREP, STRIDE as the earlier job, or the runs are not comparable)
#
# Everything below can be overridden with an environment variable.

set -uo pipefail

export TMPDIR="/tmp/${USER:-user}_${SLURM_JOB_ID:-local}"
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT

WORKINGDIR="${WORKINGDIR:-/projects/u6wf/mariad}"      # results and logs go here
DATAROOT="${DATAROOT:-/projects/u6wf/scdantu/Projects/pdudynamics/mddata}"   # trajectories (read only)
# Python environment: VENV if given, else the one already active when you submit
# (a job inherits it); stop if neither exists.
VENV="${VENV:-${VIRTUAL_ENV:-}}"
OLD_REPO="${OLD_REPO:?set OLD_REPO to a checkout of tag archive/pre-restructure}"
NEW_REPO="${NEW_REPO:?set NEW_REPO to a checkout of branch incoming-only}"
# Folder holding compare_runs.py.  sbatch runs a COPY of this script from a spool folder, so look next to
# the new code first, then next to this file, then in the submit folder.
KIT="${KIT:-$NEW_REPO/tools/slice_check}"
[ -f "$KIT/compare_runs.py" ] || KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$KIT/compare_runs.py" ] || KIT="${SLURM_SUBMIT_DIR:-$PWD}"

SYSTEM="${SYSTEM:-syt}"            # syt | t4l | custom (then set TOP, XTC_FILES, PARAMS)
FRAMES="${FRAMES:-2000}"           # keep only the first N frames of each replica
STRIDE="${STRIDE:-1}"              # ... then every Nth of those
NREP="${NREP:-3}"                  # number of replicas used (0 = all)
DEVICES="${DEVICES:-cuda cpu}"     # which of the two to do
RUNS="${RUNS:-}"                   # empty = do every run; else only these labels (e.g. "new_cpu"); the rest are
                                   # taken from $OUTDIR if an earlier job left finished results there
THREADS="${THREADS:-1}"            # CPU threads (1 keeps CPU runs reproducible)
FEATURE_CACHE_OFF="${FEATURE_CACHE_OFF:-yes}"   # yes = compute features afresh in every run
OUTROOT="${OUTROOT:-$WORKINGDIR/results}"
OUTDIR="${OUTDIR:-$OUTROOT/slice_check_${SYSTEM}_${SLURM_JOB_ID:-local}}"
DRYRUN="${DRYRUN:-0}"

# ---- inputs (same locations as the production scripts) ---------------------
XTC=()
case "$SYSTEM" in
  syt)
    SYTDIR="${SYTDIR:-$DATAROOT/syt-upload}"
    TOP="${TOP:-$SYTDIR/hsynapto-protein.pdb}"
    PARAMS="${PARAMS:-$OLD_REPO/pdukit/params/syt_c2a_incoming.ppf}"
    while IFS= read -r f; do XTC+=( "$f" ); done < <(
        find -L "$SYTDIR" -mindepth 2 -type f -name "${XTC_NAME:-*.xtc}" 2>/dev/null | sort -V)
    ;;
  t4l)
    VARIANT="${VARIANT:-GA}"
    MDDIR="${MDDIR:-$DATAROOT}"
    TOP="${TOP:-$MDDIR/ATM-110_129-${VARIANT}.tpr}"
    PARAMS="${PARAMS:-$OLD_REPO/pdukit/params/t4l_incoming.ppf}"
    shopt -s nullglob
    XTC=( "$MDDIR"/CONV-PROD-R*-ATM-110_129-"${VARIANT}".xtc )
    shopt -u nullglob
    ;;
  custom)
    : "${TOP:?custom needs TOP}" "${XTC_FILES:?custom needs XTC_FILES (space separated)}" "${PARAMS:?custom needs PARAMS}"
    read -r -a XTC <<< "$XTC_FILES"
    ;;
  *) echo "SYSTEM must be syt, t4l or custom" >&2; exit 1 ;;
esac
if [ "$NREP" -gt 0 ] && [ "${#XTC[@]}" -gt "$NREP" ]; then XTC=( "${XTC[@]:0:$NREP}" ); fi
[ "${#XTC[@]}" -gt 0 ] || { echo "no trajectories found for SYSTEM=$SYSTEM" >&2; exit 1; }
[ -s "$TOP" ] || { echo "missing topology: $TOP" >&2; exit 1; }
[ -f "$PARAMS" ] || { echo "missing parameter file: $PARAMS" >&2; exit 1; }
for r in "$OLD_REPO" "$NEW_REPO"; do [ -d "$r/pdukit" ] || { echo "not a pdu checkout: $r" >&2; exit 1; }; done
[ "$(cd "$OLD_REPO" && pwd)" != "$(cd "$NEW_REPO" && pwd)" ] || { echo "OLD_REPO and NEW_REPO are the same folder" >&2; exit 1; }

repo_info() { echo "$1  $(git -C "$1" rev-parse --abbrev-ref HEAD 2>/dev/null) $(git -C "$1" rev-parse --short HEAD 2>/dev/null)"; }
echo "================================================================"
echo "PDU slice check  $(date +%F\ %T)  system=$SYSTEM  devices=$DEVICES"
echo "old: $(repo_info "$OLD_REPO")"
echo "new: $(repo_info "$NEW_REPO")"
echo "parameter file (used by both): $PARAMS"
echo "topology: $TOP"
echo "replicas (${#XTC[@]}):"; printf '   %s\n' "${XTC[@]}"
echo "slice: first $FRAMES frames of each replica, every $STRIDE th; feature cache off: $FEATURE_CACHE_OFF"
echo "output: $OUTDIR"
[ -f "$KIT/compare_runs.py" ] && echo "compare_runs.py: $KIT" || echo "WARNING: compare_runs.py not found (looked in $KIT); the runs will work but the comparison will not"
[ -n "$RUNS" ] && echo "runs: only $RUNS (others reused from $OUTDIR if present)"
echo "environment: ${VENV:-NONE}   data: $DATAROOT"
echo "================================================================"
[ "$DRYRUN" = 1 ] && exit 0

mkdir -p "$OUTDIR" "$WORKINGDIR/logs" 2>/dev/null
module load cray-python 2>/dev/null || true
[ -n "$VENV" ] && [ -f "$VENV/bin/activate" ] || { echo "no Python environment: activate yours before sbatch, or set VENV=<its folder>" >&2; exit 1; }
source "$VENV/bin/activate"
python3 - <<'PY'
import sys
try:
    import torch
    print("python", sys.version.split()[0], "| torch", torch.__version__, "| cuda available:", torch.cuda.is_available())
except Exception as e:
    print("python", sys.version.split()[0], "| torch not importable:", e)
PY

declare -A STATUS
run_one() {   # run_one LABEL REPO DEVICE
    local label="$1" repo="$2" dev="$3" out="$OUTDIR/$1"
    if [ -n "$RUNS" ] && [[ " $RUNS " != *" $label "* ]]; then
        if [ -s "$out/results.json" ]; then
            STATUS[$label]=0; echo; echo "---- $label  kept from an earlier job ($out)"
        else
            echo; echo "---- $label  not run (not in RUNS) and no earlier results in $out"
        fi
        return
    fi
    rm -rf "$out"; mkdir -p "$out"
    local nocache=()
    [ "$FEATURE_CACHE_OFF" = yes ] && nocache=( no-feature-cache=yes )
    echo; echo "---- $label  ($dev)  $(date +%T)"
    (
        cd "$repo" || exit 1
        export PYTHONPATH="$repo"
        if [ "$dev" = cpu ]; then
            export OMP_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS"
        fi
        python3 -m pdukit run -f "${XTC[@]}" -s "$TOP" -p "$PARAMS" -o "$out" \
            --set device="$dev" frames="$FRAMES" stride="$STRIDE" "${nocache[@]}"
    ) > "$out.log" 2>&1
    STATUS[$label]=$?
    echo "     exit ${STATUS[$label]}  log: $out.log"
    [ "${STATUS[$label]}" -eq 0 ] || tail -n 8 "$out.log" | sed 's/^/     | /'
}

for dev in $DEVICES; do
    case "$dev" in
      cuda)
        run_one old_cuda   "$OLD_REPO" cuda
        run_one new_cuda   "$NEW_REPO" cuda
        run_one old_cuda_b "$OLD_REPO" cuda
        ;;
      cpu)
        run_one old_cpu "$OLD_REPO" cpu
        run_one new_cpu "$NEW_REPO" cpu
        ;;
      *) echo "unknown device $dev (use cuda and/or cpu)" ;;
    esac
done

# ---- verdict ----------------------------------------------------------------
echo; echo "================================================================"
echo "COMPARISONS (exact unless stated)"
echo "================================================================"
cmp_pair() {  # cmp_pair LABEL_A LABEL_B TITLE [extra compare_runs.py options]
    local a="$1" b="$2" title="$3"; shift 3
    echo; echo "## $title"
    if [ "${STATUS[$a]:-1}" -ne 0 ] || [ "${STATUS[$b]:-1}" -ne 0 ]; then
        echo "   skipped: a run failed (${a}=${STATUS[$a]:-not run}, ${b}=${STATUS[$b]:-not run})"; return 2
    fi
    python3 "$KIT/compare_runs.py" "$OUTDIR/$a" "$OUTDIR/$b" "$@"
}
case " $DEVICES " in *" cuda "*)
    cmp_pair old_cuda old_cuda_b "GPU noise: the OLD code run twice (this is how much the GPU wanders by itself)"
    cmp_pair old_cuda new_cuda   "GPU: old vs new"
esac
case " $DEVICES " in *" cpu "*)
    cmp_pair old_cpu new_cpu "CPU: old vs new"
esac
echo
echo "How to read this: CPU old vs new should say IDENTICAL.  On GPU, if old vs old (noise)"
echo "says IDENTICAL then old vs new must too.  If the GPU noise is not zero, re-run the"
echo "old vs new comparison with a tolerance a little above the noise, e.g."
echo "  python3 $KIT/compare_runs.py $OUTDIR/old_cuda $OUTDIR/new_cuda --rtol 1e-6"
echo "and look at the unit membership files (incoming_split_units.csv) for any change."
echo "Finished $(date +%T)"
