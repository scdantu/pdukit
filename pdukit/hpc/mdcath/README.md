# pdu on mdCATH domains

mdCATH (Hugging Face `compsciencelab/mdCATH`) is stored as HDF5, not XTC, so
`mdcath_to_xtc.py` writes `topology.pdb` and one XTC per replica at one
temperature (default 320 K, five replicas).

1. `select_domains.py` picks domains spread over the CATH classes
   (default 30 Mainly Alpha, 30 Mainly Beta, 30 Alpha Beta, 10 Few Secondary
   Structures), architectures and homologous superfamilies.  It needs
   `mdCATH_domains.txt` and CATH's `cath-domain-list.txt`.
2. `submit_mdcath_array.slurm` is an array job, one domain per task: download,
   convert, then `pdu run` with the CURRENT pipeline (`pdukit/params/mdcath_incoming.ppf`,
   `--grow-objective incoming`; output `<domain>/pdu_incoming_<job>/`) and
   `analysis/mdcath/check_run.py` on the result.  Domains that are already converted
   can also be run with `pdukit/hpc/slurm/submit_incoming_mdcath.slurm`.
3. `analysis/mdcath/summarise_batch.py RESULTS --classes mdcath_selected.tsv` makes
   one table and the headline fractions (cores beating uniform random sets and,
   the real test, compact sets on the predictive graph), per CATH class.
   `collect_results.py` is the legacy collector for old COG 2.1 runs (`<domain>/pdu/`).

The steps and commands are in the header of the Slurm script.

## Checks

The HDF5 layout and the 1 ns frame spacing were confirmed on 1ztdA00 (21 Oct 2026).

## Original notes (before the layout was checked)

1. The internal HDF5 layout (see `pdukit/io/mdcath/layout.py`).
   Run `python mdcath_to_xtc.py --inspect FILE.h5` on one real file first.
2. The frame spacing (assumed 1 ns; `--dt-ns` and `frame-ps` in the preset).
3. Trajectories are short (about 464 ns per replica on average), so the
   amount of data per domain is far below the T4L and Syt runs and the
   detection threshold will be correspondingly higher.
4. `lag1ns.ppf` is the legacy COG 2.1 preset (units capped at 60); do not use it for
   new PDU runs.

## Whole mdCATH (about 5400 domains)

1. `select_domains.py --all --mdcath-list ... --cath-list ... -o mdcath_all.tsv`
   writes every domain (CATH columns empty when CATH has no record).
2. Several domains per array task keep the array under the site `MaxArraySize`
   (`scontrol show config | grep -i MaxArraySize`):

       sbatch --array=1-540%40 --time=24:00:00 \
           --export=ALL,SELECTED=$PWD/mdcath_all.tsv,PER_TASK=10 \
           /projects/u6wf/applications/pdu-extractor/pdukit/hpc/mdcath/submit_mdcath_array.slurm

   `OFFSET=<rows>` shifts the window for a second array if needed.  A domain that
   already has `pdu_incoming_*/check_run.json` is skipped (`FORCE=1` reruns), so a
   resubmission only does what is missing.  A failing domain does not stop the task;
   each task writes `RESULTS/_status_<job>_<task>.tsv` (domain, ok/FAILED, task, time).
3. Summarise on Isambard from the per-run JSON (no bootstrap files needed):

       python3 analysis/mdcath/summarise_batch.py results/mdcath_T320 --from-json \
           --classes mdcath_all.tsv --out mdcath_T320_all_summary.tsv
