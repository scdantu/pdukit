# pdu on mdCATH domains

mdCATH (Hugging Face `compsciencelab/mdCATH`) is stored as HDF5, not XTC, so
`mdcath_to_xtc.py` writes `topology.pdb` and one XTC per replica at one
temperature (default 320 K, five replicas).

1. `select_domains.py` picks domains spread over the CATH classes
   (default 30 Mainly Alpha, 30 Mainly Beta, 30 Alpha Beta, 10 Few Secondary
   Structures), architectures and homologous superfamilies.  It needs
   `mdCATH_domains.txt` and CATH's `cath-domain-list.txt`.
2. `submit_mdcath_array.slurm` is an array job, one domain per task: download,
   convert, then `pdu run` with `pdukit/params/lag1ns.ppf`.
3. `collect_results.py` gathers all `results.json` files into one CSV and
   prints a per-class summary.

The steps and commands are in the header of the Slurm script.

## Not verified yet (tested only on synthetic HDF5 files)

1. The internal HDF5 layout (see `pdukit/lib/pdu/extract/hdf5_layout.py`).
   Run `python mdcath_to_xtc.py --inspect FILE.h5` on one real file first.
2. The frame spacing (assumed 1 ns; `--dt-ns` and `frame-ps` in the preset).
3. Trajectories are short (about 464 ns per replica on average), so the
   amount of data per domain is far below the T4L and Syt runs and the
   detection threshold will be correspondingly higher.
4. Unit sizes are capped at 60 in `lag1ns.ppf` so small domains are valid;
   this limits large units in big domains.
