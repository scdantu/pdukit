# HPC scripts

Settings that depend on the machine are not kept in the scripts.  Create
`~/.pdu_site.env` (or point `PDU_SITE` at another file) with the paths of
your environments; the mdCATH Slurm scripts read it first:

    PDU_VENV=/path/to/venv/with/pdukit

Anything given at submission (`sbatch --export=ALL,PDU_VENV=...`) overrides
the file only if the file does not set it, so keep one place per setting.

Folders: `slurm/` (benchmark, fast test, T4L and Syt runs), `mdcath/`
(pdu on mdCATH domains).
