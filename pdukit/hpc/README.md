# HPC scripts

Settings that depend on the machine are not kept in the scripts.  Create
`~/.pdu_site.env` (or point `PDU_SITE` at another file) with the paths of
your environments; the BioEmu2 and mdCATH Slurm scripts read it first:

    PDU_VENV=/path/to/venv/with/pdukit
    BIOEMU_VENV=/path/to/venv/with/bioemu2

Anything given at submission (`sbatch --export=ALL,PDU_VENV=...`) overrides
the file only if the file does not set it, so keep one place per setting.

Folders: `slurm/` (benchmark, fast test, T4L and Syt runs), `bioemu2/`
(1 ns rollouts), `mdcath/` (pdu on mdCATH domains).
