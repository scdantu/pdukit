# pdu — COG (v1.0.0-alpha.1)

> **Alpha release.** The method and the code are under active development; interfaces, output
> formats and defaults may change before v1.0. Internally this is algorithm lineage "COG 2.1".

COG finds **Protein Dynamical Units (PDUs)** in molecular dynamics (MD) data: groups of
residues whose motions are coupled to each other more strongly than to the rest of the
protein, at a chosen lag time and for chosen per-residue features. It uses dynamics only;
spatial contacts are not used in the search or in the baselines.

This repository is the first public pre-release (`v1.0.0-alpha.1`) of the COG pipeline (`pdukit` Python package).
It is the code behind the T4 lysozyme and synaptotagmin-1 C2A analyses.

Every unit is scored two ways:

1. **Closure** (the objective that is optimised): how many independent channels link the unit's
   present and future to the rest of the protein, compared with random regions of the same size.
2. **Predictive cohesion** (a check, reported at the end): whether residues inside the unit
   predict each other's futures better than they predict, or are predicted by, outside residues.

Per-residue labels (driver / listener / both / none) say which way each member's coupling points.
See [`docs/METHODS.md`](docs/METHODS.md) for definitions, stages and outputs.

## Install

```bash
git clone https://github.com/scdantu/pdu.git && cd pdu
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"            # numpy, scipy, threadpoolctl + pytest
pip install -e ".[xtc]"            # to read XTC/DCD trajectories (MDAnalysis)
pip install -e ".[gpu]"            # optional: batched scoring on a CUDA GPU (torch)
```

Python 3.10 or newer. Runs on CPU with numpy only; a GPU speeds up large proteins.

## Quick check on a planted system (minutes, CPU)

The benchmark builds a synthetic protein with known units and checks that COG recovers them.

```bash
pdu bench -o out/planted
pdu fasttest --out out/fast --designs base --configs dyn_r2_mc --seeds 0 --jobs 2
```

## Run on your own trajectories

```bash
pdu run -f rep1.xtc rep2.xtc rep3.xtc -s protein.tpr \
        -p pdukit/params/t4l_channels.ppf -o out/my_protein
```

The command is `pdu <module>`: `run`, `bench` (planted system with a known answer), `check`
(print the resolved settings, run nothing), `summarise`, `compare` and `fasttest`.
`pdu run -h` lists every option. File options are short, as in GROMACS: `-f` trajectories,
`-s` topology, `-p` parameter file, `-o` output directory. Lags and windows are in **frames**.
Residue indices in all outputs are **0-based**. Each stage rewrites `results.json`, so an
interrupted run keeps what it finished.

### Parameter files (`.ppf`)

Settings live in a parameter file, one `key = value` per line (`;` or `#` starts a comment),
in the style of a GROMACS `.mdp`:

```
lag             = 25      ; frames
highpass-window = 500
faces           = 0       ; flags take 0/1, yes/no, on/off
n-holdout       = 0       ; 0 = discover on all replicas
```

A key is the name of a command-line option without `--` (case, `-` and `_` do not matter).
Precedence: command line, then parameter file, then default. An unknown key is an error with a
suggestion, so a typo cannot fall back to a default silently. Every run writes its fully
resolved settings to `pdu_out.ppf` in the output directory; `pdu run -p pdu_out.ppf` repeats it.
`pdu check -p file.ppf -f ... -s ...` prints the resolved settings without running anything.

The settings of the published runs are in `pdukit/params/` (`t4l_channels.ppf`,
`syt_c2a_channels.ppf`, and the experimental `*_cohesion.ppf`), used by the example Slurm
scripts in `pdukit/hpc/slurm/`. Edit the data paths at the top of those scripts for your
cluster; see [`pdukit/hpc/slurm/README.md`](pdukit/hpc/slurm/README.md).

## Outputs

`results.json` (everything), `units.csv`, `units_stable.csv`, `unit_directions.csv`,
`unit_cohesion.csv`, `secondary_assignments.csv`, `residue_ledger.csv`, `calibration.json`,
`pdu_out.ppf` (the resolved settings). Details in `docs/METHODS.md`.

## Tests

```bash
pytest                 # a few minutes on CPU
```

## Limitations

Results describe the sampled ensemble only: replicas that do not converge give
sampling-dependent units. Channel counts saturate with long simulations unless an effect size is
declared. Details and open issues are in `docs/METHODS.md`.

## Citation

If you use this code, please cite it (see `CITATION.cff`; a manuscript reference will be added
here when available).

## Licence

GNU General Public License v3.0 or later, see `LICENSE`.
