# pdu — COG (v1.0.5a4)

> **Alpha release.** The method and the code are under active development; interfaces, output
> formats and defaults may change before v1.0. Internally this is algorithm lineage "COG 2.1".

COG finds **Protein Dynamical Units (PDUs)** in molecular dynamics (MD) data: groups of
residues whose motions are coupled to each other more strongly than to the rest of the
protein, at a chosen lag time and for chosen per-residue features. It uses dynamics only;
spatial contacts are not used in the search or in the baselines.

This repository is the alpha pre-release (`v1.0.5a4`) of the COG pipeline (`pdukit` Python package).
It is the code behind the T4 lysozyme and synaptotagmin-1 C2A analyses.

Every unit is scored two ways:

1. **Closure** (the objective that is optimised): how many independent channels link the unit's
   present and future to the rest of the protein, compared with random regions of the same size.
2. **Predictive cohesion** (a check, reported at the end): whether residues inside the unit
   predict each other's futures better than they predict, or are predicted by, outside residues.

Per-residue labels (driver / listener / both / none) say which way each member's coupling points.
A short account of how COG works is in section 0 of [`docs/TUTORIAL.md`](docs/TUTORIAL.md).

## In a notebook (Colab) -- four objects

```python
!pip install "pdukit[xtc,plot,notebook]"
import pdukit as pdu

ens    = pdu.Ensemble(["rep1.xtc", "rep2.xtc"], "top.pdb")   # or pdu.demo()
cfg    = pdu.Config.from_ppf("my.ppf", lag=20)                # cfg.lag = 30 to change
result = pdu.Cog(cfg).run(ens, out="runs/x")

result.protein            # residues: unit, label (core/port/bridge/loose/shared), confidence
result.table("units")     # pandas DataFrame
result.plot_units()       # figure;  result.view() = 3D view;  result.save_pml()
```

`cfg2 = cfg.copy()` makes an independent copy (`cfg2 = cfg` is the same object).
The `pdu` command line uses the same objects. A runnable version is
`examples/quickstart.ipynb`.

## Install

```bash
git clone https://github.com/scdantu/pdukit.git && cd pdukit
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"            # numpy, scipy, threadpoolctl + pytest
pip install -e ".[xtc]"            # to read XTC/DCD trajectories (MDAnalysis)
pip install -e ".[gpu]"            # optional: batched scoring on a CUDA GPU (torch)
```

Python 3.10 or newer. Runs on CPU with numpy only; a GPU speeds up large proteins.

**GPU.** torch must match your CUDA driver, so install it from PyTorch's index first, e.g. for
CUDA 12.1: `pip install "torch<2.6" --index-url https://download.pytorch.org/whl/cu121`
(see pytorch.org for other versions). `device = auto` (the default) then uses the GPU; the log
line `WSP ... backend torch on cuda` confirms it, and a `!!` warning says why when a GPU is
present but not used. The feature step always runs on the CPU.

**Workstation (32-64 GB RAM, consumer GPU such as an RTX 3070, or none).** Start from the
`workstation` preset: `pdu check -p workstation > my.ppf`, set `frame-ps`, `lag` and
`highpass-window` in it, then `pdu run -p my.ppf ... --nt 8`. It runs on the CPU (consumer GPUs
are slow in double precision) and fits a 400-residue protein in about 16 GB. Memory, time and
GPU notes are in [`docs/OPTIONS.md`](docs/OPTIONS.md#memory-time-and-where-to-run).

## Quick check on a planted system (minutes, CPU)

The benchmark builds a synthetic protein with known units and checks that COG recovers them.

```bash
pdu bench -o out/planted
```

### Synthetic test suite (the manuscript tests)

Named designs with a known answer live in `pdukit.bench.designs` and are
chosen with `bench-design`: `units`, `mixed`, `nocomm`, `medium`, `tau`
(100 residues, one frame = 0.1 ns) and the 36-residue `small-*` set.

```bash
pdu bench -p pdukit/params/synthetic100.ppf -o runs/tau_g0_lag5 \
    --set bench-design=tau bench-seed=0 lag=5
bash pdukit/hpc/synthetic/run_suite.sh pdukit/params/synthetic100.ppf \
    pdukit/hpc/synthetic/jobs100_tau.txt runs 2      # a whole job list
pdu score-bench runs/* -o tables                     # recovery tables
```

| Test | Job list | Question |
|---|---|---|
| 1 | `jobs100_test1.txt` | units found when they exist (`units`), none when they do not (`mixed`, `medium`) |
| 2-3 | `jobs100_tau.txt` | units found vs lag time; is a unit with processes on every time scale found at every lag? |
| 4 | `jobs100_test4.txt` | units without channels (`nocomm`) vs number of growth seeds |

## Run on your own trajectories

```bash
pdu run -f rep1.xtc rep2.xtc rep3.xtc -s protein.tpr \
        -p pdukit/params/t4l_incoming.ppf -o out/my_protein
```

The command is `pdu <module>`: `run`, `bench` (planted system with a known answer), `check`
(print the resolved settings, run nothing), `export`, `plot`, `summarise`, `compare` and `score-bench` (synthetic suite tables).
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

`pdu run` takes only eight arguments (`-f` trajectories, `-s` topology, `-p` ppf, `-o` output
folder, `--nt` CPU threads (default -1 = all cores), `--set`, `--log-level`, `--check`); everything else is a key in the ppf, and any key can
be overridden for one run with `--set lag=20 dec-seeds=64`. Case, `-` and `_` do not matter in
keys. Precedence: default, then parameter file, then command line. An unknown key is an error
with a suggestion, so a typo cannot fall back to a default silently. Every run writes its fully
resolved settings to `pdu_out.ppf` in the output directory; `pdu run -p pdu_out.ppf` repeats it.
`pdu run -p file.ppf --check` prints the resolved settings, and where each value came from,
without running anything.

`pdu export RUN_DIR` turns a run's `results.json` into CSV and aligned ASCII text tables
(final units, calibration, split tree, ports, timings, settings, ...); see the tutorial.

**New here?** Start with the step-by-step [`docs/TUTORIAL.md`](docs/TUTORIAL.md) (install, run, analyse).

`pdu plot <tool> RUN_DIR` makes the figures: `pml` (PyMOL script with unit, detail and confidence scenes),
`units` (unit sizes with confidence scores UCS / RCS), `lag`, `network`, `compare` (residue networks of
variants) and `movie` (all growth seeds growing, merging and finalising). Install with
`pip install "pdukit[plot]"`; see the tutorial and `pdu plot <tool> -h`.

Every setting, with its default, allowed values, meaning and the presets that change it, is
listed in [`docs/OPTIONS.md`](docs/OPTIONS.md). [`pdukit/params/default.ppf`](pdukit/params/default.ppf)
is a parameter file with every setting at its default and its meaning written above it: copy
it, delete the lines you do not change, and use it with `-p`. Both files are generated from the
settings table in `pdukit/config/schema.py` (`python -m pdukit.config.reference --write`) and a test fails if
they fall out of date.

The settings of the published runs are in `pdukit/params/` (`t4l_incoming.ppf`,
`syt_c2a_incoming.ppf`, `mdcath_incoming.ppf` and `betalactamase_incoming.ppf`), used by the example Slurm
scripts in `pdukit/hpc/slurm/`. Edit the data paths at the top of those scripts for your
cluster; see [`pdukit/hpc/slurm/README.md`](pdukit/hpc/slurm/README.md).

## Outputs

`results.json` (everything), `units.csv`, `units_stable.csv`, `unit_directions.csv`,
`unit_cohesion.csv`, `secondary_assignments.csv`, `residue_ledger.csv`, `calibration.json`,
`pdu_out.ppf` (the resolved settings). Use `pdu export` for tables; see the tutorial.

## Tests

```bash
pytest                 # a few minutes on CPU
```

## Limitations

Results describe the sampled ensemble only: replicas that do not converge give
sampling-dependent units. Channel counts saturate with long simulations unless an effect size is
declared. See the end of section 0 of the tutorial.

## AI usage

This software was prepared with the assistance of Claude Code (Anthropic); the scientific
method and its interpretation are the authors' own. See `AI_USAGE.md` for details and verification.

## Citation

If you use this code, please cite it (see `CITATION.cff`; a manuscript reference will be added
here when available).

## Licence

GNU General Public License v3.0 or later, see `LICENSE`.
