# Tutorial: install, run, analyse

This walks you from a clean machine to figures for a protein. About 30 minutes, most of it waiting for the computer.

**The idea in one sentence.** `pdu` reads MD trajectories, finds groups of residues that move together and pass information on with a delay (the units), and tells you how sure it is about each one.

**Conventions.** Residue numbers in outputs are 0-based. Lags and windows are in frames. Every setting can be given on the command line or in a `.ppf` file; the command line wins.

## 0. How COG works, in brief

**The question.** Inside a protein, which groups of residues move as a team, and which residues pass information from one team to another? COG answers this from MD alone: it looks at how motion at one residue *predicts* later motion at another, never at contacts or distances in space when it searches.

**The steps** (the log phase codes in section 4 follow these):

1. **Features.** Each residue gets a few numbers per frame describing how its surroundings move (distances to its neighbours, reduced to its slowest tICA directions: by default as many as the median residue needs to keep `rse-coverage` = 90% of its kinetic variance, or a fixed number with `rse-components` = K). Slow drifts are removed (high-pass) and each replica is centred, so replicas can be pooled (DAT, PRE).
2. **Coupling.** For every ordered pair, `rho[i, j]` measures how well residue j's motion now predicts residue i's motion `lag` frames later, *beyond* what i's own motion already predicts. The arrow is j to i. To know what counts as strong, the same quantity is computed on time-shifted copies of the data, which destroys real coupling and gives a null level `theta`. Pairs with `rho` above `theta` are significant edges (WSP, GRA, CAL).
3. **Closed groups.** A set of residues S is a good unit when little information comes in from outside. COG takes S's slowest modes and the rest of the protein's modes, finds their canonical correlations `rho_k`, and counts the channels above the null: `nu_in = sum_k max(0, 1 - theta^2 / rho_k^2)`. Small `nu_in` means a closed group. Random regions of the same size give the baseline to beat.
4. **Growth.** From many random seed residues (`dec-seeds`), a group is grown one residue at a time, accepting a residue only if it does not raise `nu_in`. The result is a candidate core (GRO).
5. **Merge.** Seeds that land on the same group are merged: candidates overlapping at least `merge-overlap` (0.5) with a family's majority core join it (MRG). How many seeds agree becomes the confidence score (RCS and UCS, section 5).
6. **Split.** A merged group that is really two is cut by recursive spectral splitting. A cut is kept only if the two parts are less coupled to each other than a null model allows (SPL).
7. **Ports and bridges.** Residues left out of every core are not discarded. A residue that talks mostly to one unit becomes that unit's **port**; one that links several units becomes a **bridge**; one that fits nowhere stays **loose** (POR).
8. **Check.** The units are used as a basis to rebuild the slow, protein-wide motions found by plain tICA; how well they do it is reported as global-mode recovery (REF and the recovery tables).

**What the output means, and what it does not.** A unit is a group whose motion is coupled within itself and weakly fed from outside, *at the chosen lag and with these features*. Arrows are predictive (statistical), not proof of physical causation. The splitting null is an exploratory shuffling model, not a full temporal null. Results describe the sampled ensemble: short or unconverged replicas give units for what was sampled, nothing more. Try several lags before trusting a boundary.

---

## 1. Install

Needs Python 3.10+ (and `ffmpeg` only if you want the movie).

```bash
git clone https://github.com/scdantu/pdukit.git      # use the repo address you were given
cd pdukit
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[xtc,plot,dev]"
```

Extras: `xtc` reads XTC/DCD/TRR (via MDAnalysis), `plot` adds matplotlib, `dev` adds the tests, `gpu` is for CUDA machines (`pip install -e ".[gpu]"`).

Check it works:

```bash
pdu -h                       # lists the modules
pytest -q                    # all tests should pass
```

If `pdu` is not found, use `python -m pdukit` instead.

## 2. First run: the planted benchmark (about 10 minutes on a CPU)

`pdu bench` builds a fake protein whose answer is known, and runs the pipeline on it. If it finishes with the log line `UNI   units: 4 closed, 0 weak`, your install is good (the planted answer is 4 units).

```bash
pdu bench -o out/planted
```

## 3. Run on trajectories

### 3a. Try the demo trajectories (1 minute)

Make a small synthetic protein (40 residues, 3 replicas, one XTC each):

```bash
python tools/make_demo_trajectories.py demo_data
pdu run -p pdukit/params/tutorial.ppf -f demo_data/rep*.xtc -s demo_data/top.pdb \
        -o out/demo --check    # prints the resolved settings and where each came from; runs nothing
pdu run -p pdukit/params/tutorial.ppf -f demo_data/rep*.xtc -s demo_data/top.pdb -o out/demo
```

Be aware: the demo is a **toy for learning the workflow**, not a test of accuracy. Its signal is noisy and the units it finds only partly line up with the four blocks it was built from. Use `pdu bench` (section 2) to judge recovery.

### 3b. Your own protein

You need:

1. **One trajectory file per replica** (XTC, DCD, TRR...). Independent replicas are what make the results trustworthy; one long run is much weaker.
2. **A topology** for the same atoms (PDB, GRO, TPR...). Strip water and ions first, keep the protein only.
3. **The time between frames** in ps (`frame-ps`).

```bash
pdu run -p pdukit/params/t4l_incoming.ppf -f rep1.xtc rep2.xtc rep3.xtc \
        -s protein.tpr -o out/my_protein --set frame-ps=100
```

Choosing a preset (all in `pdukit/params/`):

| File | Use for |
|---|---|
| `default.ppf` | general starting point |
| `t4l_incoming.ppf` | the settings used for T4 lysozyme in the manuscript; good for ~150-residue proteins |
| `mdcath_incoming.ppf` | short mdCATH-style trajectories |
| `tutorial.ppf` | the demo only |

Copy a preset, edit it, keep it with your results. Format is `key = value`, `;` starts a comment.

**Settings and the command line.** `pdu run` takes only seven arguments: `-f` (trajectories), `-s` (topology), `-p` (the ppf), `-o` (output folder), `--set`, `--log-level` and `--check`. Everything else is a key in the ppf. To change one setting for one run without editing the file, use `--set`, for example `--set lag=20 dec-seeds=64`. The order of priority is: built-in default, then the ppf, then the command line. `--check` prints what will really be used and where each value came from (`default`, `ppf:LINE` or `cli`). `-f`, `-s` and `-o` can also be written in the ppf (`xtc = rep1.xtc rep2.xtc`). Old command lines such as `--dec-seeds 64` still work but print a deprecation warning. The full list of settings is in `docs/OPTIONS.md`.

The setting that matters most is **`lag`** (in frames): how far ahead one residue's motion is predicted from another's. Pick it so lag x `frame-ps` is a few hundred ps to a few ns, then check how the result changes with it (section 4, `plot lag`).

Long runs: use the Slurm scripts in `pdukit/hpc/slurm/` (see the README there).

Every run writes `pdu_out.ppf` into its output folder: the exact settings used. Keep it.

## 4. Read the log

Each line starts with a time and a 3-letter phase:

| Code | What is happening |
|---|---|
| INI, DAT | start, loading trajectories |
| PRE, REP | feature building and replica handling |
| WSP, GRA | pairwise coupling and the graph of significant couplings |
| CAL | calibrating thresholds against shuffled data |
| GAT, COH | acceptance gates and cohesion tests |
| GRO | growing candidate units from many random seeds |
| MRG, SPL | merging candidates found repeatedly; splitting ones that are really two |
| POR, REF | adding ports; refinement |
| SCR, FAC, UNI, RES, LED, STB, DEF, VER, CMP, MOD | optional extra checks |
| OUT | writing results |

Make it quieter or noisier with `--log-level quiet|info|debug` (or `log-level` in the ppf). Lines starting `!!` are warnings: read them.

## 5. Analyse the results

### 5a. Tables

`results.json` has everything. Turn it into spreadsheets or text:

```bash
pdu export out/demo                 # writes out/demo/export/*.csv and *.txt
pdu export out/demo --list          # names of all tables
pdu export out/demo --print units   # show one table in the terminal
```

Start with `overview` (what ran), `units` (the answer), `report.txt` (everything short). 

### 5b. What a unit is

A **unit** is a set of residues that move together. Residues in a unit have a class:

| Class | Meaning |
|---|---|
| core | belongs firmly to one unit |
| port | edge of a unit where it connects outward |
| bridge | links two or more units |
| shared | sits in the core of more than one candidate |
| loose | not assigned to any unit |

The **% of loose/unassigned residues** is a useful health number: a high value means the pipeline is not finding structure at that lag.

### 5c. How sure is it? (RCS and UCS)

- **RCS** (residue confidence score, 0-100): the percentage of independent growth seeds that put this residue in this unit.
- **UCS** (unit confidence score): the median RCS of the unit's core residues.

Bands work like pLDDT (high, good, low, very low). A unit supported by fewer than 3 seeds is flagged **low evidence**: treat it as a hint, not a result. If many units are flagged, run more seeds (`--set dec-seeds=64`).

### 5d. Figures

All need `pip install -e ".[plot]"`. Run `pdu plot <name> -h` for the options of each.

```bash
pdu plot units   out/demo                          # bar chart of units, sizes, confidence
pdu plot network out/demo                          # residue-to-residue map (arrow j -> i: j predicts i)
pdu plot pml     out/demo --pdb demo_data/top.pdb  # PyMOL script
pdu plot movie   out/demo --pdb demo_data/top.pdb  # needs ffmpeg: seeds growing into units
pdu plot lag     lag10=out/demo lag20=out/demo_lag20  # unassigned % versus lag
pdu plot compare WT=out/wt L99A=out/l99a --pdb wt.pdb # networks of variants and their differences
```

The PyMOL script has three scenes (keys F1, F2, F3): units as coloured cartoon; unit detail (core sticks, ports and bridges as spheres, transparent surface, significant couplings); confidence colouring. Open PyMOL in the folder holding the PDB, then `@out/demo/units.pml`.

### 5e. Many runs

```bash
pdu summarise out             # one table, one row per run folder inside out/
pdu compare out/a out/b      # do two runs find the same units?
```

Use these for lag scans, mutants, and replicas split into halves.

## 6. Common problems

| Problem | Fix |
|---|---|
| `pdu: command not found` | activate the venv, or use `python -m pdukit` |
| "needs MDAnalysis" | `pip install -e ".[xtc]"` |
| Plot says matplotlib missing | `pip install -e ".[plot]"` |
| Atom count differs between topology and trajectory | strip the same atoms in both |
| Many units flagged low evidence | more seeds (`--set dec-seeds=64`), more or longer replicas |
| High unassigned % | try other lags; check replicas really are independent |
| Run too slow | use a GPU (`.[gpu]`, `--set backend=...`), or the Slurm scripts |
| Want to know what a setting does | `pdu run -h`, `docs/OPTIONS.md` |
