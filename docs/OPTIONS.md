# PDU settings reference

Generated from the settings table by `python -m pdukit.config.reference --write`; do not edit by hand (the explanations are in `pdukit/config/guide.py`).

Every setting is a `key = value` line in a parameter file (a `.ppf`).  Start
from a shipped preset (listed at the end of this page; `pdu check -p
workstation > my.ppf` writes a copy) or from `pdu check > my.ppf`, which
writes every setting at its default with a comment.  Then run

    pdu run -p my.ppf -f rep1.xtc rep2.xtc -s top.pdb -o out

`pdu run` has eight arguments: `-f` trajectories (one file per replica), `-s`
topology, `-p` parameter file, `-o` output folder, `--nt` CPU threads,
`--set` (change any setting for one run: `--set lag=20 n-null=32`),
`--log-level` and `--check` (print what will be used, and where each value
came from, then stop).  Precedence: built-in default, then the parameter
file, then the command line.  Case, `-` and `_` do not matter in keys; an
unknown key is an error with a suggestion.  Flags are written `1` or `0`.

Times are in **frames**, not ps: `lag = 10` with 20 ps frames is 200 ps.

Every run writes `pdu.log` (the screen log) and `pdu_out.ppf` (every setting
it really used) into the output folder; `pdu run -p out/pdu_out.ppf ...`
repeats it.

## Start here: the settings most runs change

| setting | default | what it means |
|---|---|---|
| `lag` | 1 | The time step at which coupling is measured, in frames. A unit is a group of residues whose future depends on the rest of the protein as little as possible, AT THIS TIME SCALE. Choose it from the motion you care about: 100 ps - 10 ns are typical. Units change with the lag; a scan over several lags is the honest way to report them. |
| `frame-ps` | 0.0 | Time between saved frames in ps. Only used to print times in the log; set it so the log speaks in ps and ns. |
| `highpass-window` | 0 | Removes motions slower than this many frames inside every replica before anything is measured. Slow drift that a trajectory does not sample makes everything look correlated with everything and hides real units. 20 x lag is our rule; 0 = off; 'auto' scans a few windows to choose one (slow: about one workspace per window). |
| `highpass-before-features` | off | 1 = filter the raw distances before each residue's slow modes are chosen (recommended with a high-pass window). |
| `rse-mode` | unset | How K, the number of slow modes (tICA) kept per residue, is set: fixed (you give K in rse-components) or auto. K sets the size of the problem: memory grows with (residues x K)^2. At short lags each residue has many similar fast modes (no gap in its spectrum), so the automatic K grows quickly as the lag shortens. |
| `rse-coverage` | 0.9 | Automatic K, condition 1: keep the fewest modes that hold this share of a residue's slow motion (sum of lambda^2), e.g. 0.9. K is the median over residues. |
| `rse-min-share` | 0.0 | Automatic K, condition 2: keep only modes that each add at least this share (e.g. 0.03 = 3%). Each residue stops at whichever condition comes first, so K gets as close to rse-coverage as modes worth keeping allow. 0 = off. |
| `rse-max-components` | 0 | Upper limit on the automatic K: a safety cap for memory; 8-12 fits a workstation. |
| `n-null` | 64 | Time shifts used to build the null (the 'what would chance give' reference). 64 is precise; 32 halves the memory and time of the null with slightly noisier thresholds. |
| `dec-seeds` | 24 | Number of growth starts. Keep seeds at or below a quarter of the residues (see seed-fraction) so units are found by growth, not by cutting overlapping candidates apart. |
| `seed-fraction` | 0.0 | Caps the seeds at this fraction of the residues (0.25 recommended). |
| `seed-group` | 1 | Residues in each growth start: 2 = the seed plus its strongest partner. A single residue has too few slow modes to predict anything, so the first addition often looks worse; starting from pairs avoids that barrier. |
| `min-size` | 3 | Smallest unit kept. Use at least twice seed-group (4 with pairs), or pairs that never grew are reported as units. |
| `device` | auto | Where the heavy algebra runs: auto (a CUDA GPU when torch can use it), cpu or cuda. pdu works in double precision: data-centre GPUs (A100, H100, GH200) are fast at it, consumer cards (GeForce, RTX A-series) are not, so use cpu there. |
| `nt` | -1 | CPU threads (`--nt N` on the command line). -1 = all the cores the job may use; on a shared machine give the number you were allocated. |
| `log-level` | info | quiet, info (one line per step) or debug (every growth step). Lines starting with ! are warnings. |

## Memory, time and where to run

**Memory.**  The estimator stacks every residue's K modes into one vector of
length D = residues x K and keeps D x D matrices in double precision: three
for the data and two for every null shift.  So

    bytes ~ (6 + 2 x n-null) x (residues x K)^2 x 8  +  12 x frames x D

| residues | K | n-null | matrices (without the frames term) |
|---|---|---|---|
| 160 | 8 | 64 | ~2 GB + frames term |
| 400 | 8 | 32 | ~6 GB |
| 400 | 12 | 32 | ~13 GB |
| 400 | 20 | 64 | ~69 GB |

The frames term matters for long trajectories: 2.6 million frames at
D = 4400 add about 140 GB.  Before each large step the log prints the
estimate and the free memory (`memory for the workspace: ~13.0 GB ...`);
if it does not fit, the run stops at once and names the settings that do
(`rse-max-components`, `n-null`).  `no-memory-check = 1` skips this.

**Time.**  Building the null costs about frames x D^2 x n-null operations,
so the same three numbers set the run time.  Fewer frames (`stride = 2`
keeps every second frame; then halve `lag` and `highpass-window`, which
are in frames) is the other lever.

**On a workstation** (32-64 GB, consumer GPU or none): start from the
`workstation` preset.  It runs on the CPU, keeps at most 12 modes per
residue and 32 shifts, and fits a 400-residue protein with 60,000 frames in
about 16 GB.  Write your own copy, set the time settings (`frame-ps`, `lag`,
`highpass-window`) in it, and run:

    pdu check -p workstation > my.ppf
    pdu run -p my.ppf -f rep1.xtc rep2.xtc -s top.pdb -o out --nt 8

(A shipped preset can be named without its path: `-p workstation`.)

**On a GPU**: install torch for your CUDA first (CUDA 12.1:
`pip install "torch<2.6" --index-url https://download.pytorch.org/whl/cu121`).
The log line `WSP ... backend torch on cuda` confirms the GPU is used; a
warning names cards that are slow in double precision.  The GPU must also
hold the matrices above, so on an 8 GB card with 32 shifts D = residues x K
must stay below about 3,800 (e.g. 300 residues at K = 12).

## All settings

The last column lists the shipped presets that change the option from its default, with their value.

### Files and general options

Run bookkeeping: random seed, log detail, threads, and where the files are.

| setting | default | values | what it does | presets |
|---|---|---|---|---|
| `seed` | 0 |  | random seed | planted_fast: 1, synthetic100: 1, synthetic36: 1 |
| `log-level` | info | quiet, info, debug | progress log detail: quiet = warnings only, info = one line per step, debug = also per-growth-step traces |  |
| `nt` | -1 |  | CPU threads for numpy/scipy and torch (-1 = all cores this process may use) (run-specific, not in parameter files) |  |
| `outdir` | unset |  | output directory (run-specific, not in parameter files) |  |
| `params` | unset |  | parameter file (.ppf, key = value); command-line options win (run-specific, not in parameter files) |  |

### Planted benchmark

`pdu bench` runs the pipeline on a synthetic system whose units are known, to check an install or a setting.  These settings describe that system; they are ignored for MD data.

| setting | default | values | what it does | presets |
|---|---|---|---|---|
| `planted` | off |  | run on a synthetic planted system with a known answer instead of MD files (-s, -f) | planted_fast: 1, synthetic100: 1, synthetic36: 1 |
| `n-units` | 5 |  | (planted) number of units | planted_fast: 6, synthetic100: 6, synthetic36: 6 |
| `unit-size` | 10 |  | (planted) residues per unit | planted_fast: 6, synthetic100: 6, synthetic36: 6 |
| `planted-m` | 4 |  | (planted) features per residue |  |
| `latent-dim` | 5 |  | (planted) slow latent coordinates per unit (the largest possible coupling rank) |  |
| `frames-per-replica` | 10000 |  | (planted) frames in each replica | planted_fast: 3000, synthetic100: 3000, synthetic36: 3000 |
| `n-replicas` | 3 |  | (planted) number of replicas (independent trajectories) | planted_fast: 2, synthetic100: 2, synthetic36: 2 |
| `coupling-gain` | 0.35 |  | (planted) strength of the couplings between units |  |
| `face-residues` | 1 |  | (planted) residues per unit that carry its coupling (the face) | synthetic100: 0, synthetic36: 0 |
| `decoy-pairs` | 3-4 |  | comma list like '3-4' of uncoupled adjacent units |  |
| `discontinuous` | off |  | (planted) build each unit from two stretches of chain that are far apart |  |
| `bench-seed` | 0 |  | (planted) random seed of the generated system |  |
| `scatter` | off |  | permute planted residue positions so units are not spatially contiguous (tests a dynamics-only search) |  |
| `bench-design` | planted | planted, units, mixed, nocomm, medium, tau, small-units, small-mixed, small-nocomm, small-medium, small-hierarchy | synthetic system for pdu bench: 'planted' (built from n-units, unit-size, ... above) or a named design from pdukit.bench.designs: units, mixed, nocomm, medium, tau (100 residues, 1 frame = 0.1 ns); small-units, small-mixed, small-nocomm, small-medium, small-hierarchy (the 36-residue suite) | synthetic100: units, synthetic36: small-units |
| `design-obs-noise` | 0.25 |  | (named design) observation noise added to every residue feature |  |
| `design-frames` | 0 |  | (named design) frames per replica; 0 = the design default |  |
| `design-replicas` | 0 |  | (named design) number of replicas; 0 = the design default |  |
| `design-rho-fast` | 0.8 |  | (small-hierarchy) per-frame autocorrelation of the fast sub-unit modes |  |

### MD data

How trajectories become per-residue features: which frames are read, and how each residue's environment (its distances to every residue within rse-cutoff) is compressed into K slow modes.

| setting | default | values | what it does | presets |
|---|---|---|---|---|
| `xtc-topology` | unset |  | topology (tpr, pdb, gro, ...) (run-specific, not in parameter files) |  |
| `xtc` | unset | one or more | trajectory file(s), one per replica (run-specific, not in parameter files) |  |
| `frames` | 0 |  | first N frames per replica (0 = all); also applied to synthetic systems (pdu bench) |  |
| `stride` | 1 |  | use every Nth frame of each replica (then lag and highpass-window, which are in frames, must be divided by N); also applied to synthetic systems |  |
| `rse-cutoff` | 15.0 |  | radius in angstrom of a residue's structural environment: neighbours inside it enter its descriptor |  |
| `rse-mode` | unset | auto, fixed | how K, the number of slow tICA modes per residue, is set: 'fixed' = rse-components; 'auto' = chosen from rse-coverage and rse-min-share (unset: auto when rse-components = -1, else fixed) |  |
| `rse-components` | -1 |  | K with rse-mode = fixed: slow tICA modes kept per residue after compressing its neighbour distances (a number >= 1); -1 = choose it automatically (two passes over the distances) | betalactamase_incoming: 8, mdcath_incoming: 8, planted_fast: 8, synthetic100: 8, synthetic36: 8, syt_c2a_incoming: 8, t4l_incoming: 8, tutorial: 8 |
| `rse-coverage` | 0.9 |  | automatic K, condition 1: the fewest modes at which a residue keeps this share of its kinetic variance (sum of lambda^2 over its whole spectrum); K is the median over residues; always reported in results tica_spectrum.coverage |  |
| `rse-min-share` | 0.0 |  | automatic K, condition 2: keep only modes that each add at least this share of a residue's sum of lambda^2 (e.g. 0.03); a residue stops at whichever condition comes first, so K gets as close to rse-coverage as modes worth keeping allow (0 = condition off) |  |
| `rse-max-components` | 0 |  | automatic K: upper limit (0 = none); memory grows with (residues x K)^2 | workstation: 12 |
| `min-seq-sep` | 2 |  | neighbours closer than this in sequence are left out of the descriptor (trivially rigid) |  |
| `contact-cutoff` | 8.0 |  | angstrom cutoff of the residue contact graph (growth only follows contacts) |  |
| `pair-features` | off |  | add pair-owned contact-distance features (Stage 4) |  |
| `pair-cutoff` | 8.0 |  | (pair features) angstrom cutoff for pair contacts |  |

### Estimator

How coupling is measured: the lag, how many slow coordinates describe a unit and its environment, the null that sets the noise threshold, and where the algebra runs.

| setting | default | values | what it does | presets |
|---|---|---|---|---|
| `lag` | 1 |  | lag in frames between the two time points whose coupling is measured (one frame = frame-ps) | syt_c2a_incoming: 5, t4l_incoming: 25, tutorial: 10, workstation: 10 |
| `unit-dim` | 8 |  | slow coordinates kept to describe each unit |  |
| `env-dim` | 20 |  | slow coordinates kept to describe the environment (the rest of the protein) |  |
| `ridge` | 0.01 |  | regularisation added to the covariance matrices before they are inverted |  |
| `n-null` | 64 |  | time shifts used to build the null distribution of the canonical correlations | betalactamase_incoming: 128, mdcath_incoming: 128, syt_c2a_incoming: 128, t4l_incoming: 128, tutorial: 32, workstation: 32 |
| `shift-mode` | auto | auto, global, within, cross | how the null displaces the environment in time: global (pooled), within (each replica), cross (a different replica); auto = cross with 2+ replicas, else within |  |
| `backend` | flux | flux, schur | estimator backend: flux (FluxWorkspace) or schur (Schur- complement form) |  |
| `combine` | max | max, sum, in, out | how incoming and outgoing channels form the cost: max, sum, in (incoming only) or out (outgoing only) |  |
| `device` | auto |  | where the estimator runs: auto (CUDA GPU when torch can use one, else CPU), cuda, cpu or mps | workstation: cpu |
| `no-memory-check` | off |  | skip the check, before each workspace is built, that it fits in the free RAM (and GPU memory); the check stops the run with advice instead of letting it be killed |  |
| `cond-unit` | full | full, reduced | unit's present subtracted whole (COG 2.1) or as unit_dim slow modes (v1.0) |  |
| `cond-env` | full |  | environment's present subtracted whole ('full'), as N slow modes (an integer), or as env_dim modes ('reduced', v1.0) |  |

### Calibration

Before the search, random groups of each size are scored to learn what a group of that size looks like by chance.  Units must beat that baseline.  These settings trade calibration precision for time.

| setting | default | values | what it does | presets |
|---|---|---|---|---|
| `sizes` | unset |  | calibration sizes, e.g. '2-40' (default 2..max-unit) | betalactamase_incoming: 2-120, mdcath_incoming: 2-120, planted_fast: 2-14, synthetic100: 2-24, synthetic36: 2-14, syt_c2a_incoming: 2-120, t4l_incoming: 2-120, tutorial: 2-20 |
| `target-rel-width` | 0.03 |  | (calibration) stop adding supports once the bootstrap interval of a threshold is this fraction of its value |  |
| `min-supports` | 40 |  | (calibration) fewest random supports scored per size | tutorial: 20 |
| `max-supports` | 300 |  | (calibration) most random supports scored per size | tutorial: 80 |
| `batch-supports` | 32 |  | calibration supports scored per batch |  |
| `calibrate-only` | off |  | stop after the calibration (to check thetas, the baseline and the time before a full run) |  |

### Discovery

The search itself: high-pass filtering and its window scan, seeds, growth (add the residue that best lowers incoming dependence), merging of repeated candidates, the recursive split test, ports, and performance switches.  Most runs change only the seed settings, min-size and the high-pass window.

| setting | default | values | what it does | presets |
|---|---|---|---|---|
| `discovery-replicas` | unset |  | comma list of replica indices (0-based) |  |
| `holdout-replicas` | unset |  | comma list, or 'none' to discover on everything |  |
| `n-holdout` | unset |  | hold out the last N replicas (0 = discover on all); alternative to the two lists above | betalactamase_incoming: 0, mdcath_incoming: 0, syt_c2a_incoming: 0, t4l_incoming: 0, tutorial: 0 |
| `no-center-replicas` | off |  | keep each replica's mean (v1.0 behaviour); by default every replica is centred on its own mean |  |
| `highpass-before-features` | off |  | apply --highpass-window to the raw neighbour distances BEFORE each residue's slow directions are chosen (MD data) | betalactamase_incoming: 1, mdcath_incoming: 1, syt_c2a_incoming: 1, t4l_incoming: 1, tutorial: 1, workstation: 1 |
| `fit` | off |  | remove overall rotation and translation (Kabsch) before building features | betalactamase_incoming: 1, mdcath_incoming: 1, syt_c2a_incoming: 1, t4l_incoming: 1, tutorial: 1 |
| `calibrate-coarse` | 0 |  | calibrate only ~N sizes spread over the range and interpolate the rest (0 = every size) | betalactamase_incoming: 8, mdcath_incoming: 8, synthetic100: 6, syt_c2a_incoming: 8, t4l_incoming: 8, tutorial: 4, workstation: 4 |
| `skip-frames` | 0 |  | drop the first N frames of every replica |  |
| `highpass-window` | 0 |  | remove a running mean of this many frames within every replica (0 = off; 'auto' = the smallest window whose null threshold is within --null-cal-tol of the lowest in the null calibration scan) | betalactamase_incoming: 50, mdcath_incoming: 50, syt_c2a_incoming: 100, t4l_incoming: 500, tutorial: 300, workstation: 200 |
| `no-null-calibration` | off |  | never run the highpass-window scan, not even for highpass-window = auto (which then stops with an error) | betalactamase_incoming: 1, mdcath_incoming: 1, syt_c2a_incoming: 1, t4l_incoming: 1, tutorial: 1 |
| `raw-feature-pass` | off |  | with a fixed highpass-window and highpass-before-features: also build the unfiltered features first (costs one extra feature pass and its memory; only the replica-overlap check uses them, on unfiltered data) |  |
| `no-residue-affinity` | on |  | do not record every residue's affinity to every unit (residue_affinity.csv: change in nu_in, nu_out and cohesion when the residue joins or leaves each unit's core; costs about residues x units extra scores) |  |
| `null-cal-check` | off |  | run the highpass-window scan even when the window is fixed or off, as a check (it is always run for highpass-window = auto); costs about one workspace per window tried |  |
| `null-cal-windows` | auto |  | comma list of windows (frames) to scan; 'auto' = geometric from 10 x lag to a replica / 8 |  |
| `null-cal-n-windows` | 6 |  | (highpass-window scan) number of windows tried |  |
| `null-cal-sizes` | 3,10 |  | (highpass-window scan) comma list of support sizes used |  |
| `null-cal-supports` | 48 |  | (highpass-window scan) random supports per size |  |
| `null-cal-n-null` | 32 |  | (highpass-window scan) time shifts per null |  |
| `null-cal-target` | 0.06 |  | largest acceptable null threshold (theta_in and theta_out); a warning is printed if no window reaches it |  |
| `seeding` | random | random, spread | 'random' = N residues drawn at random (dynamics only); 'spread' = farthest-point on the graph |  |
| `search-graph` | predictive | predictive, contact, union | graph used everywhere a neighbourhood is needed (growth, round 2, and the random comparison regions of the baseline/null): 'predictive' = residues whose present predicts another's future beyond a shift null (dynamics only); 'contact' = spatial contacts (old behaviour); 'union' = both |  |
| `baseline` | graph | graph, random, contact | comparison regions for the cost baseline: 'graph' = random connected sets on the search graph; 'random' = uniformly random residue sets; 'contact' = compact spatial balls (old) |  |
| `graph-k` | 5 |  | strongest predictive partners kept per residue |  |
| `graph-null-shifts` | 16 |  | time shifts used for the null of the predictive graph |  |
| `rho-min` | 0.0 |  | minimum channel strength: a channel counts only if its correlation exceeds max(noise threshold, this) (effect-size floor; 0 = off) |  |
| `block-us` | 1.0 |  | block length in us (MD data) |  |
| `block-frames` | 0 |  | block length in frames (planted data; 0 = auto) |  |
| `faces` | off |  | cores and faces: residues that carry a second unit's motion become faces (attached to both units); cores are scored with faces removed from both sides |  |
| `no-ledger` | on |  | do not record the per-residue ledger | betalactamase_incoming: 0, mdcath_incoming: 0, planted_fast: 0, synthetic100: 0, synthetic36: 0, syt_c2a_incoming: 0, t4l_incoming: 0, tutorial: 0 |
| `resolve` | off |  | second round for residues in groups with cost > 0: grow new units inside that pool, then let pool residues join a unit only if its cost improves; the rest get a secondary (best-place) assignment |  |
| `stability` | off |  | after round 2: steepest descent over one-residue changes for every unit, then n random kicks; reports whether each unit is a stable minimum (cog2_stability) |  |
| `grow-objective` | incoming | cohesion, channels, incoming | incoming (default): strictly decrease nu_in at every addition; no J or outgoing penalty; save growth paths and apply the selected incoming-postprocess mode. channels and cohesion are the archived COG 2.1 modes (git tag archive/pre-incoming-only) and are refused by pdu run |  |
| `incoming-min-gain` | 0.0 |  | incoming growth: require nu_in to decrease by more than this value (default strictly decreasing; 1e-12 numerical tolerance) |  |
| `start-min-gain` | 0.0 |  | incoming growth: the FIRST addition to a start must lower nu_in by more than this (later additions use incoming-min-gain); stops starts that only grow on noise. 0 = off |  |
| `incoming-starts` | unset |  | JSON file of zero-based starting residue lists; otherwise use the configured single-residue seeds |  |
| `incoming-postprocess` | split | none, split | incoming growth: recursive cohesion splitting (default); none is growth only (the older mc and merge-mc refinement is archived) |  |
| `split-method` | recursive-spectral | recursive-spectral, directed-modularity, markov-stability | within-parent cut proposal after incoming-only growth |  |
| `split-markov-time` | 16.0 |  | graph diffusion time, separate from the MD lag |  |
| `split-test` | both | both, graph-null, bootstrap, none | graph-exchangeability null and/or selected-cut block bootstrap |  |
| `split-null` | 199 |  | reciprocal-dyad graph nulls; repeat cut search for each | planted_fast: 39, synthetic100: 39, synthetic36: 39 |
| `split-alpha` | 0.05 |  | (recursive split) significance level of the cohesion test (halved at each depth) |  |
| `split-max-depth` | 8 |  | (recursive split) how many times a unit may be split in turn |  |
| `port-z` | 1.645 |  | ports: a left-out residue joins the unit it prefers both ways if neither the step change in cohesion nor the change against the core exceeds this many bootstrap standard deviations (guarded rounds) |  |
| `mode-units` | extended | extended, core | units given to global-mode recovery after splitting: core + ports (extended) or cores only |  |
| `score-sets` | unset |  | JSON file of fixed residue sets (zero-based); score their incoming/outgoing channels and cohesion instead of searching. Sets with mode_unit true are used for global-mode recovery |  |
| `merge-overlap` | 0.5 |  | incoming split path: before splitting, group grown supports (best nu_in first) into families when they share at least this fraction of the smaller of the support and the family's majority core; parent = union if its nu_in is not higher than the best member's, else the best member; the split then re-tests each parent |  |
| `cohesion-boot` | 20 |  | bootstrap resamples for the cohesion noise margin | planted_fast: 8, synthetic100: 8, synthetic36: 8 |
| `cohesion-blocks` | 20 |  | at least this many blocks (whole replicas if there are enough, else replicas split in contiguous blocks) |  |
| `null-cal-tol` | 0.15 |  | 'auto' window = the smallest window whose null threshold is within this fraction of the lowest one in the scan |  |
| `null-cal-max-replicas` | 20 |  | scan on at most this many replicas (evenly spaced) |  |
| `frame-ps` | 0.0 |  | time per frame in ps, for the logs only | betalactamase_incoming: 1000.0, mdcath_incoming: 1000.0, syt_c2a_incoming: 100.0, t4l_incoming: 20.0, tutorial: 20, workstation: 20 |
| `batched` | auto | auto, on, off | score candidates in batches on the GPU engine (auto: when the workspace is on CUDA) |  |
| `env-basis` | exact | exact, ritz | 'ritz' = environment slow basis from the global slow modes + block inverse (faster, approximate) |  |
| `feature-cache` | unset |  | feature cache directory (default: <outdir>/../_feature_cache) (run-specific, not in parameter files) |  |
| `no-feature-cache` | off |  | do not read or write the feature cache |  |
| `no-replica-overlap` | off |  | skip the tICA check of whether replicas visit the same states | betalactamase_incoming: 1, mdcath_incoming: 1, syt_c2a_incoming: 1, t4l_incoming: 1, tutorial: 1 |
| `dec-seeds` | 24 |  | (decomposition) number of seed residues | betalactamase_incoming: 32, mdcath_incoming: 32, planted_fast: 16, synthetic100: 32, synthetic36: 16, syt_c2a_incoming: 32, t4l_incoming: 32, tutorial: 16, workstation: 32 |
| `seed-fraction` | 0.0 |  | cap the number of seeds at this fraction of the residues (e.g. 0.25), so detection relies on growth, not on splitting overlaps; 0 = no cap (dec-seeds only) | workstation: 0.25 |
| `seed-group` | 1 |  | residues in each growth start: 1 = the seed alone (default); N = the seed plus its N-1 strongest predictive partners; 0 = just enough residues to fill unit-dim (ceil(unit-dim / features per residue)), so the first growth step compares sets with the same number of coordinates | workstation: 2 |
| `max-size` | 16 |  | (decomposition) largest candidate grown from one seed (residues) | betalactamase_incoming: 120, mdcath_incoming: 120, planted_fast: 14, synthetic100: 24, synthetic36: 14, syt_c2a_incoming: 120, t4l_incoming: 120 |
| `min-size` | 3 |  | (decomposition) smallest unit kept (residues) | workstation: 4 |
| `max-unit` | 40 |  | (decomposition) largest unit after merging (residues); also the top calibration size | betalactamase_incoming: 120, mdcath_incoming: 120, planted_fast: 14, synthetic100: 24, synthetic36: 14, syt_c2a_incoming: 120, t4l_incoming: 120, tutorial: 16 |
| `add-gate` | ratio | ratio, none | ratio: a residue new to a unit may join (merge, reassignment, round-2 joins, MC) only if it predicts the unit's members more than the rest of the protein; none = old behaviour |  |
| `n-controls` | 8 |  | matched control sets (random, contiguous, ball) scored per unit size |  |

### Verification

After discovery the units are scored again at other lags, to show which hold across time scales.

| setting | default | values | what it does | presets |
|---|---|---|---|---|
| `verify-lags` | 1,5,25 |  | comma list of lags (frames) at which the units are re-scored; 'none' skips this | betalactamase_incoming: none, mdcath_incoming: none, syt_c2a_incoming: none, t4l_incoming: none, tutorial: none |
| `verify-null` | 32 |  | time shifts used for the verification null |  |

### Global-mode recovery

At the end the units' own slow motions are combined to see how much of the protein's global slow motion they explain.

| setting | default | values | what it does | presets |
|---|---|---|---|---|
| `mode-recovery` | on | on, off | at the end, assemble the local unit dynamics into global modes (on or off) |  |
| `recovery-modes` | 10 |  | (global-mode recovery) number of reference global slow modes | planted_fast: 4, synthetic100: 4, synthetic36: 4 |
| `recovery-local-dim` | 0 |  | local slow coordinates per PDU; 0 uses unit-dim |  |
| `recovery-channel-ranks` | 1,2,4 |  | incoming-rank ablation budgets; off and full on are automatic |  |
| `recovery-combinations` | auto | auto, all, singles-pairs | auto: exhaustive up to 8 PDUs; otherwise singles, pairs, leave-one-out and all |  |
| `recovery-max-combinations` | 256 |  | (global-mode recovery) most unit combinations tried |  |
| `recovery-subsets` | unset |  | optional JSON list of combinations of zero-based final PDU IDs |  |

## Shipped presets

In `pdukit/params/` (installed with the package; `python -c "from importlib.resources import files; print(files('pdukit.params'))"` prints the folder).

- `betalactamase_incoming.ppf`: Beta-lactamases (DyNoPy_Data: KPC-2, PDC-3, SHV-1, TEM-1): current COG pipeline, 1 ns frames
- `mdcath_incoming.ppf`: mdCATH domains: current COG pipeline (incoming growth, merge, split, ports, mode recovery)
- `planted_fast.ppf`: planted fast test (6 units x 6 residues, 2 replicas x 3000 frames); reference at 7acf4cf: core 25, port 5, loose 6, 5 units
- `synthetic100.ppf`: Synthetic suite, 100 residues (pdukit.bench.designs): units, mixed, nocomm, medium, tau
- `synthetic36.ppf`: Synthetic suite, 36 residues (the first suite; pdukit.bench.designs small-*)
- `syt_c2a_incoming.ppf`: Synaptotagmin-1 C2A (100 ps frames, lag 5 = 500 ps): incoming-only growth
- `t4l_incoming.ppf`: T4 lysozyme (164 residues, 20 ps frames, lag 25 = 500 ps): incoming-only growth
- `tutorial.ppf`: Tutorial settings: small and fast (a few minutes on a laptop CPU) for the demo trajectories made by tools/make_demo_trajectories.py (40 residues, 1500 frames of 20 ps)
- `workstation.ppf`: pdu preset for a workstation: 32-64 GB of RAM and a consumer GPU (for example an RTX 3070 with 8 GB) or no GPU at all
- `final/`: the final runs of the paper (T4L WT and L99A, Syt1 C2A, mdCATH) at seven lags, with `submit_final.sh`
