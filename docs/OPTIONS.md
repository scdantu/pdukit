# PDU settings reference

Generated from the settings table by `python -m pdukit.config.reference --write`; do not edit by hand.

Every setting is a key in a parameter file (`lag = 10`; see `pdukit/params/default.ppf` for a file with every setting at its default).  To change one for a single run, use `--set` (`pdu run -p my.ppf --set lag=10`).  Precedence: built-in default (below), then the parameter file (`-p`), then `--set` and the other command-line arguments.  Flags are written `1` or `0`.  The command-line arguments of `pdu run` are only `-f`, `-s`, `-p`, `-o`, `--set`, `--log-level` and `--check`.

The last column lists the shipped presets that change the option from its default, with their value.

## Files and general options

| setting | default | values | what it does | presets |
|---|---|---|---|---|
| `seed` | 0 |  | random seed | planted_fast: 1, synthetic100: 1, synthetic36: 1 |
| `log-level` | info | quiet, info, debug | progress log detail: quiet = warnings only, info = one line per step, debug = also per-growth-step traces |  |
| `nt` | -1 |  | CPU threads for numpy/scipy and torch (-1 = all cores this process may use) (run-specific, not in parameter files) |  |
| `outdir` | unset |  | output directory (run-specific, not in parameter files) |  |
| `params` | unset |  | parameter file (.ppf, key = value); command-line options win (run-specific, not in parameter files) |  |

## Planted benchmark

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

## MD data

| setting | default | values | what it does | presets |
|---|---|---|---|---|
| `xtc-topology` | unset |  | topology (tpr, pdb, gro, ...) (run-specific, not in parameter files) |  |
| `xtc` | unset | one or more | trajectory file(s), one per replica (run-specific, not in parameter files) |  |
| `frames` | 0 |  | first N frames per replica (0 = all) |  |
| `stride` | 1 |  | use every Nth frame of each replica |  |
| `rse-cutoff` | 15.0 |  | radius in angstrom of a residue's structural environment: neighbours inside it enter its descriptor |  |
| `rse-components` | -1 |  | slow tICA components kept per residue after compressing its neighbour distances: a number >= 1, or -1 = choose it from rse-coverage (two passes over the distances) | betalactamase_incoming: 8, mdcath_incoming: 8, planted_fast: 8, synthetic100: 8, synthetic36: 8, syt_c2a_incoming: 8, t4l_incoming: 8, tutorial: 8 |
| `rse-coverage` | 0.9 |  | with rse-components = -1: smallest number of tICA modes at which the median residue keeps this share of its kinetic variance (sum of lambda^2 over its whole spectrum); always reported in results tica_spectrum.coverage |  |
| `rse-max-components` | 0 |  | with rse-components = -1: upper limit on the chosen number of tICA modes (0 = none); memory is frames x residues x modes x 4 bytes per feature set |  |
| `min-seq-sep` | 2 |  | neighbours closer than this in sequence are left out of the descriptor (trivially rigid) |  |
| `contact-cutoff` | 8.0 |  | angstrom cutoff of the residue contact graph (growth only follows contacts) |  |
| `pair-features` | off |  | add pair-owned contact-distance features (Stage 4) |  |
| `pair-cutoff` | 8.0 |  | (pair features) angstrom cutoff for pair contacts |  |

## Estimator

| setting | default | values | what it does | presets |
|---|---|---|---|---|
| `lag` | 1 |  | lag in frames between the two time points whose coupling is measured (one frame = frame-ps) | syt_c2a_incoming: 5, t4l_incoming: 25, tutorial: 10 |
| `unit-dim` | 8 |  | slow coordinates kept to describe each unit |  |
| `env-dim` | 20 |  | slow coordinates kept to describe the environment (the rest of the protein) |  |
| `ridge` | 0.01 |  | regularisation added to the covariance matrices before they are inverted |  |
| `n-null` | 64 |  | time shifts used to build the null distribution of the canonical correlations | betalactamase_incoming: 128, mdcath_incoming: 128, syt_c2a_incoming: 128, t4l_incoming: 128, tutorial: 32 |
| `shift-mode` | auto | auto, global, within, cross | how the null displaces the environment in time: global (pooled), within (each replica), cross (a different replica); auto = cross with 2+ replicas, else within |  |
| `backend` | flux | flux, schur | estimator backend: flux (FluxWorkspace) or schur (Schur- complement form) |  |
| `combine` | max | max, sum, in, out | how incoming and outgoing channels form the cost: max, sum, in (incoming only) or out (outgoing only) |  |
| `device` | auto |  | where the estimator runs: auto (CUDA GPU when torch can use one, else CPU), cuda, cpu or mps (run-specific, not in parameter files) |  |
| `cond-unit` | full | full, reduced | unit's present subtracted whole (COG 2.1) or as unit_dim slow modes (v1.0) |  |
| `cond-env` | full |  | environment's present subtracted whole ('full'), as N slow modes (an integer), or as env_dim modes ('reduced', v1.0) |  |

## Calibration

| setting | default | values | what it does | presets |
|---|---|---|---|---|
| `sizes` | unset |  | calibration sizes, e.g. '2-40' (default 2..max-unit) | betalactamase_incoming: 2-120, mdcath_incoming: 2-120, planted_fast: 2-14, synthetic100: 2-24, synthetic36: 2-14, syt_c2a_incoming: 2-120, t4l_incoming: 2-120, tutorial: 2-20 |
| `target-rel-width` | 0.03 |  | (calibration) stop adding supports once the bootstrap interval of a threshold is this fraction of its value |  |
| `min-supports` | 40 |  | (calibration) fewest random supports scored per size | tutorial: 20 |
| `max-supports` | 300 |  | (calibration) most random supports scored per size | tutorial: 80 |
| `batch-supports` | 32 |  | calibration supports scored per batch |  |
| `calibrate-only` | off |  | stop after the calibration (to check thetas, the baseline and the time before a full run) |  |

## Discovery

| setting | default | values | what it does | presets |
|---|---|---|---|---|
| `discovery-replicas` | unset |  | comma list of replica indices (0-based) |  |
| `holdout-replicas` | unset |  | comma list, or 'none' to discover on everything |  |
| `n-holdout` | unset |  | hold out the last N replicas (0 = discover on all); alternative to the two lists above | betalactamase_incoming: 0, mdcath_incoming: 0, syt_c2a_incoming: 0, t4l_incoming: 0, tutorial: 0 |
| `no-center-replicas` | off |  | keep each replica's mean (v1.0 behaviour); by default every replica is centred on its own mean |  |
| `highpass-before-features` | off |  | apply --highpass-window to the raw neighbour distances BEFORE each residue's slow directions are chosen (MD data) | betalactamase_incoming: 1, mdcath_incoming: 1, syt_c2a_incoming: 1, t4l_incoming: 1, tutorial: 1 |
| `fit` | off |  | remove overall rotation and translation (Kabsch) before building features | betalactamase_incoming: 1, mdcath_incoming: 1, syt_c2a_incoming: 1, t4l_incoming: 1, tutorial: 1 |
| `calibrate-coarse` | 0 |  | calibrate only ~N sizes spread over the range and interpolate the rest (0 = every size) | betalactamase_incoming: 8, mdcath_incoming: 8, synthetic100: 6, syt_c2a_incoming: 8, t4l_incoming: 8, tutorial: 4 |
| `skip-frames` | 0 |  | drop the first N frames of every replica |  |
| `highpass-window` | 0 |  | remove a running mean of this many frames within every replica (0 = off; 'auto' = the smallest window whose null threshold is within --null-cal-tol of the lowest in the null calibration scan) | betalactamase_incoming: 50, mdcath_incoming: 50, syt_c2a_incoming: 100, t4l_incoming: 500, tutorial: 300 |
| `no-null-calibration` | off |  | skip the highpass-window scan before discovery | betalactamase_incoming: 1, mdcath_incoming: 1, syt_c2a_incoming: 1, t4l_incoming: 1, tutorial: 1 |
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
| `frame-ps` | 0.0 |  | time per frame in ps, for the logs only | betalactamase_incoming: 1000.0, mdcath_incoming: 1000.0, syt_c2a_incoming: 100.0, t4l_incoming: 20.0, tutorial: 20 |
| `batched` | auto | auto, on, off | score candidates in batches on the GPU engine (auto: when the workspace is on CUDA) |  |
| `env-basis` | exact | exact, ritz | 'ritz' = environment slow basis from the global slow modes + block inverse (faster, approximate) |  |
| `feature-cache` | unset |  | feature cache directory (default: <outdir>/../_feature_cache) (run-specific, not in parameter files) |  |
| `no-feature-cache` | off |  | do not read or write the feature cache |  |
| `no-replica-overlap` | off |  | skip the tICA check of whether replicas visit the same states | betalactamase_incoming: 1, mdcath_incoming: 1, syt_c2a_incoming: 1, t4l_incoming: 1, tutorial: 1 |
| `dec-seeds` | 24 |  | (decomposition) number of seed residues | betalactamase_incoming: 32, mdcath_incoming: 32, planted_fast: 16, synthetic100: 32, synthetic36: 16, syt_c2a_incoming: 32, t4l_incoming: 32, tutorial: 16 |
| `seed-fraction` | 0.0 |  | cap the number of seeds at this fraction of the residues (e.g. 0.25), so detection relies on growth, not on splitting overlaps; 0 = no cap (dec-seeds only) |  |
| `seed-group` | 1 |  | residues in each growth start: 1 = the seed alone (default); N = the seed plus its N-1 strongest predictive partners; 0 = just enough residues to fill unit-dim (ceil(unit-dim / features per residue)), so the first growth step compares sets with the same number of coordinates |  |
| `max-size` | 16 |  | (decomposition) largest candidate grown from one seed (residues) | betalactamase_incoming: 120, mdcath_incoming: 120, planted_fast: 14, synthetic100: 24, synthetic36: 14, syt_c2a_incoming: 120, t4l_incoming: 120 |
| `min-size` | 3 |  | (decomposition) smallest unit kept (residues) |  |
| `max-unit` | 40 |  | (decomposition) largest unit after merging (residues); also the top calibration size | betalactamase_incoming: 120, mdcath_incoming: 120, planted_fast: 14, synthetic100: 24, synthetic36: 14, syt_c2a_incoming: 120, t4l_incoming: 120, tutorial: 16 |
| `add-gate` | ratio | ratio, none | ratio: a residue new to a unit may join (merge, reassignment, round-2 joins, MC) only if it predicts the unit's members more than the rest of the protein; none = old behaviour |  |
| `n-controls` | 8 |  | matched control sets (random, contiguous, ball) scored per unit size |  |

## Verification

| setting | default | values | what it does | presets |
|---|---|---|---|---|
| `verify-lags` | 1,5,25 |  | comma list of lags (frames) at which the units are re-scored; 'none' skips this | betalactamase_incoming: none, mdcath_incoming: none, syt_c2a_incoming: none, t4l_incoming: none, tutorial: none |
| `verify-null` | 32 |  | time shifts used for the verification null |  |

## Global-mode recovery

| setting | default | values | what it does | presets |
|---|---|---|---|---|
| `mode-recovery` | on | on, off | at the end, assemble the local unit dynamics into global modes (on or off) |  |
| `recovery-modes` | 10 |  | (global-mode recovery) number of reference global slow modes | planted_fast: 4, synthetic100: 4, synthetic36: 4 |
| `recovery-local-dim` | 0 |  | local slow coordinates per PDU; 0 uses unit-dim |  |
| `recovery-channel-ranks` | 1,2,4 |  | incoming-rank ablation budgets; off and full on are automatic |  |
| `recovery-combinations` | auto | auto, all, singles-pairs | auto: exhaustive up to 8 PDUs; otherwise singles, pairs, leave-one-out and all |  |
| `recovery-max-combinations` | 256 |  | (global-mode recovery) most unit combinations tried |  |
| `recovery-subsets` | unset |  | optional JSON list of combinations of zero-based final PDU IDs |  |
