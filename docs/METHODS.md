# COG 2.1: what it computes, stage by stage

Residue indices in all outputs are 0-based. Lags and windows are in frames.

## 1. Goal and the two scores

COG splits a protein into residue groups (PDUs) whose dynamics are coupled to each other more
than to the rest of the protein, at a chosen lag τ and with chosen per-residue features.
Dynamics only: no spatial contacts in the search or the baselines.

**Closure (optimised).** For a unit S, ν = Σ_k max(0, 1 − θ²/ρ_k²) over the canonical
correlations ρ_k between the unit's 8 slowest modes and the environment's 20 modes, incoming and
outgoing. θ is the 95th percentile of the same statistic under a time-shift null.
`cost = ν − baseline(|S|)`, where the baseline is the median ν of random regions of the same
size grown on the predictive graph. Status: `closed` (cost ≤ 0 and p_size ≤ 0.1), `weak`
(cost ≤ 0), `unresolved` (cost > 0).

**Predictive cohesion (reported).** ρ[i, j] is the leading partial canonical correlation of
residue j's present with residue i's future, given residue i's own present: row = target,
column = source. For S: inside links ρ[a, b]; driver boundary ρ[outsider, member]; listener
boundary ρ[member, outsider]. `AUC_driver = P(inside > driver boundary)`, `AUC_listener =
P(inside > listener boundary)`; cohesion is their mean (random groups ≈ 0.5).

**Per-residue directions.** driver test: ρ[members, r] versus ρ[outsiders, r]; listener test:
ρ[r, members] versus ρ[r, outsiders]; one-sided Mann–Whitney at α = 0.05. Labels: both /
driver / listener / none. A labelled member is protected from removal by the descent; if its
removal would still lower the cost by more than the margin it is reported as a face.

**Final category:** PDU (cohesive and closed) / cohesive, wide interface / closed, not cohesive
/ not a unit.

## 2. Pipeline

`cog2_run.main` parses the options (`cog2_cli`) and calls `cog2_pipeline.Cog2Run.run`,
which runs these stages in order (one method each) and rewrites `results.json` after each.

| # | stage | code |
|---|---|---|
| 1 | load features and segments, centre each replica, high-pass | `Cog2Run.load_features`, `preprocess` (`cog2_data`) |
| 2 | split (pooled by default) | `Cog2Run.split_replicas` |
| 3 | workspace: lagged covariances and shift-null pairs; channel calibration | `interface.FluxWorkspace`, `cog2.calibrate_cog2` |
| 3a | predictive coupling ρ and graph (k strongest partners) | `cog2_graph` |
| 3b | addition gate (ratio > 1) and direction protection | `cog2_gate` |
| 3c | cohesion scorer, block-bootstrap noise margin | `cog2_cohesion` |
| 4 | discovery: seeds, greedy growth on cost, knee, re-earn merge, reassignment | `interface.decompose` |
| 5 | cohesion split and status | `cog2.split_incoherent`, `cog2_scoring.unit_status` |
| 5b | round 2 for residues in unresolved groups | `cog2_resolve` |
| 5c | MC stable minimum, faces, directions, write-back, secondary labels | `cog2_stability`, `cog2_faces` |
| 5d | PDU definition with linear estimators (off by default) | `cog2_pdudef` |
| 6 | verification: blocks, replicas, rho scan, held-out, lags | `cog2_checks` |

The ledger of per-residue labels is `cog2_ledger`; the CSV tables are written by `cog2_outputs`.

## 3. Reference settings

| | T4 lysozyme | Synaptotagmin-1 C2A |
|---|---|---|
| frame | 20 ps | 100 ps |
| lag τ | 25 frames (500 ps) | 5 frames (500 ps) |
| high-pass window | 500 frames (10 ns) | 100 frames (10 ns) |
| params file | `pdukit/hpc/slurm/params/t4l_channels.ppf` | `.../syt_c2a_channels.ppf` |

Common: unit_dim 8, env_dim 20, 128 null shifts, predictive graph k = 5, 32 random seeds,
max unit 120, sizes 2–120, size penalty 0, minimum size 3, MC until convergence, margin 0.1.

## 4. Outputs

`results.json`, `calibration.json`, `units.csv`, `units_resolved.csv`, `units_stable.csv`,
`unit_directions.csv`, `unit_cohesion.csv`, `secondary_assignments.csv`, `stability_trace.csv`,
`residue_ledger.csv`, `unit_rho.csv`, `replicas.txt`, `pdu_out.ppf`.

## 5. Tests

`pytest` runs the unit tests. The planted-system test
(`python -m pdukit.lib.pdu.cog2_fasttest -h`) builds six units of six residues from two
simulated replicas and reports how many are recovered (best Jaccard ≥ 0.67).

## 6. Known limitations

- Channel counts saturate with long simulations: with a very small noise threshold random
  regions already use most of the 8 channels. Declaring an effect-size threshold is planned.
- Replicas that do not converge give results that describe the sampled ensemble only.
- Coupling inside large lobes is weak and nearly uniform; small cores are the detectable units.
- Fair-test p-values have a floor of 1/(n+1) for n null regions; use more nulls (`--stab-null`)
  for smaller p.
- Experimental options kept for reproducibility, not part of the stable configuration:
  cohesion as optimisation objective, `--pdudef`, per-block, per-replica and held-out checks.
