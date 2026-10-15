"""Command-line interface of the COG pipeline.

`build_parser` declares every option; `parse_args` parses and normalises them
(so later stages never re-parse strings such as ``--verify-lags``).
"""

from __future__ import annotations

import argparse

from .ppf import PpfError, format_ppf, ppf_defaults


def _add_planted_args(group):
    """Planted benchmark system."""
    group.add_argument("--planted", action="store_true")
    group.add_argument("--n-units", type=int, default=5)
    group.add_argument("--unit-size", type=int, default=10)
    group.add_argument("--planted-m", type=int, default=4)
    group.add_argument("--latent-dim", type=int, default=5)
    group.add_argument("--frames-per-replica", type=int, default=10000)
    group.add_argument("--n-replicas", type=int, default=3)
    group.add_argument("--coupling-gain", type=float, default=0.35)
    group.add_argument("--face-residues", type=int, default=1)
    group.add_argument(
        "--decoy-pairs",
        default="3-4",
        help="comma list like '3-4' of uncoupled adjacent units",
    )
    group.add_argument("--discontinuous", action="store_true")
    group.add_argument("--bench-seed", type=int, default=0)
    group.add_argument(
        "--scatter",
        action="store_true",
        help="permute planted residue positions so units are not "
        "spatially contiguous (tests a dynamics-only search)",
    )


def _add_md_args(group):
    """MD trajectories and feature construction."""
    group.add_argument(
        "-s", "--xtc-topology", help="topology (tpr, pdb, gro, ...)"
    )
    group.add_argument(
        "-f", "--xtc", nargs="+", help="trajectory file(s), one per replica"
    )
    group.add_argument(
        "--frames",
        type=int,
        default=0,
        help="first N frames per replica (0 = all)",
    )
    group.add_argument("--stride", type=int, default=1)
    group.add_argument("--rse-cutoff", type=float, default=15.0)
    group.add_argument("--rse-components", type=int, default=8)
    group.add_argument("--min-seq-sep", type=int, default=2)
    group.add_argument("--contact-cutoff", type=float, default=8.0)
    group.add_argument(
        "--pair-features",
        action="store_true",
        help="add pair-owned contact-distance features (Stage 4)",
    )
    group.add_argument("--pair-cutoff", type=float, default=8.0)


def _add_estimator_args(group):
    """Channel estimator settings."""
    group.add_argument("--lag", type=int, default=1)
    group.add_argument("--unit-dim", type=int, default=8)
    group.add_argument("--env-dim", type=int, default=20)
    group.add_argument("--ridge", type=float, default=1e-2)
    group.add_argument("--n-null", type=int, default=64)
    group.add_argument(
        "--shift-mode",
        default="auto",
        choices=("auto", "global", "within", "cross"),
    )
    group.add_argument("--backend", default="flux", choices=("flux", "schur"))
    group.add_argument(
        "--combine", default="max", choices=("max", "sum", "in", "out")
    )
    group.add_argument("--device", default=None)
    group.add_argument(
        "--cond-unit",
        default="full",
        choices=("full", "reduced"),
        help="unit's present subtracted whole (COG 2.1) or as "
        "unit_dim slow modes (v1.0)",
    )
    group.add_argument(
        "--cond-env",
        default="full",
        help="environment's present subtracted whole ('full'), "
        "as N slow modes (an integer), or as env_dim modes "
        "('reduced', v1.0)",
    )


def _add_calibration_args(group):
    """Calibration of thresholds and baselines."""
    group.add_argument(
        "--sizes",
        default=None,
        help="calibration sizes, e.g. '2-40' (default 2..max-unit)",
    )
    group.add_argument("--target-rel-width", type=float, default=0.03)
    group.add_argument("--min-supports", type=int, default=40)
    group.add_argument("--max-supports", type=int, default=300)
    group.add_argument(
        "--batch-supports",
        type=int,
        default=32,
        help="calibration supports scored per batch",
    )
    group.add_argument(
        "--calibrate-only",
        action="store_true",
        help="stop after the calibration (to check thetas, the "
        "baseline and the time before a full run)",
    )


def _add_replica_args(group):
    """Replica split and per-replica checks."""
    group.add_argument(
        "--discovery-replicas",
        default=None,
        help="comma list of replica indices (0-based)",
    )
    group.add_argument(
        "--holdout-replicas",
        default=None,
        help="comma list, or 'none' to discover on everything",
    )
    group.add_argument(
        "--n-holdout",
        type=int,
        default=None,
        help="hold out the last N replicas (0 = discover on all); "
        "alternative to the two lists above",
    )
    group.add_argument(
        "--per-replica-check",
        action="store_true",
        help="score the units on every replica separately, with "
        "that replica's own calibration",
    )


def _add_preprocessing_args(group):
    """Centring, high-pass, rigid-body fit and frame trimming."""
    group.add_argument(
        "--no-center-replicas",
        action="store_true",
        help="keep each replica's mean (v1.0 behaviour); by "
        "default every replica is centred on its own mean",
    )
    group.add_argument(
        "--highpass-before-features",
        action="store_true",
        help="apply --highpass-window to the raw neighbour "
        "distances BEFORE each residue's slow directions are "
        "chosen (MD data)",
    )
    group.add_argument(
        "--fit",
        action="store_true",
        help="remove overall rotation and translation (Kabsch) "
        "before building features",
    )
    group.add_argument(
        "--calibrate-coarse",
        type=int,
        default=0,
        help="calibrate only ~N sizes spread over the range and "
        "interpolate the rest (0 = every size)",
    )
    group.add_argument(
        "--skip-frames",
        type=int,
        default=0,
        help="drop the first N frames of every replica",
    )
    group.add_argument(
        "--highpass-window",
        default="0",
        help="remove a running mean of this many frames within "
        "every replica (0 = off; 'auto' = the smallest window "
        "whose null threshold is within --null-cal-tol of the "
        "lowest in the null calibration scan)",
    )


def _add_null_calibration_args(group):
    """Null-calibration scan of the high-pass window."""
    group.add_argument(
        "--no-null-calibration",
        action="store_true",
        help="skip the highpass-window scan before discovery",
    )
    group.add_argument(
        "--null-cal-windows",
        default="auto",
        help="comma list of windows (frames) to scan; 'auto' = "
        "geometric from 10 x lag to a replica / 8",
    )
    group.add_argument("--null-cal-n-windows", type=int, default=6)
    group.add_argument("--null-cal-sizes", default="3,10")
    group.add_argument("--null-cal-supports", type=int, default=48)
    group.add_argument("--null-cal-n-null", type=int, default=32)
    group.add_argument(
        "--null-cal-target",
        type=float,
        default=0.06,
        help="largest acceptable null threshold (theta_in and "
        "theta_out); a warning is printed if no window "
        "reaches it",
    )


def _add_search_args(group):
    """Search graph, seeding, baseline and amount-of-data scans."""
    group.add_argument(
        "--seeding",
        default="random",
        choices=("random", "spread"),
        help="'random' = N residues drawn at random (dynamics "
        "only); 'spread' = farthest-point on the graph",
    )
    group.add_argument(
        "--search-graph",
        default="predictive",
        choices=("predictive", "contact", "union"),
        help="graph used everywhere a neighbourhood is needed "
        "(growth, round 2, and the random comparison "
        "regions of the baseline/null): 'predictive' = "
        "residues whose present predicts another's future "
        "beyond a shift null (dynamics only); 'contact' = "
        "spatial contacts (old behaviour); 'union' = both",
    )
    group.add_argument(
        "--baseline",
        default="graph",
        choices=("graph", "random", "contact"),
        help="comparison regions for the cost baseline: 'graph' "
        "= random connected sets on the search graph; "
        "'random' = uniformly random residue sets; "
        "'contact' = compact spatial balls (old)",
    )
    group.add_argument(
        "--graph-k",
        type=int,
        default=5,
        help="strongest predictive partners kept per residue",
    )
    group.add_argument("--graph-null-shifts", type=int, default=16)
    group.add_argument(
        "--rho-min",
        type=float,
        default=0.0,
        help="minimum channel strength: a channel counts only if "
        "its correlation exceeds max(noise threshold, this)"
        " (effect-size floor; 0 = off)",
    )
    group.add_argument(
        "--rho-scan",
        default="",
        help="comma list of data fractions (e.g. 0.1,0.25,0.5,1) "
        "at which the final units and random regions are "
        "re-scored: thresholds, channel strengths (rho) and "
        "counts vs the amount of data",
    )
    group.add_argument(
        "--rho-scan-us",
        action="store_true",
        help="--rho-scan values are amounts of data in us "
        "(built from equal blocks) instead of fractions",
    )
    group.add_argument(
        "--block-check",
        action="store_true",
        help="score the final units on every equal-length block "
        "(see --block-us) with that block's own "
        "calibration; replaces the per-replica check",
    )
    group.add_argument(
        "--block-us",
        type=float,
        default=1.0,
        help="block length in us (MD data)",
    )
    group.add_argument(
        "--block-frames",
        type=int,
        default=0,
        help="block length in frames (planted data; 0 = auto)",
    )
    group.add_argument(
        "--rho-scan-null",
        type=int,
        default=32,
        help="shifts per subset in the rho scan",
    )


def _add_face_args(group):
    """Cores and faces, and the residue ledger."""
    group.add_argument(
        "--faces",
        action="store_true",
        help="cores and faces: residues that carry a second "
        "unit's motion become faces (attached to both "
        "units); cores are scored with faces removed from "
        "both sides",
    )
    group.add_argument(
        "--face-delta",
        type=float,
        default=0.0,
        help="face candidate if cost(unit) - cost(unit without "
        "the residue) >= this (keeping it does not lower "
        "the unit's cost)",
    )
    group.add_argument(
        "--face-balance",
        type=float,
        default=0.8,
        help="coupling to the second unit / coupling to its own",
    )
    group.add_argument(
        "--no-ledger",
        dest="ledger",
        action="store_false",
        help="do not record the per-residue ledger",
    )
    group.add_argument(
        "--no-ledger-loo",
        dest="ledger_loo",
        action="store_false",
        help="ledger without leave-one-out scores (faster)",
    )


def _add_resolve_args(group):
    """Second round for unresolved residues."""
    group.add_argument(
        "--resolve",
        action="store_true",
        help="second round for residues in groups with cost > 0:"
        " grow new units inside that pool, then let pool "
        "residues join a unit only if its cost improves; "
        "the rest get a secondary (best-place) assignment",
    )
    group.add_argument(
        "--resolve-margin",
        type=float,
        default=0.1,
        help="improvement in cost (channels) needed to accept a "
        "new unit or a join",
    )
    group.add_argument("--resolve-seeds", type=int, default=12)


def _add_stability_args(group):
    """Monte Carlo stable minimum and the optimisation objective."""
    group.add_argument(
        "--stability",
        action="store_true",
        help="after round 2: steepest descent over one-residue "
        "changes for every unit, then n random kicks; "
        "reports whether each unit is a stable minimum "
        "(cog2_stability)",
    )
    group.add_argument(
        "--stab-cycles",
        type=int,
        default=-1,
        help="kick cycles per unit; -1 = until convergence",
    )
    group.add_argument(
        "--stab-patience",
        type=int,
        default=100,
        help="stop after this many cycles in a row that do not "
        "change the unit's minimum",
    )
    group.add_argument(
        "--stab-kmax",
        type=int,
        default=3,
        help="each kick changes 1..kmax residues",
    )
    group.add_argument(
        "--stab-margin",
        type=float,
        default=0.1,
        help="a change is accepted only if it lowers the cost by "
        "more than this (channels)",
    )
    group.add_argument(
        "--stab-null",
        type=int,
        default=20,
        help="random regions per size put through the same descent (0 = none)",
    )
    group.add_argument(
        "--stab-steps", type=int, default=200, help="maximum descent steps"
    )
    group.add_argument(
        "--stab-protect",
        default="direction",
        choices=["direction", "none"],
        help="direction: a member that predicts the other members"
        " (driver) or is predicted by them (listener) better "
        "than by the outside (one-sided Mann-Whitney) cannot "
        "be removed by J alone; such members whose removal "
        "would lower J are reported as faces.  none = 13 Oct"
        " behaviour (J decides removals)",
    )
    group.add_argument(
        "--stab-protect-alpha",
        type=float,
        default=0.05,
        help="significance level of the driver/listener tests",
    )
    group.add_argument(
        "--stab-objective",
        default="channels",
        choices=["cohesion", "channels"],
        help="channels (default, COG 2.1 stable): the MC minimises "
        "the channel cost J.  cohesion (14 Oct, experimental): "
        "the MC maximises "
        "the directed predictive-cohesion AUC (cog2_cohesion)."
        "  Either way the cohesion of the final units is "
        "reported as a check (unit_cohesion.csv)",
    )
    group.add_argument(
        "--grow-objective",
        default="channels",
        choices=["cohesion", "channels"],
        help="channels (default, COG 2.1 stable): growth on the "
        "channel cost J.  cohesion (experimental): discovery "
        "growth, merging, "
        "reassignment and round 2 use the cohesion cost "
        "-cohesion (cog2_cohesion); finds only the coarsest "
        "split on T4L / Syt (runs 7405961 / 7405962)",
    )


def _add_cohesion_args(group):
    """Predictive-cohesion scorer and its noise margin."""
    group.add_argument(
        "--cohesion-margin",
        default="auto",
        help="'auto': noise margin from block-bootstrap "
        "re-estimates of rho (size dependent); or a number",
    )
    group.add_argument(
        "--cohesion-boot",
        type=int,
        default=20,
        help="bootstrap resamples for the cohesion noise margin",
    )
    group.add_argument(
        "--cohesion-blocks",
        type=int,
        default=20,
        help="at least this many blocks (whole replicas if there"
        " are enough, else replicas split in contiguous "
        "blocks)",
    )
    group.add_argument(
        "--channel-shadow",
        default="auto",
        choices=["auto", "on", "off"],
        help="report only: also grow from the same seeds on the "
        "channel cost J and classify every channel-closed "
        "group the cohesion search did not find ('auto' = "
        "on with --grow-objective cohesion)",
    )
    group.add_argument(
        "--cohesion-baseline-n",
        type=int,
        default=200,
        help="random regions per size for the cohesion q95 baseline",
    )


def _add_pdudef_args(group):
    """Linear-estimator PDU definition (off by default)."""
    group.add_argument(
        "--pdudef",
        action="store_true",
        help="score units with Varun's PDU definition, linear "
        "estimators: Q = r / d_pred (cog2_pdudef)",
    )
    group.add_argument(
        "--pdudef-random",
        type=int,
        default=20,
        help="random sets per size for the Q null",
    )
    group.add_argument(
        "--pdudef-shifts",
        type=int,
        default=32,
        help="time shifts used to set eta",
    )
    group.add_argument(
        "--pdudef-frames",
        type=int,
        default=2000,
        help="frames subsampled for the box-counting estimate",
    )


def _add_run_control_args(group):
    """Add unit source, window tolerance, batching, cache and logging."""
    group.add_argument(
        "--units-from",
        default=None,
        help="results.json of an earlier run: test the stability "
        "of ITS units (units_resolved, else units_discovery)"
        " instead of this run's",
    )
    group.add_argument(
        "--null-cal-tol",
        type=float,
        default=0.15,
        help="'auto' window = the smallest window whose null "
        "threshold is within this fraction of the lowest "
        "one in the scan",
    )
    group.add_argument(
        "--null-cal-max-replicas",
        type=int,
        default=20,
        help="scan on at most this many replicas (evenly spaced)",
    )
    group.add_argument(
        "--frame-ps",
        type=float,
        default=0.0,
        help="time per frame in ps, for the logs only",
    )
    group.add_argument(
        "--batched",
        default="auto",
        choices=("auto", "on", "off"),
        help="score candidates in batches on the GPU engine "
        "(auto: when the workspace is on CUDA)",
    )
    group.add_argument(
        "--env-basis",
        default="exact",
        choices=("exact", "ritz"),
        help="'ritz' = environment slow basis from the global "
        "slow modes + block inverse (faster, approximate)",
    )
    group.add_argument(
        "--feature-cache",
        default=None,
        help="feature cache directory (default: <outdir>/../_feature_cache)",
    )
    group.add_argument("--no-feature-cache", action="store_true")
    group.add_argument(
        "--no-replica-overlap",
        action="store_true",
        help="skip the tICA check of whether replicas visit the same states",
    )


def _add_decomposition_args(group):
    """Seeds, sizes, merging and assignment."""
    group.add_argument("--dec-seeds", type=int, default=24)
    group.add_argument("--dec-iters", type=int, default=6)
    group.add_argument("--max-size", type=int, default=16)
    group.add_argument("--min-size", type=int, default=3)
    group.add_argument("--max-unit", type=int, default=40)
    group.add_argument("--boundary-tol", type=float, default=0.15)
    group.add_argument("--merge-jaccard", type=float, default=0.5)
    group.add_argument(
        "--merge-mode",
        default="reearn",
        choices=("reearn", "union"),
        help="reearn (13 Oct 2026): overlapping candidates merge "
        "pairwise; the shared residues are the core and every"
        " non-shared residue must pass the addition gate and "
        "lower the cost by more than --stab-margin; union = "
        "old behaviour (non-shared residues join untested)",
    )
    group.add_argument(
        "--add-gate",
        default="ratio",
        choices=("ratio", "none"),
        help="ratio: a residue new to a unit may join (merge, "
        "reassignment, round-2 joins, MC) only if it predicts"
        " the unit's members more than the rest of the "
        "protein; none = old behaviour",
    )
    group.add_argument("--size-penalty", type=float, default=0.02)
    group.add_argument("--n-controls", type=int, default=8)
    group.add_argument(
        "--no-cohesion-split",
        action="store_true",
        help="do not split units whose halves are independent",
    )
    group.add_argument(
        "--no-keep-best",
        action="store_true",
        help="return the last assignment iteration, as COG v1.0 "
        "did, instead of the one with the lowest objective",
    )


def _add_verification_args(group):
    """Verification: lags and the COG v1.0 comparison."""
    group.add_argument("--verify-lags", default="1,5,25")
    group.add_argument("--verify-null", type=int, default=32)
    group.add_argument("--compare-cog1", action="store_true")
    group.add_argument(
        "--cog1-null",
        type=int,
        default=20,
        help="shifts for the COG v1.0 comparison workspace",
    )


def build_parser(description):
    """Return the argument parser with all option groups."""
    ap = argparse.ArgumentParser(prog="pdu run", description=description)
    ap._optionals.title = "files and general options"
    _add_planted_args(ap.add_argument_group("planted benchmark"))
    _add_md_args(ap.add_argument_group("MD data"))
    _add_estimator_args(ap.add_argument_group("estimator"))
    _add_calibration_args(ap.add_argument_group("calibration"))
    discovery = ap.add_argument_group("discovery")
    _add_replica_args(discovery)
    _add_preprocessing_args(discovery)
    _add_null_calibration_args(discovery)
    _add_search_args(discovery)
    _add_face_args(discovery)
    _add_resolve_args(discovery)
    _add_stability_args(discovery)
    _add_cohesion_args(discovery)
    _add_pdudef_args(discovery)
    _add_run_control_args(discovery)
    _add_decomposition_args(discovery)
    _add_verification_args(ap.add_argument_group("verification"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-o", "--outdir", help="output directory")
    ap.add_argument(
        "-p",
        "--params",
        help="parameter file (.ppf, key = value); command-line options win",
    )
    return ap


def parse_args(argv=None, description=None, require_outdir=True):
    """Parse and normalise the command line (and the ``-p`` parameter file).

    Precedence is command line, then parameter file, then built-in default.
    Beyond argparse's own checks this validates option combinations, turns
    ``--verify-lags`` into a list of ints, maps ``--cond-env reduced`` to
    ``None`` and ``--highpass-window`` to an int or ``"auto"``.  The resolved
    settings are kept as text in ``args._ppf`` (written to ``pdu_out.ppf``).
    """
    ap = build_parser(description)
    first, _ = ap.parse_known_args(argv)
    if first.params:
        try:
            ap.set_defaults(**ppf_defaults(ap, first.params))
        except (OSError, PpfError) as exc:
            ap.error(str(exc))
    args = ap.parse_args(argv)
    if require_outdir and not args.outdir:
        ap.error("give an output directory (-o / --outdir)")
    if not args.planted and not (args.xtc_topology and args.xtc):
        ap.error("give --planted or -s/--xtc-topology and -f/--xtc")
    if args.pair_features and args.planted:
        ap.error("--pair-features needs coordinates (MD data)")
    if args.pair_features and args.backend == "schur":
        ap.error("--pair-features cannot be used with --backend schur")
    if args.n_holdout is not None and (
        args.discovery_replicas is not None
        or args.holdout_replicas is not None
    ):
        ap.error(
            "--n-holdout cannot be combined with --discovery-replicas "
            "or --holdout-replicas"
        )
    args._ppf = format_ppf(
        ap, args, header="; resolved parameters of this run (pdu run -p)"
    )
    no_lags = str(args.verify_lags).strip().lower() in ("", "none", "0", "off")
    args.verify_lags = (
        []
        if no_lags
        else [int(x) for x in str(args.verify_lags).split(",") if x]
    )
    if args.cond_env == "reduced":
        args.cond_env = None
    if str(args.highpass_window).lower() != "auto":
        args.highpass_window = int(args.highpass_window)
    else:
        args.highpass_window = "auto"
    return args
