#!/usr/bin/env python3
"""Real-data test of the C6 size-conditioned null calibration.

    python -m pdukit.lib.pdu.null_calibration_run \
        --xtc-topology ATM-110_129-GA.tpr \
        --xtc CONV-PROD-R0001-ATM-110_129-GA.xtc \
        --frames 10000 --outdir OUTDIR

On one trajectory this:

  1. loads the first --frames frames (stride 1) and builds RSE features, the
     contact graph and a FluxWorkspace exactly as `flux_pilot.run_domain`
     does (timed);
  2. CORRECTNESS CHECK -- for every size, a fixed set of --n-check-supports
     supports x every shift, compared value by value across
     `FluxWorkspace` (`_project_and_residualise` + `_leading_null_rho`, the
     code `interface.calibrate_null_thresholds` is built from), the CPU
     module and the GPU module.  Independent of the stopping rule;
  3. runs the full calibration three ways -- `null_calibration_gpu`,
     `null_calibration_cpu`, and `interface.calibrate_null_thresholds` (the
     reference) -- with the same seed and settings, recording theta_k,
     diagnostics, total and per-size wall-clock time, and time per support.
     Each size stops once a support-level bootstrap CI is narrower than
     --target-rel-width after at least --min-supports supports, or at
     --max-pool-size; the naive value-level CI width is reported beside it;
  4. ORIGINAL VS REDESIGN -- every calibration support is scored against all
     shifts, so its own 95th percentile theta_S is exactly the threshold the
     original per-candidate null would have given it.  Per size this reports
     how widely theta_S spreads around the pooled theta_k, how wide that
     spread would be if size alone determined the null (theta_S then being
     just the 95th percentile of n_shift draws from the pooled distribution),
     and how strongly theta_S tracks contact density and effective dimension.

Writes results.json, theta_table.csv, per_support.csv and
size_vs_structure.csv to --outdir, rewriting results.json after every stage
so a job that runs out of time still leaves what it finished.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from .interface import (
    FluxWorkspace,
    _grow_ball_support,
    calibrate_null_thresholds,
    contact_graph,
    median_distance_matrix,
    rse_features,
)
from .null_calibration_cpu import (
    _available_cores,
    _CpuEvaluator,
    calibrate_null_thresholds_cpu,
)

IMPLS = ("interface", "cpu", "gpu")


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

_T0 = time.perf_counter()


def log(msg):
    """Print a time-stamped progress message."""
    print(f"[{time.perf_counter() - _T0:8.1f}s] {msg}", flush=True)


def _parse_sizes(text):
    out = []
    for part in str(text).split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return sorted(set(out))


def _jsonable(v):
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, np.ndarray):
        return _jsonable(v.tolist())
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def _environment(repo_root):
    env = {
        "host": socket.gethostname(),
        "python": platform.python_version(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "cores_available": _available_cores(),
        "numpy": np.__version__,
    }
    for mod in ("scipy", "MDAnalysis", "torch", "threadpoolctl"):
        try:
            env[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            env[mod] = None
    try:
        import torch

        env["cuda_available"] = bool(torch.cuda.is_available())
        env["cuda_device"] = (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        )
        env["torch_cuda"] = torch.version.cuda
    except Exception:  # noqa: BLE001
        env["cuda_available"] = False
    for key, cmd in (
        ("git_commit", ["rev-parse", "HEAD"]),
        ("git_branch", ["rev-parse", "--abbrev-ref", "HEAD"]),
    ):
        try:
            env[key] = subprocess.run(
                ["git", "-C", str(repo_root), *cmd],
                capture_output=True,
                text=True,
                timeout=10,
                env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
            ).stdout.strip()
        except Exception:  # noqa: BLE001
            env[key] = None
    return env


def _sample_exact(N, k, adjacency, rng, tries=40):
    """Sample a best-effort exact-size ball, as the calibrations do."""
    s = _grow_ball_support(N, k, adjacency, rng)
    for _ in range(tries - 1):
        if len(s) == k:
            break
        s = _grow_ball_support(N, k, adjacency, rng)
    return s


def _sync():
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:  # noqa: BLE001
        pass


def _spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 4 or np.ptp(x[ok]) == 0 or np.ptp(y[ok]) == 0:
        return float("nan"), float("nan")
    from scipy.stats import spearmanr

    r, p = spearmanr(x[ok], y[ok])
    return float(r), float(p)


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------


def build_inputs(a, timings):
    """Build the workspace, graph and sizes of a calibration run."""
    from .flux_pilot import load_xtc_segments

    t = time.perf_counter()
    coords, segments, meta = load_xtc_segments(
        a.xtc_topology,
        a.xtc,
        stride=a.stride,
        max_frames=a.frames or None,
        verbose=True,
    )
    timings["load_seconds"] = time.perf_counter() - t
    log(
        f"loaded {coords.shape[0]} frames x {coords.shape[1]} residues "
        f"in {timings['load_seconds']:.1f}s"
    )

    t = time.perf_counter()
    dmed = median_distance_matrix(coords)
    feats, neighbours, explained = rse_features(
        coords,
        cutoff=a.rse_cutoff,
        n_components=a.rse_components,
        min_seq_sep=a.min_seq_sep,
        dmed=dmed,
        lag=a.lag,
        segments=segments,
        koopman=True,
        ridge=a.ridge,
    )
    adj = contact_graph(
        coords, cutoff=a.contact_cutoff, dmed=dmed, units="auto"
    )
    timings["features_seconds"] = time.perf_counter() - t
    log(
        f"RSE features {feats.shape} and contact graph "
        f"(mean contacts/residue {adj.sum(axis=1).mean():.1f}) "
        f"in {timings['features_seconds']:.1f}s"
    )

    t = time.perf_counter()
    ws = FluxWorkspace(
        feats,
        lag=a.lag,
        segments=segments,
        n_null=a.n_null,
        seed=a.seed,
        ridge=a.ridge,
        device=a.device,
        reversible=True,
        truncation="koopman",
    )
    timings["workspace_seconds"] = time.perf_counter() - t
    log(
        f"FluxWorkspace D={ws.D} (N={ws.N} x m={ws.m}), {ws.n} lagged pairs, "
        f"{len(ws.Cs00)} shifts, backend={ws.backend} device={ws.device}, "
        f"built in {timings['workspace_seconds']:.1f}s"
    )
    if a.device not in (None, "cpu") and ws.backend != "torch":
        log("!! GPU requested but the workspace fell back to numpy")
    data = {
        "n_frames": int(coords.shape[0]),
        "n_residues": int(ws.N),
        "n_feat": int(ws.m),
        "D": int(ws.D),
        "n_lagged_pairs": int(ws.n),
        "n_shifts": int(len(ws.Cs00)),
        "segments": segments,
        "replicas": meta,
        "workspace_backend": ws.backend,
        "workspace_device": ws.device,
        "mean_contacts_per_residue": float(adj.sum(axis=1).mean()),
    }
    return ws, np.asarray(adj, dtype=bool), data


def correctness_check(a, ws, adj, sizes, gpu_state, timings):
    """Check that the fast calibrations agree with the reference."""
    from .null_calibration_gpu import leading_null_rho_batch_gpu

    rng = np.random.default_rng(a.seed + 12345)
    N = ws.N
    dims = dict(unit_dim=a.unit_dim, env_dim=a.env_dim)
    supports = {
        k: [_sample_exact(N, k, adj, rng) for _ in range(a.n_check_supports)]
        for k in sizes
    }

    # GPU warm-up: CUDA context / kernel start-up kept out of every timing.
    t = time.perf_counter()
    leading_null_rho_batch_gpu(
        None,
        None,
        None,
        None,
        None,
        supports[sizes[0]][:2],
        n_residues=N,
        ridge=a.ridge,
        _state=gpu_state,
        **dims,
    )
    _sync()
    timings["gpu_warmup_seconds"] = time.perf_counter() - t
    log(f"GPU warm-up {timings['gpu_warmup_seconds']:.2f}s")

    rows, worst = [], {"cpu_vs_ref": 0.0, "gpu_vs_ref": 0.0, "cpu_vs_gpu": 0.0}
    n_workers = min(a.n_check_supports, a.cpu_workers)
    with _CpuEvaluator(
        ws.C00,
        ws.C01,
        ws.C11,
        ws.Cs00,
        ws.Cs01,
        N=N,
        ridge=a.ridge,
        n_workers=n_workers,
        **dims,
    ) as ev:
        for k in sizes:
            sup = supports[k]
            t = time.perf_counter()
            ref = []
            for s in sup:
                (_, S, E, Vu, Ve, Ih, Ryy, Chy, Czz, _, _) = (
                    ws._project_and_residualise(s, a.unit_dim, a.env_dim)
                )
                ref.append(
                    [
                        ws._leading_null_rho(
                            Ih, Ryy, Chy, Vu, Ve, Czz, S, E, c0, c1
                        )
                        for c0, c1 in zip(ws.Cs00, ws.Cs01)
                    ]
                )
            ref = np.asarray(ref)
            t_ref = time.perf_counter() - t
            t = time.perf_counter()
            cpu = ev.rows(sup)
            t_cpu = time.perf_counter() - t
            t = time.perf_counter()
            gpu = leading_null_rho_batch_gpu(
                None,
                None,
                None,
                None,
                None,
                sup,
                n_residues=N,
                ridge=a.ridge,
                _state=gpu_state,
                **dims,
            )
            _sync()
            t_gpu = time.perf_counter() - t
            d = {
                "cpu_vs_ref": float(np.max(np.abs(cpu - ref))),
                "gpu_vs_ref": float(np.max(np.abs(gpu - ref))),
                "cpu_vs_gpu": float(np.max(np.abs(cpu - gpu))),
            }
            for key in worst:
                worst[key] = max(worst[key], d[key])
            rows.append(
                {
                    "size": k,
                    "n_supports": len(sup),
                    "n_values": int(ref.size),
                    **{f"max_abs_{key}": v for key, v in d.items()},
                    "seconds_interface": t_ref,
                    "seconds_cpu": t_cpu,
                    "seconds_gpu": t_gpu,
                }
            )
            log(
                f"check size {k:2d}: max|cpu-ref| {d['cpu_vs_ref']:.2e}  "
                f"max|gpu-ref| {d['gpu_vs_ref']:.2e}  "
                f"max|cpu-gpu| {d['cpu_vs_gpu']:.2e}"
            )
    passed = all(v <= a.check_tol for v in worst.values())
    log(
        f"correctness check {'PASSED' if passed else 'FAILED'} "
        f"(tolerance {a.check_tol:g}; worst {max(worst.values()):.2e})"
    )
    return {
        "passed": bool(passed),
        "tolerance": a.check_tol,
        "worst": worst,
        "per_size": rows,
        "supports": {str(k): v for k, v in supports.items()},
    }


def run_calibrations(a, ws, adj, sizes, results, save):
    """Run every requested calibration and record its result."""
    from .null_calibration_gpu import calibrate_null_thresholds_gpu

    common = dict(
        unit_dim=a.unit_dim,
        env_dim=a.env_dim,
        target_rel_width=a.target_rel_width,
        batch_supports=a.batch_supports,
        min_supports=a.min_supports,
        max_pool_size=a.max_pool_size,
        seed=a.seed,
    )
    runners = {
        "gpu": lambda: calibrate_null_thresholds_gpu(
            ws.C00,
            ws.C01,
            ws.C11,
            adj,
            sizes,
            ws.Cs00,
            ws.Cs01,
            ridge=a.ridge,
            device=a.device or "cuda",
            on_log=log,
            **common,
        ),
        "cpu": lambda: calibrate_null_thresholds_cpu(
            ws.C00,
            ws.C01,
            ws.C11,
            adj,
            sizes,
            ws.Cs00,
            ws.Cs01,
            ridge=a.ridge,
            n_workers=a.cpu_workers,
            on_log=log,
            **common,
        ),
        "interface": lambda: calibrate_null_thresholds(
            ws, adj, sizes, on_log=log, **common
        ),
    }
    for name in ("gpu", "cpu", "interface"):
        log(f"=== calibration: {name} ===")
        t = time.perf_counter()
        try:
            res = runners[name]()
            _sync()
            res["wall_seconds"] = time.perf_counter() - t
            # What the table belongs to, so decompose() can refuse it on
            # other data or settings (COG 2.1 fix).
            from .interface import workspace_fingerprint

            res.setdefault("config", {})["fingerprint"] = (
                workspace_fingerprint(ws)
            )
            results["calibrations"][name] = res
            log(f"{name} done in {res['wall_seconds']:.1f}s")
        except Exception as exc:  # noqa: BLE001
            results["calibrations"][name] = {
                "error": f"{type(exc).__name__}: {exc}",
                "wall_seconds": time.perf_counter() - t,
            }
            log(f"!! {name} failed: {type(exc).__name__}: {exc}")
        save()


def size_vs_structure(cal, n_boot=2000, seed=0):
    """Per size: does a support's own threshold theta_S vary with its
    structure, or only by the sampling noise of a 95th percentile of
    n_shift draws?  See the module docstring, step 4.
    """
    rng = np.random.default_rng(seed)
    out = []
    for k, diag in sorted(cal["diagnostics"].items(), key=lambda t: int(t[0])):
        ps = diag["per_support"]
        if not ps:
            continue
        th = np.array([e["theta_S"] for e in ps])
        pool = np.array([v for e in ps for v in e["leading_rho"]])
        J = len(ps[0]["leading_rho"])
        iid = np.percentile(
            rng.choice(pool, size=(n_boot, J), replace=True), 95, axis=1
        )
        sd_obs = float(np.std(th, ddof=1)) if th.size > 1 else float("nan")
        sd_iid = float(np.std(iid, ddof=1))
        r_den, p_den = _spearman(th, [e["contact_density"] for e in ps])
        r_eff, p_eff = _spearman(th, [e["effective_dim"] for e in ps])
        out.append(
            {
                "size": int(k),
                "theta_k": float(cal["theta"][k]),
                "n_supports": len(ps),
                "n_shifts": J,
                "theta_S_mean": float(th.mean()),
                "theta_S_sd": sd_obs,
                "theta_S_min": float(th.min()),
                "theta_S_max": float(th.max()),
                "frac_theta_S_above_theta_k": float(
                    np.mean(th > cal["theta"][k])
                ),
                "sd_if_size_alone": sd_iid,
                "sd_ratio": (sd_obs / sd_iid) if sd_iid > 0 else float("nan"),
                "spearman_density": r_den,
                "p_density": p_den,
                "spearman_effdim": r_eff,
                "p_effdim": p_eff,
            }
        )
    return out


def _theta_table(results, sizes):
    rows = []
    cals = results["calibrations"]
    for k in sizes:
        row = {"size": k}
        thetas = []
        for name in IMPLS:
            c = cals.get(name, {})
            if "theta" not in c:
                continue
            d = c["diagnostics"][k]
            row[f"theta_{name}"] = c["theta"][k]
            row[f"n_supports_{name}"] = d["n_supports"]
            row[f"n_pairs_{name}"] = d["n_pairs"]
            row[f"converged_{name}"] = d["converged"]
            row[f"stopped_by_{name}"] = d.get("stopped_by")
            row[f"rel_ci_width_{name}"] = d["rel_ci_width"]
            row[f"rel_ci_width_naive_{name}"] = d.get("rel_ci_width_naive")
            row[f"seconds_{name}"] = d["elapsed_seconds"]
            row[f"seconds_per_support_{name}"] = (
                d["elapsed_seconds"] / d["n_supports"]
                if d["n_supports"]
                else float("nan")
            )
            thetas.append(c["theta"][k])
        row["max_abs_theta_diff"] = (
            float(np.ptp(thetas)) if len(thetas) > 1 else float("nan")
        )
        rows.append(row)
    return rows


def _write_csv(path, rows):
    if not rows:
        return
    keys = []
    for r in rows:
        keys.extend(k for k in r if k not in keys)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _print_summary(results, sizes):
    cals = results["calibrations"]
    names = [n for n in IMPLS if "theta" in cals.get(n, {})]
    print("\n" + "=" * 78, flush=True)
    print("theta_k by implementation  (n_supports in brackets)")
    print(
        f"{'k':>3} "
        + " ".join(f"{n:>20}" for n in names)
        + f"{'max|diff|':>12}"
    )
    for row in results["theta_table"]:
        cells = [
            f"{row[f'theta_{n}']:.6f} ({row[f'n_supports_{n}']:>3})"
            for n in names
        ]
        print(
            f"{row['size']:>3} "
            + " ".join(f"{c:>20}" for c in cells)
            + f"{row['max_abs_theta_diff']:>12.2e}"
        )
    ref = names[0] if names else None
    if ref is not None:
        print(
            f"\nstopping (from {ref}): 95% CI width relative to theta_k -- "
            f"support-level (used to stop) vs naive value-level"
        )
        print(
            f"{'k':>3} {'supports':>9} {'support CI':>11} {'naive CI':>9}  "
            "stopped by"
        )
        for row in results["theta_table"]:
            print(
                f"{row['size']:>3} {row[f'n_supports_{ref}']:>9} "
                f"{row[f'rel_ci_width_{ref}']:>11.4f} "
                f"{row[f'rel_ci_width_naive_{ref}']:>9.4f}  "
                f"{row[f'stopped_by_{ref}']}"
            )
    print("\nwall-clock time (s), total and per support")
    for n in names:
        c = cals[n]
        n_sup = sum(c["diagnostics"][k]["n_supports"] for k in sizes)
        print(
            f"  {n:>9}: {c['elapsed_seconds']:9.1f} s   "
            f"{c['elapsed_seconds'] / max(n_sup, 1):.4f} s/support "
            f"({n_sup} supports)"
        )
        if "pool_start_seconds" in c.get("config", {}):
            print(
                f"  {'':>9}  (of which CPU worker start-up "
                f"{c['config']['pool_start_seconds']:.1f} s)"
            )
    if "interface" in names:
        for n in names:
            if n != "interface":
                speedup = (
                    cals["interface"]["elapsed_seconds"]
                    / cals[n]["elapsed_seconds"]
                )
                print(f"  speed-up {n} vs interface: {speedup:.1f}x")
    chk = results.get("check")
    if chk:
        print(
            f"\ncorrectness check: {'PASSED' if chk['passed'] else 'FAILED'} "
            f"(worst {max(chk['worst'].values()):.2e}, tol "
            f"{chk['tolerance']:g})"
        )
    svs = results.get("size_vs_structure")
    if svs:
        print(
            "\nsize alone enough?  sd(theta_S) observed vs if size alone "
            "set the null; Spearman rho of theta_S with density / eff-dim"
        )
        for r in svs:
            print(
                f"  k={r['size']:2d}  sd {r['theta_S_sd']:.4f} vs "
                f"{r['sd_if_size_alone']:.4f} (x{r['sd_ratio']:.2f})  "
                f"density {r['spearman_density']:+.2f} "
                f"(p={r['p_density']:.2g})  eff-dim "
                f"{r['spearman_effdim']:+.2f} (p={r['p_effdim']:.2g})"
            )
    print("=" * 78, flush=True)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv=None):
    """Run the null calibration from the command line."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--xtc-topology", required=True)
    ap.add_argument(
        "--xtc",
        nargs="+",
        required=True,
        help="trajectory file(s); for this test, one replica",
    )
    ap.add_argument(
        "--frames",
        type=int,
        default=10000,
        help="first N frames of each replica (default 10000; 0 = every frame)",
    )
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--lag", type=int, default=1)
    ap.add_argument("--rse-cutoff", type=float, default=15.0)
    ap.add_argument("--rse-components", type=int, default=8)
    ap.add_argument("--min-seq-sep", type=int, default=2)
    ap.add_argument("--contact-cutoff", type=float, default=8.0)
    ap.add_argument("--unit-dim", type=int, default=8)
    ap.add_argument("--env-dim", type=int, default=20)
    ap.add_argument("--n-null", type=int, default=20)
    ap.add_argument("--ridge", type=float, default=1e-2)
    ap.add_argument(
        "--sizes", default="2-16", help="e.g. '2-16' or '2,4,8' (default 2-16)"
    )
    ap.add_argument("--target-rel-width", type=float, default=0.02)
    ap.add_argument("--batch-supports", type=int, default=10)
    ap.add_argument(
        "--min-supports",
        type=int,
        default=50,
        help="supports a size needs before it may stop as "
        "converged (default 50)",
    )
    ap.add_argument(
        "--max-pool-size",
        type=int,
        default=20000,
        help="cap on pooled values per size; with 20 shifts the "
        "default 20000 is 1000 supports",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-check-supports", type=int, default=5)
    ap.add_argument("--check-tol", type=float, default=1e-6)
    ap.add_argument(
        "--cpu-workers",
        type=int,
        default=None,
        help="default: --batch-supports",
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args(argv)
    if a.n_null < 8:
        # interface.calibrate_null_thresholds silently grows the shift pool to
        # 8; the CPU/GPU modules use what they are given.  Below 8 the three
        # would calibrate against different shift pools.
        ap.error(
            "--n-null must be at least 8 (interface.calibrate_null_"
            "thresholds extends the shift pool to 8 otherwise)"
        )
    if a.min_supports * a.n_null > a.max_pool_size:
        ap.error(
            f"--min-supports {a.min_supports} x --n-null {a.n_null} "
            f"exceeds --max-pool-size {a.max_pool_size}"
        )
    if a.cpu_workers is None:
        a.cpu_workers = a.batch_supports
    sizes = _parse_sizes(a.sizes)

    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parents[3]
    timings = {}
    results = {
        "args": vars(a),
        "sizes": sizes,
        "environment": _environment(repo_root),
        "timings": timings,
        "calibrations": {},
    }

    def save():
        with (out / "results.json").open("w") as fh:
            json.dump(_jsonable(results), fh, indent=1, allow_nan=True)

    log(f"null calibration real-data test -> {out}")
    log(f"environment: {results['environment']}")
    save()

    ws, adj, data = build_inputs(a, timings)
    results["data"] = data
    save()

    from .null_calibration_gpu import _DeviceState

    t = time.perf_counter()
    gpu_state = _DeviceState(
        ws.C00,
        ws.C01,
        ws.C11,
        ws.Cs00,
        ws.Cs01,
        N=ws.N,
        device=a.device or "cuda",
    )
    _sync()
    timings["check_gpu_upload_seconds"] = time.perf_counter() - t
    results["check"] = correctness_check(a, ws, adj, sizes, gpu_state, timings)
    del gpu_state
    save()

    run_calibrations(a, ws, adj, sizes, results, save)

    results["theta_table"] = _theta_table(results, sizes)
    ref = next(
        (n for n in IMPLS if "theta" in results["calibrations"].get(n, {})),
        None,
    )
    if ref is not None:
        results["size_vs_structure_source"] = ref
        results["size_vs_structure"] = size_vs_structure(
            results["calibrations"][ref], seed=a.seed
        )
    save()

    _write_csv(out / "theta_table.csv", results["theta_table"])
    _write_csv(
        out / "size_vs_structure.csv", results.get("size_vs_structure", [])
    )
    ps_rows = []
    for name in IMPLS:
        c = results["calibrations"].get(name, {})
        for k, d in c.get("diagnostics", {}).items():
            for i, e in enumerate(d["per_support"]):
                ps_rows.append(
                    {
                        "impl": name,
                        "size": int(k),
                        "index": i,
                        "residues": " ".join(map(str, e["support"])),
                        "theta_S": e["theta_S"],
                        "contact_density": e["contact_density"],
                        "effective_dim": e["effective_dim"],
                    }
                )
    _write_csv(out / "per_support.csv", ps_rows)

    _print_summary(results, sizes)
    log(f"done; outputs in {out}")
    ok = results["check"]["passed"] and all(
        "theta" in results["calibrations"].get(n, {}) for n in IMPLS
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
