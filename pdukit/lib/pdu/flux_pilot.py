r"""mdCATH pilot: how many channels connect a candidate PDU to the rest.

This is the Gate A measurement of the v3.0 research plan, in its smallest
honest form.  For each domain it

  1. loads CA trajectories over HTTPS (no local download), pooling replicates
     as SEGMENTS so no lagged pair ever crosses a replicate boundary;
  2. builds residue-supported local descriptors -- no global modes, no
     whole-protein alignment;
  3. grows candidate supports greedily from seed residues, minimising the
     effective number of boundary channels;
  4. scores matched controls at the same sizes: random, contiguous-in-sequence,
     and spatially compact ("ball") sets;
  5. writes per-domain JSON plus a summary.

The headline number is the median effective channel count k_eff of discovered
supports versus the ball control at matched size.  If discovered supports are
not clearly below the ball control, the honest conclusion is that compactness
explains the result and there is no dynamical unit -- report that.

Resolution caveat, stated once and meant: mdCATH frames are ~1 ns apart, so
nothing faster than roughly 10 ns is resolved and rare transitions are not
sampled.  Every number here is a claim at that resolution only.

Usage
-----
    python -m pdukit.lib.pdu.flux_pilot --domains 5bqpD00 1nayB00 \\
        --temperature 320 --replicates 0 1 2 3 4 --out
        analysis/results/flux_pilot

    # smoke test with no network at all
    python -m pdukit.lib.pdu.flux_pilot --synthetic --out /tmp/flux_smoke
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from .interface import (
    FluxWorkspace,
    contact_graph,
    control_supports,
    decompose,
    detect_length_unit,
    grow_support,
    median_distance_matrix,
    residue_budget,
    residue_local_features,
    rse_features,
    score_support,
)

PILOT_DOMAINS = ("5bqpD00", "1nayB00", "1cmwA01", "2ifsA00", "2e5zA01")

# residue numbering and names from the last local trajectory load, so PyMOL
# selections use the trajectory's own numbering rather than 1..N positions
_RESIDS = {}


_LAST_TICK = {}


def _progress(done, total, prefix, t0, verbose=True):
    """Progress counter in the style already used elsewhere in pdukit (a
    carriage-return line, no tqdm dependency).

    On a terminal it redraws one line.  When stdout is a file -- a
    batch-scheduler job log, say -- carriage returns are useless, so it falls
    back to one plain line per quarter, which keeps logs readable instead of
    producing a single enormous line.
    """
    if not verbose or not total:
        return
    frac = done / total
    w = len(str(total))
    if sys.stdout.isatty():
        bar_w = 24
        filled = int(round(bar_w * frac))
        sys.stdout.write(
            f"\r    {prefix} [{'#' * filled}{'-' * (bar_w - filled)}] "
            f"{done:<{w}}/{total}, {frac:.0%}  {time.time() - t0:5.1f}s"
        )
        sys.stdout.flush()
        if done >= total:
            sys.stdout.write("\n")
            sys.stdout.flush()
    else:
        step = max(1, total // 4)
        if done >= total or done % step == 0:
            if _LAST_TICK.get(prefix) != done:
                _LAST_TICK[prefix] = done
                print(
                    f"    {prefix} {done:<{w}}/{total}, {frac:.0%}  "
                    f"{time.time() - t0:5.1f}s",
                    flush=True,
                )


def _log(msg, verbose=True):
    if verbose:
        print(msg, flush=True)


def _fmt(x, nd=2):
    try:
        return (
            "  n/a"
            if x is None or not np.isfinite(x)
            else f"{x:{nd + 4}.{nd}f}"
        )
    except TypeError:
        return "  n/a"


def _jsonable(v):
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return v


# ---------------------------------------------------------------------------


def load_segments(
    domain,
    temperature="320",
    replicates=("0",),
    stride=1,
    max_frames=None,
    dt_ns=1.0,
    remote=True,
    cache_dir=None,
    verbose=True,
):
    """Return (ca_coords (Ftot, N, 3), segments, meta).

    Replicates are concatenated along time and their row ranges returned as
    segments, so `flux_spectrum` never forms a lagged pair across a boundary --
    the same discipline `koopman.tica_multi` applies to covariances.
    """
    from .extract import io as mio
    from .extract import remote_io
    from .extract.hdf5_layout import HF_FILE_TEMPLATE

    blocks, segments, cursor, meta = [], [], 0, []
    t0 = time.time()
    _log(
        f"  loading {len(replicates)} replicate(s) at T={temperature}", verbose
    )
    _progress(0, len(replicates), "load    ", t0, verbose)
    for ri, rep in enumerate(replicates, 1):
        kw = dict(
            domain=domain,
            temperature=str(temperature),
            replicate=str(rep),
            stride=stride,
            max_frames=max_frames,
            dt_ns=dt_ns,
            atom_selection="CA",
        )
        if remote:
            traj = remote_io.load_domain_remote(**kw)
        else:
            path = Path(cache_dir) / HF_FILE_TEMPLATE.format(domain=domain)
            traj = mio.load_domain(path, **kw)
        X = np.asarray(traj.coords, dtype=np.float64)
        blocks.append(X)
        segments.append((cursor, cursor + X.shape[0]))
        cursor += X.shape[0]
        _progress(ri, len(replicates), "load    ", t0, verbose)
        meta.append(
            {
                "replicate": str(rep),
                "n_frames": int(X.shape[0]),
                "source": str(traj.source),
            }
        )

    total_frames = sum(b.shape[0] for b in blocks)
    _log(
        f"    {total_frames} frames total across {len(blocks)} replicate(s), "
        f"{blocks[0].shape[1] if blocks else 0} residues",
        verbose,
    )
    shapes = {b.shape[1] for b in blocks}
    if len(shapes) != 1:
        raise ValueError(
            f"{domain}: replicates disagree on residue count {shapes}"
        )
    return np.concatenate(blocks, axis=0), segments, meta


def load_xtc_segments(
    topology,
    replica_paths,
    stride=1,
    max_frames=None,
    atom_selection="CA",
    verbose=True,
):
    """Load local GROMACS replicas into (coords, segments, meta).

    Each replica becomes its own segment, so no lagged pair is ever formed
    across a replica seam -- the same discipline `koopman.tica_multi` applies
    to mdCATH replicates, and it matters more here: these are independent runs
    from different seeds, not one continuous trajectory.

    A replica may be given as SEVERAL files (a list, or one string with the
    parts separated by commas): they are read in the order given as one
    continuous trajectory (MDAnalysis ChainReader, continuous=True, so a
    frame repeated at a restart boundary is read once).

    Needs MDAnalysis (`pip install MDAnalysis`, or the repo's `.[xtc]` extra).
    """
    try:
        import MDAnalysis
    except ImportError as exc:  # noqa: BLE001
        raise ImportError(
            "MDAnalysis is required to read local trajectories: "
            "pip install MDAnalysis"
        ) from exc
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning)

    sel = {"CA": "protein and name CA", "all": "protein"}.get(
        atom_selection, atom_selection
    )
    blocks, segments, cursor, meta = [], [], 0, []
    t0 = time.time()
    _log(f"  loading {len(replica_paths)} replica(s) via MDAnalysis", verbose)
    for ri, path in enumerate(replica_paths, 1):
        parts = (
            list(path)
            if isinstance(path, (list, tuple))
            else [p_ for p_ in str(path).split(",") if p_]
        )
        if len(parts) > 1:
            u = MDAnalysis.Universe(topology, parts, continuous=True)
        else:
            u = MDAnalysis.Universe(topology, parts[0])
        atoms = u.select_atoms(sel)
        n_total = len(u.trajectory)
        frames = range(0, n_total, max(1, int(stride)))
        if max_frames:
            frames = list(frames)[: int(max_frames)]
        frames = list(frames)
        X = np.empty((len(frames), atoms.n_atoms, 3), dtype=np.float32)
        for k, fi in enumerate(frames):
            u.trajectory[fi]
            X[k] = atoms.positions
            if k % 2000 == 0:
                _progress(
                    cursor + k,
                    cursor + len(frames),
                    f"rep {ri}/{len(replica_paths)}",
                    t0,
                    verbose,
                )
        _progress(1, 1, f"rep {ri}/{len(replica_paths)}", t0, verbose)
        blocks.append(X)  # keep float32: 114k frames is 450 MB in f64
        segments.append((cursor, cursor + len(frames)))
        cursor += len(frames)
        dt_ps = float(u.trajectory.dt) * max(1, int(stride))
        meta.append(
            {
                "replicate": str(ri - 1),
                "n_frames": len(frames),
                "dt_ps": dt_ps,
                "source": ",".join(parts),
                "n_parts": len(parts),
            }
        )
        _log(
            f"    {Path(parts[0]).name}"
            + (f" (+{len(parts) - 1} parts)" if len(parts) > 1 else "")
            + f": {len(frames)} frames, dt {dt_ps:.1f} ps, "
            f"{atoms.n_atoms} atoms",
            verbose,
        )
    shapes = {b.shape[1] for b in blocks}
    if len(shapes) != 1:
        raise ValueError(f"replicas disagree on atom count {shapes}")
    total = sum(b.shape[0] for b in blocks)
    _log(
        f"    {total} frames total "
        f"({total * meta[0]['dt_ps'] / 1000:.0f} ns aggregate)",
        verbose,
    )
    return np.concatenate(blocks, axis=0), segments, meta


def synthetic_segments(n_res=48, n_frames=1500, n_seg=3, seed=0):
    """Make a two-lobe synthetic protein for smoke tests (no network).

    Each lobe is driven by three AR(1) latent modes.  The lobes are joined by a
    SINGLE shared channel, so a correct pilot should report roughly one channel
    for a lobe and more for sets that straddle the two.
    """
    rng = np.random.default_rng(seed)
    half = n_res // 2
    ref = np.zeros((n_res, 3))
    ref[:, 0] = np.arange(n_res) * 3.8
    ref[half:, 1] = 12.0

    def ar1(k, rho):
        z = np.zeros((n_frames, k))
        for t in range(1, n_frames):
            z[t] = rho * z[t - 1] + rng.normal(scale=1.0, size=k)
        return z

    za, zb = ar1(3, 0.90), ar1(3, 0.90)
    shared = ar1(1, 0.95)  # the one channel joining them
    za[:, 0] += 0.8 * shared[:, 0]
    zb[:, 0] += 0.8 * shared[:, 0]

    # amplitudes kept small enough that consecutive-CA spacing stays near the
    # physical 3.8 A, so the smoke test does not trip its own geometry check
    Wa = rng.normal(size=(3, half, 3)) * 0.12
    Wb = rng.normal(size=(3, n_res - half, 3)) * 0.12
    coords = ref[None] + rng.normal(scale=0.20, size=(n_frames, n_res, 3))
    coords[:, :half, :] += np.einsum("tk,kij->tij", za, Wa)
    coords[:, half:, :] += np.einsum("tk,kij->tij", zb, Wb)

    seg = n_frames // n_seg
    segments = [(i * seg, (i + 1) * seg) for i in range(n_seg)]
    meta = [
        {"replicate": f"synthetic{i}", "n_frames": seg, "source": "synthetic"}
        for i in range(n_seg)
    ]
    return coords, segments, meta


# ---------------------------------------------------------------------------


def run_domain(
    coords,
    segments,
    *,
    lag=2,
    n_neighbours=8,
    min_seq_sep=2,
    contact_cutoff=8.0,
    n_seeds=6,
    max_size=18,
    unit_dim=8,
    env_dim=40,
    ridge=1e-2,
    n_null=12,
    size_penalty=0.02,
    n_controls=8,
    seed=0,
    verbose=True,
    budget=False,
    min_size=4,
    knee_tol=0.15,
    device=None,
    budget_max_frames=20000,
    exclude_termini=2,
    rse_cutoff=None,
    rse_components=8,
    resids=None,
    resnames=None,
    control_overlap=0.3,
    decompose_mode=False,
    dec_seeds=24,
    dec_iters=6,
    boundary_tol=0.15,
    merge_jaccard=0.5,
    reversible=True,
    truncation="koopman",
):
    """Run the channel analysis on one domain and return its report."""
    rng = np.random.default_rng(seed)
    _log(
        "  median inter-residue distances (rotation/PBC-robust reference) ...",
        verbose,
    )
    dmed = median_distance_matrix(coords)
    units = detect_length_unit(coords, dmed=dmed)
    t_f = time.time()
    if rse_cutoff:
        _log(
            f"  building RSE features (all residues within {rse_cutoff:.0f} "
            "A, "
            f"compressed to {rse_components} components) ...",
            verbose,
        )
        feats, neighbours, explained = rse_features(
            coords,
            cutoff=rse_cutoff,
            n_components=rse_components,
            min_seq_sep=min_seq_sep,
            dmed=dmed,
            lag=lag,
            segments=segments,
            koopman=True,
            ridge=ridge,
            on_progress=lambda d, t: _progress(d, t, "RSE     ", t_f, verbose),
        )
        descriptor = _describe_rse(
            neighbours, explained, rse_cutoff, rse_components
        )
        _log(_rse_line(descriptor), verbose)
    else:
        _log("  building residue-local features ...", verbose)
        feats, neighbours = residue_local_features(
            coords,
            n_neighbours=n_neighbours,
            min_seq_sep=min_seq_sep,
            dmed=dmed,
            on_progress=lambda d, t: _progress(d, t, "features", t_f, verbose),
        )
        descriptor = _describe_rse(neighbours, None, None, n_neighbours)
        _log(_rse_line(descriptor), verbose)
    adj = contact_graph(coords, cutoff=contact_cutoff, dmed=dmed, units="auto")
    N = feats.shape[1]
    _log(
        f"  coordinates look like {units['unit']} (consecutive CA spacing "
        f"{units['median_ca_spacing']:.3f} = {units['spacing_angstrom']:.2f} "
        "A); "
        f"contact cutoff {contact_cutoff:.1f} A applied accordingly",
        verbose,
    )
    if units.get("suspect"):
        _log(
            f"  !! SUSPECT GEOMETRY: consecutive CA spacing works out at "
            f"{units['spacing_angstrom']:.2f} A, but a polypeptide gives 3.8. "
            f"Check the CA selection (io._guess_ca_from_z returns ALL atoms "
            f"when the PDB text is unusable) and whether the trajectory needs "
            f"unwrapping. Distances below are not trustworthy.",
            verbose,
        )
    n_pairs = sum(max(0, b - a - lag) for a, b in segments)
    _log(
        f"  {N} residues, {coords.shape[0]} frames, {n_pairs} lagged pairs "
        f"at lag {lag}; mean contacts/residue {adj.sum(axis=1).mean():.1f}",
        verbose,
    )

    t_ws = time.time()
    _log(
        f"  building covariance workspace (D={N * n_neighbours}, "
        f"{n_null} null shifts) ...",
        verbose,
    )
    ws = FluxWorkspace(
        feats,
        lag=lag,
        segments=segments,
        n_null=n_null,
        seed=seed,
        ridge=ridge,
        device=device,
        reversible=reversible,
        truncation=truncation,
        on_progress=lambda d, t: _progress(d, t, "cov     ", t_ws, verbose),
    )
    _log(
        f"    truncation={ws.truncation} "
        f"({'reversible tICA' if ws.reversible else 'VAMP'})",
        verbose,
    )
    _log(
        f"    backend={ws.backend} device={ws.device}  "
        f"{ws.n} lagged pairs  built in {time.time() - t_ws:.1f}s",
        verbose,
    )
    try:
        import resource

        peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
        _log(
            f"    peak memory {peak_gb:.2f} GB",
            verbose,
        )
    except Exception:  # noqa: BLE001
        pass
    if device not in (None, "cpu") and ws.backend != "torch":
        _log(
            "  !! GPU was requested but torch/CUDA is not available here; "
            "running on CPU.",
            verbose,
        )

    spec_kw = dict(unit_dim=unit_dim, env_dim=env_dim, workspace=ws)

    if decompose_mode:
        _log(
            "  self-consistent decomposition (every residue -> >=1 PDU) ...",
            verbose,
        )
        t_d = time.time()
        dec = decompose(
            feats,
            adj,
            ws,
            unit_dim=unit_dim,
            env_dim=env_dim,
            n_seeds=dec_seeds,
            max_size=max_size,
            min_size=min_size,
            boundary_tol=boundary_tol,
            merge_jaccard=merge_jaccard,
            size_penalty=size_penalty,
            max_iter=dec_iters,
            knee_tol=knee_tol,
            seed=seed,
            n_controls=n_controls,
            control_overlap=control_overlap,
            on_log=lambda m: _log("    " + m, verbose),
        )
        if (
            len(dec["trace"]) > 1
            and dec["trace"][-1]["objective"]
            > dec["trace"][0]["objective"] + 1e-9
        ):
            _log(
                "    note: the objective rose across iterations. The "
                "assignment step is not an exact block minimisation, so v3.0 "
                "sec 10 step 6's monotonicity does not apply here; read the "
                "trace rather than assuming convergence.",
                verbose,
            )
        _log(
            f"    {dec['n_units']} units, coverage "
            f"{100 * dec['coverage']:.0f}%, "
            f"{len(dec['boundary_residues'])} boundary residues, "
            f"{len(dec['orphans'])} orphans  ({time.time() - t_d:.0f}s)",
            verbose,
        )
        gaps = dec.get("gap_vs_best_control", {})
        for sc in dec["scores"]:
            ke = sc.get("k_eff")
            ke = "  n/a" if ke is None or not np.isfinite(ke) else f"{ke:5.2f}"
            g = gaps.get(sc["unit"], gaps.get(str(sc["unit"])))
            gs = "" if g is None else f"  gap_vs_control={g:+.2f}"
            _log(
                f"      unit {sc['unit']:>2}  size {sc['size']:>3}  "
                f"k_eff={ke}{gs}  residues={sc['residues']}",
                verbose,
            )
        if gaps:
            gv = [v for v in gaps.values() if np.isfinite(v)]
            gr = [
                v
                for v in (dec.get("gap_vs_random") or {}).values()
                if np.isfinite(v)
            ]
            if gr:
                _log(
                    f"    MEDIAN GAP vs RANDOM       : {np.median(gr):+.2f}  "
                    f"({sum(1 for v in gr if v > 0)}/{len(gr)} positive)",
                    verbose,
                )
            _log(
                f"    MEDIAN GAP vs BEST CONTROL : {np.median(gv):+.2f}  "
                f"({sum(1 for v in gv if v > 0)}/{len(gv)} positive)  "
                f"[controls disjoint from units at overlap > "
                f"{dec.get('control_overlap_threshold', 0.3):.2f}]",
                verbose,
            )
        dec["resids"] = None if resids is None else [int(r) for r in resids]
        return {
            "n_residues": int(N),
            "n_frames": int(coords.shape[0]),
            "segments": [list(map(int, sg)) for sg in segments],
            "lag_frames": int(lag),
            "descriptor": descriptor,
            "length_unit": units,
            "decomposition": dec,
            "backend": ws.backend,
            "device": ws.device,
            "config": {
                "unit_dim": unit_dim,
                "env_dim": env_dim,
                "ridge": ridge,
                "n_null": n_null,
                "rse_cutoff": rse_cutoff,
                "rse_components": rse_components,
                "min_seq_sep": min_seq_sep,
                "truncation": truncation,
                "reversible": reversible,
            },
        }

    # ---- power pre-check, before spending minutes on growth ----------------
    probe_sizes = [max(2, min(6, N)), max(3, min(12, N))]
    floors = []
    for ps in probe_sizes:
        probe = sorted(rng.choice(N, size=ps, replace=False).tolist())
        try:
            pr = score_support(feats, probe, lag=lag, **spec_kw)
        except Exception:  # noqa: BLE001
            continue
        floors.append(pr["rho_null_p95"])
        _log(
            f"  power probe (size {ps}): null floor {pr['rho_null_p95']:.3f}, "
            f"leading rho {pr['rho'][0]:.3f}",
            verbose,
        )
    probe_floor = float(np.median(floors)) if floors else float("nan")
    if np.isfinite(probe_floor) and probe_floor > 0.30:
        _log(
            f"  !! LOW POWER: null floor {probe_floor:.2f} > 0.30. Nothing "
            "can "
            f"be resolved with {n_pairs} pairs at env_dim={env_dim}. Add "
            f"replicates, drop --max-frames, or lower --env-dim.",
            verbose,
        )

    # Chain termini are floppy and dominate their own local dynamics, so seeds
    # placed there walk straight into the terminus and stay: on mdCATH,
    # independent seeds on 1jfkA02 and 1vq8L01 both converged to [0,1,2,3] and
    # the seed-agreement Jaccard came out at a meaningless 1.0.  Termini are
    # excluded from SEEDING only -- growth may still absorb them if they
    # genuinely belong to a unit.
    e = int(exclude_termini)
    pool = np.arange(e, N - e) if 0 < e < N // 2 else np.arange(N)
    if e:
        _log(
            f"  seeding from residues {pool[0]}-{pool[-1]} "
            f"({e} excluded at each terminus)",
            verbose,
        )
    seeds = rng.choice(pool, size=min(n_seeds, pool.size), replace=False)
    discovered = []
    for si, s in enumerate(seeds, 1):
        _log(
            f"  [seed {si}/{len(seeds)}] residue {int(s)}: growing "
            f"(max_size={max_size})",
            verbose,
        )

        def _step(info, _si=si):
            flag = "" if info["resolved"] else "  (unresolved)"
            _log(
                f"    size {info['size']:>3}  +res {info['added']:>4}  "
                f"tried {info['n_tried']:>3}  k*={info['k_star']:>2}  "
                f"k_eff={_fmt(info['k_eff'])}  null={info['null']:.3f}{flag}",
                verbose,
            )

        t_seed = time.time()

        def _cand(done, total, _t=t_seed):
            _progress(done, total, "frontier", _t, verbose)

        try:
            r = grow_support(
                feats,
                int(s),
                adj,
                max_size=max_size,
                lag=lag,
                size_penalty=size_penalty,
                min_size=min_size,
                knee_tol=knee_tol,
                on_step=_step if verbose else None,
                on_candidate=_cand if verbose else None,
                **spec_kw,
            )
        except Exception as exc:  # noqa: BLE001
            _log(f"    FAILED: {exc}", verbose)
            discovered.append({"seed": int(s), "error": str(exc)})
            continue
        _log(
            f"    -> knee at size {r['size']} ({r['status']}): "
            f"k*={r['k_star']}, k_eff={_fmt(r['k_eff'])}; full curve saved",
            verbose,
        )
        r["rho_top"] = (r.get("rho") or [])[:5]
        r.pop("rho", None)
        discovered.append(r)

    ok = [d for d in discovered if "k_eff" in d]
    sizes = sorted({d["size"] for d in ok})
    _log(
        f"  scoring controls at sizes {sizes} "
        f"({3 * n_controls} sets per size) ...",
        verbose,
    )
    controls = {}
    for size in sizes:
        cs = control_supports(
            feats,
            size,
            adjacency=adj,
            n_random=n_controls,
            n_segment=n_controls,
            n_ball=n_controls,
            seed=seed + size,
        )
        row = {}
        t_ctl = time.time()
        n_total = sum(len(v) for v in cs.values())
        n_done = 0
        for kind, sets in cs.items():
            vals = []
            for S in sets:
                n_done += 1
                _progress(n_done, n_total, f"ctl s{size:<3}", t_ctl, verbose)
                try:
                    r = score_support(feats, S, lag=lag, **spec_kw)
                except Exception:  # noqa: BLE001
                    continue
                vals.append(
                    {
                        "k_eff": r["k_eff"],
                        "k_star": r["k_star"],
                        "margin": r.get("margin"),
                        "coupling_total": r["coupling_total"],
                    }
                )
            row[kind] = vals
        for kind, vals in row.items():
            med = _median([v["k_eff"] for v in vals])
            mmed = _median([v.get("margin") for v in vals])
            _log(
                f"    size {size:>3}  {kind:<8} k_eff median {_fmt(med)}  "
                f"margin median {mmed:+.3f}  ({len(vals)} sets)",
                verbose,
            )
        controls[str(size)] = row

    budget_result = None
    if budget:
        _log(
            "  residue variance budget (inside / outside / irreducible) ...",
            verbose,
        )
        t_b = time.time()
        b_feats, b_segments = feats, segments
        if feats.shape[0] > budget_max_frames:
            k = int(np.ceil(feats.shape[0] / budget_max_frames))
            b_feats = feats[::k]
            b_segments = [(a // k, b // k) for a, b in segments]
            _log(
                f"    subsampling every {k}th frame for the budget "
                f"({b_feats.shape[0]} frames); the budget is a per-residue "
                f"variance split, not a rate estimate, so this costs "
                f"resolution at short lags only",
                verbose,
            )
        budget_result = residue_budget(
            b_feats,
            adjacency=adj,
            lag=lag,
            ridge=ridge,
            env_dim=env_dim,
            n_null=max(4, n_null // 2),
            seed=seed,
            segments=b_segments,
            on_progress=(
                lambda d, t: _progress(d, t, "budget  ", t_b, verbose)
            ),
        )
        ins = np.array(budget_result["inside"])
        out = np.array(budget_result["outside"])
        res = np.array(budget_result["resolved"], dtype=bool)
        _log(
            f"    inside median {np.median(ins):.3f}, outside median "
            f"{np.median(out):.3f}, resolved {res.mean():.0%} of residues",
            verbose,
        )
        if res.any():
            top = np.argsort(-out)[:8]
            _log(
                f"    largest outside share (channel residues): "
                f"{[int(i) for i in top if res[i]]}",
                verbose,
            )

    # support agreement across seeds: how often do two discovered supports that
    # share any residue agree overall?  This is the sec 7.1 deliverable -- the
    # narrowness of the family, not a single partition.
    jac = []
    for i in range(len(ok)):
        for j in range(i + 1, len(ok)):
            a, b = set(ok[i]["support"]), set(ok[j]["support"])
            if a & b:
                jac.append(len(a & b) / len(a | b))

    return {
        "n_residues": int(N),
        "n_frames": int(coords.shape[0]),
        "segments": [list(map(int, s)) for s in segments],
        "lag_frames": int(lag),
        "discovered": discovered,
        "controls": controls,
        "overlapping_pair_jaccard": jac,
        "power_probe_null_floor": probe_floor,
        "backend": ws.backend,
        "device": ws.device,
        "length_unit": units,
        "residue_budget": budget_result,
        "config": {
            "n_neighbours": n_neighbours,
            "min_seq_sep": min_seq_sep,
            "contact_cutoff": contact_cutoff,
            "unit_dim": unit_dim,
            "env_dim": env_dim,
            "ridge": ridge,
            "n_null": n_null,
            "size_penalty": size_penalty,
            "max_size": max_size,
        },
    }


def _describe_rse(neighbours, explained, cutoff, n_components):
    """Summarise a descriptor: how big is each residue's environment, and how
    much of it is long-range in sequence.

    This is the number that decides whether a descriptor is three-dimensional
    or just local chain geometry.  Six nearest neighbours on lysozyme are 78%
    within +-5 in sequence; a 15 A RSE is ~40 residues with 64% more than 10
    apart.  Print it every run so the distinction can never be assumed again.
    """
    sizes = np.array([len(nb) for nb in neighbours])
    seps = np.concatenate(
        [np.abs(np.asarray(nb) - i) for i, nb in enumerate(neighbours)]
    )
    out = {
        "cutoff_angstrom": cutoff,
        "n_components": int(n_components),
        "size_median": int(np.median(sizes)),
        "size_min": int(sizes.min()),
        "size_max": int(sizes.max()),
        "frac_seq_sep_gt10": float((seps > 10).mean()),
        "frac_seq_sep_le5": float((seps <= 5).mean()),
    }
    if explained is not None:
        out["variance_retained_median"] = float(np.median(explained))
        out["variance_retained_min"] = float(np.min(explained))
    return out


def _rse_line(d):
    v = (
        ""
        if "variance_retained_median" not in d
        else f"; variance retained median {d['variance_retained_median']:.2f} "
        f"(min {d['variance_retained_min']:.2f})"
    )
    what = "RSE" if d["cutoff_angstrom"] else "nearest-neighbour"
    return (
        f"    {what} size: median {d['size_median']} "
        f"(range {d['size_min']}-{d['size_max']}); "
        f"{100 * d['frac_seq_sep_gt10']:.0f}% of neighbours >10 apart in "
        f"sequence, {100 * d['frac_seq_sep_le5']:.0f}% within +-5{v}"
    )


PYMOL_COLOURS = (
    "skyblue",
    "salmon",
    "palegreen",
    "wheat",
    "lightpink",
    "paleyellow",
    "lightblue",
    "palecyan",
    "lightorange",
    "bluewhite",
    "violetpurple",
    "deepteal",
    "firebrick",
    "splitpea",
    "sand",
    "olive",
)


def write_pdu_pdb(path, coords, decomposition, resids=None, resnames=None):
    """CA-only PDB of the reference structure, B-factor = participation weight.

    The B-factor carries the residue's weight in its dominant unit (x100), so
    `spectrum b` in PyMOL shades each unit from core to periphery.  Boundary
    residues, which by construction have no dominant unit, land near 50.
    """
    ref = (
        np.asarray(coords[0], dtype=np.float64) if coords.ndim == 3 else coords
    )
    N = ref.shape[0]
    wres = decomposition.get("weight_residue") or [{} for _ in range(N)]
    lines = []
    for i in range(N):
        w = wres[i] if i < len(wres) else {}
        b = 100.0 * (max(w.values()) if w else 0.0)
        rid = int(resids[i]) if resids is not None else i + 1
        rnm = (resnames[i] if resnames is not None else "ALA")[:3].upper()
        x, y, z = ref[i]
        lines.append(
            f"ATOM  {i + 1:>5}  CA  {rnm:>3} A{rid:>4}    "
            f"{x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00{b:>6.2f}           C"
        )
    lines.append("END")
    Path(path).write_text("\n".join(lines) + "\n")
    return path


def write_pdu_pml(path, decomposition, pdb_name, label="PDU", resids=None):
    """Write a PyMOL script: one selection and colour per unit.

    Boundary residues are marked.

    Selections use the PDB's residue numbering, which is the trajectory's own
    numbering when resids were available -- so `unit_3` can be compared
    directly with a mutation site or an NMR peak list.
    """
    units = decomposition.get("units", [])
    boundary = {
        b["residue"]: b for b in decomposition.get("boundary_detail", [])
    }
    scores = {s["unit"]: s for s in decomposition.get("scores", [])}
    gaps = decomposition.get("gap_vs_best_control", {}) or {}

    def rid(i):
        return int(resids[i]) if resids is not None else i + 1

    L = [
        f"# {label}: {len(units)} PDUs, "
        f"coverage {100 * decomposition.get('coverage', 0):.0f}%, "
        f"{len(boundary)} boundary residues",
        "# B-factor = participation weight in the residue's dominant unit "
        "(x100)",
        "#",
        "# To use your own structure instead of the CA-only reference:",
        f"#   delete {label}; load your_structure.pdb, {label}",
        "",
        f"load {pdb_name}, {label}",
        "hide everything",
        f"show ribbon, {label}",
        "set ribbon_width, 4",
        "bg_color white",
        "set ray_opaque_background, 0",
        "",
    ]

    for a, u in enumerate(units):
        sc = scores.get(a, {})
        ke = sc.get("k_eff")
        ke = "n/a" if ke is None or not np.isfinite(ke) else f"{ke:.2f}"
        g = gaps.get(a, gaps.get(str(a)))
        gs = "" if g is None else f", gap vs control {g:+.2f}"
        sel = "+".join(str(rid(i)) for i in sorted(u))
        colour = PYMOL_COLOURS[a % len(PYMOL_COLOURS)]
        L += [
            f"# unit {a}: {len(u)} residues, k_eff {ke}{gs}",
            f"select unit_{a}, {label} and resi {sel}",
            f"color {colour}, unit_{a}",
            f"show spheres, unit_{a} and name CA",
            f"set sphere_scale, 0.35, unit_{a}",
            "",
        ]

    if boundary:
        sel = "+".join(str(rid(i)) for i in sorted(boundary))
        L += [
            "# boundary residues: assigned to more than one unit.",
            "# 'balance' is the weight ratio between their units -- near 1.0",
            "# means genuinely shared, near 0 means one unit dominates.",
            f"select boundary, {label} and resi {sel}",
            "color grey40, boundary",
            "show spheres, boundary and name CA",
            "set sphere_scale, 0.6, boundary",
            "",
        ]
        for i, b in sorted(boundary.items()):
            L.append(
                f"#   residue {rid(i)}: units {b['units']}, "
                f"weights {b['weights']}, balance {b['balance']}"
            )
        L.append("")

    L += [
        "# colour by participation weight instead (core -> periphery):",
        f"#   spectrum b, blue_white_red, {label} and name CA",
        "",
        "deselect",
        "orient",
        f"# png {label}_pdu.png, width=1600, height=1200, dpi=300, ray=1",
    ]
    Path(path).write_text("\n".join(L) + "\n")
    return path


def power_scan(
    coords,
    segments,
    *,
    lag=2,
    n_neighbours=6,
    min_seq_sep=2,
    rse_cutoff=None,
    rse_components=8,
    unit_dims=(4, 8, 16),
    env_dims=(5, 10, 20, 40),
    ridges=(1e-3, 1e-2, 1e-1),
    frame_fractions=(0.5, 1.0),
    lags=(1, 2, 4, 8),
    probe_size=8,
    n_probes=3,
    n_null=20,
    seed=0,
    verbose=True,
):
    """Measure what actually sets the null floor, on real data.

    The floor is what decides whether anything can be resolved, and reasoning
    about it from simulation is not enough: on synthetic AR(1) noise the floor
    falls steeply with environment dimension, while on 5bqpD00 halving env_dim
    barely moved it.  That discrepancy is a fact about protein features -- they
    are far more collinear and far more slowly varying than simulated noise --
    and the only way to settle it is to vary one factor at a time on the actual
    trajectory.

    Scores random probe supports (no growth, so it is cheap) and reports the
    median floor for each setting.  Use it to choose parameters, and to decide
    whether a domain is workable at all before spending an hour on it.
    """
    dmed = median_distance_matrix(coords)
    descriptor = {}
    if rse_cutoff:
        _log(
            f"  building RSE features (all residues within {rse_cutoff:.0f} "
            "A, "
            f"compressed to {rse_components} components) ...",
            verbose,
        )
        feats, nbs, explained = rse_features(
            coords,
            cutoff=rse_cutoff,
            n_components=rse_components,
            min_seq_sep=min_seq_sep,
            dmed=dmed,
            lag=lag,
            segments=segments,
            koopman=True,
        )
        descriptor = _describe_rse(nbs, explained, rse_cutoff, rse_components)
        _log(_rse_line(descriptor), verbose)
    else:
        _log(
            f"  building nearest-neighbour features (n={n_neighbours}) ...",
            verbose,
        )
        feats, nbs = residue_local_features(
            coords,
            n_neighbours=n_neighbours,
            min_seq_sep=min_seq_sep,
            dmed=dmed,
        )
        descriptor = _describe_rse(nbs, None, None, n_neighbours)
        _log(_rse_line(descriptor), verbose)
    N = feats.shape[1]
    rng = np.random.default_rng(seed)
    probes = [
        sorted(rng.choice(N, size=min(probe_size, N), replace=False).tolist())
        for _ in range(n_probes)
    ]

    base = dict(unit_dim=8, env_dim=20, ridge=1e-2, frac=1.0, lag=lag)

    def _floor(unit_dim, env_dim, ridge, frac, lag):
        segs = segments
        if frac < 1.0:
            segs = [
                (a, a + max(lag + 2, int((b - a) * frac))) for a, b in segments
            ]
        vals = []
        for S in probes:
            try:
                r = score_support(
                    feats,
                    S,
                    lag=lag,
                    ridge=ridge,
                    unit_dim=unit_dim,
                    env_dim=env_dim,
                    n_null=n_null,
                    seed=seed,
                    segments=segs,
                )
            except Exception:  # noqa: BLE001
                continue
            vals.append(
                (r["rho_null_p95"], r["rho"][0], r["n_pairs"], r["k_star"])
            )
        if not vals:
            return None
        fl = float(np.median([v[0] for v in vals]))
        r1 = float(np.median([v[1] for v in vals]))
        return {
            "floor": fl,
            "rho1": r1,
            "margin": r1 - fl,
            "k_star": float(np.median([v[3] for v in vals])),
            "n_pairs": int(vals[0][2]),
        }

    out = {
        "baseline": base,
        "n_residues": int(N),
        "axes": {},
        "descriptor": descriptor,
    }
    axes = [
        ("lag", lags),
        ("unit_dim", unit_dims),
        ("env_dim", env_dims),
        ("ridge", ridges),
        ("frac", frame_fractions),
    ]
    for name, values in axes:
        _log(f"  scanning {name} ...", verbose)
        rows = []
        for v in values:
            kwargs = dict(base)
            kwargs[name] = v
            res = _floor(**kwargs)
            if res is None:
                continue
            rows.append({name: v, **res})
            star = (
                "  <<< best"
                if res["margin"] >= 0.10
                else (
                    "  <- detected" if res["margin"] > 0.0 else "  (nothing)"
                )
            )
            _log(
                f"    {name}={v!r:<8} n_pairs={res['n_pairs']:<6} "
                f"floor={res['floor']:.3f}  rho1={res['rho1']:.3f}  "
                f"MARGIN={res['margin']:+.3f}  k*={res['k_star']:.0f}{star}",
                verbose,
            )
        out["axes"][name] = rows
        if rows:
            best = max(rows, key=lambda r: r["margin"])
            _log(
                f"    best {name}={best[name]!r} (margin "
                f"{best['margin']:+.3f})",
                verbose,
            )
    return out


def _median(vals):
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    return float(np.median(vals)) if vals else float("nan")


def summarise(report):
    """Return a short text summary of a pilot report."""
    have = [d for d in report["discovered"] if "k_eff" in d]
    disc = [d["k_eff"] for d in have]
    unresolved = sum(1 for v in disc if not np.isfinite(v))
    rows = {}
    for size, row in report["controls"].items():
        for kind, vals in row.items():
            rows.setdefault(kind, []).extend(v["k_eff"] for v in vals)
    floors = [d.get("rho_null_p95") for d in have]
    disc_margin = _median([d.get("margin") for d in have])
    ctl_margin = {}
    for size, row in report["controls"].items():
        for kind, vals in row.items():
            ctl_margin.setdefault(kind, []).extend(
                v.get("margin") for v in vals
            )
    ctl_margin = {k: _median(v) for k, v in ctl_margin.items()}
    best_ctl = max(
        [v for v in ctl_margin.values() if np.isfinite(v)],
        default=float("nan"),
    )
    return {
        "discovered_margin_median": disc_margin,
        "control_margin_median": ctl_margin,
        # THE decisive comparison: are grown supports better separated from the
        # null than arbitrary residue sets of the same size?  If not, growth is
        # not finding anything special and the whole premise fails, whatever
        # the channel counts look like.
        "margin_gain_over_controls": (disc_margin - best_ctl),
        "discovered_k_eff_median": _median(disc),
        "null_floor_median": _median(floors),
        "low_power": bool(_median(floors) > 0.30) if floors else None,
        "n_unresolved": sum(
            1 for d in have if d.get("status") == "unresolved"
        ),
        "discovered_unresolved_fraction": (unresolved / len(have))
        if have
        else float("nan"),
        "control_k_eff_median": {k: _median(v) for k, v in rows.items()},
        "discovered_sizes": [
            d["size"] for d in report["discovered"] if "size" in d
        ],
        "mean_overlap_jaccard": _median(report["overlapping_pair_jaccard"]),
    }


# ---------------------------------------------------------------------------


def main(argv=None):
    """Run the pilot from the command line."""
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--domains", nargs="+", default=list(PILOT_DOMAINS))
    ap.add_argument("--out", default="analysis/results/flux_pilot")
    ap.add_argument("--temperature", default="320")
    ap.add_argument(
        "--replicates", nargs="+", default=["0", "1", "2", "3", "4"]
    )
    ap.add_argument("--local-cache", action="store_true")
    ap.add_argument("--cache-dir", default="analysis/data/.mdcath_cache")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--dt-ns", type=float, default=1.0)
    ap.add_argument("--lag", type=int, default=2)
    ap.add_argument(
        "--n-neighbours",
        type=int,
        default=8,
        help="nearest-neighbour descriptor size (ignored when "
        "--rse-cutoff is given)",
    )
    ap.add_argument(
        "--rse-cutoff",
        type=float,
        default=None,
        help="use the v3.0 RSE instead: ALL residues within this "
        "radius in Angstrom (15 is eq. 5's value). Strongly "
        "preferred -- a nearest-neighbour descriptor is "
        "sequence-local and makes every statistic track "
        "contiguity",
    )
    ap.add_argument(
        "--rse-components",
        type=int,
        default=8,
        help="principal components kept per residue RSE",
    )
    ap.add_argument("--contact-cutoff", type=float, default=8.0)
    ap.add_argument("--n-seeds", type=int, default=6)
    ap.add_argument("--max-size", type=int, default=18)
    ap.add_argument("--unit-dim", type=int, default=8)
    ap.add_argument("--env-dim", type=int, default=40)
    ap.add_argument("--ridge", type=float, default=1e-2)
    ap.add_argument("--n-null", type=int, default=12)
    ap.add_argument(
        "--size-penalty",
        type=float,
        default=0.0,
        help="reported tie-break only; selection uses the knee of "
        "the k_eff-vs-size curve, see grow_support",
    )
    ap.add_argument(
        "--exclude-termini",
        type=int,
        default=2,
        help="residues at each chain end excluded from SEEDING "
        "(growth may still absorb them); 0 to disable",
    )
    ap.add_argument(
        "--min-size",
        type=int,
        default=4,
        help="smallest support the knee rule may return",
    )
    ap.add_argument(
        "--knee-tol",
        type=float,
        default=0.15,
        help="how close to the best k_eff still counts as the knee",
    )
    ap.add_argument("--n-controls", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--decompose",
        action="store_true",
        help="self-consistent decomposition: assign EVERY residue "
        "to at least one PDU, allowing shared membership",
    )
    ap.add_argument(
        "--dec-seeds",
        type=int,
        default=24,
        help="spread seeds for the initial candidate growth",
    )
    ap.add_argument(
        "--dec-iters", type=int, default=6, help="assignment/refit iterations"
    )
    ap.add_argument(
        "--boundary-tol",
        type=float,
        default=0.15,
        help="a residue joins every unit within this fraction of "
        "its best affinity -> boundary residues",
    )
    ap.add_argument(
        "--min-seq-sep",
        type=int,
        default=2,
        help="exclude neighbours closer than this in sequence from "
        "the descriptor. i+-2 and i+-3 CA distances are set by "
        "one backbone conformation and barely move; tICA "
        "whitens by C(0), so a near-constant coordinate gets "
        "its slow drift amplified. Try 3 or 4",
    )
    ap.add_argument(
        "--control-overlap",
        type=float,
        default=0.3,
        help="resample control sets overlapping a discovered unit "
        "above this Jaccard -- an unfiltered ball control is "
        "contaminated with true positives",
    )
    ap.add_argument(
        "--merge-jaccard",
        type=float,
        default=0.5,
        help="candidate units overlapping above this are merged",
    )
    ap.add_argument(
        "--truncation",
        default="koopman",
        choices=("koopman", "pca"),
        help="koopman = slow directions (tICA/VAMP); pca is the "
        "equal-time reduction, for comparison only",
    )
    ap.add_argument(
        "--vamp",
        action="store_true",
        help="non-reversible VAMP truncation instead of tICA",
    )
    ap.add_argument(
        "--budget",
        action="store_true",
        help="also compute the per-residue inside/outside/"
        "irreducible variance budget (one extra pass)",
    )
    ap.add_argument(
        "--quiet",
        action="store_true",
        help="suppress per-step progress output",
    )
    ap.add_argument(
        "--xtc-topology",
        default=None,
        help="topology (.tpr/.gro/.pdb) for local trajectories",
    )
    ap.add_argument(
        "--xtc",
        nargs="+",
        default=None,
        help="one trajectory file per replica; each becomes its "
        "own segment so no lagged pair crosses a seam",
    )
    ap.add_argument(
        "--label",
        default=None,
        help="name for a local-trajectory system in the output",
    )
    ap.add_argument(
        "--device",
        default=None,
        help="cpu (default), cuda, mps, or auto -- GPU path via "
        "PyTorch for the covariance build and per-candidate "
        "algebra",
    )
    ap.add_argument(
        "--scan-lags",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
        help="lags (in FRAMES) for --power-scan. At 20 ps/frame, "
        "1 5 25 100 500 spans 20 ps to 10 ns",
    )
    ap.add_argument(
        "--power-scan",
        action="store_true",
        help="measure what sets the null floor on this data "
        "(one factor at a time) and exit; cheap, no growth",
    )
    ap.add_argument(
        "--synthetic",
        action="store_true",
        help="skip mdCATH entirely and run one synthetic domain",
    )
    a = ap.parse_args(argv)

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    kw = dict(
        lag=a.lag,
        n_neighbours=a.n_neighbours,
        contact_cutoff=a.contact_cutoff,
        n_seeds=a.n_seeds,
        max_size=a.max_size,
        unit_dim=a.unit_dim,
        env_dim=a.env_dim,
        ridge=a.ridge,
        n_null=a.n_null,
        size_penalty=a.size_penalty,
        n_controls=a.n_controls,
        seed=a.seed,
        verbose=not a.quiet,
        budget=a.budget,
        min_size=a.min_size,
        knee_tol=a.knee_tol,
        device=a.device,
        exclude_termini=a.exclude_termini,
        min_seq_sep=a.min_seq_sep,
        control_overlap=a.control_overlap,
        rse_cutoff=a.rse_cutoff,
        rse_components=a.rse_components,
        decompose_mode=a.decompose,
        dec_seeds=a.dec_seeds,
        dec_iters=a.dec_iters,
        boundary_tol=a.boundary_tol,
        merge_jaccard=a.merge_jaccard,
        reversible=not a.vamp,
        truncation=a.truncation,
    )

    if a.xtc:
        if not a.xtc_topology:
            ap.error("--xtc requires --xtc-topology")
        jobs = [(a.label or Path(a.xtc_topology).stem, None)]
    elif a.synthetic:
        jobs = [("synthetic", None)]
    else:
        jobs = [(d, None) for d in a.domains]
    reports, failures = [], []
    for i, (domain, _) in enumerate(jobs, 1):
        print(f"\n[{i}/{len(jobs)}] {domain}", flush=True)
        try:
            if a.xtc:
                coords, segments, meta = load_xtc_segments(
                    a.xtc_topology,
                    a.xtc,
                    stride=a.stride,
                    max_frames=a.max_frames,
                    verbose=not a.quiet,
                )
            elif a.synthetic:
                coords, segments, meta = synthetic_segments()
            else:
                coords, segments, meta = load_segments(
                    domain,
                    temperature=a.temperature,
                    replicates=a.replicates,
                    stride=a.stride,
                    max_frames=a.max_frames,
                    dt_ns=a.dt_ns,
                    remote=not a.local_cache,
                    cache_dir=a.cache_dir,
                    verbose=not a.quiet,
                )
            if a.power_scan:
                scan = power_scan(
                    coords,
                    segments,
                    lag=a.lag,
                    n_neighbours=a.n_neighbours,
                    seed=a.seed,
                    rse_cutoff=a.rse_cutoff,
                    rse_components=a.rse_components,
                    lags=tuple(a.scan_lags),
                    verbose=not a.quiet,
                )
                scan["domain"] = domain
                with (out_dir / f"{domain}_power_scan.json").open("w") as fh:
                    json.dump(_jsonable(scan), fh, indent=2, allow_nan=True)
                reports.append(
                    {
                        "domain": domain,
                        "summary": {"power_scan": True},
                        "power_scan": scan,
                    }
                )
                continue
            # resolved AFTER loading, so the trajectory's own numbering is used
            rep = run_domain(
                coords,
                segments,
                resids=_RESIDS.get("resids"),
                resnames=_RESIDS.get("resnames"),
                **kw,
            )
            rep["domain"], rep["replicates"] = domain, meta
            if "decomposition" in rep:
                d = rep["decomposition"]
                rep["summary"] = {
                    "n_units": d["n_units"],
                    "coverage": d["coverage"],
                    "n_boundary": len(d["boundary_residues"]),
                    "n_orphans": len(d["orphans"]),
                }
                reports.append(rep)
                with (out_dir / f"{domain}_flux.json").open("w") as fh:
                    json.dump(_jsonable(rep), fh, indent=2, allow_nan=True)
                try:
                    pdb = out_dir / f"{domain}_pdu.pdb"
                    pml = out_dir / f"{domain}_pdu.pml"
                    write_pdu_pdb(
                        pdb,
                        coords,
                        d,
                        resids=_RESIDS.get("resids"),
                        resnames=_RESIDS.get("resnames"),
                    )
                    write_pdu_pml(
                        pml,
                        d,
                        pdb.name,
                        label=domain,
                        resids=_RESIDS.get("resids"),
                    )
                    print(
                        f"  wrote {pml.name} (+ {pdb.name}) -- "
                        f"open with: pymol {pml.name}",
                        flush=True,
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"  could not write PyMOL files: {exc}", flush=True)
                print(
                    f"  {d['n_units']} PDUs | coverage "
                    f"{100 * d['coverage']:.0f}% | "
                    f"{len(d['boundary_residues'])} boundary residues",
                    flush=True,
                )
                continue
            rep["summary"] = summarise(rep)
            reports.append(rep)
            with (out_dir / f"{domain}_flux.json").open("w") as fh:
                json.dump(_jsonable(rep), fh, indent=2, allow_nan=True)
            s = rep["summary"]
            ctl = s["control_k_eff_median"]
            if s.get("low_power"):
                print(
                    f"  !! LOW POWER: null floor {s['null_floor_median']:.2f} "
                    f"(> 0.30). Nothing can be resolved at this frame count "
                    f"and env_dim -- add replicates, drop --max-frames, or "
                    f"lower --env-dim before reading anything below."
                )
            _cm = [
                v
                for v in s["control_margin_median"].values()
                if v is not None and np.isfinite(v)
            ]
            _best_cm = max(_cm) if _cm else float("nan")
            print(
                f"  MARGIN discovered {s['discovered_margin_median']:+.3f} "
                f"vs best control {_best_cm:+.3f} "
                f"(gain {s['margin_gain_over_controls']:+.3f})",
                flush=True,
            )
            print(
                "  discovered k_eff median = "
                f"{s['discovered_k_eff_median']:.2f} "
                f"| controls ball={ctl.get('ball', float('nan')):.2f} "
                f"random={ctl.get('random', float('nan')):.2f} "
                f"segment={ctl.get('segment', float('nan')):.2f} "
                f"| seed agreement={s['mean_overlap_jaccard']:.2f}"
            )
        except Exception as exc:  # noqa: BLE001
            failures.append({"domain": domain, "error": str(exc)})
            print(f"  FAILED: {exc}")

    summary = {
        "domains": [r["domain"] for r in reports],
        "configuration": vars(a),
        "failures": failures,
        "per_domain": [
            r["summary"] | {"domain": r["domain"]} for r in reports
        ],
    }
    with (out_dir / "summary.json").open("w") as fh:
        json.dump(_jsonable(summary), fh, indent=2, allow_nan=True)
    print(f"\nwrote {out_dir}/summary.json")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
