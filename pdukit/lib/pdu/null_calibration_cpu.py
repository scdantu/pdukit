"""Standalone, multi-core CPU reimplementation of the null-threshold
calibration logic in `interface.py`'s `calibrate_null_thresholds`.

This module is a validation harness, not a replacement for that function:
it operates on plain covariance arrays handed in by the caller (rather than
a `FluxWorkspace`), so it can be exercised and cross-checked against
`null_calibration_gpu.py` and against `interface.calibrate_null_thresholds`
before either is trusted at scale. It is not wired into
`decompose()`/`score()`/`calibrate_null_thresholds()`.

Only the reversible (`FluxWorkspace`'s default) Koopman-basis branch is
reproduced -- see the companion plan document for the non-reversible/VAMP
and PCA-truncation branches this deliberately leaves out.

RESIDUES VS FEATURE COLUMNS.  Supports are sets of RESIDUES (indices into
the N x N contact graph), but the covariances are D x D over FEATURE columns,
D = N * m with m features per residue, laid out residue-major exactly as
`FluxWorkspace` lays them out (`feats.reshape(F, N * m)`).  Every covariance
slice therefore uses the support's feature columns -- residue i owns columns
i*m .. i*m+m-1 -- as `FluxWorkspace._cols` does.  m is inferred as D // N and
must divide exactly.

PARALLELISM.  Within one calibration round the supports are independent, so
they are spread over worker processes.  The covariances (C00, C01, C11 and
every shift replica) are placed ONCE in a shared-memory block that every
worker maps without copying, and the pool is created once per calibration
run, so each task sends only its support and receives one row of numbers.
Each worker is limited to one BLAS thread so the workers do not oversubscribe
the cores between them.
"""

from __future__ import annotations

import contextlib
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import shared_memory

import numpy as np

from .interface import (
    _bootstrap_ci_rel_width,
    _bootstrap_ci_rel_width_two_way,
    _grow_ball_support,
    _inv_sqrt,
    _support_descriptors,
)

# ---------------------------------------------------------------------------
# residues -> feature columns
# ---------------------------------------------------------------------------


def _infer_n_feat(D, N):
    """Features per residue, m = D // N; D must be an exact multiple of N."""
    D, N = int(D), int(N)
    if N <= 0 or D % N != 0:
        raise ValueError(
            f"covariance dimension D={D} is not a whole multiple "
            f"of the number of residues N={N}"
        )
    return D // N


def _support_indices(support, N, m):
    """Residue indices of `support` and its feature columns, matching
    `FluxWorkspace._cols`.

    Returns (residues, S_cols, E_cols): `residues` the sorted unique residue
    indices, `S_cols` the D-space columns those residues own (residue i owns
    i*m .. i*m+m-1), `E_cols` every other column.
    """
    residues = np.asarray(sorted(set(int(i) for i in support)), dtype=int)
    if residues.size == 0 or residues.min() < 0 or residues.max() >= N:
        raise ValueError("support out of range")
    mask = np.zeros(N, dtype=bool)
    mask[residues] = True
    col = np.repeat(mask, m)
    S_cols, E_cols = np.where(col)[0], np.where(~col)[0]
    if E_cols.size == 0:
        raise ValueError("candidate covers the whole protein: no environment")
    return residues, S_cols, E_cols


# ---------------------------------------------------------------------------
# per-support linear algebra (numpy), matching FluxWorkspace's CPU path
# ---------------------------------------------------------------------------


def _koopman_basis_np(C0, C1, k, ridge):
    """Reversible-branch Koopman basis, matching `FluxWorkspace._koopman_basis`
    (reversible=True, numpy path): whiten with C0, symmetrise C1, take the
    leading-k eigenvectors of the whitened lag covariance -- with scipy's
    subset eigensolver when k < n // 4, exactly as production does, since
    for the environment block (order ~D) that is most of the per-support
    cost.
    """
    n = C0.shape[0]
    k = int(min(k, n))
    L = _inv_sqrt(C0, ridge)
    A = L @ (0.5 * (C1 + C1.T)) @ L
    A = 0.5 * (A + A.T)
    if k < n // 4:
        try:
            from scipy.linalg import eigh as _seigh

            _, U = _seigh(
                A,
                subset_by_index=[n - k, n - 1],
                check_finite=False,
                driver="evx",
            )
            U = np.ascontiguousarray(U[:, ::-1])
        except Exception:  # noqa: BLE001
            w, U = np.linalg.eigh(A)
            U = np.ascontiguousarray(U[:, np.argsort(w)[::-1][:k]])
    else:
        w, U = np.linalg.eigh(A)
        U = np.ascontiguousarray(U[:, np.argsort(w)[::-1][:k]])
    return np.ascontiguousarray(L @ U)


def _project_and_residualise_np(C00, C01, C11, S, E, unit_dim, env_dim, ridge):
    """Match `FluxWorkspace._project_and_residualise`.

    S and E are feature COLUMNS; returns the per-support quantities
    reused for every shift replica.
    """
    C00_SS = C00[np.ix_(S, S)]
    C01_SS = C01[np.ix_(S, S)]
    C00_EE = C00[np.ix_(E, E)]
    Vu = _koopman_basis_np(C00_SS, C01_SS, unit_dim, ridge)
    Ve = _koopman_basis_np(C00_EE, C01[np.ix_(E, E)], env_dim, ridge)

    Chh = Vu.T @ C00_SS @ Vu
    Cyy = Vu.T @ C11[np.ix_(S, S)] @ Vu
    Chy = Vu.T @ C01_SS @ Vu
    Czz = Ve.T @ C00_EE @ Ve

    Ih = _inv_sqrt(Chh, ridge)
    Ih = Ih @ Ih  # ridge-regularised Chh^-1
    Ryy = Cyy - Chy.T @ Ih @ Chy

    return Vu, Ve, Ih, Ryy, Chy, Czz


def _leading_null_rho_np(
    Ih, Ryy, Chy, Vu, Ve, Czz, S, E, Cs00_j, Cs01_j, ridge
):
    """Match `FluxWorkspace._leading_null_rho`/`_spectrum`.

    Returns the leading canonical correlation against one shift-null
    replica.
    """
    Chz_n = Vu.T @ Cs00_j[np.ix_(E, S)].T @ Ve
    Czy_n = Ve.T @ Cs01_j[np.ix_(E, S)] @ Vu
    Rzz_n = Czz - Chz_n.T @ Ih @ Chz_n
    M_n = (
        _inv_sqrt(Ryy, ridge)
        @ (Czy_n - Chz_n.T @ Ih @ Chy).T
        @ _inv_sqrt(Rzz_n, ridge)
    )
    return float(np.clip(np.linalg.svd(M_n, compute_uv=False), 0.0, 1.0)[0])


def _support_row(arrays, support, N, m, unit_dim, env_dim, ridge):
    """One support's basis/projection once, then every shift replica
    against it.  `arrays` holds C00, C01, C11 (D, D) and Cs00, Cs01 (J, D, D).
    """
    _, S, E = _support_indices(support, N, m)
    C00, C01, C11 = arrays["C00"], arrays["C01"], arrays["C11"]
    Cs00, Cs01 = arrays["Cs00"], arrays["Cs01"]
    Vu, Ve, Ih, Ryy, Chy, Czz = _project_and_residualise_np(
        C00, C01, C11, S, E, unit_dim, env_dim, ridge
    )
    row = np.empty(Cs00.shape[0], dtype=np.float64)
    for j in range(Cs00.shape[0]):
        row[j] = _leading_null_rho_np(
            Ih, Ryy, Chy, Vu, Ve, Czz, S, E, Cs00[j], Cs01[j], ridge
        )
    return row


# ---------------------------------------------------------------------------
# shared memory + worker pool
# ---------------------------------------------------------------------------

_ARRAY_NAMES = ("C00", "C01", "C11", "Cs00", "Cs01")
_BLAS_ENV = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

# Set in each worker by `_worker_init`; read by `_worker_row`.
_WORKER = {}


def _worker_init(shm_name, layout, params):
    """Run once per worker process: map the shared block, cap BLAS threads."""
    try:
        from threadpoolctl import threadpool_limits

        _WORKER["_limits"] = threadpool_limits(limits=1)
    except Exception:  # noqa: BLE001
        pass  # env vars set by the parent around pool start cover this
    shm = shared_memory.SharedMemory(name=shm_name)
    _WORKER["_shm"] = shm  # keep the mapping alive in the worker
    _WORKER["arrays"] = {
        name: np.ndarray(shape, dtype=np.float64, buffer=shm.buf, offset=off)
        for name, (off, shape) in layout.items()
    }
    _WORKER.update(params)


def _worker_ping(_):
    return os.getpid()


def _worker_row(support):
    w = _WORKER
    return _support_row(
        w["arrays"],
        support,
        w["N"],
        w["m"],
        w["unit_dim"],
        w["env_dim"],
        w["ridge"],
    )


def _available_cores():
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # not on Linux
        return os.cpu_count() or 1


class _CpuEvaluator:
    """Holds the covariances and (for n_workers > 1) one process pool for the
    lifetime of a calibration run.  Use as a context manager.
    """

    def __init__(
        self,
        C00,
        C01,
        C11,
        Cs00,
        Cs01,
        *,
        N,
        unit_dim,
        env_dim,
        ridge,
        n_workers,
    ):
        arrays = {
            "C00": np.asarray(C00, dtype=np.float64),
            "C01": np.asarray(C01, dtype=np.float64),
            "C11": np.asarray(C11, dtype=np.float64),
            "Cs00": np.stack([np.asarray(c, dtype=np.float64) for c in Cs00]),
            "Cs01": np.stack([np.asarray(c, dtype=np.float64) for c in Cs01]),
        }
        self.N = int(N)
        self.m = _infer_n_feat(arrays["C00"].shape[0], self.N)
        self.params = {
            "N": self.N,
            "m": self.m,
            "unit_dim": int(unit_dim),
            "env_dim": int(env_dim),
            "ridge": float(ridge),
        }
        self.n_shift = arrays["Cs00"].shape[0]
        self.n_workers = max(1, min(int(n_workers), _available_cores()))
        self._arrays, self._shm, self._ex, self._saved_env = (
            arrays,
            None,
            None,
            {},
        )
        self.start_seconds = 0.0
        if self.n_workers > 1:
            t = time.perf_counter()
            self._start_pool(arrays)
            self.start_seconds = time.perf_counter() - t

    def _start_pool(self, arrays):
        layout, off = {}, 0
        for name in _ARRAY_NAMES:
            a = arrays[name]
            layout[name] = (off, a.shape)
            off += a.nbytes
        self._shm = shared_memory.SharedMemory(create=True, size=max(off, 1))
        for name in _ARRAY_NAMES:
            o, shape = layout[name]
            np.ndarray(
                shape, dtype=np.float64, buffer=self._shm.buf, offset=o
            )[...] = arrays[name]
        # Workers inherit the environment when they start; one BLAS thread
        # each.  The parent's own BLAS is already initialised, so this does
        # not change it; the old values are restored on close().
        for k in _BLAS_ENV:
            self._saved_env[k] = os.environ.get(k)
            os.environ[k] = "1"
        # 'forkserver' (or 'spawn'), never 'fork': the caller may already have
        # CUDA/OpenMP state (e.g. a torch-backed workspace) that is not
        # fork-safe.  The shared block makes the cleaner start cheap.
        methods = mp.get_all_start_methods()
        ctx = mp.get_context(
            "forkserver" if "forkserver" in methods else "spawn"
        )
        self._ex = ProcessPoolExecutor(
            max_workers=self.n_workers,
            mp_context=ctx,
            initializer=_worker_init,
            initargs=(self._shm.name, layout, self.params),
        )
        # Start every worker now (one quick task each, submitted together so
        # none is idle yet), so start-up is paid here and reported as
        # `start_seconds` rather than landing inside the first size's timing.
        list(self._ex.map(_worker_ping, range(self.n_workers)))

    def rows(self, supports):
        supports = [list(map(int, s)) for s in supports]
        if self._ex is None:
            p = self.params
            out = [
                _support_row(
                    self._arrays,
                    s,
                    p["N"],
                    p["m"],
                    p["unit_dim"],
                    p["env_dim"],
                    p["ridge"],
                )
                for s in supports
            ]
        else:
            out = list(self._ex.map(_worker_row, supports))
        if not out:
            return np.empty((0, self.n_shift), dtype=np.float64)
        return np.stack(out, axis=0)

    def close(self):
        if self._ex is not None:
            self._ex.shutdown(wait=True)
            self._ex = None
        if self._shm is not None:
            with contextlib.suppress(Exception):
                self._shm.close()
                self._shm.unlink()
            self._shm = None
        for k, v in self._saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        self._saved_env = {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def leading_null_rho_batch_cpu(
    C00,
    C01,
    C11,
    Cs00,
    Cs01,
    supports,
    *,
    n_residues,
    unit_dim,
    env_dim,
    ridge=1e-2,
    n_workers=None,
):
    """Leading null-correlation for every (support, shift) pair.

    Returns a numpy array of shape (len(supports), len(Cs00)): row i,
    column j is the leading canonical correlation of supports[i] against
    shift replica j (Cs00[j], Cs01[j]).

    `supports` are lists of RESIDUE indices in range(n_residues); the
    covariances are (D, D) over feature columns with D = n_residues * m.
    `C11` is the one-lag-later equal-time covariance (`FluxWorkspace.C11`),
    needed for Cyy/Ryy in the projection step.

    Each support's basis + projection + residualisation is computed once
    and reused for every shift evaluated against it.  Parallelised across
    supports (n_workers processes, default min(len(supports), available
    cores)).  For repeated calls -- a calibration run -- use
    `calibrate_null_thresholds_cpu`, which keeps one pool and one shared
    copy of the covariances for the whole run instead of one per call.
    """
    if n_workers is None:
        n_workers = max(1, min(len(supports), _available_cores()))
    with _CpuEvaluator(
        C00,
        C01,
        C11,
        Cs00,
        Cs01,
        N=n_residues,
        unit_dim=unit_dim,
        env_dim=env_dim,
        ridge=ridge,
        n_workers=n_workers,
    ) as ev:
        return ev.rows(supports)


def calibrate_null_thresholds_cpu(
    C00,
    C01,
    C11,
    adjacency,
    sizes,
    Cs00,
    Cs01,
    *,
    unit_dim,
    env_dim,
    ridge=1e-2,
    target_rel_width=0.02,
    batch_supports=10,
    min_supports=50,
    max_pool_size=5000,
    seed=0,
    n_workers=None,
    on_log=None,
):
    """Calibrate null thresholds on CPU processes.

    Reimplements interface.py's calibrate_null_thresholds, operating on
    caller-supplied covariance arrays instead of a FluxWorkspace.  Same
    support sampling, same random stream, same stopping rule, same return
    shape:

      {"theta": {k: float},
       "diagnostics": {k: {n_pairs, n_supports, rel_ci_width,
                           rel_ci_width_naive, converged, stopped_by,
                           elapsed_seconds, contact_density, effective_dim,
                           per_support}},
       "config": {unit_dim, env_dim, ridge, n_shifts, target_rel_width,
                  batch_supports, min_supports, max_pool_size, seed,
                  n_workers, pool_start_seconds},

    The stopping rule is interface.calibrate_null_thresholds's: a
    support-level bootstrap (whole supports resampled) must be narrower than
    `target_rel_width`, after at least `min_supports` supports.
       "elapsed_seconds": float}

    `unit_dim`/`env_dim` are required and must match what the table will be
    scored with.  `n_workers` defaults to `batch_supports`: one calibration
    round only ever has that many independent supports, so more workers
    would sit idle.  `elapsed_seconds` (total and per size) is wall-clock
    time from time.perf_counter(); the total includes pool start-up, which is
    also reported on its own as config["pool_start_seconds"] and is NOT in
    the per-size times -- a first-look timing, not a benchmark.
    """
    log = on_log if on_log is not None else (lambda *_: None)
    t_start = time.perf_counter()
    rng = np.random.default_rng(seed)
    adjacency = np.asarray(adjacency, dtype=bool)
    N = adjacency.shape[0]
    n_shift_given = len(Cs00)
    if int(min_supports) * n_shift_given > int(max_pool_size):
        raise ValueError(
            f"min_supports={min_supports} x {n_shift_given} shifts exceeds "
            f"max_pool_size={max_pool_size}: no size could ever be declared "
            f"converged"
        )
    if n_workers is None:
        n_workers = batch_supports

    def sample_support(k, tries=40):
        s = _grow_ball_support(N, k, adjacency, rng)
        for _ in range(tries - 1):
            if len(s) == k:
                break
            s = _grow_ball_support(N, k, adjacency, rng)
        return s

    theta, diagnostics = {}, {}
    with _CpuEvaluator(
        C00,
        C01,
        C11,
        Cs00,
        Cs01,
        N=N,
        unit_dim=unit_dim,
        env_dim=env_dim,
        ridge=ridge,
        n_workers=n_workers,
    ) as ev:
        C00_np = ev._arrays["C00"]
        for k in sizes:
            k = int(k)
            t_size = time.perf_counter()
            pool, densities, eff_dims, per_support = [], [], [], []
            support_rows = []
            n_supports = 0
            rel_width, converged, stopped_by = float("inf"), False, None

            while True:
                batch = [sample_support(k) for _ in range(batch_supports)]
                rows = ev.rows(batch)

                for support, row in zip(batch, rows):
                    residues, S, _ = _support_indices(support, N, ev.m)
                    density, eff_dim = _support_descriptors(
                        residues, S, adjacency, C00_np
                    )
                    densities.append(density)
                    eff_dims.append(eff_dim)
                    n_supports += 1
                    pool.extend(row.tolist())
                    support_rows.append(row)
                    per_support.append(
                        {
                            "support": [int(i) for i in residues],
                            "leading_rho": [float(v) for v in row],
                            "theta_S": float(np.percentile(row, 95)),
                            "contact_density": float(density),
                            "effective_dim": float(eff_dim),
                        }
                    )

                # The support-level bootstrap is only informative once there
                # are enough supports, so it is not run before min_supports.
                if n_supports >= min_supports:
                    _, rel_width = _bootstrap_ci_rel_width_two_way(
                        support_rows, rng=rng
                    )
                if n_supports >= min_supports and rel_width < target_rel_width:
                    converged, stopped_by = True, "converged"
                    break
                if len(pool) >= max_pool_size:
                    stopped_by = "max_pool_size"
                    log(
                        f"calibrate_null_thresholds_cpu: size {k} hit "
                        f"max_pool_size ({max_pool_size}) before converging "
                        f"(rel CI width {rel_width:.4f} >= {target_rel_width})"
                    )
                    break

            _, rel_width_naive = _bootstrap_ci_rel_width(pool, rng=rng)
            theta[k] = float(np.percentile(pool, 95))
            diagnostics[k] = {
                "n_pairs": len(pool),
                "n_supports": n_supports,
                "rel_ci_width": float(rel_width),
                "rel_ci_width_naive": float(rel_width_naive),
                "converged": bool(converged),
                "stopped_by": stopped_by,
                "elapsed_seconds": time.perf_counter() - t_size,
                "contact_density": densities,
                "effective_dim": eff_dims,
                "per_support": per_support,
            }
            log(
                f"calibrate_null_thresholds_cpu: size {k} "
                f"theta={theta[k]:.4f} "
                f"n_pairs={len(pool)} n_supports={n_supports} "
                f"rel_ci_width={rel_width:.4f} (naive {rel_width_naive:.4f}) "
                f"converged={converged} "
                f"({diagnostics[k]['elapsed_seconds']:.1f}s)"
            )
        n_workers_used = ev.n_workers
        n_shifts = ev.n_shift
        pool_start = ev.start_seconds

    config = {
        "unit_dim": int(unit_dim),
        "env_dim": int(env_dim),
        "ridge": float(ridge),
        "n_shifts": int(n_shifts),
        "target_rel_width": float(target_rel_width),
        "batch_supports": int(batch_supports),
        "min_supports": int(min_supports),
        "max_pool_size": int(max_pool_size),
        "seed": int(seed),
        "bootstrap": "two_way",
        "n_workers": int(n_workers_used),
        "pool_start_seconds": float(pool_start),
    }
    return {
        "theta": theta,
        "diagnostics": diagnostics,
        "config": config,
        "elapsed_seconds": time.perf_counter() - t_start,
    }
