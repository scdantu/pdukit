"""The reference scoring kernel: canonical correlations and the flux spectrum
of a support set, as plain functions with no state.
"""

from __future__ import annotations

import numpy as np


def _inv_sqrt(C, ridge):
    """Ridge-regularised inverse square root of a symmetric PSD matrix."""
    C = 0.5 * (C + C.T)
    p = C.shape[0]
    scale = max(np.trace(C) / max(p, 1), 1e-30)
    C = C + ridge * scale * np.eye(p)
    w, V = np.linalg.eigh(C)
    w = np.maximum(w, 1e-14 * max(w.max(), 1e-30))
    return (V / np.sqrt(w)) @ V.T


def _center(A):
    return A - A.mean(axis=0, keepdims=True)


def _residualise(Y, H, ridge):
    """Y - E[Y | H] under a ridge linear model (both already centred)."""
    if H.shape[1] == 0:
        return Y
    Chh = H.T @ H / max(len(H), 1)
    Chy = H.T @ Y / max(len(H), 1)
    p = Chh.shape[0]
    scale = max(np.trace(Chh) / max(p, 1), 1e-30)
    beta = np.linalg.solve(Chh + ridge * scale * np.eye(p), Chy)
    return Y - H @ beta


def _pca(A, n_components, ridge=0.0):
    """Return truncated PCA scores (n, k) of centred A, k <= n_comp."""
    if n_components is None or n_components >= A.shape[1]:
        return A
    # economy SVD is enough; we only need the leading directions
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    k = int(min(n_components, s.size))
    return A @ Vt[:k].T


def _canonical_correlations(Ytil, Ztil, ridge):
    n = Ytil.shape[0]
    Cyy = Ytil.T @ Ytil / n
    Czz = Ztil.T @ Ztil / n
    Cyz = Ytil.T @ Ztil / n
    M = _inv_sqrt(Cyy, ridge) @ Cyz @ _inv_sqrt(Czz, ridge)
    s = np.linalg.svd(M, compute_uv=False)
    return np.clip(s, 0.0, 1.0)


def flux_spectrum(
    unit,
    env,
    lag=1,
    ridge=1e-2,
    unit_dim=None,
    env_dim=40,
    n_null=12,
    null_min_shift=None,
    seed=0,
    segments=None,
):
    """Canonical-correlation spectrum of the environment's back-action.

    Parameters
    ----------
    unit : (F, p) time series of the candidate's local features
    env  : (F, q) time series of the rest of the protein's local features
    lag  : integer lag in frames (tau)
    ridge: relative ridge added to every covariance before inversion
    unit_dim / env_dim : PCA truncation applied BEFORE the spectrum.  The
        environment truncation is a stated limitation: channels living outside
        the leading env_dim directions cannot be found.  Raise it when frames
        allow.
    n_null : number of circular time-shift replicas used to build the null.
    segments : optional list of (start, stop) row ranges that must not be
        crossed when forming lagged pairs (e.g. replicate boundaries).  If None
        the whole array is one segment.

    Returns
    -------
    dict with rho, null threshold, k_star, k_eff, coupling_total, curve.

    Notes
    -----
    The null is a circular time shift of the environment block by a large
    offset.  It preserves each block's own marginal covariance and
    autocorrelation but destroys the cross-relation, so it gives the level of
    canonical correlation expected from finite samples and high dimension
    alone.  This matters: with p, q comparable to n, canonical correlations are
    strongly biased upward and a raw spectrum would look like many channels
    everywhere.

    """
    rng = np.random.default_rng(seed)
    U = np.asarray(unit, dtype=np.float64)
    E = np.asarray(env, dtype=np.float64)
    if U.shape[0] != E.shape[0]:
        raise ValueError("unit and env must have the same number of frames")
    F = U.shape[0]
    lag = int(lag)
    if lag < 1:
        raise ValueError("lag must be >= 1")

    if segments is None:
        segments = [(0, F)]
    rows0, rows1 = [], []
    for a, b in segments:
        if b - a <= lag:
            continue
        rows0.append(np.arange(a, b - lag))
        rows1.append(np.arange(a + lag, b))
    if not rows0:
        raise ValueError("no usable lagged pairs; lag too large for segments")
    rows0 = np.concatenate(rows0)
    rows1 = np.concatenate(rows1)
    n = rows0.size

    Uc = _center(U)
    Ec = _center(E)
    Uc = _pca(Uc, unit_dim)
    Ec = _pca(Ec, env_dim)

    H = Uc[rows0]  # unit present
    Y = Uc[rows1]  # unit future
    Z = Ec[rows0]  # environment present

    H, Y, Z = _center(H), _center(Y), _center(Z)
    Ytil = _center(_residualise(Y, H, ridge))
    Ztil = _center(_residualise(Z, H, ridge))

    rho = _canonical_correlations(Ytil, Ztil, ridge)

    # --- shift null -------------------------------------------------------
    if null_min_shift is None:
        null_min_shift = max(10 * lag, F // 10, 5)
    null_leading = []
    for _ in range(int(n_null)):
        shift = int(
            rng.integers(
                null_min_shift, max(null_min_shift + 1, F - null_min_shift)
            )
        )
        Zs = np.roll(Ec, shift, axis=0)[rows0]
        Zs = _center(_residualise(_center(Zs), H, ridge))
        null_leading.append(_canonical_correlations(Ytil, Zs, ridge)[0])
    thr = float(np.percentile(null_leading, 95)) if null_leading else 0.0

    rho2 = rho**2
    total = float(rho2.sum())
    k_star = int(np.sum(rho > thr))

    # Null-corrected spectrum.  Subtracting the null floor from rho^2 stops
    # finite-sample noise channels from inflating the channel count: with
    # p, q comparable to n every canonical correlation is biased upward, so an
    # uncorrected participation ratio reports many channels even for a fully
    # decoupled block.  Channels at or below the floor contribute nothing.
    rho2_corr = np.maximum(rho2 - thr**2, 0.0)
    total_corr = float(rho2_corr.sum())
    if total_corr > 0:
        k_eff = float(total_corr**2 / np.sum(rho2_corr**2))
        curve = (np.cumsum(rho2_corr) / total_corr).tolist()
    else:
        # No coupling resolved above the null.  This is not a failure: it is
        # the most closed a candidate can look at this resolution.  k_eff is
        # undefined, and callers that need a search objective should treat it
        # as zero channels (see grow_support).
        k_eff = float("nan")
        curve = []

    return {
        "n_pairs": int(n),
        "lag": lag,
        "rho": rho.tolist(),
        "rho_null_p95": thr,
        "resolved": bool(k_star > 0),
        # Margin of the leading canonical correlation over the null floor.
        # THIS, not the floor's absolute value, is what says whether anything
        # was detected: a high floor is harmless if the signal clears it.
        "margin": float(rho[0] - thr) if rho.size else 0.0,
        # Detection is margin-based.  An earlier version flagged any floor
        # above 0.30 as a power failure, which was wrong: on 5bqpD00 the floor
        # sat at 0.36 while the leading rho sat consistently above it at every
        # parameter setting, and halving the data collapsed the margin to zero
        # -- the signature of a real, weak signal, not of noise.
        "low_power": bool(rho.size == 0 or (rho[0] - thr) <= 0.0),
        "k_star": k_star,
        "k_eff": k_eff,
        "coupling_total": total,
        "coupling_resolved": total_corr,
        "delta_int_curve": curve,
        "unit_dim": int(Uc.shape[1]),
        "env_dim": int(Ec.shape[1]),
    }


def _split(feats, support):
    """(F, N, m) residue features -> unit block and environment block."""
    F, N, m = feats.shape
    support = np.asarray(sorted(set(int(i) for i in support)), dtype=int)
    if support.size == 0 or support.min() < 0 or support.max() >= N:
        raise ValueError("support out of range")
    mask = np.zeros(N, dtype=bool)
    mask[support] = True
    unit = feats[:, mask, :].reshape(F, -1)
    env = feats[:, ~mask, :].reshape(F, -1)
    return support, unit, env


def score_support(feats, support, workspace=None, **kw):
    """Flux spectrum for one candidate residue set.

    With ``workspace`` (a `FluxWorkspace`) the cost is independent of the
    number of frames; ``lag``, ``ridge``, ``n_null``, ``seed`` and ``segments``
    are then fixed by the workspace and ignored here.
    """
    if workspace is not None:
        return workspace.score(
            support,
            unit_dim=kw.get("unit_dim", 8),
            env_dim=kw.get("env_dim", 40),
        )
    support, unit, env = _split(feats, support)
    out = flux_spectrum(unit, env, **kw)
    out["support"] = support.tolist()
    out["size"] = int(support.size)
    return out
