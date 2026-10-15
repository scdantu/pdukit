"""Interface-cost / flux-dimension diagnostics for candidate PDUs (v3.0 line).

WHAT THIS MODULE IS FOR
-----------------------
The v3.0 reference asks a different question from the established extractor.
The extractor builds candidates by rotating global tICA modes into a localised
basis.  v3.0 Section 10 step 2 explicitly forbids that route for *discovery*
("Do not obtain them by rotating global slow modes").  This module therefore
never touches a global mode.  It starts from residues and asks, for a proposed
residue set S:

    given the unit's own present state, how many INDEPENDENT CHANNELS does the
    rest of the protein need in order to say anything further about the unit's
    future?

That number -- not the strength of the coupling -- is the proposed defining
property of a PDU.  A unit may be strongly coupled to its environment and still
be a unit, provided the coupling arrives through few channels.  Requiring the
coupling to be *small* instead would reduce the criterion to near-independence,
i.e. to the iVAMPnets assumption that v3.0 deliberately drops.

RELATION TO THE OPERATOR PICTURE
--------------------------------
Section 3 of the reference splits the transfer operator at lag tau into

    A = P K P,  B = P K R,  C = R K P,  D = R K R

with P the projection onto resolved (unit) observables.  The environment's
back-action on the unit is the self-energy  B (zI - D)^{-1} C.  Lyna's
boundary-flux proposal is ||(I - P) L P||, i.e. the NORM of the cross block.

In the linear-Gaussian case the two pictures coincide exactly, and the
rank-resolved version of that norm is computable without ever forming a large
operator: the singular values of the cross block, after whitening, ARE the
canonical correlations between

    Ytil = unit future  residualised on unit present
    Ztil = environment present residualised on unit present

So `flux_spectrum` below returns the singular-value spectrum of the boundary
flux.  ||.||  is its first singular value; what we report instead is how many
singular values there are.

WHAT THIS MODULE IS NOT
-----------------------
* Not a validity threshold.  Every number here is a descriptor with a stated
  null; nothing is pass/fail.
* Not causal.  Canonical correlations are observational.  A channel is a
  predictive channel, not a demonstrated mechanical force.
* Not nonlinear.  The estimator is linear-Gaussian by choice, because it is
  exactly solvable, has an analytic null, and cannot silently absorb the whole
  protein into a flexible encoder.  A nonlinear version is a later step and
  must be charged for its capacity.
* Not a slow-mode estimator.  At mdCATH's 1 ns spacing nothing faster than
  ~10 ns is resolved, and rare transitions are not sampled.  Everything here is
  a statement at the declared resolution R = (dt, tau, features).

Numpy only, deliberately: no scipy, no torch, matching the repo's existing
policy of not depending on mdtraj/biopython in the extract layer.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np

__all__ = [
    "residue_local_features",
    "rse_features",
    "detect_length_unit",
    "median_distance_matrix",
    "contact_graph",
    "residue_budget",
    "flux_spectrum",
    "FluxResult",
    "FluxWorkspace",
    "score_support",
    "grow_support",
    "control_supports",
    "calibrate_null_thresholds",
    "decompose",
]


# ---------------------------------------------------------------------------
# 1. Residue-supported local features.  No global alignment anywhere.
# ---------------------------------------------------------------------------


def residue_local_features(
    ca_coords,
    n_neighbours=8,
    min_seq_sep=2,
    dmed=None,
    max_frames=200,
    dtype=np.float32,
    chunk=4096,
    on_progress=None,
):
    """Per-residue local descriptors from CA coordinates.

    Residue i is described by its distances to the ``n_neighbours`` residues
    nearest to it by MEDIAN distance across frames (see
    `median_distance_matrix`), excluding sequence neighbours closer than
    ``min_seq_sep`` which are trivially rigid.

    Support honesty: the descriptor of residue i depends on i and on its fixed
    neighbour list, so its support is i's RSE, not the whole protein.
    Distances are internal, so no whole-protein alignment is used or needed
    (v3.0 sec 1.2).

    MEMORY.  Built frame-chunk by frame-chunk, and the source positions are
    broadcast rather than tiled.  The obvious implementation materialises three
    (F, N, m, 3) float64 arrays at once, which on 114,000 frames of a
    164-residue protein peaks at 7.5 GB and simply fails.  Chunked and in
    float32 the same job holds a few tens of MB at a time; the output is
    float32 by default because these are distances in Angstrom, where float32
    carries ~7 significant figures and the covariances are accumulated in
    float64 downstream anyway.

    Parameters
    ----------
    ca_coords : (F, N, 3)
    dmed      : precomputed median distance matrix, or None to compute it
    chunk     : frames processed at once
    on_progress : optional callback(done_frames, total_frames)

    Returns
    -------
    feats     : (F, N, n_neighbours)
    neighbours: (N, n_neighbours) int   -- the support of each descriptor

    """
    X = np.asarray(ca_coords)
    if X.ndim != 3 or X.shape[2] != 3:
        raise ValueError("ca_coords must be (F, N, 3)")
    F, N, _ = X.shape
    m = int(n_neighbours)
    if m < 1 or m >= N:
        raise ValueError(f"n_neighbours must be in [1, {N - 1}]")

    d_ref = (
        median_distance_matrix(X, max_frames=max_frames)
        if dmed is None
        else np.array(dmed, dtype=np.float64, copy=True)
    )
    idx = np.arange(N)
    sep = np.abs(idx[:, None] - idx[None, :])
    d_ref = np.where(sep < max(1, min_seq_sep), np.inf, d_ref)
    np.fill_diagonal(d_ref, np.inf)
    neighbours = np.argsort(d_ref, axis=1)[:, :m]  # (N, m)

    feats = np.empty((F, N, m), dtype=dtype)
    step = max(1, int(chunk))
    for a in range(0, F, step):
        b = min(a + step, F)
        Xc = np.asarray(X[a:b], dtype=np.float32)  # (c, N, 3)
        # (c, N, 1, 3) broadcast against (c, N, m, 3): no tiling of the source
        diff = Xc[:, :, None, :] - Xc[:, neighbours, :]
        feats[a:b] = np.sqrt(np.einsum("cnmk,cnmk->cnm", diff, diff))
        if on_progress is not None:
            on_progress(b, F)
    return feats, neighbours


def median_distance_matrix(ca_coords, max_frames=200, seed=0):
    """(N, N) median pairwise CA distance taken ACROSS frames.

    Why not the distance matrix of the mean structure?  Because mdCATH
    trajectories are neither superposed nor guaranteed unwrapped.  Averaging
    coordinates that rotate, translate or jump across a periodic boundary
    produces a reference structure that never existed, and both neighbour
    selection and the contact graph then rest on it.  The observed symptom on
    5bqpD00 is a median consecutive-CA spacing of 4.98 A where the polypeptide
    value is 3.8 A.

    Distances are internal, so taking them per frame and reducing with a MEDIAN
    across frames is invariant to rotation and translation, and robust to a
    minority of boundary-corrupted frames.  Frames are subsampled evenly to
    keep the (frames, N, N) intermediate small.
    """
    X = np.asarray(ca_coords)
    F, N, _ = X.shape
    if F > max_frames:
        idx = np.linspace(0, F - 1, int(max_frames)).astype(int)
        X = X[idx]
    X = np.asarray(X, dtype=np.float32)
    d = np.empty((X.shape[0], N, N), dtype=np.float32)
    for k in range(X.shape[0]):  # (N,N,3) at a time, not (F,N,N,3)
        diff = X[k][:, None, :] - X[k][None, :, :]
        d[k] = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))
    return np.median(d, axis=0).astype(np.float64)


def reference_spacing(dmed):
    """Median distance between residues adjacent in sequence, from `dmed`."""
    if dmed.shape[0] < 2:
        return float("nan")
    return float(np.median(np.diag(dmed, k=1)))


def detect_length_unit(ca_coords, dmed=None, max_frames=200):
    """Infer whether CA coordinates are in Angstrom or nanometre.

    Consecutive CA atoms sit 3.8 A = 0.38 nm apart, one of the most reliable
    numbers in structural biology, so the median over the chain identifies the
    unit unambiguously.  The spacing is taken from `median_distance_matrix`,
    not from a mean structure -- see that function for why.

    This is not defensive padding.  mdCATH is GROMACS-derived and stores
    nanometres, while `hdf5_layout.py` documents Angstrom as an assumption, so
    every absolute-distance cutoff in the package is otherwise wrong by 10x.
    The symptom is a contact graph in which every residue contacts every other.

    Returns a dict with 'unit', the observed 'median_ca_spacing', the
    'to_angstrom' factor, and 'spacing_angstrom' -- the spacing expressed in
    Angstrom.  If that last value is far from 3.8, the atom selection or the
    trajectory is suspect and 'suspect' is set.
    """
    X = np.asarray(ca_coords, dtype=np.float64)
    if dmed is None:
        dmed = median_distance_matrix(X, max_frames=max_frames)
    step = reference_spacing(dmed)
    if not np.isfinite(step):
        return {
            "unit": "unknown",
            "median_ca_spacing": step,
            "to_angstrom": 1.0,
            "spacing_angstrom": step,
            "suspect": True,
        }
    if 2.5 <= step <= 5.5:
        unit, to_ang = "angstrom", 1.0
    elif 0.25 <= step <= 0.55:
        unit, to_ang = "nanometre", 10.0
    else:
        unit, to_ang = "unknown", 1.0
    ang = step * to_ang
    return {
        "unit": unit,
        "median_ca_spacing": step,
        "to_angstrom": to_ang,
        "spacing_angstrom": ang,
        "suspect": bool(unit == "unknown" or not (3.2 <= ang <= 4.4)),
    }


def contact_graph(
    ca_coords, cutoff=8.0, dmed=None, units="auto", max_frames=200
):
    """Boolean (N, N) contact adjacency from the MEDIAN inter-residue distance.

    ``cutoff`` is always in ANGSTROM.  With ``units='auto'`` the coordinates'
    own unit is detected and the cutoff converted, so the same call is correct
    for Angstrom and nanometre inputs.  Pass 'angstrom' or 'nanometre' to force
    it, or 'none' to compare raw numbers.

    Growth follows SPATIAL neighbours, not sequence neighbours, because v3.0 D1
    allows supports discontinuous in sequence.
    """
    X = np.asarray(ca_coords, dtype=np.float64)
    if dmed is None:
        dmed = median_distance_matrix(X, max_frames=max_frames)
    if units == "auto":
        scale = 1.0 / detect_length_unit(X, dmed=dmed)["to_angstrom"]
    elif units == "nanometre":
        scale = 0.1
    elif units in ("angstrom", "none"):
        scale = 1.0
    else:
        raise ValueError(f"unknown units {units!r}")
    adj = dmed <= float(cutoff) * scale
    np.fill_diagonal(adj, False)
    return adj


# ---------------------------------------------------------------------------
# 2. Linear algebra helpers
# ---------------------------------------------------------------------------


def _center(A):
    return A - A.mean(axis=0, keepdims=True)


def _inv_sqrt(C, ridge):
    """Ridge-regularised inverse square root of a symmetric PSD matrix."""
    C = 0.5 * (C + C.T)
    p = C.shape[0]
    scale = max(np.trace(C) / max(p, 1), 1e-30)
    C = C + ridge * scale * np.eye(p)
    w, V = np.linalg.eigh(C)
    w = np.maximum(w, 1e-14 * max(w.max(), 1e-30))
    return (V / np.sqrt(w)) @ V.T


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


# ---------------------------------------------------------------------------
# 3. The flux spectrum
# ---------------------------------------------------------------------------


@dataclass
class FluxResult:
    """Boundary-flux spectrum for one candidate support."""

    support: list  # residue indices
    size: int
    n_pairs: int  # usable lagged pairs
    lag: int
    rho: list  # canonical correlations, descending
    rho_null_p95: float  # 95th pct of leading rho under shift null
    k_star: int  # channels above the null
    k_eff: float  # participation ratio of rho^2 (continuous)
    coupling_total: float  # sum rho^2 -- the MAGNITUDE of the flux
    delta_int_curve: list  # cumulative fraction of flux by channel
    unit_dim: int
    env_dim: int

    def to_dict(self):
        """Return the result as a plain dictionary."""
        return asdict(self)


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


# ---------------------------------------------------------------------------
# 4. Scoring a support, growing one, and matched controls
# ---------------------------------------------------------------------------


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


def _objective(result, size_penalty=0.0):
    """Channel count for a candidate whose coupling was actually resolved.

    Returns +inf when nothing was resolved above the null.  That case is
    "untested at this resolution", NOT "perfectly closed": with a high null
    floor every candidate looks channel-free, and treating that as zero
    channels lets the trivial solutions win.  v3.0 is explicit that
    insufficient data must yield uncertainty rather than a claim.

    ``size_penalty`` defaults to zero and is a reported tie-break only.  It is
    deliberately NOT what picks the answer: k_eff has a floor of 1 -- once a
    candidate is down to a single channel no addition can lower it -- so any
    positive size charge makes the SMALLEST support win, and growth collapses
    to one or two residues.  Selection uses the knee of the k_eff-versus-size
    curve instead; see `grow_support`.
    """
    if not result.get("resolved", False):
        return float("inf")
    k = result.get("k_eff", float("nan"))
    if not np.isfinite(k):
        return float("inf")
    return float(k) + float(size_penalty) * float(result.get("size", 0))


def _explore_key(result):
    """Ordering for candidates that resolved nothing: head towards resolvable.

    Ratio of the leading canonical correlation to the null floor.  It carries
    no interpretation on its own -- it exists so growth keeps moving instead of
    stalling on the first unresolved candidate it meets.
    """
    thr = float(result.get("rho_null_p95", 0.0)) or 1e-9
    rho = result.get("rho") or [0.0]
    return -(float(rho[0]) / thr)


def grow_support(
    feats,
    seed_residue,
    adjacency,
    max_size=20,
    lag=1,
    min_size=4,
    knee_tol=0.15,
    size_penalty=0.0,
    on_step=None,
    on_candidate=None,
    **kw,
):
    """Grow a support from a seed and return the knee of its k_eff curve.

    At each step every residue adjacent to the current support is scored and
    the best is added; growth runs all the way to ``max_size`` so the whole
    k_eff-versus-size curve is recorded.  Candidates whose coupling resolved
    above the null are preferred, ordered by ``_objective``; when none
    resolved, growth follows ``_explore_key`` towards a resolvable candidate
    rather than stopping.

    SELECTION IS THE KNEE, not an argmin.  k_eff cannot fall below 1, so a
    scalar objective with any size charge is minimised by the smallest support
    and growth degenerates to one or two residues -- which is what happened on
    5bqpD00.  Instead the returned support is the SMALLEST one, of at least
    ``min_size`` residues, whose k_eff is within ``knee_tol`` of the best value
    on the curve: the point at which adding residues stops buying fewer
    channels.  The full curve is returned in ``growth_trace`` and is the real
    deliverable -- plot it against the matched controls.

    ``on_step`` is called after every accepted addition; ``on_candidate(done,
    total)`` while a frontier is being scored.

    Growth follows the contact graph, so supports may be discontinuous in
    sequence (v3.0 D1).  The result is a LOCAL minimum; different seeds may
    converge elsewhere, which is the intended output -- v3.0 sec 7.1 concedes
    the decomposition is not generally unique, so the deliverable is the family
    of local minima and their agreement, not a single partition.
    """
    adjacency = np.asarray(adjacency, dtype=bool)
    support = [int(seed_residue)]
    first = score_support(feats, support, lag=lag, **kw)
    history = [(list(support), _objective(first, size_penalty), first)]
    if on_step is not None:
        on_step(
            {
                "size": 1,
                "added": int(seed_residue),
                "n_tried": 0,
                "k_star": first["k_star"],
                "k_eff": first["k_eff"],
                "resolved": first.get("resolved", False),
                "null": first["rho_null_p95"],
                "objective": _objective(first, size_penalty),
            }
        )

    while len(support) < max_size:
        frontier = np.where(adjacency[support].any(axis=0))[0]
        frontier = [int(j) for j in frontier if j not in support]
        if not frontier:
            break
        resolved, unresolved = [], []
        for ci, j in enumerate(frontier, 1):
            if on_candidate is not None:
                on_candidate(ci, len(frontier))
            try:
                r = score_support(feats, support + [j], lag=lag, **kw)
            except Exception:  # noqa: BLE001
                continue
            (resolved if r.get("resolved") else unresolved).append(
                (_objective(r, size_penalty), j, r)
            )
        if resolved:
            resolved.sort(key=lambda t: t[0])
            obj, j, r = resolved[0]
        elif unresolved:
            unresolved.sort(key=lambda t: _explore_key(t[2]))
            obj, j, r = unresolved[0]
        else:
            break
        support.append(j)
        history.append((list(support), obj, r))
        if on_step is not None:
            on_step(
                {
                    "size": len(support),
                    "added": int(j),
                    "n_tried": len(resolved) + len(unresolved),
                    "k_star": r["k_star"],
                    "k_eff": r["k_eff"],
                    "resolved": r.get("resolved", False),
                    "null": r["rho_null_p95"],
                    "objective": obj,
                }
            )

    curve = [
        {
            "size": len(sup),
            "k_eff": res.get("k_eff"),
            "k_star": res.get("k_star"),
            "resolved": res.get("resolved"),
            "null": res.get("rho_null_p95"),
            "support": sup,
        }
        for sup, _, res in history
    ]

    eligible = [h for h in history if np.isfinite(h[1])]
    if eligible:
        best = min(h[1] for h in eligible)
        at_knee = [
            h
            for h in eligible
            if h[1] <= best + knee_tol and len(h[0]) >= min_size
        ]
        if not at_knee:  # never reached min_size resolved
            at_knee = [h for h in eligible if h[1] <= best + knee_tol]
        chosen = min(at_knee, key=lambda h: len(h[0]))
        status = "resolved"
    else:
        best = float("inf")
        chosen = max(history, key=lambda h: len(h[0]))
        status = "unresolved"

    out = dict(chosen[2])
    out["status"] = status
    out["seed"] = int(seed_residue)
    out["best_objective"] = float(best)
    out["min_size"] = int(min_size)
    out["knee_tol"] = float(knee_tol)
    out["growth_trace"] = curve
    return out


def _grow_ball_support(N, size, adjacency, rng):
    """Grow one spatially compact support of `size` residues by breadth-first
    search on the contact graph `adjacency` (N, N) bool, starting from a
    random residue.

    Shared by `control_supports`'s 'ball' controls and
    `calibrate_null_thresholds`'s calibration supports, both of which want
    supports grown the same way real candidates are grown -- not uniform-
    random residue sets, not sequence segments.
    """
    start = int(rng.integers(0, N))
    chosen, frontier = [start], [start]
    while len(chosen) < size and frontier:
        nxt = []
        for u in frontier:
            for v in np.where(adjacency[u])[0]:
                v = int(v)
                if v not in chosen:
                    chosen.append(v)
                    nxt.append(v)
                    if len(chosen) >= size:
                        break
            if len(chosen) >= size:
                break
        frontier = nxt
    return chosen


def control_supports(
    feats,
    size,
    adjacency=None,
    n_random=8,
    n_segment=8,
    n_ball=8,
    seed=0,
    exclude=None,
    max_overlap=0.3,
    max_tries=40,
):
    """Return matched nulls for a support of a given size.

    random  : residues drawn uniformly, spatially scattered
    segment : contiguous stretches in sequence
    ball    : spatially compact sets grown by breadth-first search on contacts

    ``exclude`` is the list of DISCOVERED units, and it matters more than it
    looks.  A compact blob dropped at random into a protein will sometimes BE a
    unit, or overlap one heavily -- nothing in v3.0 D1 forbids a PDU from being
    spatially compact, it permits discontinuity without requiring it.  So an
    unfiltered ball control is contaminated with true positives, and demanding
    that a discovered unit beat it is testing a drug against a lower dose of
    itself.  On the first lysozyme decomposition that single effect moved the
    verdict from +1.36 (11/11 units better) to +0.07 (6/11).

    Control sets whose Jaccard overlap with any discovered unit exceeds
    ``max_overlap`` are therefore resampled.  The geometric matching is kept --
    which is the point of these controls -- while the contamination is removed.
    """
    rng = np.random.default_rng(seed)
    N = feats.shape[1]
    size = int(size)
    excl = [set(u) for u in (exclude or [])]

    def overlap(S):
        s_ = set(S)
        return max(
            (len(s_ & u) / max(len(s_ | u), 1) for u in excl), default=0.0
        )

    def sample(bucket, gen):
        """Draw a control, preferring one disjoint from every discovered unit.

        When the units cover the protein there may be NO disjoint set of the
        required size -- a disjoint ball of 40 residues in a 164-residue
        protein with eleven units is often impossible.  Rejecting outright then
        silently yields an empty bucket and a nan control, which is the worst
        possible failure: it looks like a missing number rather than a broken
        comparison. So the least-overlapping candidate is kept instead, and its
        overlap is recorded, turning contamination from a hidden binary into a
        reported quantity.
        """
        best, best_ov = None, np.inf
        for _try in range(max_tries):
            S = gen()
            if S is None or len(S) != size:
                continue
            ov = overlap(S)
            if ov < best_ov:
                best, best_ov = S, ov
            if ov <= max_overlap:
                break
        if best is None:
            return None
        bucket.append(sorted(int(i) for i in best))
        achieved.append(float(best_ov))
        return best_ov

    out = {"random": [], "segment": [], "ball": []}
    achieved = []
    rejected = 0

    for _ in range(n_random):
        ov = sample(
            out["random"],
            lambda: rng.choice(N, size=size, replace=False).tolist(),
        )
        rejected += int(ov is not None and ov > max_overlap)

    for _ in range(n_segment):
        if N - size <= 0:
            break

        def _seg():
            a = int(rng.integers(0, N - size))
            return list(range(a, a + size))

        ov = sample(out["segment"], _seg)
        rejected += int(ov is not None and ov > max_overlap)

    if adjacency is not None:
        adjacency = np.asarray(adjacency, dtype=bool)

        for _ in range(n_ball):
            ov = sample(
                out["ball"],
                lambda: _grow_ball_support(N, size, adjacency, rng),
            )
            rejected += int(ov is not None and ov > max_overlap)

    out["_rejected_for_overlap"] = rejected
    out["_overlap_median"] = (
        float(np.median(achieved)) if achieved else float("nan")
    )
    out["_overlap_max"] = float(np.max(achieved)) if achieved else float("nan")
    return out


# ---------------------------------------------------------------------------
# 5. Per-residue variance budget: inside / outside / irreducible
# ---------------------------------------------------------------------------


def residue_budget(
    feats,
    adjacency=None,
    lag=1,
    ridge=1e-2,
    env_dim=40,
    n_null=8,
    seed=0,
    segments=None,
    on_progress=None,
):
    """Three-way variance decomposition of each residue's own local future.

    For residue i, with d_i(t) its RSE distance descriptor:

        Var[ d_i(t+tau) ]  =  explained from INSIDE i's own RSE
                           +  explained from OUTSIDE it
                           +  irreducible

    The rigid-protein baseline makes this well posed.  A perfectly rigid
    protein with thermal noise has NO predictability at lags beyond the noise
    correlation time: d_i(t) says nothing about d_i(t+tau) beyond the mean.  So
    the baseline is zero explained variance, not a tolerance on the size of the
    fluctuation, and anything explained is structure.

    ORDERING CONVENTION, which must be reported.  Inside and outside are
    correlated, so the variance they share can be attributed to either and the
    answer depends on which is fitted first.  This routine fits INSIDE first,
    giving the RSE the benefit of the doubt and making the outside share as
    small as it can be.  That is the conservative direction for a PDU claim:
    outside contribution that survives is real.  v3.0 sec 4.2 already calls the
    internal/coupling split a convention plus an invariance test; this is that
    convention made explicit.

    Parameters
    ----------
    feats     : (F, N, m) residue-local descriptors
    adjacency : (N, N) bool or None.  When given, "inside" means residue i's
                own descriptor plus those of its contacts.  PASS IT.  With
                None, "inside" is residue i alone and its own neighbourhood
                counts as outside, so every residue shows an outside share
                whether or not anything couples to it -- on the planted
                synthetic that collapses the coupled/decoupled contrast from
                a factor of ~2800 to a factor of 1.0.
    Remaining arguments match `flux_spectrum`.

    Returns
    -------
    dict of arrays, each length N:
        inside      fraction of d_i(t+tau) variance explained from inside
        outside     additional fraction explained by the rest of the protein
        irreducible remainder (1 - inside - outside)
        resolved    bool, whether the outside share cleared the shift null
        outside_null the null level for the outside share, per residue

    Residues with ``resolved == False`` are unresolved at this resolution and
    should be greyed out in any figure rather than read as "no coupling".

    """
    F, N, m = np.asarray(feats).shape
    if segments is None:
        segments = [(0, F)]
    lag = int(lag)
    rows0, rows1 = [], []
    for a, b in segments:
        if b - a > lag:
            rows0.append(np.arange(a, b - lag))
            rows1.append(np.arange(a + lag, b))
    if not rows0:
        raise ValueError("no usable lagged pairs; lag too large for segments")
    rows0 = np.concatenate(rows0)
    rows1 = np.concatenate(rows1)
    rng = np.random.default_rng(seed)

    flat = np.asarray(feats, dtype=np.float64).reshape(F, N * m)
    flat = _center(flat)
    adjacency = (
        None if adjacency is None else np.asarray(adjacency, dtype=bool)
    )

    inside = np.zeros(N)
    outside = np.zeros(N)
    resolved = np.zeros(N, dtype=bool)
    out_null = np.zeros(N)

    def _explained(Y, X):
        """Fraction of Y's variance a ridge linear model on X explains."""
        if X.shape[1] == 0:
            return 0.0, Y
        R = _center(_residualise(Y, X, ridge))
        v0 = float((Y**2).sum())
        if v0 <= 0:
            return 0.0, R
        return float(max(0.0, 1.0 - (R**2).sum() / v0)), R

    for i in range(N):
        if on_progress is not None:
            on_progress(i + 1, N)
        own = np.arange(i * m, (i + 1) * m)
        if adjacency is None:
            in_cols = own
        else:
            nb = np.where(adjacency[i])[0]
            in_cols = (
                np.concatenate(
                    [own] + [np.arange(j * m, (j + 1) * m) for j in nb]
                )
                if nb.size
                else own
            )
        mask = np.zeros(N * m, dtype=bool)
        mask[in_cols] = True

        Y = _center(flat[rows1][:, own])
        H = _center(flat[rows0][:, in_cols])
        Zc = flat[rows0][:, ~mask]
        if Zc.shape[1] == 0:
            inside[i], _ = _explained(Y, H)
            continue
        Z = _center(_pca(_center(Zc), env_dim))

        f_in, Ytil = _explained(Y, H)
        Ztil = _center(_residualise(Z, H, ridge))
        f_out, _ = _explained(Ytil, Ztil)

        # shift null for the OUTSIDE share only
        nulls = []
        min_shift = max(10 * lag, F // 10, 5)
        for _ in range(int(n_null)):
            sh = int(
                rng.integers(min_shift, max(min_shift + 1, F - min_shift))
            )
            Zs = _center(
                _residualise(
                    _center(np.roll(Z, sh, axis=0)[np.arange(len(rows0))]),
                    H,
                    ridge,
                )
            )
            nulls.append(_explained(Ytil, Zs)[0])
        thr = float(np.percentile(nulls, 95)) if nulls else 0.0

        inside[i] = f_in
        outside[i] = max(0.0, f_out - thr) * (1.0 - f_in)
        out_null[i] = thr
        resolved[i] = f_out > thr

    irreducible = np.clip(1.0 - inside - outside, 0.0, 1.0)
    return {
        "inside": inside.tolist(),
        "outside": outside.tolist(),
        "irreducible": irreducible.tolist(),
        "resolved": resolved.tolist(),
        "outside_null": out_null.tolist(),
        "lag": lag,
        "n_pairs": int(rows0.size),
    }


# ---------------------------------------------------------------------------
# 6. Covariance workspace: the same estimator, O(D^3) per candidate
# ---------------------------------------------------------------------------


def _resolve_backend(device):
    """Pick numpy or torch.

    Returns (backend, torch_module_or_None, device_str).

    ``device`` None or "cpu" -> numpy.  "cuda"/"mps"/"auto" -> torch when it is
    installed and the device is actually visible, else numpy: a silent fallback
    to CPU on a GPU job would be far worse than a slow one you know about, so
    `FluxWorkspace.backend` records what was really used and the pilot prints
    it.
    """
    if device in (None, "cpu", "numpy"):
        return "numpy", None, "cpu"
    try:
        import torch
    except ImportError:
        return "numpy", None, "cpu"
    if device == "auto":
        device = (
            "cuda"
            if torch.cuda.is_available()
            else (
                "mps"
                if getattr(torch.backends, "mps", None)
                and torch.backends.mps.is_available()
                else "cpu"
            )
        )
    if device == "cuda" and not torch.cuda.is_available():
        return "numpy", None, "cpu"
    if device == "mps" and not (
        getattr(torch.backends, "mps", None)
        and torch.backends.mps.is_available()
    ):
        return "numpy", None, "cpu"
    if device == "mps":
        # MPS accelerates the covariance build (large float32 matmuls) but not
        # the per-candidate eigendecompositions, which need double precision
        # MPS does not have.  Expect the build to speed up and growth not to.
        return "torch", torch, "mps"
    if device == "cpu":
        return "numpy", None, "cpu"
    return "torch", torch, device


class FluxWorkspace:
    """Precomputed lagged covariances, so scoring a candidate never touches
    the trajectory again.

    WHY THIS EXISTS.  The direct implementation PCAs the environment inside
    every call, which costs O(n q^2).  At mdCATH size (n ~ 2500) that is a
    second; on 114,000 frames of lysozyme it is a minute per candidate, and a
    single growth run needs hundreds of candidates.  Every quantity the
    estimator needs is a function of three D x D matrices, so those are built
    once in a couple of passes and each candidate afterwards is submatrix
    algebra independent of n.

    The mathematics is unchanged -- same residualisation, same canonical
    correlations, same shift null.  `test_workspace_matches_direct_path`
    asserts the two agree.

    Notation, with X0 the resolved features at t and X1 at t+tau, both centred:

        C00 = X0'X0/n     C01 = X0'X1/n     C11 = X1'X1/n

    For a candidate with feature columns S and environment columns E, writing
    Vu, Ve for the unit and environment PCA bases:

        Chh = Vu' C00[S,S] Vu    Cyy = Vu' C11[S,S] Vu    Chy = Vu' C01[S,S] Vu
        Czz = Ve' C00[E,E] Ve    Chz = Vu' C00[S,E] Ve    Czy = Ve' C01[E,S] Vu

    and residualising the unit's future and the environment on the unit's own
    present gives, with no approximation,

        Ryy = Cyy - Chy' Chh^-1 Chy
        Rzz = Czz - Chz' Chh^-1 Chz
        Rzy = Czy - Chz' Chh^-1 Chy

    whose whitened singular values are the flux spectrum.

    The shift null needs the environment displaced in time against a fixed
    unit, so it additionally stores, per shift j,

        Cs00[j] = X0_shifted' X0 / n      Cs01[j] = X0_shifted' X1 / n

    The shifted environment keeps its own covariance (a circular shift is a
    permutation of rows), so Czz and both PCA bases are reused unchanged.

    GPU.  Pass ``device="cuda"`` (or "auto") to run through PyTorch.  Both
    phases benefit: the build is a handful of (n x D)' (n x D) matmuls, and
    each candidate is an eigh plus an svd on matrices of order D.  The
    accumulation runs in float32 and the D x D algebra in float64, which keeps
    the conditioning of the whitening step intact on a GH200.  Without torch,
    or without a visible GPU, it falls back to numpy -- and records that in
    ``backend``, because a silent fallback to CPU on a GPU job would be worse
    than a slow one you know about.
    """

    def __init__(
        self,
        feats,
        lag=1,
        segments=None,
        n_null=8,
        seed=0,
        ridge=1e-2,
        dtype=np.float32,
        on_progress=None,
        device=None,
        chunk=8192,
        reversible=True,
        truncation="koopman",
        shift_mode="global",
    ):
        """Prepare the covariances of the replicas for scoring supports."""
        A = np.asarray(feats)
        if A.ndim != 3:
            raise ValueError("feats must be (F, N, m)")
        self.F, self.N, self.m = A.shape
        self.D = self.N * self.m
        self.lag = int(lag)
        self.ridge = float(ridge)
        self.n_null = int(n_null)
        self.reversible = bool(reversible)
        if truncation not in ("koopman", "pca"):
            raise ValueError("truncation must be 'koopman' or 'pca'")
        # 'pca' exists only so the equal-time path can be compared against the
        # direct implementation; 'koopman' is the estimator.
        self.truncation = truncation
        # How the null displaces the environment in time (COG 2.1 fix):
        #   global  circular shift over the concatenated trajectory, as
        #           before -- can wrap across replica seams; kept as the
        #           default so earlier results reproduce exactly
        #   within  circular shift inside each replica (segment)
        #   cross   environment taken from a DIFFERENT replica, at a random
        #           offset -- exactly independent of the unit, no residual
        #           slow correlation from a short shift
        if shift_mode not in ("global", "within", "cross"):
            raise ValueError(
                "shift_mode must be 'global', 'within' or 'cross'"
            )
        self.shift_mode = shift_mode
        self.backend, self._torch, self.device = _resolve_backend(device)

        # A view when feats is already the right dtype and C-contiguous, so the
        # trajectory is not duplicated just to be reshaped.  Kept as an
        # attribute (not a local) so `extend_null_shifts` can build further
        # shift pairs later without re-deriving it.
        self._flat = np.ascontiguousarray(A, dtype=dtype).reshape(
            self.F, self.D
        )

        if segments is None:
            segments = [(0, self.F)]
        self.segments = [tuple(map(int, sg)) for sg in segments]
        usable = [(a, b) for a, b in self.segments if b - a > self.lag]
        if not usable:
            raise ValueError(
                "no usable lagged pairs; lag too large for segments"
            )
        self.n = int(sum(b - a - self.lag for a, b in usable))
        self.rows0 = np.concatenate(
            [np.arange(a, b - self.lag) for a, b in usable]
        )
        self.rows1 = np.concatenate(
            [np.arange(a + self.lag, b) for a, b in usable]
        )

        self._step = max(1024, int(chunk))
        total_ticks = 1 + 2 * self.n_null
        tick = [0]

        def _tick():
            tick[0] += 1
            if on_progress is not None:
                on_progress(tick[0], total_ticks)

        S00, s0, _ = self._gram(self.rows0, self.rows0)
        S01, _, s1 = self._gram(self.rows0, self.rows1)
        S11, _, _ = self._gram(self.rows1, self.rows1)
        self.m0 = s0 / self.n
        self.m1 = s1 / self.n
        self.C00 = S00 / self.n - np.outer(self.m0, self.m0)
        self.C01 = S01 / self.n - np.outer(self.m0, self.m1)
        self.C11 = S11 / self.n - np.outer(self.m1, self.m1)
        _tick()

        # Kept so `extend_null_shifts` can continue the same draw sequence:
        # a workspace built with n_null=k and then extended by n_more matches
        # one built with n_null=k+n_more directly, because both draw from the
        # same np.random.default_rng(seed) stream in the same order.
        self._rng = np.random.default_rng(seed)
        self._usable = usable
        if self.shift_mode == "cross" and len(usable) < 2:
            raise ValueError(
                "shift_mode='cross' needs at least two segments (replicas)"
            )
        self.Cs00, self.Cs01 = [], []
        for _ in range(self.n_null):
            rs = self._shifted_rows(self._rng)
            Ss0, ms, _ = self._gram(rs, self.rows0)
            self.Cs00.append(Ss0 / self.n - np.outer(ms / self.n, self.m0))
            _tick()
            Ss1, ms1, _ = self._gram(rs, self.rows1)
            self.Cs01.append(Ss1 / self.n - np.outer(ms1 / self.n, self.m1))
            _tick()

    def extend_null_shifts(self, n_more, seed=None):
        """Append `n_more` more (Cs00, Cs01) shift pairs to the shared null
        pool, using the same construction `__init__` uses for its own.

        With `seed` left as `None` this continues the SAME rng stream
        `__init__` started, so a workspace built with `n_null=k` and then
        extended by `n_more` here draws identically to one built with
        `n_null=k + n_more` directly -- there is no special case for shifts
        added later.  Pass an explicit `seed` to draw an independent batch
        instead.
        """
        rng = self._rng if seed is None else np.random.default_rng(seed)
        for _ in range(int(n_more)):
            rs = self._shifted_rows(rng)
            Ss0, ms, _ = self._gram(rs, self.rows0)
            self.Cs00.append(Ss0 / self.n - np.outer(ms / self.n, self.m0))
            Ss1, ms1, _ = self._gram(rs, self.rows1)
            self.Cs01.append(Ss1 / self.n - np.outer(ms1 / self.n, self.m1))
        self.n_null = len(self.Cs00)

    # -- helpers ----------------------------------------------------------
    def _shifted_rows(self, rng):
        """Row indices of the displaced environment, one per lagged pair.

        'global' draws exactly as before (one integer per shift), so a
        workspace built with the default reproduces earlier results.
        """
        if self.shift_mode == "global":
            min_shift = max(10 * self.lag, self.F // 10, 5)
            sh = int(
                rng.integers(min_shift, max(min_shift + 1, self.F - min_shift))
            )
            return (self.rows0 - sh) % self.F
        segs = [(int(a), int(b)) for a, b in self._usable]
        out = np.empty_like(self.rows0)
        pos = 0
        if self.shift_mode == "within":
            u = float(rng.random())
            for a, b in segs:
                L = b - a
                n_seg = L - self.lag
                ms = max(10 * self.lag, L // 10, 5)
                span = max(L - 2 * ms, 1)
                sh = ms + int(u * span)
                r = self.rows0[pos : pos + n_seg]
                out[pos : pos + n_seg] = a + ((r - a - sh) % L)
                pos += n_seg
            return out
        # cross: a random derangement of the segments plus a random offset
        k = len(segs)
        while True:
            perm = rng.permutation(k)
            if np.all(perm != np.arange(k)):
                break
        u = float(rng.random())
        for i, (a, b) in enumerate(segs):
            n_seg = (b - a) - self.lag
            ta, tb = segs[int(perm[i])]
            Lt = tb - ta
            off = int(u * Lt)
            r = self.rows0[pos : pos + n_seg]
            out[pos : pos + n_seg] = ta + ((r - a + off) % Lt)
            pos += n_seg
        return out

    def _gram(self, a_rows, b_rows):
        """(sum a'b, sum a, sum b) over paired row-index arrays, chunked.

        Never materialises a full (n x D) copy: at 114k frames that is
        450 MB per copy and there would be four of them live at once.
        """
        S = np.zeros((self.D, self.D), dtype=np.float64)
        sa = np.zeros(self.D, dtype=np.float64)
        sb = np.zeros(self.D, dtype=np.float64)
        use_torch = self.backend == "torch"
        if use_torch:
            th = self._torch
            dev = th.device(self.device)
            T = th.zeros((self.D, self.D), dtype=th.float32, device=dev)
        for c0 in range(0, a_rows.size, self._step):
            c1 = min(c0 + self._step, a_rows.size)
            Ac = self._flat[a_rows[c0:c1]]
            Bc = self._flat[b_rows[c0:c1]]
            if use_torch:
                ta = self._torch.as_tensor(Ac, device=dev)
                tb = self._torch.as_tensor(Bc, device=dev)
                T += ta.T @ tb
            else:
                S += Ac.T.astype(np.float64) @ Bc.astype(np.float64)
            sa += Ac.sum(axis=0, dtype=np.float64)
            sb += Bc.sum(axis=0, dtype=np.float64)
        if use_torch:
            # Apple MPS has no float64 at all, so the cast to double must
            # happen after the tensor is back on the host, never on device.
            S = T.cpu().numpy().astype(np.float64)
        return S, sa, sb

    def _cols(self, support):
        support = np.asarray(sorted(set(int(i) for i in support)), dtype=int)
        if support.size == 0 or support.min() < 0 or support.max() >= self.N:
            raise ValueError("support out of range")
        mask = np.zeros(self.N, dtype=bool)
        mask[support] = True
        col = np.repeat(mask, self.m)
        return support, np.where(col)[0], np.where(~col)[0]

    def _basis(self, C, k):
        """Top-k eigenvectors of a symmetric matrix (variance directions).

        Retained only for the equal-time fallback; the estimator uses
        `_koopman_basis`.  See that method for why.
        """
        n = C.shape[0]
        k = int(min(k, n))
        if k < n // 4:
            try:
                from scipy.linalg import eigh as _seigh

                w, V = _seigh(
                    C,
                    subset_by_index=[n - k, n - 1],
                    check_finite=False,
                    driver="evx",
                )
                return np.ascontiguousarray(V[:, ::-1])
            except Exception:  # noqa: BLE001
                pass
        w, V = np.linalg.eigh(C)
        return np.ascontiguousarray(V[:, np.argsort(w)[::-1][:k]])

    def _koopman_basis(self, idx, k, reversible=True):
        """Top-k SLOW directions of a block -- tICA/VAMP, not PCA.

        WHY THIS REPLACED PCA.  Truncating with the equal-time covariance keeps
        the directions of largest AMPLITUDE.  Slowness and amplitude are
        different things: a slow collective mode can be small and be discarded
        in the first few components, while a fast high-amplitude local rattle
        is kept.  Doing that before a lagged-covariance estimator throws away
        exactly the dynamics the estimator exists to find -- the classic reason
        `koopman.py` whitens with C(0) and then diagonalises C(tau) rather than
        running PCA first.

        Reversible branch (the default, matching `koopman.tica`): whiten the
        block with its equal-time covariance, symmetrise the lagged covariance,
        and take the leading eigenvectors --

            L = C00^-1/2 ,  A = L (C01 + C01') L / 2 ,  V = L U_k

        Non-reversible branch (VAMP), for driven or non-equilibrium data:

            M = C00^-1/2 C01 C11^-1/2 ,  V = C00^-1/2 U_k  from the SVD of M

        Returns the (block_dim, k) map from block features to slow coordinates.
        """
        C0 = self.C00[np.ix_(idx, idx)]
        C1 = self.C01[np.ix_(idx, idx)]
        n = C0.shape[0]
        k = int(min(k, n))

        # GPU path.  This is the per-candidate hot spot once the truncation is
        # Koopman -- two inverse square roots plus a decomposition on a matrix
        # of order D -- so on CUDA it belongs on the device.  Needs float64, so
        # CUDA only; Apple MPS has no double precision and stays on the host.
        if self.backend == "torch" and self.device == "cuda" and n > 256:
            th = self._torch
            dev = th.device("cuda")
            t0 = th.as_tensor(C0, device=dev, dtype=th.float64)
            t1 = th.as_tensor(C1, device=dev, dtype=th.float64)
            t0 = 0.5 * (t0 + t0.T)
            scale = th.clamp(th.trace(t0) / max(n, 1), min=1e-30)
            t0 = t0 + self.ridge * scale * th.eye(
                n, device=dev, dtype=th.float64
            )
            w0, V0 = th.linalg.eigh(t0)
            w0 = th.clamp(w0, min=1e-14 * float(w0.max()))
            L = (V0 / th.sqrt(w0)) @ V0.T
            if reversible:
                A = L @ (0.5 * (t1 + t1.T)) @ L
                A = 0.5 * (A + A.T)
                w, U = th.linalg.eigh(A)
                U = U[:, th.argsort(w, descending=True)[:k]]
            else:
                tt = th.as_tensor(
                    self.C11[np.ix_(idx, idx)], device=dev, dtype=th.float64
                )
                tt = 0.5 * (tt + tt.T)
                st = th.clamp(th.trace(tt) / max(n, 1), min=1e-30)
                tt = tt + self.ridge * st * th.eye(
                    n, device=dev, dtype=th.float64
                )
                wt, Vt = th.linalg.eigh(tt)
                wt = th.clamp(wt, min=1e-14 * float(wt.max()))
                Ly = (Vt / th.sqrt(wt)) @ Vt.T
                U, _, _ = th.linalg.svd(L @ t1 @ Ly, full_matrices=False)
                U = U[:, :k]
            return np.ascontiguousarray((L @ U).cpu().numpy())

        L = _inv_sqrt(C0, self.ridge)
        if reversible:
            A = L @ (0.5 * (C1 + C1.T)) @ L
            A = 0.5 * (A + A.T)
            if k < n // 4:
                try:
                    from scipy.linalg import eigh as _seigh

                    w, U = _seigh(
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
        else:
            Ly = _inv_sqrt(self.C11[np.ix_(idx, idx)], self.ridge)
            M = L @ C1 @ Ly
            U, sv, _ = np.linalg.svd(M, full_matrices=False)
            U = np.ascontiguousarray(U[:, :k])
        return np.ascontiguousarray(L @ U)

    def _spectrum(self, Ryy, Rzz, Rzy):
        M = _inv_sqrt(Ryy, self.ridge) @ Rzy.T @ _inv_sqrt(Rzz, self.ridge)
        return np.clip(np.linalg.svd(M, compute_uv=False), 0.0, 1.0)

    def _project_and_residualise(self, support, unit_dim=8, env_dim=40):
        """Unit/environment bases, residualised covariances, and the real
        (non-null) canonical-correlation spectrum for one candidate.

        Everything `score()` needs before it either recomputes a per-
        candidate shift null (the default) or looks one up in a pre-
        calibrated table (`null_thresholds`).  Split out so
        `calibrate_null_thresholds` can share this exact computation instead
        of duplicating it.
        """
        support, S, E = self._cols(support)
        if E.size == 0:
            raise ValueError(
                "candidate covers the whole protein: no environment"
            )

        if self.truncation == "koopman":
            Vu = self._koopman_basis(S, unit_dim, self.reversible)
            Ve = self._koopman_basis(E, env_dim, self.reversible)
        else:
            Vu = self._basis(self.C00[np.ix_(S, S)], unit_dim)
            Ve = self._basis(self.C00[np.ix_(E, E)], env_dim)

        Chh = Vu.T @ self.C00[np.ix_(S, S)] @ Vu
        Cyy = Vu.T @ self.C11[np.ix_(S, S)] @ Vu
        Chy = Vu.T @ self.C01[np.ix_(S, S)] @ Vu
        Czz = Ve.T @ self.C00[np.ix_(E, E)] @ Ve
        Chz = Vu.T @ self.C00[np.ix_(S, E)] @ Ve

        Ih = _inv_sqrt(Chh, self.ridge)
        Ih = Ih @ Ih  # ridge-regularised Chh^-1
        Ryy = Cyy - Chy.T @ Ih @ Chy
        Rzz = Czz - Chz.T @ Ih @ Chz

        Czy = Ve.T @ self.C01[np.ix_(E, S)] @ Vu
        rho = self._spectrum(Ryy, Rzz, Czy - Chz.T @ Ih @ Chy)

        return support, S, E, Vu, Ve, Ih, Ryy, Chy, Czz, Cyy, rho

    def _leading_null_rho(
        self, Ih, Ryy, Chy, Vu, Ve, Czz, S, E, Cs00_j, Cs01_j
    ):
        """Leading canonical correlation against one shift-null replica,
        for a candidate whose real-spectrum quantities are already computed
        (by `_project_and_residualise`).
        """
        Chz_n = Vu.T @ Cs00_j[np.ix_(E, S)].T @ Ve
        Czy_n = Ve.T @ Cs01_j[np.ix_(E, S)] @ Vu
        Rzz_n = Czz - Chz_n.T @ Ih @ Chz_n
        return float(self._spectrum(Ryy, Rzz_n, Czy_n - Chz_n.T @ Ih @ Chy)[0])

    # -- public -----------------------------------------------------------
    def score(self, support, unit_dim=8, env_dim=40, null_thresholds=None):
        """Flux spectrum for one candidate; same output shape as
        `flux_spectrum`.

        With `null_thresholds` left as `None` (the default), the null
        threshold is recomputed for this one candidate from the workspace's
        own shift replicas, exactly as before.  Passing a `{size: theta}`
        mapping -- e.g. the `"theta"` table from `calibrate_null_thresholds`
        -- skips that per-candidate computation entirely and looks up
        `thr = null_thresholds[support.size]` instead; a size missing from
        the table is an error rather than a silent fall-back, since mixing
        the two threshold designs in one run would be silent and wrong.
        """
        support, S, E, Vu, Ve, Ih, Ryy, Chy, Czz, Cyy, rho = (
            self._project_and_residualise(support, unit_dim, env_dim)
        )

        if null_thresholds is None:
            leading = [
                self._leading_null_rho(
                    Ih, Ryy, Chy, Vu, Ve, Czz, S, E, Cs00, Cs01
                )
                for Cs00, Cs01 in zip(self.Cs00, self.Cs01)
            ]
            thr = float(np.percentile(leading, 95)) if leading else 0.0
        else:
            size = int(support.size)
            if size not in null_thresholds:
                raise KeyError(
                    f"null_thresholds has no entry for support size {size}"
                )
            thr = float(null_thresholds[size])

        rho2 = rho**2
        total = float(rho2.sum())
        k_star = int(np.sum(rho > thr))
        rho2c = np.maximum(rho2 - thr**2, 0.0)
        totc = float(rho2c.sum())
        if totc > 0:
            k_eff = float(totc**2 / np.sum(rho2c**2))
            curve = (np.cumsum(rho2c) / totc).tolist()
        else:
            k_eff, curve = float("nan"), []

        # SELF-PREDICTION (v3.1 item 1).  Eq. (13) residualises the unit's
        # future by subtracting the part its own present explains; sigma is
        # that subtracted fraction -- how much of the unit's future the unit
        # itself accounts for.  k_eff is monotone non-increasing in |S| and
        # floors at 1, so it carries no size information; sigma is the term
        # that might.  Reported always, used only when cohesion_lambda > 0.
        tr_yy = float(np.trace(Cyy))
        sigma = (
            (float(np.trace(Chy.T @ Ih @ Chy)) / tr_yy)
            if tr_yy > 0
            else float("nan")
        )

        # NUCLEAR-TO-SPECTRAL RATIO (v3.1 item 6) on the null-corrected
        # amplitudes.  k_eff is a participation ratio, so (0.8, 0.2) reads
        # 1.12 and a weak second channel is nearly free to a greedy search;
        # this reads 1.25 and charges for it.
        rhoc = np.sqrt(rho2c)
        nuc_spec = (
            float(rhoc.sum() / rhoc[0])
            if rhoc.size and rhoc[0] > 0
            else float("nan")
        )

        return {
            "support": support.tolist(),
            "size": int(support.size),
            "n_pairs": self.n,
            "lag": self.lag,
            "rho": rho.tolist(),
            "rho_null_p95": thr,
            "resolved": bool(k_star > 0),
            "margin": float(rho[0] - thr) if rho.size else 0.0,
            "low_power": bool(rho.size == 0 or (rho[0] - thr) <= 0.0),
            "k_star": k_star,
            "k_eff": k_eff,
            "coupling_total": total,
            "sigma": sigma,
            "nuc_spec": nuc_spec,
            "coupling_resolved": totc,
            "delta_int_curve": curve,
            "unit_dim": int(Vu.shape[1]),
            "env_dim": int(Ve.shape[1]),
        }


# ---------------------------------------------------------------------------
# 7. True RSE features: every residue within a cutoff (v3.0 eq. 5)
# ---------------------------------------------------------------------------


def rse_features(
    ca_coords,
    cutoff=15.0,
    n_components=8,
    min_seq_sep=2,
    dmed=None,
    units="auto",
    dtype=np.float32,
    chunk=8192,
    max_frames=200,
    on_progress=None,
    lag=1,
    segments=None,
    koopman=True,
    ridge=1e-2,
    reversible=True,
    distance_filter=None,
    spectra=None,
):
    """Residue Structural Environment descriptors (v3.0 eq. 5).

    ``distance_filter`` (COG 2.1, optional): a callable f(Di, segments)
    applied in place to each residue's raw (F, n_i) distance matrix BEFORE
    the slow-direction reduction -- e.g. removing within-replica drift, so
    each residue's slowest directions are chosen inside the time band that
    is analysed rather than from the drift.  None leaves this unchanged.

    Residue i is described by its distances to EVERY residue within ``cutoff``
    (15 A by default), not to a fixed small number of nearest neighbours.  That
    distinction is not cosmetic.  On lysozyme:

        6 nearest neighbours : 78% lie within +-5 residues in sequence
        RSE within 15 A      : 64% lie MORE than 10 residues away (~40 each)

    A nearest-neighbour descriptor therefore encodes local chain geometry, and
    every statistic built on it ranks candidates by sequence contiguity.  The
    RSE spans the fold and carries the long-range contacts that make a
    residue's environment three-dimensional.

    COMPRESSION BY SLOWNESS, NOT VARIANCE.  With ``koopman=True`` (the default)
    each residue's RSE distances are reduced by tICA at lag ``lag`` -- whiten
    with the equal-time covariance, diagonalise the lagged one -- so the kept
    components are the residue's SLOWEST local modes.  Reducing by PCA instead
    keeps its largest-amplitude modes, which is a different thing and can throw
    away the slow local dynamics before the estimator ever sees it.  Pass
    ``koopman=False`` for the equal-time reduction.

    ``segments`` (row ranges, one per replica) is respected so no lagged pair
    crosses a replica seam.

    COMPRESSION, and why it is declared rather than hidden.  Forty distances
    per residue would put the workspace covariances at (164 x 40)^2, some 14 GB
    across the null shifts.  Each residue's distance vector is therefore
    compressed to ``n_components`` principal components, fitted on that
    residue's own RSE distances alone.  The support is unchanged -- the
    components are a function of exactly the atoms in the RSE, so the
    descriptor stays residue-supported in the sense of v3.0 sec 1.2 -- and D2
    explicitly lists allowed coordinate compression as a protocol choice.  It
    IS a fitted encoder, so it must be charged for in any model comparison, and
    for a transfer test it must be fitted on source proteins only.

    Parameters
    ----------
    cutoff       : RSE radius in ANGSTROM (converted if coordinates are nm)
    n_components : components kept per residue
    min_seq_sep  : exclude near-in-sequence neighbours as trivially rigid

    Returns
    -------
    feats        : (F, N, n_components)
    neighbours   : list of N index arrays -- each residue's actual RSE
    explained    : (N,) fraction of that residue's RSE variance retained

    """
    X = np.asarray(ca_coords)
    F, N, _ = X.shape
    if dmed is None:
        dmed = median_distance_matrix(X, max_frames=max_frames)
    if units == "auto":
        scale = 1.0 / detect_length_unit(X, dmed=dmed)["to_angstrom"]
    elif units == "nanometre":
        scale = 0.1
    else:
        scale = 1.0
    cut = float(cutoff) * scale

    idx = np.arange(N)
    sep = np.abs(idx[:, None] - idx[None, :])
    ok = (dmed <= cut) & (sep >= max(1, min_seq_sep))

    k = int(n_components)
    feats = np.empty((F, N, k), dtype=dtype)
    neighbours, explained = [], np.zeros(N)

    lag = int(lag)
    if segments is None:
        segments = [(0, F)]
    r0i = np.concatenate(
        [np.arange(a, b - lag) for a, b in segments if b - a > lag]
    )
    r1i = np.concatenate(
        [np.arange(a + lag, b) for a, b in segments if b - a > lag]
    )

    for i in range(N):
        nb = np.where(ok[i])[0]
        if nb.size == 0:  # fall back to nearest few
            d = np.where(sep[i] >= max(1, min_seq_sep), dmed[i], np.inf)
            nb = np.argsort(d)[:k]
        neighbours.append(nb)

        # distances from i to its RSE, frame by frame (F, n_i)
        Di = np.empty((F, nb.size), dtype=np.float32)
        for a in range(0, F, chunk):
            b = min(a + chunk, F)
            Xc = np.asarray(X[a:b], dtype=np.float32)
            diff = Xc[:, i, None, :] - Xc[:, nb, :]
            Di[a:b] = np.sqrt(np.einsum("cnk,cnk->cn", diff, diff))

        if distance_filter is not None:
            distance_filter(Di, segments)
        Di -= Di.mean(axis=0, dtype=np.float64).astype(np.float32)
        D64 = Di.astype(np.float64)
        C = (D64.T @ D64) / max(F, 1)
        C = 0.5 * (C + C.T)
        kk = int(min(k, C.shape[0]))
        wv = np.linalg.eigvalsh(C)
        tot = float(np.sum(np.maximum(wv, 0.0)))

        if koopman:
            A0 = D64[r0i]
            A1 = D64[r1i]
            nn = max(A0.shape[0], 1)
            C0 = (A0.T @ A0) / nn
            C1 = (A0.T @ A1) / nn
            L = _inv_sqrt(0.5 * (C0 + C0.T), ridge)
            if reversible:
                M = L @ (0.5 * (C1 + C1.T)) @ L
                M = 0.5 * (M + M.T)
                w, U = np.linalg.eigh(M)
                U = U[:, np.argsort(w)[::-1][:kk]]
                if spectra is not None:  # lag-tau autocorrelations of
                    spectra.append(  # this residue's slow modes
                        np.sort(w)[::-1][:24].astype(float).tolist()
                    )
            else:
                Ct = (A1.T @ A1) / nn
                U, sv, _ = np.linalg.svd(
                    L @ C1 @ _inv_sqrt(0.5 * (Ct + Ct.T), ridge),
                    full_matrices=False,
                )
                U = U[:, :kk]
                if spectra is not None:
                    spectra.append(sv[:24].astype(float).tolist())
            P = np.ascontiguousarray(L @ U)
            # variance retained by these SLOW directions (not the top-variance
            # ones), so the number is comparable across the two reductions
            Q = P / np.maximum(np.linalg.norm(P, axis=0, keepdims=True), 1e-12)
            explained[i] = (
                float(np.trace(Q.T @ C @ Q)) / tot if tot > 0 else 0.0
            )
        else:
            w, V = np.linalg.eigh(C)
            order = np.argsort(w)[::-1]
            explained[i] = (
                float(np.sum(np.maximum(w[order[:kk]], 0.0))) / tot
                if tot > 0
                else 0.0
            )
            P = np.ascontiguousarray(V[:, order[:kk]])
        proj = Di @ P.astype(np.float32)
        if kk < k:  # pad if the RSE was tiny
            feats[:, i, :kk] = proj
            feats[:, i, kk:] = 0.0
        else:
            feats[:, i, :] = proj
        if on_progress is not None:
            on_progress(i + 1, N)

    return feats, neighbours, explained


# ---------------------------------------------------------------------------
# 8. Self-consistent decomposition: every residue in at least one PDU
# ---------------------------------------------------------------------------


def _unit_cost(result, rank_stat="keff", cohesion_lambda=0.0):
    """Objective for a candidate unit.

    Channel count, with 'nothing resolved' scored as zero channels: unresolved
    is the most closed a unit can look, so it is the best cost.  The size
    charge and the requirement that a unit leave a non-empty environment are
    what stop the whole protein from winning that way.

    `rank_stat` selects the continuous relaxation of rank (v3.1 item 6):
      keff      participation ratio of the null-corrected spectrum
      nuc_spec  nuclear-to-spectral ratio, which charges more for weak channels

    `cohesion_lambda` subtracts the unit's self-prediction (v3.1 item 1):

        J'(S) = rank(S) - lambda * sigma(S)

    At lambda = 0 this is exactly the previous objective, so the default path
    is unchanged.
    """
    if not result.get("resolved", False):
        base = 0.0
    else:
        k = result.get(
            "k_eff" if rank_stat == "keff" else "nuc_spec", float("nan")
        )
        base = 0.0 if not np.isfinite(k) else float(k)
    if cohesion_lambda:
        sig = result.get("sigma", float("nan"))
        if np.isfinite(sig):
            base -= float(cohesion_lambda) * float(sig)
    return base


def _farthest_point_seeds(adjacency, n, rng):
    """Spread seeds over the contact graph by farthest-point sampling.

    Random seeds clump; spread seeds cover the fold with far fewer growths, and
    coverage of every residue comes from the assignment step rather than from
    seeding each one.
    """
    N = adjacency.shape[0]
    n = int(min(n, N))
    INF = N + 5
    dist = np.full(N, INF)
    chosen = [int(rng.integers(N))]

    def bfs(src):
        d = np.full(N, INF)
        d[src] = 0
        frontier = [src]
        while frontier:
            nxt = []
            for u in frontier:
                for v in np.where(adjacency[u])[0]:
                    if d[v] > d[u] + 1:
                        d[v] = d[u] + 1
                        nxt.append(int(v))
            frontier = nxt
        return d

    dist = np.minimum(dist, bfs(chosen[0]))
    while len(chosen) < n:
        nxt = int(np.argmax(dist))
        if dist[nxt] <= 0:
            remaining = [i for i in range(N) if i not in chosen]
            if not remaining:
                break
            nxt = int(rng.choice(remaining))
        chosen.append(nxt)
        dist = np.minimum(dist, bfs(nxt))
    return sorted(set(chosen))


def _merge_reearn(
    supports, threshold, cost, prefetch, gate, margin=0.1, min_size=3, log=None
):
    """Merge overlapping candidates WITHOUT assuming the non-shared residues
    belong (13 Oct 2026).

    Pairs are merged one at a time (the most-overlapping pair first, then
    the overlaps are recomputed), so there are no chains.  For a pair with
    Jaccard >= threshold the shared residues are the core; each non-shared
    residue must re-earn its place, best first: it must pass `gate(r, set)`
    (dynamics consistency with the current members) AND lower the cost by
    more than `margin`.  Residues that do not re-earn a place are left out
    (they become orphans for the reassignment step).  A pair whose core is
    smaller than `min_size` is not merged.
    """
    sets = [set(int(x) for x in s) for s in supports]
    skip = set()
    stats = {"merges": 0, "re_earned": 0, "dropped": 0}
    while True:
        best = None
        for a in range(len(sets)):
            for b in range(a + 1, len(sets)):
                if (a, b) in skip:
                    continue
                j = len(sets[a] & sets[b]) / max(len(sets[a] | sets[b]), 1)
                if j >= threshold and (best is None or j > best[0]):
                    best = (j, a, b)
        if best is None:
            break
        _, a, b = best
        core = sets[a] & sets[b]
        if len(core) < min_size:
            skip.add((a, b))
            continue
        cand = (sets[a] | sets[b]) - core
        cur = set(core)
        c_cur = cost(sorted(cur))
        while cand:
            ok = [r for r in sorted(cand) if gate is None or gate(r, cur)]
            if not ok:
                break
            prefetch([sorted(cur | {r}) for r in ok])
            cs = [(cost(sorted(cur | {r})), r) for r in ok]
            c_best, r_best = min(cs)
            mg = margin(len(cur)) if callable(margin) else margin
            if not c_best < c_cur - mg:
                break
            cur.add(r_best)
            cand.discard(r_best)
            c_cur = c_best
            stats["re_earned"] += 1
        stats["dropped"] += len(cand)
        stats["merges"] += 1
        keep = [x for t, x in enumerate(sets) if t not in (a, b)]
        sets = keep + [cur]
        skip = set()  # indices changed: recompute every overlap
    if log is not None and stats["merges"]:
        log(
            f"merge (core and re-earn): {stats['merges']} merges, "
            f"{stats['re_earned']} non-shared residues re-earned a place, "
            f"{stats['dropped']} left out"
        )
    return [sorted(x) for x in sets]


def _merge_by_overlap(supports, threshold):
    """Union-merge supports whose Jaccard overlap exceeds the threshold."""
    sets = [set(s) for s in supports]
    changed = True
    while changed:
        changed = False
        out = []
        while sets:
            cur = sets.pop()
            merged = True
            while merged:
                merged = False
                rest = []
                for other in sets:
                    j = len(cur & other) / max(len(cur | other), 1)
                    if j >= threshold:
                        cur |= other
                        merged = True
                        changed = True
                    else:
                        rest.append(other)
                sets = rest
            out.append(cur)
        sets = out
    return [sorted(s) for s in sets]


# ---------------------------------------------------------------------------
# 8a. C6 redesign: size-conditioned null calibration, done once and reused
# ---------------------------------------------------------------------------


def _bootstrap_ci_rel_width(
    pool, percentile=95, n_boot=2000, ci=0.95, rng=None
):
    """Bootstrap a percentile of `pool` and the relative width of its CI.

    Resamples `pool` (a 1-D array of leading-rho values) with replacement
    `n_boot` times at its current size, computes `percentile` on each
    resample, and returns `(point_estimate, relative_width)` where
    `relative_width = (hi - lo) / point_estimate` for the `ci`-level interval
    (the (1-ci)/2 and 1-(1-ci)/2 percentiles of the bootstrap replicates).

    General-purpose: knows nothing about supports, sizes, or C6, so it stays
    this simple.
    """
    pool = np.asarray(pool, dtype=np.float64)
    n = pool.size
    if n == 0:
        return float("nan"), float("inf")
    point = float(np.percentile(pool, percentile))
    if rng is None:
        rng = np.random.default_rng()
    idx = rng.integers(0, n, size=(int(n_boot), n))
    boot = np.percentile(pool[idx], percentile, axis=1)
    lo, hi = np.percentile(
        boot, [100 * (1 - ci) / 2, 100 * (1 - (1 - ci) / 2)]
    )
    rel_width = float((hi - lo) / point) if point != 0 else float("inf")
    return point, rel_width


def _bootstrap_ci_rel_width_by_support(
    rows, percentile=95, n_boot=2000, ci=0.95, rng=None, chunk=100
):
    """Like `_bootstrap_ci_rel_width`, but resampling whole SUPPORTS.

    `rows` is (n_supports, n_shifts): one row of leading null correlations
    per calibration support.  Each bootstrap replicate draws `n_supports`
    rows with replacement and takes `percentile` over all their values, so
    the interval includes the support-to-support variation in null level.
    Resampling the pooled values individually instead treats the n_shifts
    values of one support as independent -- they are not, they share that
    support's own null level -- and gives an interval that is too narrow.

    The point estimate is the same as the pooled one: `percentile` of every
    value in `rows`.  Returns `(point_estimate, relative_width)`.
    """
    rows = np.asarray(rows, dtype=np.float64)
    if rows.ndim != 2 or rows.size == 0:
        return float("nan"), float("inf")
    n_sup = rows.shape[0]
    point = float(np.percentile(rows, percentile))
    if rng is None:
        rng = np.random.default_rng()
    idx = rng.integers(0, n_sup, size=(int(n_boot), n_sup))
    boot = np.empty(int(n_boot), dtype=np.float64)
    for c0 in range(0, int(n_boot), int(chunk)):
        block = rows[idx[c0 : c0 + int(chunk)]]  # (c, n_sup, n_shifts)
        boot[c0 : c0 + block.shape[0]] = np.percentile(
            block.reshape(block.shape[0], -1), percentile, axis=1
        )
    lo, hi = np.percentile(
        boot, [100 * (1 - ci) / 2, 100 * (1 - (1 - ci) / 2)]
    )
    rel_width = float((hi - lo) / point) if point != 0 else float("inf")
    return point, rel_width


def _bootstrap_ci_rel_width_two_way(
    rows, percentile=95, n_boot=2000, ci=0.95, rng=None, chunk=50
):
    """Like `_bootstrap_ci_rel_width_by_support`, but resampling SUPPORTS
    AND SHIFTS independently.

    Every calibration support is scored against the SAME shift replicas, so
    the shifts are a shared source of noise: if one shift happens to leave
    slow motions aligned, it raises every support's null together.
    Resampling supports alone cannot see that and reports an interval that
    is too narrow (on the planted test system the shift-to-shift variance was
    ~7x the support-to-support variance).  Each replicate here draws
    `n_supports` rows AND `n_shifts` columns with replacement.
    Returns `(point_estimate, relative_width)`.
    """
    rows = np.asarray(rows, dtype=np.float64)
    if rows.ndim != 2 or rows.size == 0:
        return float("nan"), float("inf")
    n_sup, n_sh = rows.shape
    point = float(np.percentile(rows, percentile))
    if rng is None:
        rng = np.random.default_rng()
    n_boot = int(n_boot)
    boot = np.empty(n_boot, dtype=np.float64)
    for c0 in range(0, n_boot, int(chunk)):
        c = min(int(chunk), n_boot - c0)
        i_s = rng.integers(0, n_sup, size=(c, n_sup))
        i_j = rng.integers(0, n_sh, size=(c, n_sh))
        block = rows[i_s[:, :, None], i_j[:, None, :]]  # (c, n_sup, n_sh)
        boot[c0 : c0 + c] = np.percentile(
            block.reshape(c, -1), percentile, axis=1
        )
    lo, hi = np.percentile(
        boot, [100 * (1 - ci) / 2, 100 * (1 - (1 - ci) / 2)]
    )
    rel_width = float((hi - lo) / point) if point != 0 else float("inf")
    return point, rel_width


def _variance_components(rows):
    """Between-support, between-shift and residual variance of a
    (n_supports, n_shifts) table of leading null correlations.
    """
    rows = np.asarray(rows, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[0] < 2 or rows.shape[1] < 2:
        return {
            "between_supports": float("nan"),
            "between_shifts": float("nan"),
            "residual": float("nan"),
        }
    grand = rows.mean()
    sup = rows.mean(axis=1) - grand
    sh = rows.mean(axis=0) - grand
    resid = rows - grand - sup[:, None] - sh[None, :]
    return {
        "between_supports": float(sup.var(ddof=1)),
        "between_shifts": float(sh.var(ddof=1)),
        "residual": float(resid.var(ddof=1)),
    }


# ---------------------------------------------------------------------------
# null-table bookkeeping (COG 2.1 fixes)
# ---------------------------------------------------------------------------


def workspace_fingerprint(ws):
    """Return what a null table depends on beyond unit_dim and env_dim.

    That is the data (how many lagged pairs, which segments, a numerical
    signature of the covariances) and the estimator settings.  Stored in
    every calibration's config and checked by `check_null_calibration`.
    """
    return {
        "n_pairs": int(ws.n),
        "N": int(ws.N),
        "m": int(ws.m),
        "D": int(ws.D),
        "lag": int(ws.lag),
        "ridge": float(ws.ridge),
        "truncation": str(ws.truncation),
        "reversible": bool(ws.reversible),
        "shift_mode": str(getattr(ws, "shift_mode", "global")),
        "segments": [[int(a), int(b)] for a, b in ws.segments],
        "trace_C00": float(np.trace(ws.C00)),
        "trace_C01": float(np.trace(ws.C01)),
    }


def check_null_calibration(calibration, ws, unit_dim, env_dim, *, rtol=1e-4):
    """Raise ValueError if `calibration` was not made on this workspace's
    data and settings.  `calibration` is the full dict a calibration returns.
    Calibrations made before fingerprints existed are checked on
    unit_dim/env_dim only.
    """
    config = calibration.get("config") or {}
    problems = []
    for key, want in (("unit_dim", unit_dim), ("env_dim", env_dim)):
        if (
            key in config
            and config[key] is not None
            and want is not None
            and int(config[key]) != int(want)
        ):
            problems.append(f"{key}: table {config[key]} vs run {want}")
    fp = config.get("fingerprint")
    if fp:
        now = workspace_fingerprint(ws)
        for key, got in fp.items():
            if key not in now:
                continue
            want = now[key]
            if isinstance(got, float) or isinstance(want, float):
                if not np.isclose(
                    float(got), float(want), rtol=rtol, atol=1e-12
                ):
                    problems.append(f"{key}: table {got} vs run {want}")
            elif got != want:
                problems.append(f"{key}: table {got} vs run {want}")
    if problems:
        raise ValueError(
            "null calibration does not match this workspace -- "
            "a null threshold is only valid for the data and "
            "settings it was calibrated on: " + "; ".join(problems)
        )


def complete_theta_table(theta, max_size, *, mode="interpolate"):
    """Return a {size: theta} table covering every size 1..max_size.

    A size missing from the table used to make `score()` raise KeyError,
    which `decompose()` silently turned into an infinite cost -- so the
    calibrated sizes, not the data, decided which unit sizes could exist.
    'interpolate' fills gaps linearly and holds the end values beyond the
    calibrated range (theta is close to flat in size); 'strict' raises.
    Returns (table, filled_sizes).
    """
    th = {int(k): float(v) for k, v in theta.items()}
    if not th:
        raise ValueError("empty null table")
    ks = sorted(th)
    vals = [th[k] for k in ks]
    out, filled = {}, []
    for k in range(1, int(max_size) + 1):
        if k in th:
            out[k] = th[k]
            continue
        if mode == "strict":
            raise ValueError(
                f"null table has no entry for size {k}; "
                f"calibrated sizes are {ks}"
            )
        out[k] = float(np.interp(k, ks, vals))
        filled.append(k)
    return out, filled


def _support_descriptors(support, S, adjacency, C00):
    """Two cheap descriptors for a calibration support, recorded alongside
    its leading-null-rho contributions in `calibrate_null_thresholds` in case
    `|S|` alone turns out not to be a fine enough grouping for the null
    threshold.

    Returns `(contact_density, effective_dim)`:
      contact_density -- fraction of internal residue pairs in `support`
                         that are contacts.
      effective_dim    -- participation ratio of the eigenvalues of the
                         support's own raw (not null-corrected) unit
                         covariance block `C00[S, S]` -- the same formula
                         `k_eff` uses, applied to the equal-time spectrum
                         instead of the canonical-correlation one.
    """
    s = np.asarray(sorted(set(int(i) for i in support)), dtype=int)
    k = s.size
    if k < 2:
        density = float("nan")
    else:
        sub = np.asarray(adjacency, dtype=bool)[np.ix_(s, s)]
        n_pairs = k * (k - 1) // 2
        n_contacts = int(sub.sum()) // 2
        density = float(n_contacts) / float(n_pairs)

    block = C00[np.ix_(S, S)]
    w = np.maximum(np.linalg.eigvalsh(0.5 * (block + block.T)), 0.0)
    tot = float(w.sum())
    eff_dim = float(tot**2 / np.sum(w**2)) if tot > 0 else float("nan")

    return density, eff_dim


def calibrate_null_thresholds(
    workspace,
    adjacency,
    sizes,
    *,
    unit_dim,
    env_dim,
    target_rel_width=0.02,
    batch_supports=10,
    min_supports=50,
    max_pool_size=5000,
    seed=0,
    on_log=None,
    bootstrap="two_way",
    min_shifts=8,
):
    """One-time, size-conditioned null calibration for C6.

    Today, `FluxWorkspace.score()` recomputes C6's null threshold from
    scratch for every candidate it scores: the leading canonical correlation
    of `n_null` time-shifted environment replicas, thresholded at their 95th
    percentile, for that one candidate only.  This estimates the same
    threshold once per candidate SUPPORT SIZE `k` instead, pooled across many
    different supports of that size and many shift replicas, so `score()` can
    look up `theta_k` rather than recomputing it (see its `null_thresholds`
    argument).

    For each size `k` in `sizes`, this grows representative supports on the
    contact graph (the same 'ball' method `control_supports()` uses for its
    own spatially-compact controls -- not uniform-random residue sets, not
    sequence segments), scores each against the workspace's shift-null
    replicas (growing the shared shift pool via `workspace.extend_null_shifts`
    if it is smaller than a minimal working size), and pools the leading null
    correlations.  Data keeps being added for a given `k` until a bootstrap
    95% CI on the pooled 95th percentile is narrower than `target_rel_width`
    (relative to the estimate itself) AND at least `min_supports` supports
    have been scored, or until `max_pool_size` samples have been pooled for
    that `k`, whichever comes first.

    STOPPING RULE.  The bootstrap resamples whole supports
    (`_bootstrap_ci_rel_width_by_support`), not individual values: the
    n_shift values of one support share that support's own null level, so
    resampling them individually overstates the precision and stops early.
    The value-level width is still reported, once per size, as
    ``rel_ci_width_naive``.  `min_supports` keeps the support-level
    bootstrap from declaring convergence on a handful of supports, where it
    is itself unreliable; `min_supports * n_shifts` must not exceed
    `max_pool_size`.

    The shift pool (`workspace.Cs00` / `workspace.Cs01`) is shared across
    every size `k`, so it is grown at most once for the whole calibration
    run, not once per size.

    `unit_dim` and `env_dim` are REQUIRED and must be the values the
    calibrated table will later be scored with (`score()` / `decompose()`):
    the null correlation depends on both, so a table calibrated at one
    `env_dim` and used at another is a different null applied silently.  They
    are recorded in the returned ``"config"``, and `decompose()` refuses a
    calibration whose recorded dimensions differ from its own.

    Returns
    -------
    dict with:
      "theta"       : {k: float}            -- the frozen thresholds
      "diagnostics" : {k: {"n_pairs": int, "n_supports": int,
                            "rel_ci_width": float,
                            "rel_ci_width_naive": float,
                            "converged": bool, "stopped_by": str,
                            "elapsed_seconds": float,
                            "contact_density": list[float],
                            "effective_dim": list[float],
                            "per_support": list[dict]}}
          ``per_support`` keeps each support's own shift row together:
          {"support": residues, "leading_rho": [one per shift],
           "theta_S": 95th percentile of that row -- exactly the threshold
           the default per-candidate null in `score()` gives that support,
           "contact_density": float, "effective_dim": float}
      "config"      : the settings the table was calibrated with
                      (unit_dim, env_dim, ridge, n_shifts, lag, truncation,
                      reversible, target_rel_width, batch_supports,
                      min_supports, max_pool_size, seed)
      "elapsed_seconds" : total wall-clock time of the call

    """
    log = on_log if on_log is not None else (lambda *_: None)
    t_start = time.perf_counter()
    rng = np.random.default_rng(seed)
    adjacency = np.asarray(adjacency, dtype=bool)
    N = adjacency.shape[0]
    unit_dim, env_dim = int(unit_dim), int(env_dim)

    if bootstrap not in ("two_way", "support"):
        raise ValueError("bootstrap must be 'two_way' or 'support'")
    _ci = (
        _bootstrap_ci_rel_width_two_way
        if bootstrap == "two_way"
        else _bootstrap_ci_rel_width_by_support
    )
    min_working_shifts = max(8, int(min_shifts))
    if len(workspace.Cs00) < min_working_shifts:
        workspace.extend_null_shifts(min_working_shifts - len(workspace.Cs00))
    if int(min_supports) * len(workspace.Cs00) > int(max_pool_size):
        raise ValueError(
            f"min_supports={min_supports} x {len(workspace.Cs00)} shifts "
            f"exceeds max_pool_size={max_pool_size}: no size could ever be "
            f"declared converged"
        )

    def sample_support(k, tries=40):
        """Return a ball support of exactly `k` residues, best-effort."""
        s = _grow_ball_support(N, k, adjacency, rng)
        for _ in range(tries - 1):
            if len(s) == k:
                break
            s = _grow_ball_support(N, k, adjacency, rng)
        return s

    theta, diagnostics = {}, {}
    for k in sizes:
        k = int(k)
        t_size = time.perf_counter()
        pool, densities, eff_dims, per_support = [], [], [], []
        support_rows = []
        n_supports = 0
        rel_width, converged, stopped_by = float("inf"), False, None

        while True:
            for _ in range(batch_supports):
                support = sample_support(k)
                (support_, S, E, Vu, Ve, Ih, Ryy, Chy, Czz, Cyy, rho) = (
                    workspace._project_and_residualise(
                        support, unit_dim, env_dim
                    )
                )
                density, eff_dim = _support_descriptors(
                    support_, S, adjacency, workspace.C00
                )
                densities.append(density)
                eff_dims.append(eff_dim)
                n_supports += 1
                row = [
                    workspace._leading_null_rho(
                        Ih, Ryy, Chy, Vu, Ve, Czz, S, E, Cs00, Cs01
                    )
                    for Cs00, Cs01 in zip(workspace.Cs00, workspace.Cs01)
                ]
                pool.extend(row)
                support_rows.append(row)
                per_support.append(
                    {
                        "support": [int(i) for i in support_],
                        "leading_rho": [float(v) for v in row],
                        "theta_S": float(np.percentile(row, 95)),
                        "contact_density": float(density),
                        "effective_dim": float(eff_dim),
                    }
                )

            # The support-level bootstrap is only informative once there are
            # enough supports, so it is not run before min_supports.
            if n_supports >= min_supports:
                _, rel_width = _ci(support_rows, rng=rng)
            if n_supports >= min_supports and rel_width < target_rel_width:
                converged, stopped_by = True, "converged"
                break
            if len(pool) >= max_pool_size:
                stopped_by = "max_pool_size"
                log(
                    f"calibrate_null_thresholds: size {k} hit max_pool_size "
                    f"({max_pool_size}) before converging "
                    f"(rel CI width {rel_width:.4f} >= {target_rel_width})"
                )
                break

        _, rel_width_naive = _bootstrap_ci_rel_width(pool, rng=rng)
        # Extra diagnostics draw from their OWN stream, so the support
        # sequence (and the CPU/GPU modules' parity with it) is unchanged.
        diag_rng = np.random.default_rng([int(seed), int(k), 7])
        _, rel_width_sup = _bootstrap_ci_rel_width_by_support(
            support_rows, rng=diag_rng
        )
        _, rel_width_2w = _bootstrap_ci_rel_width_two_way(
            support_rows, rng=diag_rng
        )
        theta[k] = float(np.percentile(pool, 95))
        diagnostics[k] = {
            "n_pairs": len(pool),
            "n_supports": n_supports,
            "rel_ci_width": float(rel_width),
            "rel_ci_width_naive": float(rel_width_naive),
            "rel_ci_width_support": float(rel_width_sup),
            "rel_ci_width_two_way": float(rel_width_2w),
            "bootstrap": bootstrap,
            "variance_components": _variance_components(support_rows),
            "converged": bool(converged),
            "stopped_by": stopped_by,
            "elapsed_seconds": time.perf_counter() - t_size,
            "contact_density": densities,
            "effective_dim": eff_dims,
            "per_support": per_support,
        }
        log(
            f"calibrate_null_thresholds: size {k} theta={theta[k]:.4f} "
            f"n_pairs={len(pool)} n_supports={n_supports} "
            f"rel_ci_width={rel_width:.4f} (naive {rel_width_naive:.4f}) "
            f"converged={converged} "
            f"({diagnostics[k]['elapsed_seconds']:.1f}s)"
        )

    config = {
        "unit_dim": unit_dim,
        "env_dim": env_dim,
        "ridge": float(workspace.ridge),
        "n_shifts": int(len(workspace.Cs00)),
        "lag": int(workspace.lag),
        "truncation": workspace.truncation,
        "reversible": bool(workspace.reversible),
        "target_rel_width": float(target_rel_width),
        "batch_supports": int(batch_supports),
        "min_supports": int(min_supports),
        "max_pool_size": int(max_pool_size),
        "seed": int(seed),
        "bootstrap": bootstrap,
        "fingerprint": workspace_fingerprint(workspace),
    }
    return {
        "theta": theta,
        "diagnostics": diagnostics,
        "config": config,
        "elapsed_seconds": time.perf_counter() - t_start,
    }


def decompose(
    feats,
    adjacency,
    workspace,
    *,
    unit_dim=8,
    env_dim=20,
    n_seeds=24,
    max_size=16,
    min_size=3,
    max_unit=40,
    rank_stat="keff",
    cohesion_lambda=0.0,
    boundary_tol=0.15,
    merge_jaccard=0.5,
    size_penalty=0.02,
    max_iter=6,
    frontier_cap=12,
    knee_tol=0.15,
    n_controls=8,
    control_overlap=0.3,
    seed=0,
    on_log=None,
    null_calibration=None,
    memoise=True,
    scorer=None,
    cost_fn=None,
    null_fill="interpolate",
    keep_best=False,
    batch_scorer=None,
    seed_residues=None,
    merge_mode="union",
    merge_gate=None,
    merge_margin=0.1,
):
    """COG v1.0 (formerly THOR): assign EVERY residue to at least one PDU,
    allowing shared membership.

    Version series: COG v1.0 (this function) -> HULK v1.1 (`hulk.py`) ->
    IRON MAN v1.2 (`iron_man.py`) -> Triangle v1.3 (`triangle.py`) -> COG 2.x
    (`cog2*.py`).  The superseded versions are not part of this repository.

    The seed-and-grow procedure returns candidates, not a decomposition: run
    eight seeds and you get eight answers covering whatever they happened to
    reach.  This is the alternating procedure of v3.0 sec 10 steps 5-6 --
    propose supports, reassign residues, refit, repeat until stable -- with the
    coverage guarantee made explicit.

    MEMBERSHIP.  For residue i and unit alpha the affinity is how much i lowers
    that unit's channel count:

        w(i, alpha) = cost(alpha without i) - cost(alpha with i)

    computed from the same estimator, so it is measured, not declared.  Residue
    i joins EVERY unit whose affinity is within ``boundary_tol`` of its best --
    so a residue that serves two units equally is placed in both and reported
    as a BOUNDARY residue, exactly as D1's "sets may overlap" allows.

    COVERAGE.  A residue whose best affinity is not positive is still assigned
    to its best unit and flagged ``weak``, so the decomposition covers the
    whole protein.  Whether those weak assignments are real is a question for
    the report, not something to hide by leaving residues out.

    OUTCOMES.  Convergence to a single unit spanning the protein is a
    legitimate result -- v3.0 D2 calls a whole-protein candidate an explicit
    null outcome, and it would mean lysozyme has no usable internal
    decomposition at this resolution.  ``max_unit`` stops any one unit
    swallowing everything so that outcome shows up as several large units
    rather than a crash.

    Returns a dict: units (residue lists), membership (residue -> unit ids),
    boundary residues, per-unit scores, coverage, and the objective trace.

    ``null_calibration``, if given, is the dict `calibrate_null_thresholds()`
    returns (or just its ``"theta"`` sub-dict) -- a pre-calibrated, size-
    conditioned null-threshold table that every `workspace.score()` call in
    this function then uses in place of the default per-candidate null
    computation.  Left as `None` (the default), behavior is unchanged.
    Given the full dict, its recorded ``config`` unit_dim/env_dim must equal
    this call's, or a ValueError is raised; given only the bare ``"theta"``
    table there is nothing to check against, so a warning is logged instead.
    `decompose()` never calls `calibrate_null_thresholds()` itself: like
    `feats`, `adjacency` and `workspace`, calibration is a setup step the
    caller runs once and passes in.

    ``memoise`` (default True) caches `workspace.score()` by residue set for
    the duration of this call.  The same set is scored many times across the
    growth, reassignment, participation-weight and report phases, and
    `score()` is deterministic for a fixed workspace, so the cache changes
    run time only, never the result.  ``memoise=False`` restores the old
    behaviour exactly; the returned ``score_cache`` counts hits and misses.

    COG 2.1 hooks.  ``scorer`` (callable support -> result dict) replaces
    `workspace.score(support, **kw)`, and ``cost_fn`` (callable result ->
    float) replaces `_unit_cost(result, rank_stat, cohesion_lambda)`.  Left
    as None, behaviour is unchanged.  The matched controls are now scored
    with the SAME cost as the units (before, they always used the default
    k_eff cost, so a run with rank_stat='nuc_spec' or cohesion_lambda > 0
    compared units and controls on different scales).

    ``batch_scorer`` (callable list-of-supports -> list of result dicts,
    e.g. `Cog2Scorer.score_batch`): before each step that scores many
    candidates -- a growth step's frontier, one unit's add/remove affinities,
    a size's controls -- the not-yet-cached ones are scored together and
    put in the cache.  Needs memoise=True; results are identical, only
    faster on the GPU.

    ``keep_best`` (default False, which keeps v1.0 behaviour): the
    assignment loop is not monotone -- an iteration can raise the objective
    (on a planted benchmark it went -3.46 -> -0.32 over four iterations and
    returned the worst).  With keep_best=True the units of the iteration
    with the LOWEST objective are kept and reported; ``trace`` records which.

    ``null_fill`` ('interpolate' or 'strict') says what to do with sizes
    the null table does not cover; see `complete_theta_table`.  The table is
    completed up to N-1 before the run, so no size can silently score as
    infinite any more.
    """
    log = on_log if on_log is not None else (lambda *_: None)
    rng = np.random.default_rng(seed)
    adjacency = np.asarray(adjacency, dtype=bool)
    N = adjacency.shape[0]
    kw = dict(unit_dim=unit_dim, env_dim=env_dim)
    if null_calibration is not None:
        if "theta" in null_calibration:
            if null_calibration.get("config") is None:
                log(
                    "decompose: null_calibration has no 'config'; cannot "
                    "check it was calibrated on this workspace at unit_dim/"
                    f"env_dim {unit_dim}/{env_dim}"
                )
            else:
                check_null_calibration(
                    null_calibration, workspace, unit_dim, env_dim
                )
            theta_raw = null_calibration["theta"]
        else:
            log(
                "decompose: null_calibration is a bare theta table; cannot "
                "check it was calibrated at unit_dim/env_dim "
                f"{unit_dim}/{env_dim} -- pass the full dict "
                "calibrate_null_thresholds() returns to have that checked"
            )
            theta_raw = null_calibration
        table, filled = complete_theta_table(theta_raw, N - 1, mode=null_fill)
        if filled:
            calibrated = sorted(int(k) for k in theta_raw)
            log(
                f"decompose: null table calibrated at sizes "
                f"{calibrated[0]}-{calibrated[-1]} ({len(calibrated)} sizes); "
                f"{len(filled)} other sizes filled by interpolation / end "
                f"values"
            )
        kw["null_thresholds"] = table

    # Errors that genuinely mean 'this support cannot be scored' (no
    # environment left, a singular block).  Anything else is a bug and is
    # raised rather than turned into an infinite cost.
    _UNSCORABLE = (
        ValueError,
        np.linalg.LinAlgError,
        FloatingPointError,
        ZeroDivisionError,
    )

    def _cost_of(result):
        if cost_fn is not None:
            return float(cost_fn(result))
        return _unit_cost(
            result, rank_stat=rank_stat, cohesion_lambda=cohesion_lambda
        )

    # Per-call cache of workspace.score() keyed by the sorted residue set.
    # A failed score is cached as its exception and re-raised, so callers see
    # exactly what they saw before.
    _score_cache = {}
    cache_stats = {"hits": 0, "misses": 0}

    def _score(support):
        key = tuple(sorted(int(i) for i in support))
        if memoise and key in _score_cache:
            cache_stats["hits"] += 1
            res = _score_cache[key]
        else:
            cache_stats["misses"] += 1
            try:
                res = (
                    scorer(list(key))
                    if scorer is not None
                    else workspace.score(list(key), **kw)
                )
            except _UNSCORABLE as exc:
                res = exc
            if memoise:
                _score_cache[key] = res
        if isinstance(res, Exception):
            raise res
        return res

    def prefetch(supports):
        """Score the uncached supports together (batch_scorer)."""
        if batch_scorer is None or not memoise:
            return
        keys, seen = [], set()
        for sup in supports:
            if len(sup) == 0 or len(sup) >= N:
                continue
            key = tuple(sorted(int(i) for i in sup))
            if key in _score_cache or key in seen:
                continue
            seen.add(key)
            keys.append(key)
        if not keys:
            return
        try:
            res = batch_scorer([list(k) for k in keys])
        except _UNSCORABLE:
            return  # fall back to one-by-one scoring
        for k, r in zip(keys, res):
            cache_stats["misses"] += 1
            _score_cache[k] = r

    def cost(support):
        if len(support) == 0 or len(support) >= N:
            return float("inf")
        try:
            return _cost_of(_score(support))
        except _UNSCORABLE:
            return float("inf")

    def diagnostics(support):
        """Sigma and both rank statistics for a support, for the cost curve."""
        try:
            r = _score(support)
            return (
                r.get("k_eff", float("nan")),
                r.get("nuc_spec", float("nan")),
                r.get("sigma", float("nan")),
            )
        except _UNSCORABLE:
            return (float("nan"),) * 3

    # ---- phase 1: spread seeds, grow candidates -------------------------
    if seed_residues is not None:
        seeds = [int(x) for x in seed_residues]
        log(f"phase 1: growing from {len(seeds)} randomly drawn seeds")
    else:
        seeds = _farthest_point_seeds(adjacency, n_seeds, rng)
        log(f"phase 1: growing from {len(seeds)} spread seeds")
    candidates, flat_curves = [], 0
    for si, s in enumerate(seeds, 1):
        support = [int(s)]
        curve = [(1, cost(support), list(support))]
        while len(support) < max_size:
            frontier = np.where(adjacency[support].any(axis=0))[0]
            frontier = [int(j) for j in frontier if j not in support]
            if not frontier:
                break
            if frontier_cap and len(frontier) > frontier_cap:
                d = np.array([adjacency[j][support].sum() for j in frontier])
                frontier = [frontier[t] for t in np.argsort(-d)[:frontier_cap]]
            prefetch([support + [j] for j in frontier])
            scored = [(cost(support + [j]), j) for j in frontier]
            scored.sort(key=lambda t: t[0])
            c, j = scored[0]
            support.append(j)
            curve.append((len(support), c, list(support)))

        # Choose the LARGEST support within `knee_tol` of the best cost, over
        # sizes of at least `min_size`.
        #
        # The strict argmin is wrong here and was the bug: the cost floors at
        # one channel, so a lone residue that already sits at the floor can
        # never be beaten and the candidate stays at size 1 -- nine of the
        # first nineteen seeds on lysozyme returned size 1 or 2 and were then
        # discarded by min_size.  Phase 1 supplies STARTING POINTS for the
        # assignment phase, which trims them; erring large is right, erring
        # small throws the seed away.
        pool = [c for c in curve if c[0] >= min_size] or curve
        floor = min(c[1] for c in pool)
        kt = (
            knee_tol(min(pool, key=lambda c: c[1])[0])
            if callable(knee_tol)
            else knee_tol
        )
        at_floor = [c for c in pool if c[1] <= floor + kt]
        size_sel, cost_sel, sup_sel = max(at_floor, key=lambda c: c[0])
        # A curve that sits within tolerance of its floor at every size carries
        # NO size information -- the boundary is not determined by the data for
        # this seed, and whatever size comes out is the search limit, not a
        # measurement.  Counted and reported rather than hidden.
        flat = len(at_floor) >= max(2, int(0.8 * len(pool)))
        flat_curves += int(flat)
        candidates.append(sorted(sup_sel))
        log(
            f"  seed {si}/{len(seeds)} (res {int(s)}): size {size_sel} "
            f"cost {cost_sel:.2f} (curve min {floor:.2f} over sizes "
            f"{pool[0][0]}-{pool[-1][0]})"
            f"{'  FLAT: size undetermined' if flat else ''}"
        )

    log(
        f"phase 1 done: {flat_curves}/{len(seeds)} seeds had a FLAT cost "
        f"curve (no size preference from the data)"
    )

    def _merge(sups):
        if merge_mode == "reearn":
            return _merge_reearn(
                sups,
                merge_jaccard,
                cost,
                prefetch,
                merge_gate,
                margin=merge_margin,
                min_size=min_size,
                log=log,
            )
        return _merge_by_overlap(sups, merge_jaccard)

    units = _merge(candidates)
    log(
        f"phase 2: {len(candidates)} candidates -> {len(units)} units after "
        f"merging at Jaccard {merge_jaccard}"
    )

    # ---- phase 3: alternate assignment and refit ------------------------
    trace, membership = [], None
    best_it = None  # (J, iteration, units, membership)
    for it in range(1, int(max_iter) + 1):
        prefetch(units)
        base = [cost(u) for u in units]
        aff = np.full((N, len(units)), -np.inf)
        for a, u in enumerate(units):
            uset = set(u)
            near = set(u) | {
                int(j) for j in np.where(adjacency[u].any(axis=0))[0]
            }
            prefetch(
                [
                    sorted(uset - {i}) if i in uset else sorted(uset | {i})
                    for i in sorted(near)
                ]
            )
            for i in sorted(near):
                if i in uset:
                    if len(u) <= 1:
                        continue
                    aff[i, a] = cost(uset - {i}) - base[a]
                else:
                    if (
                        merge_mode == "reearn"
                        and merge_gate is not None
                        and not merge_gate(i, uset)
                    ):
                        continue  # not dynamically consistent: no
                        # addition to this unit
                    aff[i, a] = base[a] - cost(uset | {i})

        membership = [[] for _ in range(N)]
        weak = []
        for i in range(N):
            row = aff[i]
            if not np.any(np.isfinite(row)):
                continue
            best = float(np.nanmax(row[np.isfinite(row)]))
            if best > 0:
                thr = best - boundary_tol * abs(best)
                membership[i] = [
                    a
                    for a in range(len(units))
                    if np.isfinite(row[a]) and row[a] >= thr
                ]
            else:
                membership[i] = [
                    int(np.argmax(np.where(np.isfinite(row), row, -np.inf)))
                ]
                weak.append(i)

        new_units = []
        for a in range(len(units)):
            members = [i for i in range(N) if a in membership[i]]
            if len(members) >= min_size:
                if len(members) > max_unit:
                    # keep the members with the strongest affinity rather than
                    # the lowest indices, so the cap truncates the periphery
                    members = [
                        int(i)
                        for i in sorted(members, key=lambda i: -aff[i, a])[
                            :max_unit
                        ]
                    ]
                new_units.append(sorted(members))
        if not new_units:
            new_units = units
        new_units = _merge(new_units)

        # Membership was indexed against the PRE-merge units; after merging and
        # dropping small units those indices no longer refer to the same sets.
        # Rebuild it from the final lists so membership, boundary residues and
        # the unit lists all agree -- they did not before, and 165 assignments
        # were silently inconsistent on the first lysozyme decomposition.
        membership = [
            [a for a, u in enumerate(new_units) if i in set(u)]
            for i in range(N)
        ]
        for i in range(N):
            if membership[i]:
                continue
            row = aff[i]
            if np.any(np.isfinite(row)):
                # re-home an orphan created by merging: nearest unit by
                # affinity
                order = np.argsort(np.where(np.isfinite(row), row, -np.inf))[
                    ::-1
                ]
                for a_old in order:
                    old_set = (
                        set(units[a_old]) if a_old < len(units) else set()
                    )
                    for a_new, u in enumerate(new_units):
                        if old_set & set(u):
                            if (
                                merge_mode == "reearn"
                                and merge_gate is not None
                                and not merge_gate(i, set(u))
                            ):
                                continue
                            membership[i] = [a_new]
                            break
                    if membership[i]:
                        break
            if not membership[i] and new_units and merge_mode != "reearn":
                # last resort: the unit with the most contacts to i
                best = int(
                    np.argmax([adjacency[i][u].sum() for u in new_units])
                )
                membership[i] = [best]

        J = sum(cost(u) for u in new_units) + size_penalty * sum(
            len(u) for u in new_units
        )
        covered = sum(1 for m in membership if m)
        trace.append(
            {
                "iteration": it,
                "n_units": len(new_units),
                "objective": float(J),
                "covered": int(covered),
                "weak": len(weak),
            }
        )
        log(
            f"iter {it}: {len(new_units)} units, objective {J:.2f}, "
            f"covered {covered}/{N}, weak {len(weak)}"
        )

        if best_it is None or J < best_it[0]:
            best_it = (
                float(J),
                it,
                [sorted(u) for u in new_units],
                [list(m) for m in membership],
            )

        same = [sorted(u) for u in new_units] == [sorted(u) for u in units]
        units = new_units
        if same and it > 1:
            log("converged: assignments unchanged")
            break

    if (
        keep_best
        and best_it is not None
        and best_it[1] != trace[-1]["iteration"]
    ):
        log(
            f"keep_best: returning iteration {best_it[1]} (objective "
            f"{best_it[0]:.2f}) instead of the last "
            f"({trace[-1]['objective']:.2f})"
        )
        units, membership = best_it[2], best_it[3]
    if trace:
        trace[-1]["kept_iteration"] = (
            best_it[1] if keep_best and best_it else trace[-1]["iteration"]
        )

    # ---- phase 3b: participation weights against the FINAL units ---------
    # The affinities computed during the loop were scored against the PRE-merge
    # units, so they cannot be reported as weights for the final ones.
    # Rescored here, restricted to each unit's members and contact frontier.
    #
    # TWO NORMALISATIONS, kept separate because v3.0 sec 6.3 is explicit that
    # they are different objects and must not be conflated:
    #   weight_residue[i][a]  -- residue-centric, sums to 1 over the units i
    #                            belongs to.  This is the membership field mu:
    #                            a boundary residue shows ~0.5/0.5.
    #   weight_unit[a][i]     -- unit-centric, sums to 1 over unit a's members.
    #                            This is the footprint rho of eq. (26): which
    #                            residues carry the unit.
    #
    # Both are PREDICTIVE NECESSITY (v3.0 sec 6.2), not causal influence and
    # not a conserved amount of dynamics: w is the drop in channel count when
    # the residue is withheld, so a residue with a correlated substitute
    # elsewhere in the unit can score near zero while still mattering
    # physically.
    log("phase 3b: participation weights against the final units")
    prefetch(units)
    base_final = [cost(u) for u in units]
    affinity = np.full((N, len(units)), np.nan)
    for a, u in enumerate(units):
        uset = set(u)
        near = uset | {int(j) for j in np.where(adjacency[u].any(axis=0))[0]}
        prefetch(
            [
                sorted(uset - {i}) if i in uset else sorted(uset | {i})
                for i in sorted(near)
            ]
        )
        for i in sorted(near):
            if i in uset:
                if len(u) <= 1:
                    continue
                affinity[i, a] = cost(uset - {i}) - base_final[a]
            else:
                affinity[i, a] = base_final[a] - cost(uset | {i})

    weight_residue, weight_unit = [], []
    for i in range(N):
        ws_ = {}
        vals = [
            (a, affinity[i, a])
            for a in membership[i]
            if np.isfinite(affinity[i, a])
        ]
        pos = [(a, max(v, 0.0)) for a, v in vals]
        tot = sum(v for _, v in pos)
        for a in membership[i]:
            if tot > 0:
                ws_[a] = (
                    float(max(affinity[i, a], 0.0) / tot)
                    if np.isfinite(affinity[i, a])
                    else 0.0
                )
            else:
                # every affinity zero or negative: the residue is in this unit
                # by the coverage guarantee, not because it earns its place.
                # Split evenly and let `weak` flag it.
                ws_[a] = 1.0 / max(len(membership[i]), 1)
        weight_residue.append({str(a): round(v, 4) for a, v in ws_.items()})
    for a, u in enumerate(units):
        vals = {
            i: max(float(affinity[i, a]), 0.0)
            for i in u
            if np.isfinite(affinity[i, a])
        }
        tot = sum(vals.values())
        if tot > 0:
            weight_unit.append(
                {str(i): round(v / tot, 4) for i, v in sorted(vals.items())}
            )
        else:
            weight_unit.append(
                {str(i): round(1.0 / max(len(u), 1), 4) for i in sorted(u)}
            )

    # ---- phase 4: controls at matched sizes ------------------------------
    # Eleven units with k_eff 1.7-3.4 mean nothing without this: the units are
    # the output of a search, the controls are not, so the comparison is the
    # only thing that says whether the search found structure.
    controls = {}
    if n_controls:
        for size in sorted({len(u) for u in units}):
            cs = control_supports(
                feats,
                size,
                adjacency=adjacency,
                n_random=n_controls,
                n_segment=n_controls,
                n_ball=n_controls,
                seed=seed + size,
                exclude=units,
                max_overlap=control_overlap,
            )
            rejected = cs.pop("_rejected_for_overlap", 0)
            ov_med = cs.pop("_overlap_median", float("nan"))
            ov_max = cs.pop("_overlap_max", float("nan"))
            row = {}
            prefetch([S_ for sets in cs.values() for S_ in sets])
            for kind, sets in cs.items():
                vals = []
                for S in sets:
                    try:
                        vals.append(_cost_of(_score(S)))
                    except _UNSCORABLE:
                        continue
                row[kind] = vals
            controls[str(size)] = row
            meds = {
                k: (float(np.median(v)) if v else float("nan"))
                for k, v in row.items()
            }
            log(
                f"controls size {size}: "
                + "  ".join(f"{k}={v:.2f}" for k, v in meds.items())
                + (
                    f"   [overlap with units: median {ov_med:.2f}, "
                    f"max {ov_max:.2f}"
                    + (
                        f"; {rejected} could not be made disjoint]"
                        if rejected
                        else "]"
                    )
                    if np.isfinite(ov_med)
                    else ""
                )
            )

    # ---- phase 5: report -------------------------------------------------
    scores = []
    for a, u in enumerate(units):
        try:
            r = _score(u)
            c = _cost_of(r)
        except _UNSCORABLE:
            r, c = {}, float("inf")
        entry = {
            "unit": a,
            "size": len(u),
            "residues": sorted(u),
            "k_eff": r.get("k_eff"),
            "k_star": r.get("k_star"),
            "resolved": r.get("resolved"),
            "margin": r.get("margin"),
            "cost": float(c),
        }
        if scorer is not None:
            entry["detail"] = {
                k: v
                for k, v in r.items()
                if k not in ("support", "delta_int_curve")
            }
        scores.append(entry)
    boundary = [i for i in range(N) if len(membership[i]) > 1]
    orphan = [i for i in range(N) if not membership[i]]
    # Two gaps, because they answer different questions and conflating them is
    # what produced a spurious negative:
    #   gap_vs_random   -- beats scattered residues?  spatial coherence counts.
    #   gap_vs_best     -- beats the best matched control, now that the
    #                      controls
    #                      are disjoint from the discovered units.
    gaps, gaps_random = {}, {}
    for sc in scores:
        row = controls.get(str(sc["size"]))
        # With a custom cost the comparison is cost against cost; the
        # default path keeps the previous k_eff comparison.
        k = sc.get("cost") if cost_fn is not None else sc.get("k_eff")
        if row and k is not None and np.isfinite(k):
            meds = {kk: float(np.median(v)) for kk, v in row.items() if v}
            if meds:
                gaps[sc["unit"]] = float(min(meds.values()) - k)
            if "random" in meds:
                gaps_random[sc["unit"]] = float(meds["random"] - k)
    boundary_detail = [
        {
            "residue": i,
            "units": sorted(membership[i]),
            "weights": weight_residue[i],
            "balance": round(
                min(weight_residue[i].values())
                / max(max(weight_residue[i].values()), 1e-12),
                3,
            ),
        }
        for i in range(N)
        if len(membership[i]) > 1
    ]
    return {
        "units": [sorted(u) for u in units],
        "scores": scores,
        "controls": controls,
        "gap_vs_best_control": gaps,
        "gap_vs_random": gaps_random,
        "control_overlap_threshold": float(control_overlap),
        "affinity": [
            [
                None if not np.isfinite(v) else round(float(v), 5)
                for v in affinity[i]
            ]
            for i in range(N)
        ],
        "weight_residue": weight_residue,
        "weight_unit": weight_unit,
        "boundary_detail": boundary_detail,
        "membership": [sorted(m) for m in membership],
        "boundary_residues": boundary,
        "orphans": orphan,
        "coverage": float((N - len(orphan)) / N),
        "n_units": len(units),
        "trace": trace,
        "score_cache": dict(cache_stats, memoise=bool(memoise)),
        "flat_curve_seeds": int(flat_curves),
        "n_seeds": len(seeds),
        "config": {
            "unit_dim": unit_dim,
            "env_dim": env_dim,
            "n_seeds": n_seeds,
            "max_size": max_size,
            "boundary_tol": boundary_tol,
            "merge_jaccard": merge_jaccard,
            "size_penalty": size_penalty,
            "max_iter": max_iter,
        },
    }
