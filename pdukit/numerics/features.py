"""Per-residue features from coordinates: relative-structure-error features."""

from __future__ import annotations

import numpy as np

from pdukit.numerics.geometry import detect_length_unit, median_distance_matrix
from pdukit.numerics.kernel import _inv_sqrt


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
    coverage=0.9,
    report=None,
    max_components=0,
    min_share=0.0,
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
    n_components : components kept per residue; 0 or less = choose the
                   number from ``coverage`` (see `modes_for_coverage`)
    min_seq_sep  : exclude near-in-sequence neighbours as trivially rigid
    max_components : upper limit on the chosen number (0 = none); memory
                   grows with it (frames x residues x number x 4 bytes)
    coverage     : target share of each residue's spectrum (kinetic variance
                   sum lambda^2 for tICA, variance for PCA) used when the
                   number is chosen automatically, and reported always
    min_share    : with the automatic number, a mode is kept only if it
                   adds at least this share of the residue's spectrum
                   (0 = no such condition); see `modes_for_coverage`
    spectra      : list; each residue's full eigenvalue list is appended
    report       : dict; filled with the number of components used, the
                   coverage each residue keeps with it, and the number each
                   residue would need to reach ``coverage``

    AUTOMATIC NUMBER OF COMPONENTS.  With ``n_components <= 0`` every
    residue's spectrum is computed first, then each residue needs the
    fewest modes that reach ``coverage`` of its spectrum, but no more than
    the modes that each add at least ``min_share``.  K is the median of
    that over residues (at least 2, at most ``max_components``), and the
    distances are computed a second time to project on K components
    (slower: two passes).

    Returns
    -------
    feats        : (F, N, K) with K = n_components or the chosen number
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

    auto = int(n_components) < 1
    if not 0 < float(coverage) <= 1:
        raise ValueError(f"coverage must be in (0, 1], got {coverage}")
    explained = np.zeros(N)
    lag = int(lag)
    if segments is None:
        segments = [(0, F)]
    r0i = np.concatenate(
        [np.arange(a, b - lag) for a, b in segments if b - a > lag]
    )
    r1i = np.concatenate(
        [np.arange(a + lag, b) for a, b in segments if b - a > lag]
    )

    neighbours = []
    for i in range(N):
        nb = np.where(ok[i])[0]
        if nb.size == 0:  # fall back to nearest few
            d = np.where(sep[i] >= max(1, min_seq_sep), dmed[i], np.inf)
            nb = np.argsort(d)[: int(n_components) if not auto else 8]
        neighbours.append(nb)

    def distances(i):
        """Centred distances from residue i to its RSE, (F, n_i)."""
        nb = neighbours[i]
        Di = np.empty((F, nb.size), dtype=np.float32)
        for a in range(0, F, chunk):
            b = min(a + chunk, F)
            Xc = np.asarray(X[a:b], dtype=np.float32)
            diff = Xc[:, i, None, :] - Xc[:, nb, :]
            Di[a:b] = np.sqrt(np.einsum("cnk,cnk->cn", diff, diff))
        if distance_filter is not None:
            distance_filter(Di, segments)
        Di -= Di.mean(axis=0, dtype=np.float64).astype(np.float32)
        return Di

    def basis(Di):
        """All directions, slowest (or largest) first, with eigenvalues."""
        D64 = Di.astype(np.float64)
        C = (D64.T @ D64) / max(F, 1)
        C = 0.5 * (C + C.T)
        tot = float(np.sum(np.maximum(np.linalg.eigvalsh(C), 0.0)))
        if not koopman:
            w, V = np.linalg.eigh(C)
            order = np.argsort(w)[::-1]
            return V[:, order], w[order], C, tot
        A0, A1 = D64[r0i], D64[r1i]
        nn = max(A0.shape[0], 1)
        C0 = (A0.T @ A0) / nn
        C1 = (A0.T @ A1) / nn
        L = _inv_sqrt(0.5 * (C0 + C0.T), ridge)
        if reversible:
            M = L @ (0.5 * (C1 + C1.T)) @ L
            w, U = np.linalg.eigh(0.5 * (M + M.T))
            order = np.argsort(w)[::-1]
            return L @ U[:, order], w[order], C, tot
        Ct = (A1.T @ A1) / nn
        U, sv, _ = np.linalg.svd(
            L @ C1 @ _inv_sqrt(0.5 * (Ct + Ct.T), ridge), full_matrices=False
        )
        return L @ U, sv, C, tot

    def retained(P, w, C, tot):
        """Share of the residue's RSE variance kept by the columns of P."""
        if tot <= 0:
            return 0.0
        if not koopman:
            return float(np.sum(np.maximum(w, 0.0))) / tot
        # variance retained by the SLOW directions (not the top-variance
        # ones), so the number is comparable across the two reductions
        Q = P / np.maximum(np.linalg.norm(P, axis=0, keepdims=True), 1e-12)
        return float(np.trace(Q.T @ C @ Q)) / tot

    def project(feats, i, Di, P, w, C, tot):
        k = feats.shape[2]
        kk = int(min(k, P.shape[1]))
        Pk = np.ascontiguousarray(P[:, :kk])
        explained[i] = retained(Pk, w[:kk], C, tot)
        feats[:, i, :kk] = Di @ Pk.astype(np.float32)
        if kk < k:  # pad if the RSE was tiny
            feats[:, i, kk:] = 0.0

    eigen = []
    kept_bases = [] if auto else None
    feats = None if auto else np.empty((F, N, int(n_components)), dtype=dtype)
    for i in range(N):
        Di = distances(i)
        P, w, C, tot = basis(Di)
        eigen.append(w)
        if spectra is not None and koopman:
            spectra.append(w.astype(float).tolist())
        if auto:
            kept_bases.append((P, w, C, tot))
        else:
            project(feats, i, Di, P, w, C, tot)
        if on_progress is not None:
            on_progress(i + 1, N)

    power = 2 if koopman else 1
    k_cov, _ = modes_for_coverage(eigen, coverage, power=power)
    k_need, chosen = modes_for_coverage(
        eigen, coverage, power=power, min_share=min_share
    )
    if auto:
        k = max(2, min(k_need, max(len(w) for w in eigen)))
        if max_components and max_components > 0:
            k = min(k, int(max_components))
        feats = np.empty((F, N, k), dtype=dtype)
        for i in range(N):
            project(feats, i, distances(i), *kept_bases[i])
            if on_progress is not None:
                on_progress(i + 1, N)
    k = feats.shape[2]
    if report is not None:
        curves = [coverage_curve(w, power) for w in eigen]
        report.update(
            n_components=int(k),
            chosen=bool(auto),
            coverage_target=float(coverage),
            coverage_measure="sum lambda^2" if koopman else "variance",
            coverage_kept=[
                float(c[min(k, len(c)) - 1]) if len(c) else 0.0
                for c in curves
            ],
            modes_for_target=[
                _first_reaching(c, coverage) for c in curves
            ],
            modes_for_target_median=int(k_cov),
            min_share=float(min_share),
            modes_by_share=[
                modes_above_share(w, min_share, power) for w in eigen
            ] if min_share > 0 else None,
            modes_chosen_median=int(k_need),
            limited_by=(
                "min_share" if min_share > 0 and k_need < k_cov
                else "coverage"
            ),
        )
    return feats, neighbours, explained


def coverage_curve(eigenvalues, power=2):
    """Share of a residue's spectrum kept by its first 1, 2, ... modes.

    power 2: kinetic variance sum lambda_k^2 of tICA eigenvalues (lag
    autocorrelations); power 1: variance (PCA eigenvalues).  Negative
    values count as zero.  Returns the cumulative share, ending at 1.
    """
    e = np.clip(np.asarray(eigenvalues, float), 0.0, None) ** power
    tot = e.sum()
    if tot <= 0:
        return np.ones(len(e))
    return np.cumsum(e) / tot


def _first_reaching(curve, target):
    """Smallest number of modes whose share reaches target (1-based)."""
    if len(curve) == 0:
        return 0
    return int(min(np.searchsorted(curve, target - 1e-12) + 1, len(curve)))


def modes_above_share(eigenvalues, min_share, power=2):
    """Return how many leading modes each add >= ``min_share`` (at least 1).

    The share of mode k is lambda_k^power over the sum over all modes
    (negative values count as zero).  Counting stops at the first mode
    below the threshold.
    """
    e = np.clip(np.asarray(eigenvalues, float), 0.0, None) ** power
    tot = e.sum()
    if tot <= 0 or len(e) == 0:
        return 1
    below = np.nonzero(e / tot < float(min_share))[0]
    return int(max(1, below[0] if below.size else len(e)))


def modes_for_coverage(spectra, coverage, power=2, min_share=0.0):
    """Return the number of modes needed to keep ``coverage`` of spectra.

    Each residue needs the smallest K whose `coverage_curve` reaches the
    target; with ``min_share`` > 0 it gets no more than the leading modes
    that each add at least that share (`modes_above_share`): get as close
    to the target as modes worth keeping allow.  The number returned is
    the median over residues, rounded up, so half the residues keep at
    least that.  Returns ``(K, per_residue)``.
    """
    per = [
        _first_reaching(coverage_curve(w, power), coverage) for w in spectra
    ]
    if min_share and min_share > 0:
        per = [
            min(k, modes_above_share(w, min_share, power))
            for k, w in zip(per, spectra)
        ]
    if not per:
        return 0, per
    return int(np.ceil(np.median(per))), per
