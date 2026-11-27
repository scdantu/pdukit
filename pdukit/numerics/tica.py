"""Reversible tICA on time series split into replicas (segments)."""

from __future__ import annotations

import numpy as np


def lagged_pairs(segments, lag):
    """Row indices (t, t + lag) inside each replica, never across seams."""
    r0 = [np.arange(a, b - lag) for a, b in segments if b - a > lag]
    r1 = [np.arange(a + lag, b) for a, b in segments if b - a > lag]
    if not r0:
        raise ValueError("lag too long for every replica")
    return np.concatenate(r0), np.concatenate(r1)


def reversible_tica(X, segments, lag, n_modes, ridge=1e-6):
    """Slowest ``n_modes`` directions of X (frames, features).

    Whiten with the symmetrised equal-time covariance, diagonalise the
    symmetrised lagged covariance.  Returns (eigenvalues, vectors) with
    vectors (features, n_modes) such that X_centred @ vectors are the mode
    time series with unit variance.  ``ridge`` is relative to the mean
    variance.
    """
    X = np.asarray(X, float)
    X = X - X.mean(0)
    i0, i1 = lagged_pairs(segments, int(lag))
    A, B = X[i0], X[i1]
    n = len(i0)
    C0 = 0.5 * (A.T @ A + B.T @ B) / n
    Ct = 0.5 * (A.T @ B + B.T @ A) / n
    eps = ridge * max(np.trace(C0) / len(C0), 1e-30)
    w, V = np.linalg.eigh(C0 + eps * np.eye(len(C0)))
    keep = w > 1e-12 * w.max()
    L = V[:, keep] / np.sqrt(w[keep])
    lam, U = np.linalg.eigh(L.T @ Ct @ L)
    order = np.argsort(lam)[::-1][: int(n_modes)]
    return lam[order], L @ U[:, order]


def timescale(eigenvalue, lag):
    """Implied timescale -lag / ln(lambda); None if lambda not in (0, 1)."""
    if not 0 < eigenvalue < 1:
        return None
    return float(-lag / np.log(eigenvalue))


def explained(target, predictors):
    """R^2 of a least-squares fit of ``target`` from ``predictors``."""
    y = np.asarray(target, float) - np.mean(target)
    P = np.asarray(predictors, float)
    P = P - P.mean(0)
    coef, *_ = np.linalg.lstsq(P, y, rcond=None)
    resid = y - P @ coef
    return float(1 - resid @ resid / max(y @ y, 1e-30))
