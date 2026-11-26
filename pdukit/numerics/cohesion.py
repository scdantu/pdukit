"""Predictive cohesion of a residue set, and the nulls it is compared with.

``rho[i, j]`` is how much residue j's present predicts residue i's future
beyond i's own present (directed).  For a set S:

    inside    rho[a, b] for members a != b
    driver    rho[k, i]  member i -> outsider k
    listener  rho[i, k]  outsider k -> member i
    cohesion  = (P(inside > driver) + P(inside > listener)) / 2

Each P is an area under the ROC curve (Mann-Whitney U / (n1 n2), ties
counted half): 0.5 for a random group, 1 when every inside link beats every
boundary link.  See `pdukit.cog.units.cohesion` for how COG uses it.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def auc(inside, boundary):
    """P(inside value > boundary value), ties half; nan if either is empty."""
    n1, n2 = len(inside), len(boundary)
    if n1 == 0 or n2 == 0:
        return float("nan")
    r = rankdata(np.concatenate([inside, boundary]))
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n2))


def cohesion(rho, S):
    """dict(auc_driver, auc_listener, cohesion) for the residue set S."""
    R = np.asarray(rho, float)
    N = R.shape[0]
    S = np.array(sorted({int(x) for x in S}))
    E = np.setdiff1d(np.arange(N), S)
    if len(S) < 2 or len(E) == 0:
        return {
            "auc_driver": float("nan"),
            "auc_listener": float("nan"),
            "cohesion": float("nan"),
        }
    inside = R[np.ix_(S, S)][~np.eye(len(S), dtype=bool)]
    d = auc(inside, R[np.ix_(E, S)].ravel())
    li = auc(inside, R[np.ix_(S, E)].ravel())
    return {"auc_driver": d, "auc_listener": li, "cohesion": 0.5 * (d + li)}


def random_sets(n_residues, size, n_sets, rng):
    """``n_sets`` uniformly random residue sets of ``size`` (no repeats)."""
    return [rng.choice(n_residues, size, replace=False) for _ in range(n_sets)]


def compact_region(adj, size, rng):
    """Grow a random spatially connected region on the contact graph.

    Starts at a random residue and adds a random neighbour of the region
    until it has ``size`` residues; None if the component is too small.
    """
    N = len(adj)
    s = rng.integers(N)
    S = [s]
    front = set(np.where(adj[s])[0])
    while len(S) < size and front:
        r = rng.choice(sorted(front))
        S.append(r)
        front |= set(np.where(adj[r])[0])
        front -= set(S)
    return S if len(S) == size else None


def compact_regions(adj, size, n_regions, rng, max_tries=None):
    """Up to ``n_regions`` compact regions of ``size`` (see compact_region)."""
    tries = max_tries or 2 * n_regions
    found = (compact_region(adj, size, rng) for _ in range(tries))
    return [r for r in found if r is not None][:n_regions]


def null_quantile(rho, sets, q=95.0):
    """Return the q-th percentile of cohesion over the null sets."""
    values = [cohesion(rho, s)["cohesion"] for s in sets]
    return float(np.percentile(values, q))


def boot_sd(boot_rho, S):
    """Return the sd of cohesion(S) over bootstrap copies of rho."""
    values = [cohesion(b, S)["cohesion"] for b in boot_rho]
    return float(np.std(values, ddof=1))


def above_null(value, sd, null_q, z=1.645):
    """Return True if value - z sd beats the null quantile (one-sided)."""
    return bool(value - z * sd > null_q)
