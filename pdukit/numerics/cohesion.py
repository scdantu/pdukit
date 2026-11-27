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


def blockiness(rho, units, n_random=200, rng=None):
    """How block-like the coupling is for a set of units.

    Pairs of residues that share a unit ("within") are compared with
    pairs of assigned residues that share none ("between"), using the
    two-way coupling (rho[i, j] + rho[j, i]) / 2:

        ratio  mean within / mean between
        auc    P(a within pair beats a between pair), ties half; 0.5 = no
               blocks, 1 = every within pair beats every between pair

    The null draws random residue sets of the same sizes (independently,
    so overlapping units stay overlapping on average) ``n_random`` times;
    ``z`` and ``percentile`` place the found units in that null.  A flat
    coupling landscape gives an auc near 0.5 and a found auc inside the
    null.
    """
    R = np.asarray(rho, float)
    S2 = 0.5 * (R + R.T)
    N = S2.shape[0]
    rng = np.random.default_rng(0) if rng is None else rng

    def score(sets):
        member = np.zeros((len(sets), N), bool)
        for k, s in enumerate(sets):
            member[k, sorted({int(x) for x in s})] = True
        assigned = member.any(0)
        share = (member.T.astype(int) @ member.astype(int)) > 0
        iu = np.triu_indices(N, 1)
        both = assigned[iu[0]] & assigned[iu[1]]
        within = share[iu] & both
        between = ~share[iu] & both
        w, b = S2[iu][within], S2[iu][between]
        w, b = w[np.isfinite(w)], b[np.isfinite(b)]
        if len(w) == 0 or len(b) == 0:
            return float("nan"), float("nan"), float("nan"), float("nan")
        return float(w.mean()), float(b.mean()), float(
            w.mean() / b.mean()) if b.mean() else float("nan"), auc(w, b)

    sets = [list(u) for u in units if len(u) >= 2]
    within, between, ratio, found = score(sets)
    sizes = [len(set(u)) for u in sets]
    null = []
    for _ in range(int(n_random)):
        rand = [rng.choice(N, k, replace=False) for k in sizes]
        null.append(score(rand)[3])
    null = np.array([x for x in null if np.isfinite(x)])
    sd = float(null.std(ddof=1)) if len(null) > 1 else float("nan")
    nan = float("nan")
    mean = float(null.mean()) if len(null) else nan
    return {
        "within_mean": within,
        "between_mean": between,
        "ratio": ratio,
        "auc": found,
        "null_auc_mean": mean,
        "null_auc_sd": sd,
        "z": float((found - mean) / sd) if sd and sd > 0 else nan,
        "percentile": (
            float(100 * np.mean(null < found)) if len(null) else nan
        ),
        "n_random": int(len(null)),
        "n_units": len(sets),
    }
