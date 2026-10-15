"""Dynamics consistency gate for ADDING a residue to a unit (13 Oct 2026).

A residue r that is new to a unit S may join only if it predicts the unit's
members better than it predicts the rest of the protein:

    mean_{i in S} rho[i, r]  >  mean_{o not in S, o != r} rho[o, r]

rho[i, r] = leading partial canonical correlation of r's present with i's
future given i's present (cog2_graph.pair_rho).  The test uses every member,
not one link, so membership is not assumed to be transitive.  No tuning
parameter.  On the 13 Oct runs, members of closed units pass it (T4L 82-100%,
Syt closed 96%), weak Syt units much less (53-89%).

Removals are never gated (J decides).  A residue returning to its own
original unit is not gated.

Directions (14 Oct 2026): directions matter for allostery.  For a member
r of S, with the other members m and the residues outside o:
    driver    r predicts the members' futures:   rho[m, r]  vs  rho[o, r]
    listener  the members predict r's future:     rho[r, m]  vs  rho[r, o]
Each direction is tested one-sided (Mann-Whitney U, members > outside) at
level alpha.  Label: 'both', 'driver', 'listener' or None.  A member with a
label is PROTECTED in the MC: J alone cannot remove it (cog2_stability).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import mannwhitneyu


def ratio(rho, r, members):
    """Return the inside/outside link ratio of ``r`` for ``members``."""
    rho = np.asarray(rho, float)
    N = rho.shape[0]
    m = [int(x) for x in members if int(x) != int(r)]
    if not m:
        return float("inf")
    mask = np.ones(N, bool)
    mask[m] = False
    mask[int(r)] = False
    if not mask.any():
        return float("inf")
    out = float(rho[mask, int(r)].mean())
    return float(rho[m, int(r)].mean()) / max(out, 1e-12)


def make_gate(rho):
    """gate(r, members) -> True if r predicts members more than outside."""
    R = np.asarray(rho, float)

    def gate(r, members):
        return ratio(R, r, members) > 1.0

    return gate


def directions(rho, r, members):
    """Driver and listener test of residue r against the set `members`.

    Returns dict(p_driver, p_listener, ratio_driver, ratio_listener).
    """
    R = np.asarray(rho, float)
    N = R.shape[0]
    r = int(r)
    m = sorted({int(x) for x in members} - {r})
    o = sorted(set(range(N)) - set(m) - {r})
    if not m or not o:
        return {
            "p_driver": 1.0,
            "p_listener": 1.0,
            "ratio_driver": float("nan"),
            "ratio_listener": float("nan"),
        }
    d_in, d_out = R[m, r], R[o, r]  # r -> members, r -> outside
    l_in, l_out = R[r, m], R[r, o]  # members -> r, outside -> r
    pd = float(mannwhitneyu(d_in, d_out, alternative="greater").pvalue)
    pl = float(mannwhitneyu(l_in, l_out, alternative="greater").pvalue)
    return {
        "p_driver": pd,
        "p_listener": pl,
        "ratio_driver": float(d_in.mean() / max(d_out.mean(), 1e-12)),
        "ratio_listener": float(l_in.mean() / max(l_out.mean(), 1e-12)),
    }


def direction_label(p_driver, p_listener, alpha=0.05):
    """Return 'both', 'driver', 'listener' or None from the two p-values."""
    d, l_ = p_driver <= alpha, p_listener <= alpha
    return (
        "both" if d and l_ else "driver" if d else "listener" if l_ else None
    )


def make_protect(rho, alpha=0.05):
    """protect(r, members) -> 'both' | 'driver' | 'listener' | None.

    Cached per (residue, member set).
    """
    R = np.asarray(rho, float)
    cache = {}

    def protect(r, members):
        key = (int(r), frozenset(int(x) for x in members))
        if key not in cache:
            t = directions(R, r, key[1])
            cache[key] = direction_label(t["p_driver"], t["p_listener"], alpha)
        return cache[key]

    return protect


def direction_table(rho, units, alpha=0.05):
    """One row per member of every unit: direction label and statistics."""
    rows = []
    for a, u in enumerate(units):
        for r in sorted(int(x) for x in u):
            t = directions(rho, r, u)
            rows.append(
                {
                    "unit": a,
                    "residue": r,
                    "label": direction_label(
                        t["p_driver"], t["p_listener"], alpha
                    )
                    or "none",
                    **{k: round(v, 5) for k, v in t.items()},
                }
            )
    return rows
