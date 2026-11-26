"""Predictive cohesion: the PDU definition used by COG (14 Oct 2026).

A set S of residues is a PDU when its members predict each other's dynamics
better than they predict, or are predicted by, the residues outside S.

    rho[i, j]  = how much j's present predicts i's future beyond i's own
                 present (cog2_graph.pair_rho: leading partial canonical
                 correlation at lag tau, directed)
    inside     = rho[a, b] for members a != b
    driver     boundary  = rho[k, i]   member i -> outsider k
    listener   boundary  = rho[i, k]   outsider k -> member i
    AUC_driver   = P(inside link > driver boundary link)
    AUC_listener = P(inside link > listener boundary link)
    cohesion     = (AUC_driver + AUC_listener) / 2

AUC = Mann-Whitney U / (n_inside * n_boundary), ties counted half; 0.5 for
a random group.  No threshold or quantile is chosen.  The MC maximises the
cohesion (cog2_stability with this scorer); the channel count (interface
theorem) is reported for every unit as its closure, not used as the target.

Cost used by growth, round 2 and the MC (14 Oct 2026): cohesion-based growth
with a noise-based margin:

    cost(S) = -cohesion(S)        (an AUC: already a probability, so sets
                                   of different sizes are compared directly)

Whether a set is a unit at all is a separate test, `significant(S)`:
cohesion(S) > q95(|S|), the 95th percentile of uniformly random sets of the
same size (p <= 0.05).  q95 is NOT folded into the cost: it falls with size
(planted: 0.93 at 2 residues, 0.59 at 12), so a cost q95 - cohesion
rewarded growth and merged geared units (fast test 14 Oct: scattered 19 ->
8 recovered).  Random regions grown on the predictive graph are not used
for q95 either: on planted systems they mostly lie inside real units
(median cohesion 0.93-0.98).  The selection effect of searching is handled
by the descended null reported for the final units.

Margin = noise.  rho is re-estimated on block-bootstrap resamples of the
lagged pairs (blocks = whole replicas when there are >= 20 of them,
otherwise each replica split into contiguous blocks).  For random regions of
each size and a random one-residue change, the standard deviation of the
change in cohesion over the resamples is measured; margin(k) = 1.645 x its
median (one-sided 95%).  A change is accepted only if it improves the cost
by more than margin(|S|): smaller changes are within what a different draw
of the same simulations would give.

Planted test (14 Oct, 6 x 6, 72 units): true units 0.97-1.00; halves
0.92-0.95; two geared units merged 0.89-0.91; unit + 2 random residues
0.71-0.77; random groups 0.48.  From a true unit the only additions that
raise it are planted faces.  Blind spot: two geared units merged can score
as high as either alone (truth wins 53/72).
"""

from __future__ import annotations

import numpy as np

from pdukit.numerics.cohesion import cohesion


def _interp(table, k):
    """Linear interpolation of a {size: value} table (clamped at the ends)."""
    ks = sorted(table)
    if not ks:
        return float("nan")
    return float(np.interp(k, ks, [table[x] for x in ks]))


def margin_at(margin, size):
    """Return a margin given as a number or as a function of the size."""
    return float(margin(size)) if callable(margin) else float(margin)


class CohesionScorer:
    """Scorer interface for COG (decompose, cog2_resolve, cog2_stability).

    cost = -cohesion.  `baseline` ({size: q95}) is used by `significant`
    (cohesion above the 95th percentile of random sets of the same size).
    `margin` is a number or a function of size.
    """

    def __init__(self, rho, baseline=None, margin=0.0):
        """Hold the predictive graph, the random baseline and the margin."""
        self.rho = np.asarray(rho, float)
        self.baseline = dict(baseline) if baseline else None
        self._margin = margin

    def margin(self, size):
        """Return the noise margin at a set size."""
        return margin_at(self._margin, size)

    def q95(self, size):
        """Return the 95th percentile of random-set cohesion at a size."""
        return _interp(self.baseline, size) if self.baseline else 0.0

    def cost_of(self, S):
        """Return the cohesion result and its cost (-cohesion) of a set."""
        c = cohesion(self.rho, S)
        v = c["cohesion"]
        return c, (-v if np.isfinite(v) else 0.0)

    def significant(self, S):
        """Cohesion above q95 of random sets of the same size."""
        if not self.baseline:
            return True
        v = cohesion(self.rho, S)["cohesion"]
        return bool(np.isfinite(v) and v > self.q95(len(set(S))))

    def score_batch(self, sets):
        """Score many sets; each result carries its cohesion cost."""
        out = []
        for s in sets:
            c, cost = self.cost_of(s)
            out.append({**c, "cost": cost})
        return out

    def __call__(self, support):  # decompose(scorer=...) interface
        """Score one support (the ``decompose(scorer=...)`` interface)."""
        return self.score_batch([support])[0]

    @staticmethod
    def cost_fn(result):
        """Return the cost of a scored result."""
        return float(result["cost"])


def block_bootstrap_rho(ws, n_boot=20, min_blocks=20, seed=0, log=print):
    """Rho re-estimated on n_boot block-bootstrap resamples of the lagged
    pairs of a FluxWorkspace (same features, lag and centring).
    """
    from pdukit.cog.graph import pair_rho

    if not hasattr(ws, "_gram") or not hasattr(ws, "_usable"):
        return [], {"blocks": 0}
    N, m, lag = int(ws.N), int(ws.m), int(ws.lag)
    segs = list(ws._usable)
    per = max(1, int(np.ceil(min_blocks / max(len(segs), 1))))
    blocks = []
    for a_, b_ in segs:
        r0 = np.arange(a_, b_ - lag)
        for ch in np.array_split(r0, per):
            if ch.size < 2:
                continue
            S00, s0, _ = ws._gram(ch, ch)
            S01, _, s1 = ws._gram(ch, ch + lag)
            S11, _, _ = ws._gram(ch + lag, ch + lag)
            blocks.append(
                (
                    S00.astype(np.float32),
                    S01.astype(np.float32),
                    S11.astype(np.float32),
                    s0,
                    s1,
                    int(ch.size),
                )
            )
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(int(n_boot)):
        w = np.bincount(
            rng.integers(0, len(blocks), len(blocks)), minlength=len(blocks)
        )
        S00 = np.zeros(blocks[0][0].shape)
        S01, S11 = np.zeros_like(S00), np.zeros_like(S00)
        s0 = np.zeros(blocks[0][3].shape)
        s1, n = np.zeros_like(s0), 0
        for wk, bl in zip(w, blocks):
            if wk == 0:
                continue
            S00 += wk * bl[0]
            S01 += wk * bl[1]
            S11 += wk * bl[2]
            s0 += wk * bl[3]
            s1 += wk * bl[4]
            n += wk * bl[5]
        m0, m1 = s0 / n, s1 / n
        out.append(
            pair_rho(
                S00 / n - np.outer(m0, m0),
                S01 / n - np.outer(m0, m1),
                S11 / n - np.outer(m1, m1),
                N,
                m,
            )
        )
    info = {
        "blocks": len(blocks),
        "blocks_per_replica": per,
        "replicas": len(segs),
        "n_boot": int(n_boot),
    }
    log(
        f"bootstrap rho: {n_boot} resamples, {len(blocks)} blocks "
        f"({per} per replica)"
    )
    return out, info


def random_null(rho, size, n=200, seed=0):
    """Cohesion of n uniformly random sets of `size` residues."""
    rng = np.random.default_rng(seed)
    N = np.asarray(rho).shape[0]
    return np.array(
        [
            cohesion(rho, rng.choice(N, int(size), replace=False))["cohesion"]
            for _ in range(int(n))
        ]
    )


def unit_table(rho, units, *, n_random=200, seed=0):
    """Per unit: AUCs, cohesion, p against random sets of the same size."""
    nulls, rows = {}, []
    for a, u in enumerate(units):
        k = len(u)
        c = cohesion(rho, u)
        if k not in nulls:
            nulls[k] = random_null(rho, k, n_random, seed + k)
        nl = nulls[k]
        p = (1 + int(np.sum(nl >= c["cohesion"]))) / (len(nl) + 1)
        rows.append(
            {
                "unit": a,
                "size": k,
                **{key: round(v, 4) for key, v in c.items()},
                "p_random": round(p, 4),
                "random_median": round(float(np.median(nl)), 4),
                **link_strength(rho, u),
            }
        )
    return rows


def link_strength(rho, S):
    """How strong the links are, not only how they rank (14 Oct 2026):
    median rho inside, mean rho inside and across the boundary (both
    directions), their ratio, and rho^2 of the median inside link = share
    of a member's future one other member explains.
    """
    R = np.asarray(rho, float)
    N = R.shape[0]
    S = np.array(sorted({int(x) for x in S}))
    E = np.setdiff1d(np.arange(N), S)
    if len(S) < 2 or len(E) == 0:
        return {}
    ins = R[np.ix_(S, S)][~np.eye(len(S), dtype=bool)]
    bnd = np.concatenate([R[np.ix_(E, S)].ravel(), R[np.ix_(S, E)].ravel()])
    med = float(np.median(ins))
    return {
        "rho_inside_median": round(med, 4),
        "rho_inside_mean": round(float(ins.mean()), 4),
        "rho_boundary_mean": round(float(bnd.mean()), 4),
        "inside_over_boundary": round(
            float(ins.mean() / max(bnd.mean(), 1e-12)), 3
        ),
        "explained_share_median": round(med * med, 4),
    }
