"""Second round for residues left unresolved by COG 2.1 discovery.

Rule (12 Oct 2026):
  1. Round 1 (discovery) keeps every group with cost <= 0 as a unit.
     Groups with cost > 0 are not units; their residues form the POOL.
  2. Round 2a -- own units: grow new groups inside the pool only (seeds and
     added residues from the pool; scored against the whole protein as
     environment).  A group is accepted if its cost < -margin.
  3. Round 2b -- joins: a pool residue joins an existing unit (round 1 or
     2a) if and only if that makes the unit's cost more negative by more
     than `margin`.  Best moves first; repeated until nothing improves.
  4. Leftovers -- secondary assignment: each remaining residue is recorded
     with the unit where adding it raises the cost least (its "best place"),
     flagged 'secondary', and NOT added to the unit.  Residues with no unit
     within two contacts are 'isolated'.  Both wait for more data or a
     different time band.

`margin` guards against accepting changes that are within noise (cost is a
soft channel count; 0.1 channel by default).
"""

from __future__ import annotations

import numpy as np


def _mg(margin, size):
    return float(margin(size)) if callable(margin) else float(margin)


def _cost(scorer, groups):
    return [
        float(r["cost"])
        for r in scorer.score_batch(
            [sorted(int(x) for x in g) for g in groups]
        )
    ]


def _spread_seeds(pool, adj, n):
    """Up to n seeds spread over the pool by contact-graph distance
    (farthest-point sampling within the pool).
    """
    pool = sorted(pool)
    if len(pool) <= n:
        return pool
    P = np.array(pool)
    sub = adj[np.ix_(P, P)]
    INF = 10**6

    def bfs(i):
        d = np.full(len(P), INF)
        d[i] = 0
        front = [i]
        while front:
            nxt = []
            for v in front:
                for w in np.where(sub[v])[0]:
                    if d[w] == INF:
                        d[w] = d[v] + 1
                        nxt.append(w)
            front = nxt
        return d

    chosen = [0]
    dmin = bfs(0).astype(float)
    while len(chosen) < n:
        dmin[chosen] = -1
        j = int(np.argmax(dmin))
        if dmin[j] <= 0:
            break
        chosen.append(j)
        dmin = np.minimum(dmin, bfs(j))
    return [int(P[c]) for c in chosen]


def _grow_in_pool(scorer, seed, pool, adj, max_unit, min_size):
    """Greedy growth from `seed` adding the pool neighbour that gives the
    lowest cost; returns the best prefix of size >= min_size.
    """
    grp = [seed]
    best = (None, np.inf)
    path = []
    while len(grp) < max_unit:
        front = sorted(
            {
                int(j)
                for i in grp
                for j in np.where(adj[i])[0]
                if j in pool and j not in grp
            }
        )
        if not front:
            break
        costs = _cost(scorer, [grp + [j] for j in front])
        k = int(np.argmin(costs))
        grp = grp + [front[k]]
        path.append(costs[k])
        if len(grp) >= min_size and costs[k] < best[1]:
            best = (sorted(grp), costs[k])
    return best


def resolve_unresolved(
    scorer,
    units,
    adjacency,
    *,
    max_unit=40,
    min_size=3,
    margin=0.1,
    n_seeds=12,
    max_new=20,
    assigned=None,
    log=print,
    gate=None,
    is_unit=None,
):
    """`is_unit` (optional, e.g.

    CohesionScorer.significant): decides which groups are units (round 1) and
    whether a new pool group is accepted (2a), in place of the sign of the
    cost.
    """
    assigned_extra = assigned  # e.g. face residues: never in the pool
    adj = np.asarray(adjacency, bool)
    N = adj.shape[0]
    units = [sorted(int(x) for x in u) for u in units]
    c0 = _cost(scorer, units)
    keep = [
        u
        for u, c in zip(units, c0)
        if (is_unit(u) if is_unit is not None else c <= 0)
    ]
    assigned = set(x for u in keep for x in u)
    pool = (
        set(range(N)) - assigned - set(int(x) for x in (assigned_extra or []))
    )
    out = {
        "margin": (
            margin if not callable(margin) else "noise table (size dependent)"
        ),
        "round1_units": len(keep),
        "pool_start": sorted(pool),
        "new_units": [],
        "joins": [],
        "secondary": [],
        "isolated": [],
    }
    log(
        f"resolve: {len(keep)} units kept "
        f"({'significant' if is_unit is not None else 'cost <= 0'}); "
        f"{len(pool)} residues "
        "in the unresolved pool"
    )

    # 2a. own units inside the pool --------------------------------------
    while len(pool) >= min_size and len(out["new_units"]) < max_new:
        best = (None, np.inf)
        for s in _spread_seeds(pool, adj, n_seeds):
            g, c = _grow_in_pool(scorer, s, pool, adj, max_unit, min_size)
            if g is not None and c < best[1]:
                best = (g, c)
        if best[0] is None:
            break
        if is_unit is not None:
            if not is_unit(best[0]):
                break
        elif best[1] >= -_mg(margin, len(best[0])):
            break
        keep.append(best[0])
        pool -= set(best[0])
        out["new_units"].append({"residues": best[0], "cost": best[1]})
        log(
            f"resolve 2a: new unit from the pool, size {len(best[0])}, cost "
            f"{best[1]:+.2f}; pool now {len(pool)}"
        )

    # 2b. joins: only if the unit's cost improves by more than margin ------
    cur = _cost(scorer, keep) if keep else []
    while pool and keep:
        cand = [
            (r, i)
            for r in sorted(pool)
            for i, u in enumerate(keep)
            if len(u) < max_unit
            and adj[r, u].any()
            and (gate is None or gate(r, set(u)))
        ]
        if not cand:
            break
        trial = _cost(scorer, [keep[i] + [r] for r, i in cand])
        delta = [t - cur[i] for t, (r, i) in zip(trial, cand)]
        order = np.argsort(delta)
        used_r, used_u, moved = set(), set(), 0
        for k in order:
            r, i = cand[k]
            if delta[k] >= -_mg(margin, len(keep[i]) + 1):
                continue
            if r in used_r or i in used_u:
                continue  # one move per residue and per unit per pass
            keep[i] = sorted(keep[i] + [r])
            cur[i] = trial[k]
            pool.discard(r)
            used_r.add(r)
            used_u.add(i)
            out["joins"].append(
                {
                    "residue": int(r),
                    "unit": int(i),
                    "delta_cost": round(float(delta[k]), 3),
                }
            )
            moved += 1
        if not moved:
            break
        log(f"resolve 2b: {moved} residues joined units; pool now {len(pool)}")

    # 3. secondary assignment ---------------------------------------------
    for r in sorted(pool):
        near = [i for i, u in enumerate(keep) if adj[r, u].any()]
        if not near:
            two = set(j for j in np.where(adj[r])[0]) | {r}
            two = set(k for j in two for k in np.where(adj[j])[0])
            near = [i for i, u in enumerate(keep) if two & set(u)]
        if not near:
            out["isolated"].append(int(r))
            continue
        trial = _cost(scorer, [keep[i] + [r] for i in near])
        d = [t - cur[i] for t, i in zip(trial, near)]
        k = int(np.argmin(d))
        out["secondary"].append(
            {
                "residue": int(r),
                "best_unit": int(near[k]),
                "delta_cost": round(float(d[k]), 3),
                "alternatives": len(near),
            }
        )
    out["units"] = keep
    out["unit_costs"] = [round(float(c), 3) for c in cur] if cur else []
    log(
        f"resolve: {len(keep)} units ({len(out['new_units'])} new, "
        f"{len(out['joins'])} residues joined); {len(out['secondary'])} "
        f"secondary, {len(out['isolated'])} isolated"
    )
    return out
