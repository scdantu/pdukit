"""Starting residues for unit discovery."""

from __future__ import annotations

import numpy as np


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


def seed_groups(seeds, rho, adjacency, size):
    """Grow each seed into a start of ``size`` residues.

    The next member is the graph neighbour of the group with the largest
    summed two-way coupling to the members, sum_m rho[m, j] + rho[j, m]
    (rho[i, j] = j's present predicts i's future).  When the group has no
    neighbour left, any residue may be chosen.  Ties go to the lower index.
    Identical starts are kept once.  Returns a list of sorted residue lists.
    """
    R = np.nan_to_num(np.asarray(rho, float))
    sym = R + R.T
    np.fill_diagonal(sym, -np.inf)
    adj = np.asarray(adjacency, bool)
    n = len(sym)
    size = int(max(1, min(size, n)))
    starts, seen = [], set()
    for seed in seeds:
        group = [int(seed)]
        while len(group) < size:
            inside = np.zeros(n, bool)
            inside[group] = True
            frontier = adj[group].any(axis=0) & ~inside
            pool = np.flatnonzero(frontier)
            if pool.size == 0:
                pool = np.flatnonzero(~inside)
            gain = sym[np.ix_(group, pool)].sum(axis=0)
            group.append(int(pool[np.argmax(gain)]))
        key = tuple(sorted(group))
        if key not in seen:
            seen.add(key)
            starts.append(list(key))
    return starts
