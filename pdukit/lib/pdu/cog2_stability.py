"""Test whether each unit is a stable minimum of its cost.

(13 Oct 2026)

Rule
----
A unit S is at a stable minimum when no single change of membership --
removing one member or adding one non-member -- lowers its cost by more
than `margin`.

1. Steepest descent.  Score every one-residue change (all members out, all
   non-members in; dynamics only, no graph restriction).  If the best change
   lowers the cost by more than `margin`, apply it and repeat; otherwise
   stop.  The end point is the unit's local minimum M.
2. Kicks (cycles).  Change k random residues of M (k = 1..kmax, each a
   random add or remove), then descend again.  `n_cycles` > 0 runs at most
   that many cycles; -1 runs until convergence.  Either way the test stops
   once `patience` cycles in a row have not changed M (default 100).
     - back to M                 -> "returned" (the residues are put back)
     - a set with cost lower than M by more than `margin`
                                 -> "lower": M was not the minimum; the
                                    lower set becomes M and the cycles go on
     - a different set, higher than M by more than `margin`
                                 -> "higher": put back to M (the kick
                                    landed in a worse local minimum)
     - a different set within `margin` of M
                                 -> "alternative": an equally good minimum,
                                    reported (M is kept)
3. Per residue: how often it ends a cycle with a membership different from
   M's ("flip frequency").  Residues that flip often sit on the boundary.
4. Null.  The same descent from random regions of the same size (grown the
   way the baseline regions are grown).  p_descended = fraction of descended
   random regions whose cost is at least as low as the unit's minimum.  This
   corrects for having tuned the unit on the same data.

The cost is the scorer's cost (channels minus the size-matched random-region
baseline), so a change of size is compared with regions of the new size.

Additions (13 Oct 2026): a unit may add only its own original members or
residues from the POOL (residues in no unit when the test starts) -- never
members of other units -- and a pool residue must also pass the dynamics
consistency gate (cog2_gate: it predicts the unit's members better than the
rest of the protein).  Applies to descent and kicks.  Removals are never
gated; removed residues go to round 2 after the write-back.

Protected members (14 Oct 2026): a residue that significantly predicts the
others' dynamics should not be dropped.  With
`protect` (cog2_gate.make_protect), a member that passes the driver or the
listener test against the other members cannot be removed by the descent or
by a kick: J alone does not drop it.  At the minimum, a protected member
whose removal would lower the cost by more than `margin` is reported as a
FACE: the unit predicts it (or it predicts the unit), yet it adds a channel
to the outside, i.e. it carries the unit's interface.  The descended nulls
use the same protection, so units and nulls move under the same rules.

Write-back (`reconcile`): the minima become the units.  A residue claimed by
two minima stays where removing it would raise the cost most; residues in
no minimum are left for the round-2 pool.
"""

from __future__ import annotations

import time

import numpy as np


def _mg(margin, size):
    """Margin is a number or a function of the set size (cohesion noise)."""
    return float(margin(size)) if callable(margin) else float(margin)


class _Memo:
    """Cost of a residue set, scored once (batched)."""

    def __init__(self, scorer):
        self.scorer, self.cache, self.n_scored = scorer, {}, 0

    def costs(self, sets):
        keys = [frozenset(int(x) for x in s) for s in sets]
        todo = [k for k in dict.fromkeys(keys) if k not in self.cache]
        if todo:
            res = self.scorer.score_batch([sorted(k) for k in todo])
            self.n_scored += len(todo)
            for k, r in zip(todo, res):
                self.cache[k] = float(r["cost"])
        return [self.cache[k] for k in keys]


def _allowed(r, cur, original, gate, pool=None):
    if r in original:
        return True
    if pool is not None and r not in pool:
        return False  # a member of another unit
    return gate is None or gate(r, cur)


def descend(
    memo,
    start,
    N,
    *,
    margin=0.1,
    min_size=3,
    max_unit=40,
    max_steps=200,
    gate=None,
    original=frozenset(),
    pool=None,
    protect=None,
):
    """Steepest descent over one-residue changes.

    Returns (set, cost, moves) with moves = [(+r or -r-1, new cost), ...]
    ('-r-1' = removal of r, so residue 0 is unambiguous).
    """
    cur = frozenset(int(x) for x in start)
    c_cur = memo.costs([cur])[0]
    moves = []
    for _ in range(int(max_steps)):
        cand, tag = [], []
        if len(cur) > min_size:
            for r in cur:
                if protect is not None and protect(r, cur):
                    continue  # predicts / is predicted: kept
                cand.append(cur - {r})
                tag.append(-r - 1)
        if len(cur) < max_unit:
            for r in range(N):
                if r not in cur and _allowed(r, cur, original, gate, pool):
                    cand.append(cur | {r})
                    tag.append(r)
        if not cand:
            break
        cs = memo.costs(cand)
        k = int(np.argmin(cs))
        if cs[k] >= c_cur - _mg(margin, len(cur)):
            break
        cur, c_cur = cand[k], cs[k]
        moves.append((int(tag[k]), round(float(c_cur), 4)))
    return cur, c_cur, moves


def _kick(
    rng,
    S,
    N,
    k,
    min_size,
    max_unit,
    gate=None,
    original=frozenset(),
    pool=None,
    protect=None,
):
    S = set(S)
    for _ in range(int(k)):
        add = (len(S) <= min_size) or (
            rng.random() < 0.5 and len(S) < max_unit
        )
        removable = (
            sorted(S)
            if protect is None
            else [r for r in sorted(S) if not protect(r, S)]
        )
        if not add and not removable:
            add = len(S) < max_unit
        if add:
            out = [
                r
                for r in range(N)
                if r not in S and _allowed(r, S, original, gate, pool)
            ]
            if out:
                S.add(int(rng.choice(out)))
        elif len(S) > min_size and removable:
            S.discard(int(rng.choice(removable)))
    return frozenset(S)


HARD_MAX_CYCLES = 100000  # safety stop for n_cycles = -1


def stability(
    memo,
    unit,
    N,
    *,
    n_cycles=-1,
    patience=100,
    kmax=3,
    margin=0.1,
    min_size=3,
    max_unit=40,
    max_steps=200,
    rng=None,
    log=None,
    label="",
    gate=None,
    pool=None,
    protect=None,
):
    """Find the stable minimum of one unit by kicked descent."""
    rng = rng if rng is not None else np.random.default_rng(0)
    start = frozenset(int(x) for x in unit)
    gk = dict(gate=gate, original=start, pool=pool, protect=protect)
    c0 = memo.costs([start])[0]
    M, cM, moves0 = descend(
        memo,
        start,
        N,
        margin=margin,
        min_size=min_size,
        max_unit=max_unit,
        max_steps=max_steps,
        **gk,
    )
    flips = np.zeros(N)
    outcome = {"returned": 0, "lower": 0, "higher": 0, "alternative": 0}
    alts, trace = [], []
    limit = HARD_MAX_CYCLES if int(n_cycles) < 0 else int(n_cycles)
    since_change, n_run = 0, 0
    if log:
        log(
            f"{label}first descent: cost {c0:+.3f} -> {cM:+.3f} "
            f"(dJ {cM - c0:+.3f}), size {len(start)} -> {len(M)}, "
            f"{len(moves0)} moves"
        )
    while n_run < limit and since_change < int(patience):
        n_run += 1
        k = int(rng.integers(1, int(kmax) + 1))
        K = _kick(rng, M, N, k, min_size, max_unit, **gk)
        cK = memo.costs([K])[0]
        E, cE, mv = descend(
            memo,
            K,
            N,
            margin=margin,
            min_size=min_size,
            max_unit=max_unit,
            max_steps=max_steps,
            **gk,
        )
        diff = sorted(E ^ M)
        cM_before = cM
        if not diff:
            res = "returned"
            since_change += 1
        else:
            flips[diff] += 1
            mM = _mg(margin, len(M))
            if cE < cM - mM:
                res = "lower"
                M, cM = E, cE
                since_change = 0
            elif cE > cM + mM:
                res = "higher"
                since_change += 1
            else:
                res = "alternative"
                since_change += 1
                alts.append(
                    {
                        "cost": round(float(cE), 3),
                        "size": len(E),
                        "differs_by": [int(x) for x in diff],
                    }
                )
        outcome[res] += 1
        trace.append(
            {
                "cycle": n_run,
                "k": k,
                "cost_kicked": round(float(cK), 4),
                "cost_end": round(float(cE), 4),
                "dJ_vs_minimum": round(float(cE - cM_before), 4),
                "descent_moves": len(mv),
                "size_end": len(E),
                "outcome": res,
                "minimum_cost": round(float(cM), 4),
            }
        )
        if log:
            log(
                f"{label}cycle {n_run}: kick {k} -> cost {cK:+.3f}, "
                f"descent {len(mv)} moves -> {cE:+.3f} (dJ vs minimum "
                f"{cE - cM_before:+.3f}) {res}; minimum {cM:+.3f} size "
                f"{len(M)}; {since_change} cycles without change"
            )
    converged = since_change >= int(patience)
    # directions of the members at the minimum; faces = protected members
    # whose removal would still lower the cost by more than the margin
    direction, faces = {}, {}
    if protect is not None:
        prot = [r for r in sorted(M) if protect(r, M)]
        direction = {int(r): (protect(r, M) or "none") for r in sorted(M)}
        if prot and len(M) > min_size:
            cs = memo.costs([M - {r} for r in prot])
            faces = {
                int(r): round(float(c - cM), 3)
                for r, c in zip(prot, cs)
                if c < cM - _mg(margin, len(M))
            }
        if log and faces:
            log(
                f"{label}faces (kept by dynamics, removal would lower the "
                f"cost): {faces}"
            )
    return {
        "start": sorted(start),
        "start_cost": round(float(c0), 3),
        "minimum": sorted(M),
        "minimum_cost": round(float(cM), 3),
        "first_descent_moves": moves0,
        "removed": sorted(start - M),
        "added": sorted(M - start),
        "dJ": round(float(cM - c0), 3),
        "cycles_run": n_run,
        "converged": bool(converged),
        "patience": int(patience),
        **outcome,
        "returned_fraction": round(outcome["returned"] / max(n_run, 1), 3),
        "flip_frequency": {
            int(r): round(float(f) / max(n_run, 1), 3)
            for r in np.where(flips > 0)[0]
            for f in [flips[r]]
        },
        "alternatives": alts[:10],
        "trace": trace,
        "direction": direction,
        "faces": faces,
    }


def descended_null(
    memo,
    adjacency,
    size,
    N,
    n,
    *,
    margin=0.1,
    min_size=3,
    max_unit=40,
    max_steps=200,
    rng=None,
    gate=None,
    protect=None,
):
    """Costs of random regions of `size` after the same descent."""
    from .cog2_scoring import ball_support as _ball_support

    rng = rng if rng is not None else np.random.default_rng(1)
    out = []
    for _ in range(int(n)):
        s = _ball_support(N, int(size), adjacency, rng)
        if len(s) != int(size):
            continue
        _, c, _ = descend(
            memo,
            s,
            N,
            margin=margin,
            min_size=min_size,
            max_unit=max_unit,
            max_steps=max_steps,
            gate=gate,
            original=frozenset(s),
            protect=protect,
        )
        out.append(float(c))
    return out


def run_stability(
    scorer,
    units,
    adjacency,
    *,
    n_cycles=-1,
    patience=100,
    kmax=3,
    margin=0.1,
    min_size=3,
    max_unit=40,
    max_steps=200,
    n_null=20,
    seed=0,
    log=print,
    gate=None,
    protect=None,
):
    """Find the stable minimum of every unit and the faces they keep."""
    N = int(np.asarray(adjacency).shape[0])
    memo = _Memo(scorer)
    # residues in no unit when the test starts: the only outside residues a
    # unit may take in (13 Oct: additions come from a pool)
    in_units = set(int(x) for u in units for x in u)
    pool = frozenset(r for r in range(N) if r not in in_units)
    # a pool residue taken by one unit's minimum leaves the pool for the
    # units tested after it (first claim wins; units in input order)
    avail = set(pool)
    log(
        f"stability: {len(pool)} residues in the pool (in no unit); units may"
        " add only their own original members or pool residues"
    )
    rng = np.random.default_rng(seed)
    rows, nulls = [], {}
    for i, u in enumerate(units):
        t = time.perf_counter()
        st = stability(
            memo,
            u,
            N,
            n_cycles=n_cycles,
            patience=patience,
            kmax=kmax,
            margin=margin,
            min_size=min_size,
            max_unit=max_unit,
            max_steps=max_steps,
            rng=rng,
            log=log,
            label=f"stability unit {i}: ",
            gate=gate,
            pool=frozenset(avail),
            protect=protect,
        )
        taken = set(st["added"]) & avail
        avail -= taken
        st["pool_residues_taken"] = sorted(int(x) for x in taken)
        st["unit"] = i
        sz = len(st["minimum"])
        if n_null and sz not in nulls:
            nulls[sz] = descended_null(
                memo,
                adjacency,
                sz,
                N,
                n_null,
                margin=margin,
                min_size=min_size,
                max_unit=max_unit,
                max_steps=max_steps,
                rng=rng,
                gate=gate,
                protect=protect,
            )
        nl = nulls.get(sz, [])
        st["null_descended_costs"] = [round(x, 3) for x in nl]
        st["p_descended"] = (
            round(
                (1 + sum(x <= st["minimum_cost"] for x in nl)) / (len(nl) + 1),
                3,
            )
            if nl
            else None
        )
        st["seconds"] = round(time.perf_counter() - t, 1)
        rows.append(st)
        log(
            f"stability unit {i}: size {len(st['start'])} -> "
            f"{len(st['minimum'])} (removed {len(st['removed'])}, added "
            f"{len(st['added'])}), cost {st['start_cost']:+.2f} -> "
            f"{st['minimum_cost']:+.2f} (dJ {st['dJ']:+.2f}); kicks: "
            f"{st['returned']} returned, {st['lower']} lower, "
            f"{st['higher']} higher (put back), {st['alternative']} "
            f"equally good alternative, of {st['cycles_run']} "
            f"({'converged' if st['converged'] else 'NOT converged'}); "
            f"p vs descended random regions {st['p_descended']}; "
            f"{st['seconds']}s"
        )
    J0 = sum(r["start_cost"] for r in rows)
    J1 = sum(r["minimum_cost"] for r in rows)
    log(
        f"stability: J {J0:+.2f} -> {J1:+.2f} (dJ {J1 - J0:+.2f}) over "
        f"{len(rows)} units; {sum(r['converged'] for r in rows)} converged"
    )
    return {
        "settings": {
            "n_cycles": n_cycles,
            "patience": patience,
            "kmax": kmax,
            "margin": (
                margin
                if not callable(margin)
                else "noise table (size dependent)"
            ),
            "min_size": min_size,
            "max_unit": max_unit,
            "max_steps": max_steps,
            "n_null": n_null,
            "seed": seed,
            "addition_gate": gate is not None,
            "protect_by_direction": protect is not None,
            "pool_size": len(pool),
        },
        "J_start": round(J0, 3),
        "J_minimum": round(J1, 3),
        "dJ": round(J1 - J0, 3),
        "units": rows,
        "n_scored": memo.n_scored,
    }


def reconcile(memo, minima, N, *, min_size=3, log=print):
    """Write-back: one place per residue.

    A residue in several minima stays in the unit where removing it raises
    the cost most; it is removed from the others (removals are never gated).
    Units that fall below `min_size` are dropped.  Returns (units, report).
    """
    units = [set(int(x) for x in m) for m in minima]
    owners = {}
    for a, u in enumerate(units):
        for r in u:
            owners.setdefault(r, []).append(a)
    shared = {r: a_ for r, a_ in owners.items() if len(a_) > 1}
    moves = []
    for r in sorted(shared):
        cl = [a for a in shared[r] if r in units[a]]
        if len(cl) < 2:
            continue
        # a unit that would be left with fewer than min_size residues is
        # dropped anyway, so it never keeps a contested residue
        ok = [a for a in cl if len(units[a]) - 1 >= min_size]
        base = memo.costs([units[a] for a in ok]) if ok else []
        wo = memo.costs([units[a] - {r} for a in ok]) if ok else []
        rise = dict(zip(ok, [w - b for w, b in zip(wo, base)]))
        need = [rise.get(a, -np.inf) for a in cl]  # cost rise if removed
        keep = cl[int(np.argmax(need))]
        for a in cl:
            if a != keep:
                units[a].discard(r)
        moves.append(
            {
                "residue": int(r),
                "kept_in": int(keep),
                "claimed_by": [int(a) for a in cl],
                "cost_rise_if_removed": [
                    round(float(x), 3) if np.isfinite(x) else None
                    for x in need
                ],
            }
        )
    dropped = [a for a, u in enumerate(units) if len(u) < min_size]
    out = [sorted(u) for u in units if len(u) >= min_size]
    covered = set(x for u in out for x in u)
    pool = [r for r in range(N) if r not in covered]
    log(
        f"write-back: {len(shared)} residues claimed by two or more minima "
        f"(each kept in one unit); {len(dropped)} units fell below "
        f"{min_size} residues; {len(pool)} residues in no unit (round-2 "
        "pool)"
    )
    return out, {
        "shared_resolved": moves,
        "dropped_units": dropped,
        "pool": pool,
        "n_units": len(out),
    }
