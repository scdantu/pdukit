"""Control supports: random residue sets of the size of a candidate unit, used
as the null for its score.
"""

from __future__ import annotations

import numpy as np


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
