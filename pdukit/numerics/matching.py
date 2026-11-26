"""Compare residue sets: Jaccard overlap and matching found units to truth."""

from __future__ import annotations

import numpy as np


def jaccard(a, b):
    """Return |a & b| / |a | b| of two residue sets (0 if both empty)."""
    a, b = set(map(int, a)), set(map(int, b))
    return len(a & b) / max(len(a | b), 1)


def best_jaccard(target, candidates):
    """Best Jaccard of ``target`` with any candidate set (0 if none)."""
    return max((jaccard(target, c) for c in candidates), default=0.0)


def match_units(found, truth_units, *, threshold=0.67):
    """Compare a decomposition with the planted answer.

    For every TRUE unit, the best Jaccard overlap with any found unit
    (recall side); for every FOUND unit, the best overlap with any true unit
    (precision side).  A true unit counts as recovered when its best overlap
    reaches `threshold`.  Also reports how many found units straddle two true
    units (each overlapping by >= 25 % of the found unit) -- the merge error.
    The default threshold is 0.67: at 0.5 a merged 20-residue unit counted
    as recovering both of the 10-residue units inside it.
    """
    F = [set(map(int, u)) for u in found]
    T = [set(map(int, u)) for u in truth_units]
    rec = [best_jaccard(t, F) for t in T]
    prec = [best_jaccard(f, T) for f in F]
    merged = 0
    for f in F:
        big = [t for t in T if len(f & t) >= 0.25 * max(len(f), 1)]
        merged += int(len(big) >= 2)
    return {
        "recall_jaccard": [round(x, 4) for x in rec],
        "precision_jaccard": [round(x, 4) for x in prec],
        "mean_recall": float(np.mean(rec)) if rec else 0.0,
        "mean_precision": float(np.mean(prec)) if prec else 0.0,
        "n_recovered": int(sum(x >= threshold for x in rec)),
        "n_true": len(T),
        "n_found": len(F),
        "n_merged": int(merged),
        "threshold": float(threshold),
    }
