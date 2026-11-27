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


def labels_from_units(units, n_residues):
    """Return a label per residue: unit index, or a singleton label.

    Residues in no unit each get a label of their own, so agreeing on
    "unassigned" earns nothing in `adjusted_rand`.  A residue in two units
    keeps the first.
    """
    lab = np.full(int(n_residues), -1, dtype=int)
    for k, u in enumerate(units):
        for i in u:
            if lab[int(i)] < 0:
                lab[int(i)] = k
    free = np.where(lab < 0)[0]
    lab[free] = len(units) + np.arange(free.size)
    return lab


def adjusted_rand(a, b):
    """Return the adjusted Rand index of two labelings (Hubert & Arabie).

    1 = the same partition, about 0 = no more agreement than chance.
    Defined as 1 when both partitions are all singletons or one block.
    """
    a, b = np.asarray(a), np.asarray(b)
    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)
    table = np.zeros((ai.max() + 1, bi.max() + 1))
    np.add.at(table, (ai, bi), 1)

    def pairs(x):
        return float(np.sum(x * (x - 1) / 2))

    n = pairs(np.array([a.size]))
    index = pairs(table)
    rows, cols = pairs(table.sum(1)), pairs(table.sum(0))
    expected = rows * cols / n if n else 0.0
    top = 0.5 * (rows + cols)
    if top == expected:
        return 1.0
    return (index - expected) / (top - expected)
