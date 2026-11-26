"""Bootstrap confidence intervals and variance components of calibration
estimates.
"""

from __future__ import annotations

import numpy as np


def _bootstrap_ci_rel_width_two_way(
    rows, percentile=95, n_boot=2000, ci=0.95, rng=None, chunk=50
):
    """Like `_bootstrap_ci_rel_width_by_support`, but resampling SUPPORTS
    AND SHIFTS independently.

    Every calibration support is scored against the SAME shift replicas, so
    the shifts are a shared source of noise: if one shift happens to leave
    slow motions aligned, it raises every support's null together.
    Resampling supports alone cannot see that and reports an interval that
    is too narrow (on the planted test system the shift-to-shift variance was
    ~7x the support-to-support variance).  Each replicate here draws
    `n_supports` rows AND `n_shifts` columns with replacement.
    Returns `(point_estimate, relative_width)`.
    """
    rows = np.asarray(rows, dtype=np.float64)
    if rows.ndim != 2 or rows.size == 0:
        return float("nan"), float("inf")
    n_sup, n_sh = rows.shape
    point = float(np.percentile(rows, percentile))
    if rng is None:
        rng = np.random.default_rng()
    n_boot = int(n_boot)
    boot = np.empty(n_boot, dtype=np.float64)
    for c0 in range(0, n_boot, int(chunk)):
        c = min(int(chunk), n_boot - c0)
        i_s = rng.integers(0, n_sup, size=(c, n_sup))
        i_j = rng.integers(0, n_sh, size=(c, n_sh))
        block = rows[i_s[:, :, None], i_j[:, None, :]]  # (c, n_sup, n_sh)
        boot[c0 : c0 + c] = np.percentile(
            block.reshape(c, -1), percentile, axis=1
        )
    lo, hi = np.percentile(
        boot, [100 * (1 - ci) / 2, 100 * (1 - (1 - ci) / 2)]
    )
    rel_width = float((hi - lo) / point) if point != 0 else float("inf")
    return point, rel_width


def _variance_components(rows):
    """Between-support, between-shift and residual variance of a
    (n_supports, n_shifts) table of leading null correlations.
    """
    rows = np.asarray(rows, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[0] < 2 or rows.shape[1] < 2:
        return {
            "between_supports": float("nan"),
            "between_shifts": float("nan"),
            "residual": float("nan"),
        }
    grand = rows.mean()
    sup = rows.mean(axis=1) - grand
    sh = rows.mean(axis=0) - grand
    resid = rows - grand - sup[:, None] - sh[None, :]
    return {
        "between_supports": float(sup.var(ddof=1)),
        "between_shifts": float(sh.var(ddof=1)),
        "residual": float(resid.var(ddof=1)),
    }
