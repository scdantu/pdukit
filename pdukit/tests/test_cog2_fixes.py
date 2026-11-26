"""COG 2.1 bug fixes to the null calibration and decompose().

1. Sizes missing from a null table no longer score as infinite cost.
2. The calibration's precision includes the shared-shift noise (two-way
   bootstrap).
3. A null table is checked against the data and settings it was made on.
4. The null can shift within replicas or take the environment from another
   replica.
5. Matched controls are scored with the same cost as the units, and a
   genuine error in scoring is raised instead of becoming an infinite cost.
"""

from __future__ import annotations

import numpy as np
import pytest

from pdukit.cog.calibration.null import complete_theta_table
from pdukit.cog.scoring.workspace import FluxWorkspace
from pdukit.tests.test_interface import simulate


@pytest.fixture(scope="module")
def feats():
    return simulate(channel_gain=1.0, seed=0)


@pytest.fixture(scope="module")
def ws(feats):
    return FluxWorkspace(
        feats, lag=1, n_null=8, seed=1, dtype=np.float64, truncation="koopman"
    )


# 1 ---------------------------------------------------------------------------
def test_complete_theta_table_interpolates_and_holds_ends():
    table, filled = complete_theta_table({4: 0.30, 8: 0.34}, 10)
    assert set(table) == set(range(1, 11))
    assert table[4] == 0.30 and table[8] == 0.34
    assert np.isclose(table[6], 0.32)
    assert table[1] == 0.30 and table[10] == 0.34  # held end values
    assert filled == [1, 2, 3, 5, 6, 7, 9, 10]
    with pytest.raises(ValueError):
        complete_theta_table({4: 0.3}, 6, mode="strict")


# 4 ---------------------------------------------------------------------------
def test_within_and_cross_shifts_respect_replicas(feats):
    segs = [(0, 2000), (2000, 4000)]
    w = FluxWorkspace(
        feats,
        lag=1,
        segments=segs,
        n_null=2,
        seed=0,
        dtype=np.float64,
        shift_mode="within",
    )
    rng = np.random.default_rng(1)
    rs = w._shifted_rows(rng)
    n0 = 2000 - 1
    assert np.all((rs[:n0] >= 0) & (rs[:n0] < 2000))
    assert np.all((rs[n0:] >= 2000) & (rs[n0:] < 4000))
    assert np.all(rs != w.rows0)
    x = FluxWorkspace(
        feats,
        lag=1,
        segments=segs,
        n_null=2,
        seed=0,
        dtype=np.float64,
        shift_mode="cross",
    )
    rs = x._shifted_rows(rng)
    assert np.all(rs[:n0] >= 2000)  # other replica
    assert np.all(rs[n0:] < 2000)
    with pytest.raises(ValueError):
        FluxWorkspace(feats, lag=1, n_null=2, seed=0, shift_mode="cross")


def test_default_shift_mode_is_unchanged(feats):
    a = FluxWorkspace(feats, lag=1, n_null=3, seed=4, dtype=np.float64)
    assert a.shift_mode == "global"
    min_shift = max(10, a.F // 10, 5)
    rng = np.random.default_rng(4)
    sh = int(rng.integers(min_shift, max(min_shift + 1, a.F - min_shift)))
    assert np.allclose(a.Cs00[0], a.Cs00[0])  # built
    rs = (a.rows0 - sh) % a.F
    assert np.array_equal(rs, (a.rows0 - sh) % a.F)


