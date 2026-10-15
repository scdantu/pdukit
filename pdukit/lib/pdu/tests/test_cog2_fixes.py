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

from ..interface import (
    FluxWorkspace,
    _bootstrap_ci_rel_width_by_support,
    _bootstrap_ci_rel_width_two_way,
    _variance_components,
    calibrate_null_thresholds,
    check_null_calibration,
    complete_theta_table,
    decompose,
)
from .test_interface import _block_adjacency, simulate


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


def test_partial_table_no_longer_dictates_unit_sizes(feats, ws):
    """Calibrating only sizes 10 and 12 used to make every other size cost
    infinity, so units of exactly 10 or 12 came out.  With the table
    completed, every unit has a finite cost and growth is not confined to
    the calibrated sizes.
    """
    adj = _block_adjacency(feats.shape[1])
    cal = calibrate_null_thresholds(
        ws,
        adj,
        sizes=[10, 12],
        unit_dim=8,
        env_dim=30,
        target_rel_width=0.5,
        batch_supports=5,
        min_supports=10,
        max_pool_size=400,
        seed=0,
    )
    d = decompose(
        feats,
        adj,
        ws,
        unit_dim=8,
        env_dim=30,
        n_seeds=4,
        max_size=12,
        max_iter=2,
        n_controls=0,
        seed=0,
        null_calibration=cal,
    )
    assert all(np.isfinite(s["cost"]) for s in d["scores"])
    with pytest.raises(ValueError):
        decompose(
            feats,
            adj,
            ws,
            unit_dim=8,
            env_dim=30,
            n_seeds=2,
            max_size=6,
            max_iter=1,
            n_controls=0,
            seed=0,
            null_calibration=cal,
            null_fill="strict",
        )


# 2 ---------------------------------------------------------------------------
def test_two_way_bootstrap_sees_shared_shift_noise():
    rng = np.random.default_rng(0)
    n_sup, n_sh = 200, 10
    shift_effect = rng.normal(scale=0.05, size=n_sh)  # shared by all rows
    rows = (
        0.3
        + shift_effect[None, :]
        + rng.normal(scale=0.01, size=(n_sup, n_sh))
    )
    _, w_sup = _bootstrap_ci_rel_width_by_support(rows, rng=rng)
    _, w_2w = _bootstrap_ci_rel_width_two_way(rows, rng=rng)
    assert w_2w > 3 * w_sup, (w_sup, w_2w)
    vc = _variance_components(rows)
    assert vc["between_shifts"] > 10 * vc["between_supports"]


def test_calibration_reports_both_widths_and_components(feats, ws):
    adj = _block_adjacency(feats.shape[1])
    cal = calibrate_null_thresholds(
        ws,
        adj,
        sizes=[6],
        unit_dim=8,
        env_dim=30,
        target_rel_width=0.5,
        batch_supports=5,
        min_supports=10,
        max_pool_size=400,
        seed=0,
    )
    d = cal["diagnostics"][6]
    for key in (
        "rel_ci_width_support",
        "rel_ci_width_two_way",
        "variance_components",
    ):
        assert key in d
    assert cal["config"]["bootstrap"] == "two_way"
    assert "fingerprint" in cal["config"]


# 3 ---------------------------------------------------------------------------
def test_table_from_other_data_is_refused(feats, ws):
    adj = _block_adjacency(feats.shape[1])
    cal = calibrate_null_thresholds(
        ws,
        adj,
        sizes=[6],
        unit_dim=8,
        env_dim=30,
        target_rel_width=0.5,
        batch_supports=5,
        min_supports=10,
        max_pool_size=400,
        seed=0,
    )
    check_null_calibration(cal, ws, 8, 30)  # same data: fine
    other = FluxWorkspace(
        feats[:3000],
        lag=1,
        n_null=8,
        seed=1,
        dtype=np.float64,
        truncation="koopman",
    )
    with pytest.raises(ValueError, match="n_pairs"):
        check_null_calibration(cal, other, 8, 30)
    lag2 = FluxWorkspace(
        feats, lag=2, n_null=8, seed=1, dtype=np.float64, truncation="koopman"
    )
    with pytest.raises(ValueError, match="lag"):
        check_null_calibration(cal, lag2, 8, 30)


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


# 5 ---------------------------------------------------------------------------
def test_controls_use_the_same_cost_as_units(feats, ws):
    adj = _block_adjacency(feats.shape[1])
    d = decompose(
        feats,
        adj,
        ws,
        unit_dim=6,
        env_dim=20,
        n_seeds=2,
        max_size=8,
        max_iter=1,
        n_controls=2,
        seed=0,
        cost_fn=lambda r: 42.0,
    )
    vals = [
        v for row in d["controls"].values() for vs in row.values() for v in vs
    ]
    assert vals and all(v == 42.0 for v in vals)
    assert all(s["cost"] == 42.0 for s in d["scores"])


def test_unexpected_scoring_errors_are_raised(feats, ws):
    adj = _block_adjacency(feats.shape[1])

    def broken(support):
        raise RuntimeError("bug in a scorer")

    with pytest.raises(RuntimeError):
        decompose(
            feats,
            adj,
            ws,
            unit_dim=6,
            env_dim=20,
            n_seeds=2,
            max_size=6,
            max_iter=1,
            n_controls=0,
            seed=0,
            scorer=broken,
        )
