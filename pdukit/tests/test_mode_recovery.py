"""Analytic coupled/uncoupled global modes and combination coverage."""

import numpy as np
import pytest

from pdukit.cog.modes import (
    channel_operator,
    combination_sets,
    recover_global_modes,
)
from pdukit.cog.scoring.workspace import FluxWorkspace
from pdukit.io.results.stage_files import write_mode_recovery


def workspace(operator):
    ws = object.__new__(FluxWorkspace)
    ws.N, ws.m, ws.lag, ws.ridge, ws.backend = (
        len(operator),
        1,
        1,
        0.0,
        "numpy",
    )
    ws.C00 = np.eye(len(operator))
    ws.C11 = np.eye(len(operator))
    ws.C01 = np.array(operator)
    return ws


def test_channels_restore_coupled_modes_not_just_their_span(tmp_path):
    ws = workspace([[0.85, 0.1], [0.1, 0.85]])
    report = recover_global_modes(
        ws, [[0], [1]], n_modes=2, local_dim=1, ranks=[1], frame_ps=20
    )
    full = [r for r in report["rows"] if r["combination"] == 0]
    on = [r for r in full if r["channels"] == "on"]
    off = [r for r in full if r["channels"] == "off"]
    ranked = [r for r in full if r["channels"] == "rank-1"]
    assert all(r["coverage_fraction"] == pytest.approx(1) for r in full)
    assert all(r["mode_overlap_squared"] > 0.999999 for r in on)
    assert all(abs(r["timescale_relative_error"]) < 1e-9 for r in on)
    assert abs(off[0]["timescale_relative_error"]) > 0.5
    assert all(
        r["prediction_r2_gain_vs_off"] == pytest.approx(0.01) for r in on
    )
    assert [r["prediction_r2"] for r in ranked] == pytest.approx(
        [r["prediction_r2"] for r in on]
    )
    singles = [r for r in report["rows"] if r["units"] == [0]]
    assert all(r["coverage_fraction"] == pytest.approx(0.5) for r in singles)
    assert all(abs(r["prediction_r2_gain_vs_off"]) < 1e-12 for r in singles)
    write_mode_recovery(tmp_path, report)
    assert (tmp_path / "global_mode_recovery.md").exists()


def test_independent_units_need_no_channels():
    report = recover_global_modes(
        workspace([[0.9, 0], [0, 0.6]]), [[0], [1]], n_modes=2, local_dim=1
    )
    assert all(
        abs(r["prediction_r2_gain_vs_off"]) < 1e-12 for r in report["rows"]
    )


@pytest.mark.parametrize("ridge", [0.0, 0.02])
def test_full_basis_recovers_same_tica_reference_with_directed_covariance(
    ridge,
):
    ws = workspace([[0.8, 0.15], [0.05, 0.6]])
    ws.ridge = ridge
    report = recover_global_modes(ws, [[0], [1]], n_modes=2, local_dim=1)
    rows = [
        r
        for r in report["rows"]
        if r["combination"] == 0 and r["channels"] == "on"
    ]
    assert all(r["mode_overlap_squared"] > 1 - 1e-12 for r in rows)
    assert all(abs(r["timescale_relative_error"]) < 1e-12 for r in rows)


def test_overlapping_coordinates_are_reported_and_remain_finite():
    report = recover_global_modes(
        workspace([[0.9, 0.05], [0.05, 0.8]]),
        [[0, 1], [0, 1]],
        n_modes=2,
        local_dim=2,
    )
    assert report["combinations"][0]["redundant_coordinates"] == 2
    assert report["combinations"][0]["overlapping_supports"]
    assert all(np.isfinite(r["prediction_r2"]) for r in report["rows"])


def test_channel_budget_is_joint_against_the_rest():
    rng = np.random.default_rng(5)
    operator = rng.normal(size=(6, 6))
    blocks = [np.array([0, 1]), np.array([2, 3]), np.array([4, 5])]
    reduced = channel_operator(operator, blocks, rank=1)
    for block in blocks:
        others = np.setdiff1d(np.arange(6), block)
        assert (
            np.linalg.matrix_rank(reduced[np.ix_(others, block)], tol=1e-10)
            <= 1
        )
        assert np.allclose(
            reduced[np.ix_(block, block)], operator[np.ix_(block, block)]
        )


def test_combination_cap_is_explicit_and_keeps_full_case():
    groups, report = combination_sets(12, limit=10)
    assert groups[0] == tuple(range(12))
    assert len(groups) == 10 and report["truncated"]
    assert len(combination_sets(3)[0]) == 7
    with pytest.raises(ValueError, match="out of range"):
        combination_sets(2, explicit=[[2]])
