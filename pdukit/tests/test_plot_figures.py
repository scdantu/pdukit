"""Tests for confidence scores and the matplotlib / movie plot tools."""

import json

import numpy as np
import pytest

from pdukit.numerics import confidence
from pdukit.viz import movie

N = 20


@pytest.fixture()
def results():
    runs = []
    finals = [
        [0, 1, 2, 3],
        [0, 1, 2],
        [0, 1, 2, 3],
        [10, 11, 12],
        [10, 11, 12, 13],
        [5],
    ]
    for k, fin in enumerate(finals):
        hist = [{"step": 0, "added_residue": None, "residues": fin[:1]}]
        for j in range(1, len(fin)):
            hist.append(
                {"step": j, "added_residue": fin[j], "residues": fin[: j + 1]}
            )
        runs.append(
            {
                "run": k,
                "initial": fin[:1],
                "final": fin,
                "eligible": len(fin) >= 3,
                "stop_reason": "x",
                "history": hist,
                "proposals": [],
            }
        )
    rho = np.full((N, N), 0.01)
    rho[2, 7] = rho[7, 2] = 0.5
    rho[4, 12] = 0.4
    return {
        "data": {"n_residues": N},
        "args": {"merge_overlap": 0.5, "lag": 5, "frame_ps": 20.0},
        "units_final": [
            {"unit": 0, "residues": [0, 1, 2, 3, 4], "size": 5},
            {"unit": 1, "residues": [10, 11, 12, 13], "size": 4},
        ],
        "incoming_ports": {
            "counts": {"bridge": 1, "loose": 1},
            "units": [
                {"unit": 0, "core": [0, 1, 2, 3], "ports": [4]},
                {"unit": 1, "core": [10, 11, 12, 13], "ports": []},
            ],
            "residues": [
                {"residue": 7, "cls": "bridge"},
                {"residue": 9, "cls": "loose"},
            ],
        },
        "incoming_growth": {"runs": runs},
        "predictive_graph": {"theta": 0.05, "rho": rho.tolist()},
    }


def test_rcs_is_fraction_of_supporting_seeds(results):
    res = confidence.residue_scores(results)
    # unit 0: seeds 0,1,2 support it (3 seeds); residue 3 is in 2 of them
    assert res[0]["rcs"] == 100.0 and res[0]["n_seeds"] == 3
    assert res[3]["rcs"] == pytest.approx(100 * 2 / 3)
    assert res[4]["rcs"] == 0.0 and res[4]["role"] == "port"
    assert 7 not in res  # bridge: no unit, no score


def test_ucs_is_median_of_core_and_flags_low_evidence(results):
    rows = {u["unit"]: u for u in confidence.unit_scores(results)}
    assert rows[0]["ucs"] == 100.0 and not rows[0]["low_evidence"]
    assert rows[1]["n_seeds"] == 2 and rows[1]["low_evidence"]
    assert (
        confidence.band(95) == "very high"
        and confidence.band(40) == "very low"
    )


def test_seed_assignment_skips_ineligible(results):
    a = confidence.seed_assignment(results)
    assert a[5] is None and a[0] == 0 and a[4] == 1


def test_movie_plan_phases_and_convergence(results):
    plan = movie.build_plan(results, max_growth_frames=10)
    phases = [f["phase"] for f in plan["frames"]]
    assert phases[0] == "GRO" and phases[-1] == "POR"
    assert {"GRO", "MRG", "SPL", "POR"} <= set(phases)
    first, last = plan["frames"][0], plan["frames"][-1]
    # seeds 0 and 2 have different colours at the start, the same at the end
    assert not np.allclose(first["seed_rgb"][0], first["seed_rgb"][2])
    assert np.allclose(last["seed_rgb"][0], last["seed_rgb"][2])
    assert last["bridge"][7] and not last["bridge"][0]


def test_figures_are_written(results, tmp_path):
    pytest.importorskip("matplotlib")
    from pdukit.viz import network, units

    units.units_figure(results, tmp_path / "u.png")
    network.network_figure(
        results, tmp_path / "n.png", min_seq_sep=2, top_frac=1.0
    )
    network.compare_figure(
        {"wt": results, "mut": results},
        tmp_path / "c.png",
        min_seq_sep=2,
        top_frac=1.0,
    )
    for name in ("u.png", "n.png", "c.png"):
        assert (tmp_path / name).stat().st_size > 1000


def test_lag_figure_groups_runs(results, tmp_path):
    pytest.importorskip("matplotlib")
    from pdukit.viz import units

    runs = []
    for i, lag in enumerate((5, 25)):
        d = tmp_path / f"r{i}"
        d.mkdir()
        r = json.loads(json.dumps(results))
        r["args"]["lag"] = lag
        (d / "results.json").write_text(json.dumps(r))
        runs.append(str(d))
    units.lag_figure({"prot": runs}, tmp_path / "lag.png")
    assert (tmp_path / "lag.png").stat().st_size > 1000
    assert units._lag_row(runs[0])["pct_unassigned"] == pytest.approx(
        10.0
    )


def test_directed_links_orientation(results):
    from pdukit.viz import network

    links = network.directed_links(results, min_seq_sep=2, top_frac=1.0)
    # rho[i, j] = j predicts i: rho[4, 12] -> arrow 12 -> 4
    assert (12, 4) in links and (4, 12) not in links
    assert (2, 7) in links and (7, 2) in links
    one = network.directed_links(
        results, min_seq_sep=2, top_frac=1.0, oneway=True
    )
    assert (2, 7) not in one and (12, 4) in one
