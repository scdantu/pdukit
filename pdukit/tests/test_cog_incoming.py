"""Tests of strict incoming-only selection and the pipeline entry point."""

import json

import numpy as np
import pytest

from pdukit.cog.pipeline import Pipeline
from pdukit.cog.units.grow import grow_incoming
from pdukit.io.results.stage_files import write_incoming_tables
from pdukit.tests.cfg import parse_args


def row(inc, out, cost=-100):
    return dict(
        nu_in=inc,
        nu_out=out,
        cost=cost,
        theta_in=0.01,
        theta_out=0.01,
        rho_in=[0.3, 0.1],
        rho_out=[0.4, 0.2],
    )


class Scorer:
    def __init__(self, values):
        self.values = values

    def __call__(self, residues):
        return self.values.get(tuple(sorted(residues)), row(10, 0))

    def score_batch(self, supports):
        return [self(s) for s in supports]


def graph():
    return ~np.eye(5, dtype=bool)


def test_accepts_best_incoming_even_when_outgoing_rises():
    s = Scorer({(0,): row(4, 6), (0, 1): row(3, 8), (0, 2): row(3.5, 0)})
    r = grow_incoming(s, graph(), [[0]], min_size=2, max_size=3)
    assert r["units"][0]["residues"] == [0, 1]
    assert r["units"][0]["nu_out"] == 8
    assert r["runs"][0]["stop_reason"] == "no_incoming_improvement"


def test_rejects_flat_and_uphill_even_when_j_improves():
    s = Scorer(
        {(0,): row(2, 8, 0), (0, 1): row(2, 0, -100), (0, 2): row(3, 0, -200)}
    )
    r = grow_incoming(s, graph(), [[0]], min_size=3, max_size=4)
    assert r["units"] == []
    assert r["runs"][0]["final"] == [0]
    assert not any(p["accepted"] for p in r["runs"][0]["proposals"])


def test_growth_path_gain_margin_and_output(tmp_path):
    s = Scorer(
        {
            (0,): row(4, 1),
            (0, 1): row(3, 7),
            (0, 1, 2): row(2, 8),
            (0, 1, 2, 3): row(1.95, 0),
        }
    )
    r = grow_incoming(s, graph(), [[0]], min_size=3, max_size=4, min_gain=0.1)
    assert r["units"][0]["residues"] == [0, 1, 2]
    assert [h["nu_in"] for h in r["runs"][0]["history"]] == [4, 3, 2]
    write_incoming_tables(tmp_path, r)
    assert "False" in (tmp_path / "incoming_proposals.csv").read_text()
    assert "rho_out" in (tmp_path / "incoming_growth.csv").read_text()
    json.dumps(r, allow_nan=False)


def test_frontier_and_deduplication():
    g = graph()
    g[0, 2] = g[2, 0] = False
    s = Scorer({(0,): row(4, 1), (0, 1): row(3, 2), (0, 2): row(0, 0)})
    r = grow_incoming(s, g, [[0], [0]], min_size=2, max_size=2)
    assert len(r["units"]) == 1
    assert r["units"][0]["residues"] == [0, 1]


def test_bad_scores_and_options():
    with pytest.raises(ValueError, match="invalid nu_in"):
        grow_incoming(
            Scorer({(0,): row(float("nan"), 1)}), graph(), [[0]], max_size=4
        )
    with pytest.raises(SystemExit):
        parse_args(["--planted", "-o", "out", "--incoming-min-gain", "-1"])
    a = parse_args(["--planted", "-o", "out", "--grow-objective", "incoming"])
    assert a.grow_objective == "incoming"


class FakeState:
    """Stands in for RunState: only what Pipeline.run touches."""

    def __init__(self, argv, calls):
        self.args = parse_args(argv)
        self.out = "out"
        self.calls = calls

    def save(self):
        self.calls.append("save")


def fake_stage(code, calls):
    class Fake:
        def run(self, state):
            calls.append(self.code)

    Fake.code = code
    return Fake


CODES = ["DAT", "REP", "WIN", "PRE", "WSP", "GRA", "CAL", "GRO", "MOD"]


def test_pipeline_runs_the_stages_in_order_and_saves_last():
    calls = []
    state = FakeState(
        ["--planted", "-o", "out", "--grow-objective", "incoming"], calls
    )
    stages = [fake_stage(code, calls) for code in CODES]
    assert Pipeline(state, stages).run() == 0
    assert calls == ["save", *CODES, "save"]


def test_calibrate_only_stops_after_calibration():
    calls = []
    state = FakeState(
        [
            "--planted",
            "-o",
            "out",
            "--grow-objective",
            "incoming",
            "--calibrate-only",
        ],
        calls,
    )
    stages = [fake_stage(code, calls) for code in CODES]
    assert Pipeline(state, stages).run() == 0
    assert calls == ["save", *CODES[: CODES.index("CAL") + 1]]


@pytest.mark.parametrize("objective", ["channels", "cohesion"])
def test_archived_objectives_are_refused_before_any_work(objective):
    calls = []
    state = FakeState(
        ["--planted", "-o", "out", "--grow-objective", objective], calls
    )
    with pytest.raises(SystemExit, match="archived"):
        Pipeline(state, [fake_stage("DAT", calls)]).run()
    assert calls == []


def test_start_min_gain_applies_to_the_first_addition_only():
    import numpy as np

    from pdukit.cog.units.grow import grow_incoming

    # nu_in falls by 0.05 for the first addition, then by 0.5 per residue
    def scorer(support):
        k = len(support)
        nu = 5.0 if k == 2 else 4.95 - 0.5 * (k - 3)
        return {"nu_in": nu, "nu_out": 0.0, "theta_in": 0.1,
                "theta_out": 0.1, "rho_in": [0.5], "rho_out": [0.0]}

    adj = np.ones((8, 8), bool) & ~np.eye(8, dtype=bool)
    free = grow_incoming(scorer, adj, [[0, 1]], max_size=6, min_size=3)
    assert len(free["runs"][0]["history"]) > 1
    gated = grow_incoming(scorer, adj, [[0, 1]], max_size=6, min_size=3,
                          start_min_gain=0.1)
    assert len(gated["runs"][0]["history"]) == 1
    late = grow_incoming(scorer, adj, [[0, 1]], max_size=6, min_size=3,
                         start_min_gain=0.01)
    assert len(late["runs"][0]["history"]) == 5  # later steps not gated
