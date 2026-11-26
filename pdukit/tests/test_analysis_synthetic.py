"""Scoring synthetic runs against the planted answer (pdu score-bench)."""

import csv
import json

import numpy as np
import pytest

from pdukit.analysis.synthetic import SuiteScorer, SyntheticScorer, main
from pdukit.io.results.reader import ResultsReader

N = 24
UNITS = [list(range(0, 8)), list(range(8, 16)), list(range(16, 24))]


def _rho(rng):
    """Strong links inside each planted unit, weak noise elsewhere."""
    R = rng.uniform(0.0, 0.1, size=(N, N))
    for u in UNITS:
        R[np.ix_(u, u)] += 0.5
    np.fill_diagonal(R, 0.0)
    return R


def _fake_run(path, cores, design="units", lag=1, seeds=32, gen_seed=0,
              truth_units=UNITS, ports=True, named=None, contacts=None):
    """Write the files a pdu bench run leaves behind."""
    rng = np.random.default_rng(1)
    R = _rho(rng)
    path.mkdir(parents=True)
    truth = {"units": truth_units, "design": design,
             "contacts": contacts or [[i, i + 1] for i in range(N - 1)]}
    if named:
        truth["named"] = named
    results = {
        "args": {"lag": lag, "dec_seeds": seeds, "bench_seed": gen_seed,
                 "design_obs_noise": 0.25},
        "data": {"n_residues": N,
                 "planted": {"design": design, "dt_ns": 0.1}},
        "truth": truth,
        "predictive_graph": {"rho": R.tolist()},
    }
    (path / "results.json").write_text(json.dumps(results))
    boot = np.stack([R + rng.normal(0, 0.005, R.shape) for _ in range(8)])
    np.savez(path / "bootstrap_rho.npz", rho=boot)
    if ports:
        assigned = {i for c in cores for i in c}
        p = {
            "units": [{"unit": k, "core": c} for k, c in enumerate(cores)],
            "counts": {"core": len(assigned), "loose": N - len(assigned)},
            "residues": [
                {"residue": i, "cls": "core" if i in assigned else "loose"}
                for i in range(N)
            ],
        }
        (path / "incoming_ports.json").write_text(json.dumps(p))
    return path


def test_reader_gives_ports_bootstrap_and_rho(tmp_path):
    run = _fake_run(tmp_path / "r", UNITS)
    r = ResultsReader.load(run)
    assert len(r.ports()["units"]) == 3
    assert r.bootstrap_rho().shape == (8, N, N)
    assert r.rho().shape == (N, N)
    bare = tmp_path / "bare"
    bare.mkdir()
    (bare / "results.json").write_text("{}")
    assert ResultsReader.load(bare).ports() is None
    assert ResultsReader.load(bare).bootstrap_rho() is None


def test_true_units_are_recovered_and_real(tmp_path):
    s = SyntheticScorer(_fake_run(tmp_path / "r", UNITS), n_random=200)
    score = s.score()
    assert (score.recovered, score.real_cores, score.false_units) == (3, 3, 0)
    assert score.unassigned_pct == 0.0 and score.lag_ns == 0.1


def test_random_core_is_not_real(tmp_path):
    scattered = [0, 9, 17, 5, 12]
    s = SyntheticScorer(
        _fake_run(tmp_path / "r", [UNITS[0], scattered]), n_random=200
    )
    assert s.real_flags == [True, False]
    assert s.score().false_units == 0


def test_real_core_in_a_system_without_units_is_a_false_unit(tmp_path):
    run = _fake_run(tmp_path / "r", [UNITS[1]], design="medium",
                    truth_units=[])
    score = SyntheticScorer(run, n_random=200).score()
    assert (score.n_planted, score.real_cores, score.false_units) == (0, 1, 1)


def test_compact_null_uses_recorded_contacts(tmp_path):
    # contacts chain the residues in the order 0, 5, 10, ... (mod 24):
    # compact regions mix the units, so the units beat them
    order = [(5 * k) % N for k in range(N)]
    mixed = [[a, b] for a, b in zip(order, order[1:])]
    s = SyntheticScorer(
        _fake_run(tmp_path / "r", UNITS, contacts=mixed), null="compact"
    )
    assert s.real_flags == [True, True, True]
    # contacts that follow the units: compact regions ARE the units, so
    # this stricter null does not call them real
    s2 = SyntheticScorer(_fake_run(tmp_path / "r2", UNITS), null="compact")
    assert s2.real_flags == [False, False, False]


def test_run_without_port_file_counts_all_unassigned(tmp_path):
    run = _fake_run(tmp_path / "r", [], design="mixed", truth_units=[],
                    ports=False)
    score = SyntheticScorer(run).score()
    assert score.n_found == 0 and score.unassigned_pct == 100.0


def test_suite_tables_follow_the_settings_not_folder_names(tmp_path):
    named = {"A": UNITS[0], "AB": UNITS[0] + UNITS[1]}
    runs = [
        _fake_run(tmp_path / "x1", UNITS),
        _fake_run(tmp_path / "x2", UNITS[:2], seeds=8),
        _fake_run(tmp_path / "x3", [], design="mixed", truth_units=[]),
        _fake_run(tmp_path / "x4", UNITS, design="tau", lag=5, named=named),
    ]
    suite = SuiteScorer(runs, n_random=100)
    t = suite.figure_tables()
    assert [r["design"] for r in t["synthetic_test1.csv"]] == [
        "units", "mixed"]
    assert sorted(r["growth_seeds"] for r in t["synthetic_test4.csv"]) == [
        8, 32]
    tau = {r["unit"]: r["best_jaccard"] for r in t["synthetic_tau.csv"]}
    assert tau == {"A": 1.0, "AB": 0.5}
    assert t["synthetic_tau_counts.csv"][0]["lag_ns"] == 0.5
    assert len(t["synthetic_membership.csv"]) == N


def test_cli_writes_tables(tmp_path, capsys):
    run = _fake_run(tmp_path / "r", UNITS)
    assert main([str(run), "-o", str(tmp_path / "t"), "--n-random", "50"]) == 0
    with open(tmp_path / "t" / "synthetic_runs.csv") as fh:
        rows = list(csv.DictReader(fh))
    assert rows[0]["recovered"] == "3"
    assert main([str(tmp_path / "nothing"), "-o", str(tmp_path / "u")]) == 1


@pytest.mark.parametrize("bad", ["foo"])
def test_compact_null_needs_contacts(tmp_path, bad):
    run = _fake_run(tmp_path / bad, UNITS)
    d = json.loads((run / "results.json").read_text())
    d["truth"].pop("contacts")
    (run / "results.json").write_text(json.dumps(d))
    with pytest.raises(ValueError, match="no contacts"):
        SyntheticScorer(run, null="compact").real_flags
