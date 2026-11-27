"""Convergence tests: thinning data and comparing runs (pdu converge)."""

import itertools
import json

import numpy as np
import pytest

from pdukit.analysis.convergence import ConvergenceComparer, main
from pdukit.cog.features import thin_replicas
from pdukit.numerics.matching import adjusted_rand, labels_from_units


def _ari_by_pairs(a, b):
    """Adjusted Rand index by counting pairs (slow, independent check)."""
    n = len(a)
    same = [(a[i] == a[j], b[i] == b[j])
            for i, j in itertools.combinations(range(n), 2)]
    both = sum(x and y for x, y in same)
    sa = sum(x for x, _ in same)
    sb = sum(y for _, y in same)
    total = len(same)
    exp = sa * sb / total
    return (both - exp) / (0.5 * (sa + sb) - exp)


def test_adjusted_rand_matches_pair_counting_and_is_label_free():
    rng = np.random.default_rng(0)
    for _ in range(20):
        a, b = rng.integers(0, 4, 30), rng.integers(0, 5, 30)
        assert adjusted_rand(a, b) == pytest.approx(_ari_by_pairs(a, b))
    a = np.array([0, 0, 1, 1, 2, 2])
    assert adjusted_rand(a, a) == 1.0
    assert adjusted_rand(a, 5 - a) == 1.0


def test_unassigned_residues_are_singletons():
    lab = labels_from_units([[0, 1], [3, 4]], 6)
    assert list(lab[:2]) == [0, 0] and lab[3] == lab[4] == 1
    assert len({lab[2], lab[5], 0, 1}) == 4


def test_thin_replicas_keeps_each_replica_apart():
    x = np.arange(20)[:, None]
    out, segs = thin_replicas(x, [(0, 10), (10, 20)], stride=3)
    assert out[:, 0].tolist() == [0, 3, 6, 9, 10, 13, 16, 19]
    assert segs == [(0, 4), (4, 8)]
    out, segs = thin_replicas(x, [(0, 10), (10, 20)], max_frames=4)
    assert out[:, 0].tolist() == [0, 1, 2, 3, 10, 11, 12, 13]


def test_bench_design_is_thinned():
    from pdukit.cog.features import load_data
    from pdukit.config.config import Config

    cfg = Config.build(sets=[
        ("outdir", "out"), ("planted", "1"), ("bench-design", "tau"),
        ("design-frames", "100"), ("design-replicas", "2"), ("stride", "5"),
    ])
    data = load_data(cfg, {})
    assert data["feats"].shape[0] == 40
    assert data["segments"] == [(0, 20), (20, 40)]


N = 24
UNITS = [list(range(0, 8)), list(range(8, 16)), list(range(16, 24))]


def _run(path, cores, noise=0.0, theta=0.03, seed=0):
    rng = np.random.default_rng(seed)
    R = rng.uniform(0.0, 0.1, size=(N, N))
    for u in UNITS:
        R[np.ix_(u, u)] += 0.5
    R += rng.normal(0, noise, R.shape)
    np.fill_diagonal(R, 0.0)
    path.mkdir(parents=True)
    results = {
        "args": {"stride": 1, "frames": 0, "lag": 50, "seed": seed},
        "data": {"n_residues": N, "n_frames": 1000,
                 "segments": [[0, 500], [500, 1000]]},
        "predictive_graph": {"rho": R.tolist()},
        "calibration": {"theta_in": {"2": theta, "4": theta},
                        "theta_out": {"2": theta, "4": theta}},
        "timings": {"workspace_seconds": 10.0},
        "units_final": [{"residues": c} for c in cores],
    }
    (path / "results.json").write_text(json.dumps(results))
    np.savez(path / "bootstrap_rho.npz",
             rho=np.stack([R + rng.normal(0, 0.005, R.shape)
                           for _ in range(8)]))
    assigned = {i for c in cores for i in c}
    (path / "incoming_ports.json").write_text(json.dumps({
        "units": [{"unit": k, "core": c} for k, c in enumerate(cores)],
        "counts": {"core": len(assigned), "loose": N - len(assigned)},
        "residues": [{"residue": i,
                      "cls": "core" if i in assigned else "loose"}
                     for i in range(N)],
    }))
    return path


def test_same_units_agree_and_a_lost_unit_shows(tmp_path):
    ref = _run(tmp_path / "full", UNITS)
    same = _run(tmp_path / "s2", UNITS, noise=0.01, theta=0.033, seed=1)
    lost = _run(tmp_path / "s50", UNITS[:2], noise=0.05, seed=2)
    comp = ConvergenceComparer(ref, [same, lost], n_random=200)
    rows = {r["run"]: r for r in comp.rows()}
    assert rows["full"]["ari"] == 1.0 and rows["full"]["rho_corr"] == 1.0
    assert rows["s2"]["ari"] == 1.0
    assert rows["s2"]["theta_in_change"] == pytest.approx(0.1)
    assert rows["s50"]["ari"] < 1.0 and rows["s50"]["cores"] == 2
    assert rows["s50"]["rho_corr"] < rows["s2"]["rho_corr"]
    assert len(comp.pair_rows()) == 3


def test_cli_writes_both_tables(tmp_path):
    ref = _run(tmp_path / "full", UNITS)
    other = _run(tmp_path / "half", UNITS, seed=1)
    assert main([str(ref), str(other), "-o", str(tmp_path / "t"),
                 "--n-random", "100"]) == 0
    assert (tmp_path / "t" / "convergence.csv").is_file()
    assert (tmp_path / "t" / "convergence_pairs.csv").is_file()
    assert main([str(tmp_path / "x"), str(other), "-o", "t2"]) == 1


def test_unfinished_runs_are_skipped(tmp_path, capsys):
    ref = _run(tmp_path / "full", UNITS)
    done = _run(tmp_path / "s2", UNITS, seed=1)
    busy = tmp_path / "busy"
    busy.mkdir()
    (busy / "results.json").write_text(json.dumps({"args": {}}))
    assert main([str(ref), str(done), str(busy), "-o",
                 str(tmp_path / "t"), "--n-random", "50"]) == 0
    out = capsys.readouterr().out
    assert "skipped (no finished results.json): busy" in out
