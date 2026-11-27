"""Following residues and units across lag times (pdu track)."""

import json

from pdukit.analysis.track import LagTracker, main

N = 12


def _run(path, lag, cores, loose=()):
    path.mkdir(parents=True)
    results = {"args": {"lag": lag, "frame_ps": 100.0},
               "data": {"n_residues": N}}
    (path / "results.json").write_text(json.dumps(results))
    assigned = {i for c in cores for i in c}
    res = [{"residue": i,
            "cls": "core" if i in assigned else
            ("loose" if i in loose else "port")} for i in range(N)]
    (path / "incoming_ports.json").write_text(json.dumps({
        "units": [{"unit": k, "core": c} for k, c in enumerate(cores)],
        "residues": res}))
    return path


def _three(tmp_path):
    a = _run(tmp_path / "r1", 1, [[0, 1, 2, 3], [4, 5, 6, 7]])
    b = _run(tmp_path / "r2", 5, [[0, 1, 2, 3], [4, 5], [6, 7]])
    c = _run(tmp_path / "r3", 20, [[0, 1, 2, 3, 4, 5, 6, 7]], loose=[9])
    return [c, a, b]  # deliberately out of order


def test_runs_are_ordered_by_lag_and_lineage_has_fates(tmp_path):
    tr = LagTracker(_three(tmp_path))
    assert [r.lag_ps for r in tr.runs] == [100.0, 500.0, 2000.0]
    rows = tr.lineage_rows()
    first = [r for r in rows if r["from_lag"] == "100 ps"]
    fates = {r["unit"]: r["fate"] for r in first}
    assert fates == {0: "persists", 1: "splits"}
    second = [r for r in rows if r["from_lag"] == "500 ps"]
    assert {r["fate"] for r in second} == {"merges"}


def test_residue_persistence_and_core_graph(tmp_path):
    tr = LagTracker(_three(tmp_path))
    rows = {r["residue"]: r for r in tr.residue_rows()}
    assert rows[0]["lags_core"] == 3 and rows[0]["mate_jaccard_mean"] < 1
    assert rows[9]["class_2 ns"] == "loose"
    assert rows[10]["mate_jaccard_mean"] is None
    edges, comps = tr.core_graph()
    assert comps[0]["residues"] == "0 1 2 3"   # together in all three lags
    assert any(c["residues"] == "4 5 6 7" for c in comps)


def test_cli(tmp_path, capsys):
    runs = [str(p) for p in _three(tmp_path)]
    assert main(runs + ["-o", str(tmp_path / "t")]) == 0
    assert (tmp_path / "t" / "track_lineage.csv").is_file()
    assert "unit fates" in capsys.readouterr().out
    assert main([runs[0], "-o", str(tmp_path / "u")]) == 1
