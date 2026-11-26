"""Tests for ``pdu export`` (results.json -> CSV / ASCII)."""

import csv
import json

import pytest

from pdukit.analysis import tables as export
from pdukit.cli.main import main as pdu_main

RESULTS = {
    "version": "COG 2.1",
    "mode": "incoming-only growth",
    "data": {"n_frames": 100, "n_residues": 6, "n_feat": 2},
    "args": {"lag": 1, "seed": 0},
    "timings": {"data_seconds": 0.5, "calibration_seconds": 2.0},
    "calibration": {
        "theta_in": {"2": 0.1, "4": 0.2},
        "theta_out": {"2": 0.11, "4": 0.21},
        "baseline_median": {"2": 1.5, "4": 2.5},
        "baseline_iqr": {"2": [1.0, 2.0], "4": [2.0, 3.0]},
        "n_supports": {"2": 10, "4": 12},
        "stopped_by": {"2": "converged", "4": "max_supports"},
    },
    "units_final": [
        {"unit": 0, "residues": [0, 1, 2], "size": 3},
        {"unit": 1, "residues": [3, 4], "size": 2},
    ],
    "units_split": [
        {"unit": 0, "cohesion": 0.9, "nu_in": 0.1, "nu_out": 0.8},
        {"unit": 1, "cohesion": 0.8, "nu_in": None, "nu_out": 1.1},
    ],
    "incoming_ports": {
        "counts": {"core": 5},
        "residues": [{"residue": 0, "cls": "core", "unit": "0"}],
        "units": [{"unit": 0, "ports": [5], "extended": [0, 1, 2, 5]}],
    },
}


@pytest.fixture()
def run_dir(tmp_path):
    (tmp_path / "results.json").write_text(json.dumps(RESULTS))
    return tmp_path


def test_units_table_joins_split_and_ports():
    cols, rows = export.build(RESULTS, ["units"])["units"]
    assert cols[:3] == ["unit", "size", "residues"]
    assert rows[0][cols.index("ports")] == [5]
    assert rows[1][cols.index("nu_in")] is None


def test_csv_cells_are_flat():
    cols, rows = export.build(RESULTS, ["units"])["units"]
    parsed = list(csv.reader(export.to_csv(cols, rows).splitlines()))
    assert parsed[1][2] == "0 1 2"  # residues, space separated
    assert parsed[2][cols.index("nu_in")] == ""  # missing -> empty


def test_text_table_is_aligned():
    cols, rows = export.build(RESULTS, ["timings"])["timings"]
    lines = export.to_text(cols, rows, "timings").splitlines()
    assert lines[0] == "timings" and set(lines[1]) == {"="}
    assert len({len(x.rstrip()) <= len(lines[2]) for x in lines[2:]}) == 1


def test_cli_writes_files(run_dir, capsys):
    assert pdu_main(["export", str(run_dir)]) == 0
    out = run_dir / "export"
    for name in ("units.csv", "units.txt", "calibration.csv", "report.txt"):
        assert (out / name).is_file()
    assert "predictive_edges" in capsys.readouterr().out  # reported as empty
    assert "theta_in" in (out / "calibration.csv").read_text().splitlines()[0]


def test_format_and_tables_options(run_dir):
    out = run_dir / "o"
    export.main(
        [str(run_dir), "-f", "csv", "--tables", "units", "-o", str(out)]
    )
    assert sorted(p.name for p in out.iterdir()) == ["units.csv"]


def test_key_flattens_any_part(run_dir):
    out = run_dir / "k"
    export.main([str(run_dir), "--key", "units_final", "-o", str(out)])
    text = (out / "units_final.csv").read_text().splitlines()
    assert text[0].split(",") == ["unit", "residues", "size"]


def test_bad_inputs_exit(run_dir, tmp_path):
    with pytest.raises(SystemExit):
        export.main([str(tmp_path / "nowhere")])
    with pytest.raises(SystemExit):
        export.main([str(run_dir), "--tables", "nope"])
    with pytest.raises(SystemExit):
        export.main([str(run_dir), "--key", "no.such"])


def test_list_shows_every_table(capsys):
    export.main(["--list"])
    out = capsys.readouterr().out
    assert all(name in out for name in export.TABLES)
