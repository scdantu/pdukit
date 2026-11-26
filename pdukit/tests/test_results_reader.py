"""ResultsReader reads results.json from a directory or a file."""

import json

import pytest

from pdukit.io.results import SCHEMA_VERSION
from pdukit.io.results.reader import ResultsReader


def write(path, data):
    path.mkdir(exist_ok=True)
    (path / "results.json").write_text(json.dumps(data))
    return path


def test_loads_from_directory_or_file(tmp_path):
    run = write(tmp_path / "run", {"data": {"n_residues": 5}})
    for where in (run, run / "results.json"):
        res = ResultsReader.load(where)
        assert res == {"data": {"n_residues": 5}}
        assert res.path == run / "results.json"
        assert res.rundir == run


def test_missing_results_json_is_a_clean_error(tmp_path):
    with pytest.raises(SystemExit, match="no results.json"):
        ResultsReader.load(tmp_path)


def test_units_come_from_the_last_stage_that_ran(tmp_path):
    units = lambda *sets: [{"residues": list(s)} for s in sets]  # noqa: E731
    data = {
        "units_refined": units([1, 2]),
        "units_split": units([3, 4]),
        "units_final": units([5, 6]),
        "discovery": {"units": [[9]]},
    }
    for drop, expected in (
        (None, [[5, 6]]),
        ("units_final", [[3, 4]]),
        ("units_split", [[1, 2]]),
        ("units_refined", [[9]]),
    ):
        if drop:
            data.pop(drop)
        run = write(tmp_path / f"r_{drop}", data)
        assert ResultsReader.load(run).units() == expected


def test_no_units_gives_an_empty_list(tmp_path):
    assert ResultsReader.load(write(tmp_path / "r", {})).units() == []


def test_schema_version_is_zero_before_versions_were_recorded(tmp_path):
    assert ResultsReader.load(write(tmp_path / "r", {})).schema_version == 0


def test_a_newer_layout_is_refused(tmp_path):
    run = write(tmp_path / "r", {"schema_version": SCHEMA_VERSION + 1})
    with pytest.raises(SystemExit, match="newer"):
        ResultsReader.load(run)


def test_what_the_writer_saves_loads_back(tmp_path):
    from pdukit.io.results.writer import ResultsWriter

    ResultsWriter(tmp_path).save({"units_final": [{"residues": [1, 2]}]})
    res = ResultsReader.load(tmp_path)
    assert res.schema_version == SCHEMA_VERSION
    assert res.units() == [[1, 2]]
