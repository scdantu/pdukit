"""ResultsWriter writes the named files and hands the tables to stage_files."""

import json

import numpy as np
import pytest

from pdukit.io.results import SCHEMA_VERSION, stage_files
from pdukit.io.results.writer import ResultsWriter


def test_save_rewrites_results_json(tmp_path):
    writer = ResultsWriter(tmp_path)
    writer.save({"a": float("nan"), "b": np.arange(3), "c": {"d": 1}})
    writer.save({"a": 2.5})
    saved = json.loads((tmp_path / "results.json").read_text())
    assert saved == {"schema_version": SCHEMA_VERSION, "a": 2.5}


def test_save_keeps_nan_and_arrays(tmp_path):
    ResultsWriter(tmp_path).save({"a": float("nan"), "b": np.arange(3)})
    data = json.loads((tmp_path / "results.json").read_text())
    assert data["a"] != data["a"]
    assert data["b"] == [0, 1, 2]


def test_ppf_calibration_and_bootstrap_files(tmp_path):
    writer = ResultsWriter(tmp_path)
    writer.write_ppf("lag = 25\n")
    writer.write_calibration({"theta_in": {"2": 0.1}})
    writer.write_bootstrap([[0.5, 0.25], [0.5, 0.75]])
    assert (tmp_path / "pdu_out.ppf").read_text() == "lag = 25\n"
    cal = json.loads((tmp_path / "calibration.json").read_text())
    assert cal == {"theta_in": {"2": 0.1}}
    rho = np.load(tmp_path / "bootstrap_rho.npz")["rho"]
    assert rho.dtype == np.float32 and rho.shape == (2, 2)


@pytest.mark.parametrize(
    "method",
    [
        "write_incoming_tables",
        "write_merge",
        "write_split",
        "write_classes",
        "write_mode_recovery",
    ],
)
def test_stage_methods_call_the_stage_writer(tmp_path, monkeypatch, method):
    calls = []
    monkeypatch.setattr(
        stage_files, method, lambda out, rec: calls.append((out, rec))
    )
    getattr(ResultsWriter(tmp_path), method)("record")
    assert calls == [(tmp_path, "record")]
