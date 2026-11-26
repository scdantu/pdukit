"""One run of the whole current pipeline on a small planted system.

Planted system: 5 units of 8 residues, two of them uncoupled decoys, three
replicas.  Settings are the shipped mdCATH preset (incoming growth, merge,
recursive split, ports, global-mode recovery) with a short trajectory, so
this takes about a minute on a CPU.  It checks that every stage ran and
wrote its output, that the units agree with the planted answer, and that the
run can be reproduced from the parameter file it wrote.
"""

import json
from pathlib import Path

import pytest

from pdukit.cli.main import main
from pdukit.config import Config
from pdukit.numerics.matching import match_units

PRESET = Path(__file__).resolve().parents[1] / "params" / "mdcath_incoming.ppf"
EXPECTED_FILES = [
    "results.json",
    "pdu_out.ppf",
    "calibration.json",
    "bootstrap_rho.npz",
    "incoming_growth.csv",
    "incoming_candidates.csv",
    "incoming_merge.json",
    "incoming_split.json",
    "incoming_split_units.csv",
    "incoming_ports.json",
    "incoming_residue_classes.csv",
    "incoming_units_extended.csv",
    "global_mode_recovery.json",
    "global_mode_recovery.csv",
]
EXPECTED_KEYS = [
    "version",
    "args",
    "data",
    "truth",
    "calibration",
    "incoming_growth",
    "incoming_merge",
    "incoming_split",
    "incoming_ports",
    "units_final",
    "global_mode_recovery",
]


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    out = tmp_path_factory.mktemp("planted_run")
    main(
        [
            "bench",
            "-p", str(PRESET),
            "--grow-objective", "incoming",
            "--n-units", "5",
            "--unit-size", "8",
            "--frames-per-replica", "1500",
            "--n-replicas", "3",
            "--n-null", "16",
            "--device", "cpu",
            "-o", str(out),
        ]
    )  # fmt: skip
    return out, json.loads((out / "results.json").read_text())


def test_every_stage_wrote_its_output(run):
    out, _ = run
    missing = [f for f in EXPECTED_FILES if not (out / f).is_file()]
    assert missing == []


def test_results_json_has_every_section(run):
    _, results = run
    assert [k for k in EXPECTED_KEYS if k not in results] == []
    assert results["units_final"]


def test_units_are_valid_residue_sets(run):
    _, results = run
    n_res = 5 * 8
    for unit in results["units_final"]:
        residues = unit["residues"]
        assert unit["size"] == len(residues) >= 3
        assert residues == sorted(set(residues))
        assert 0 <= residues[0] and residues[-1] < n_res


def test_units_agree_with_the_planted_answer(run):
    _, results = run
    found = [u["residues"] for u in results["units_final"]]
    match = match_units(found, results["truth"]["units"])
    # at least three of five planted units are recovered (Jaccard >= 0.67),
    # no found unit straddles two planted units, and every found unit is
    # mostly one planted unit
    assert match["n_recovered"] >= 3
    assert match["n_merged"] == 0
    assert min(match["precision_jaccard"]) >= 0.67


def test_run_can_be_reproduced_from_its_parameter_file(run):
    out, results = run
    values = Config.build(
        ppf=str(out / "pdu_out.ppf"), sets=[("outdir", str(out))]
    )
    assert values["lag"] == results["args"]["lag"]
    assert values["n_units"] == 5
    assert values["grow_objective"] == "incoming"
