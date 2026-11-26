"""Protein / Residue / Unit: the residue-level view of a run."""

import pytest

from pdukit.model import Protein, ResidueClass

RESULTS = {
    "data": {"n_residues": 6},
    "units_final": [
        {"unit": 0, "residues": [0, 1, 2], "size": 3},
        {"unit": 1, "residues": [3, 4], "size": 2},
    ],
    "incoming_growth": {"runs": []},
    "incoming_ports": {
        "units": [{"unit": 0, "core": [0, 1, 2], "ports": [3]}],
        "residues": [
            {"residue": 0, "cls": "core", "unit": "0"},
            {"residue": 1, "cls": "core", "unit": "0"},
            {"residue": 2, "cls": "shared", "unit": "0 1"},
            {"residue": 3, "cls": "port", "unit": "0",
             "best_units": "0 1", "best_drives": "1.5 0.5",
             "best_driven": "1.2 0.9"},
            {"residue": 4, "cls": "bridge", "unit": "",
             "best_units": "0 1", "best_drives": "1.1 1.0",
             "best_driven": "1.0 1.1"},
            {"residue": 5, "cls": "loose", "unit": ""},
        ],
    },
}


def test_assign_labels_and_units():
    """Labels, owners and coupling come from the run's results."""
    prot = Protein.from_results(RESULTS)
    assert len(prot) == 6
    assert prot.counts() == {"core": 2, "shared": 1, "port": 1,
                             "bridge": 1, "loose": 1}
    assert prot.residue(2).units == (0, 1)
    assert prot.residue(3).port_of == (0,)
    assert prot.residue(3).drives == {0: 1.5, 1: 0.5}
    assert prot.residue(4).bridge_between == (0, 1)
    assert prot.residue(5).label == "loose"      # str-comparable enum
    assert prot.units[0].ports == [3]
    assert prot.units[0].extended == [0, 1, 2, 3]
    assert [r.index for r in prot.by_label(ResidueClass.PORT)] == [3]


def test_blank_and_bad_label():
    """A fresh protein is unassigned; unknown labels are refused."""
    prot = Protein.blank(3)
    assert prot.counts() == {"unassigned": 3}
    with pytest.raises(ValueError):
        prot.by_label("nonsense")
    with pytest.raises(ValueError):
        Protein([prot.residue(1)])


def test_records_and_frame():
    """One plain record per residue; DataFrame when pandas is present."""
    prot = Protein.from_results(RESULTS)
    rec = prot.to_records()
    assert rec[3]["label"] == "port" and rec[3]["port_of"] == "0"
    pd = pytest.importorskip("pandas")
    assert isinstance(prot.to_frame(), pd.DataFrame)


def test_enums_match_settings_and_phases():
    """Each enum lists exactly the words the settings table allows."""
    from pdukit.config.schema import SETTINGS
    from pdukit.model import enums
    from pdukit.runlog import PHASES

    pairs = {
        "log-level": enums.LogLevel, "shift-mode": enums.ShiftMode,
        "backend": enums.Backend, "seeding": enums.Seeding,
        "search-graph": enums.SearchGraph, "split-method": enums.SplitMethod,
        "split-test": enums.SplitTest,
    }
    choices = {row[1]: list(row[5]) for row in SETTINGS if row[5]}
    for key, enum in pairs.items():
        assert [m.value for m in enum] == choices[key], key
    assert {m.value for m in enums.Phase} == set(PHASES)


def test_enum_assignment_to_config():
    """A setting accepts an enum member and stores the plain word."""
    from pdukit import Config
    from pdukit.model import Backend

    cfg = Config.from_ppf()
    cfg.backend = Backend.SCHUR
    assert cfg.backend == "schur"
    assert any(
        line.split(";")[0].split() == ["backend", "=", "schur"]
        for line in cfg.to_ppf().splitlines()
    )


def test_ensemble_checks_files(tmp_path):
    """An Ensemble refuses missing files and lists its settings."""
    from pdukit import Ensemble, PduError

    with pytest.raises(PduError):
        Ensemble(tmp_path / "nope.xtc", tmp_path / "nope.pdb")
    traj, top = tmp_path / "a.xtc", tmp_path / "t.pdb"
    traj.write_text("")
    top.write_text("")
    ens = Ensemble(traj, top)
    assert ens.n_replicas == 1
    assert ens.settings()["xtc_topology"] == str(top)
