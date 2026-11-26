"""Tests for ``pdu plot pml``."""

import json

import numpy as np
import pytest

from pdukit.cli.main import main as pdu_main
from pdukit.viz import pymol

N = 12


def _pdb(path, chain=" ", first=10):
    rows = []
    for i in range(N):
        rows.append(
            f"ATOM  {i + 1:5d}  CA  ALA {chain}{first + i:4d}    "
            f"{1.0 * i:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00           C"
        )
    path.write_text("\n".join(rows) + "\nEND\n")
    return path


@pytest.fixture()
def results():
    rho = np.zeros((N, N))
    rho[0, 6] = rho[6, 0] = 0.9  # strong, far apart in sequence
    rho[0, 1] = rho[1, 0] = 0.95  # sequence neighbours: dropped
    rho[3, 9] = 0.02  # below theta
    return {
        "data": {"n_residues": N},
        "predictive_graph": {"theta": 0.05, "rho": rho.tolist()},
        "units_final": [
            {"unit": 0, "residues": [0, 1, 2], "size": 3},
            {"unit": 1, "residues": [6, 7], "size": 2},
        ],
        "incoming_ports": {
            "units": [
                {"unit": 0, "core": [0, 1, 2], "ports": [3]},
                {"unit": 1, "core": [6, 7], "ports": []},
            ],
            "residues": [
                {"residue": 5, "cls": "bridge"},
                {"residue": 9, "cls": "loose"},
            ],
        },
    }


def test_real_residue_numbers_are_used(tmp_path, results):
    pdb = _pdb(tmp_path / "p.pdb", first=10)
    text = pymol.build_pml(results, pdb)
    assert "select u0, (resi 10+11+12+13)" in text  # 0..3 -> 10..13
    assert "select pdu_bridge, (resi 15)" in text
    assert "scene units_cartoon, store" in text
    assert "scene units_detail, store" in text
    assert "set transparency, 0.65, surf_u0" in text


def test_chain_ids_are_kept(tmp_path, results):
    pdb = _pdb(tmp_path / "p.pdb", chain="B")
    assert "(chain B and resi" in pymol.build_pml(results, pdb)


def test_wrong_pdb_is_refused(tmp_path, results):
    pdb = _pdb(tmp_path / "p.pdb")
    results["data"]["n_residues"] = N + 3
    with pytest.raises(SystemExit):
        pymol.build_pml(results, pdb)


def test_first_residue_renumbers(tmp_path, results):
    pdb = _pdb(tmp_path / "p.pdb", first=10)
    text = pymol.build_pml(results, pdb, first_residue=100)
    assert "resi 100+101+102+103" in text


def test_edges_are_significant_and_not_local(results):
    pairs = pymol.significant_edges(
        results, max_partners=3, min_seq_sep=4, top_frac=1.0
    )
    assert [(i, j) for i, j, _ in pairs] == [(0, 6)]


def test_edges_none_draws_no_network(tmp_path, results):
    pdb = _pdb(tmp_path / "p.pdb")
    text = pymol.build_pml(results, pdb, edges="none")
    assert "CYLINDER" not in text and "pdu_network" not in text


def test_cli_writes_script(tmp_path, results):
    (tmp_path / "results.json").write_text(json.dumps(results))
    pdb = _pdb(tmp_path / "p.pdb")
    assert pdu_main(["plot", "pml", str(tmp_path), "--pdb", str(pdb)]) == 0
    assert (tmp_path / "units.pml").read_text().startswith("# PDU units")
