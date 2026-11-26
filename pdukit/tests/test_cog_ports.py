"""Core / port / bridge classification and fixed-set scoring."""


import numpy as np

from pdukit.cog.units.ports import classify_residues
from pdukit.io.results.stage_files import write_classes


def blocks():
    rho = np.full((14, 14), 0.1)
    rho[:6, :6] = 0.8  # unit A: 0-5
    rho[6:12, 6:12] = 0.8  # unit B: 6-11
    rho[13, :6] = rho[:6, 13] = 0.8  # residue 13: belongs with A
    rho[12, :] = rho[:, 12] = 0.45  # residue 12: equally tied to everything
    np.fill_diagonal(rho, 0)
    return rho


def scorer(support):
    return dict(
        nu_in=10 - len(support) / 2,
        nu_out=1.0,
        theta_in=0.1,
        theta_out=0.1,
        rho_in=[0.5, 0.2],
        rho_out=[0.3],
    )


def test_port_bridge_core(tmp_path):
    rho = blocks()
    cores = [list(range(0, 5)), list(range(6, 12))]  # residue 5 left out of A
    rng = np.random.default_rng(0)
    boot = [
        np.clip(rho + rng.normal(0, 0.02, rho.shape), 0, None)
        for _ in range(10)
    ]
    res = classify_residues(rho, cores, bootstrap=boot, scorer=scorer)
    cls = {r["residue"]: r["cls"] for r in res["residues"]}
    assert cls[5] == "port" and cls[13] == "port"
    assert cls[12] == "bridge"
    assert cls[0] == "core" and cls[6] == "core"
    ext = {u["unit"]: u["extended"] for u in res["units"]}
    assert ext[0] == [0, 1, 2, 3, 4, 5, 13] and ext[1] == list(range(6, 12))
    u0 = res["units"][0]
    assert u0["nu_in_extended"] < u0["nu_in_core"] and "strength_in_core" in u0
    write_classes(tmp_path, res)
    assert (tmp_path / "incoming_residue_classes.csv").exists()
    assert (tmp_path / "incoming_units_extended.csv").exists()


def diluting(seed=0, w=0.4, N=40):
    """Core A (0-7) and B (20-29); residues 8-19 couple to A slightly less
    than A's members couple to each other."""
    g = np.random.default_rng(seed)
    rho = np.abs(g.normal(0.2, 0.08, (N, N)))
    A, W, B = list(range(8)), list(range(8, 20)), list(range(20, 30))
    rho[np.ix_(A, A)] = np.abs(g.normal(0.5, 0.08, (8, 8)))
    rho[np.ix_(W, A)] = np.abs(g.normal(w, 0.08, (12, 8)))
    rho[np.ix_(A, W)] = np.abs(g.normal(w, 0.08, (8, 12)))
    rho[np.ix_(B, B)] = np.abs(g.normal(0.5, 0.08, (10, 10)))
    np.fill_diagonal(rho, 0)
    boot = [
        np.clip(rho + g.normal(0, 0.05, rho.shape), 0, None) for _ in range(20)
    ]
    return rho, boot, A, B


def test_guard_stops_dilution():
    rho, boot, A, B = diluting()
    on = classify_residues(rho, [A, B], bootstrap=boot)
    off = classify_residues(rho, [A, B], bootstrap=boot, guard=False)
    u_on, u_off = on["units"][0], off["units"][0]
    assert u_off["n_ports"] > u_on["n_ports"]
    assert u_off["cohesion_extended"] < u_on["cohesion_extended"]
    for u in on["units"]:  # every unit stays within noise of its core
        assert u["cohesion_change"] >= -1.645 * u["cohesion_change_sd"] - 1e-12
    refused = [r for r in on["residues"] if r["cls"] == "loose"]
    assert any("guard" in r["failed"] for r in refused)
    assert on["rounds"] >= 2


