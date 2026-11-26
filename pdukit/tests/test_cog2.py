"""COG 2.1 stages 1-4: planted benchmark, two-way soft count with matched
baseline, cohesion check, pair-owned features.

Each test is a property the method must have if it measures what it says:

1. the planted generator is reproducible, stable and knows its own answer;
2. the incoming spectrum is exactly the v1.0 statistic (FluxWorkspace and
   Schur backends);
3. a unit that only DRIVES is closed to v1.0 and open to COG 2.1, and the
   direction is reported correctly;
4. the soft count is 0 at the noise edge and approaches the hard count far
   above it;
5. a calibration carries its fingerprint and is refused on other data;
6. two uncoupled units glued together are flagged 'split';
7. pair features straddling the boundary are excluded, and with no pair
   features the pair workspace reproduces the residue workspace;
8. COG 2.1 (with the cohesion split move) recovers a planted system with an
   uncoupled decoy pair, and keep_best never returns a worse objective than
   the last iteration.
"""

from __future__ import annotations

import numpy as np
import pytest

from pdukit.bench.planted import planted_system
from pdukit.cog.calibration.scorer import calibrate_cog2
from pdukit.cog.scoring.pair import PairFluxWorkspace
from pdukit.cog.scoring.scorer import FluxScorer, soft_count
from pdukit.cog.scoring.workspace import FluxWorkspace


@pytest.fixture(scope="module")
def oneway():
    """Three units; 0 drives 1 (rank 1), 2 is isolated."""
    return planted_system(
        n_units=3,
        unit_size=8,
        m=4,
        couplings=[(0, 1, 1)],
        n_frames=5000,
        n_replicas=2,
        seed=0,
    )


@pytest.fixture(scope="module")
def ws1(oneway):
    return FluxWorkspace(
        oneway["feats"].astype(np.float64),
        lag=1,
        segments=oneway["segments"],
        n_null=16,
        seed=1,
        dtype=np.float64,
        shift_mode="cross",
    )


# 2 ---------------------------------------------------------------------------
def test_incoming_direction_is_the_v1_statistic(oneway, ws1):
    # with the unit's present summarised as in v1.0 (reduced conditioning)
    sc = FluxScorer(
        ws1, unit_dim=8, env_dim=20, cond_unit="reduced", cond_env=None
    )
    for u in oneway["truth"]["units"]:
        r = ws1.score(u, unit_dim=8, env_dim=20)
        s = sc(u)
        assert np.allclose(r["rho"], s["rho_in"], atol=1e-10)


def test_schur_backend_incoming_matches_schur_workspace(oneway, ws1):
    from pdukit.cog.scoring.workspace_schur import SchurWorkspace

    sw = SchurWorkspace(ws1)
    sc = FluxScorer(
        ws1,
        unit_dim=6,
        env_dim=20,
        backend="schur",
        schur=sw,
        cond_unit="reduced",
        cond_env=None,
    )
    u = oneway["truth"]["units"][1]
    assert np.allclose(
        sw.score(u, unit_dim=6)["rho"], sc(u)["rho_in"], atol=1e-10
    )


# 3 ---------------------------------------------------------------------------
def test_driver_is_open_outward_and_listener_inward(oneway, ws1):
    sc = FluxScorer(ws1, unit_dim=8, env_dim=20)
    drv, lis, iso = (sc(u) for u in oneway["truth"]["units"])
    # v1.0 (incoming only) calls the driver closed ...
    assert drv["nu_in"] < 0.3 and drv["k_star_in"] == 0
    # ... COG 2.1 sees its one outgoing gear
    assert 0.6 < drv["nu_out"] < 1.6 and drv["k_star_out"] == 1
    assert 0.6 < lis["nu_in"] < 1.6 and lis["nu_out"] < 0.3
    assert iso["nu"] < 0.3


# 4 ---------------------------------------------------------------------------
def test_soft_count_limits():
    assert soft_count([0.1, 0.05], 0.1) == 0.0
    assert np.isclose(soft_count([0.9, 0.8, 0.05], 0.05), 2.0, atol=0.02)
    # one channel at rho^2 = 2 theta^2 counts one half
    assert np.isclose(soft_count([np.sqrt(2) * 0.1], 0.1), 0.5)
    assert soft_count([0.3, 0.2], 0.1, eta=0.0) == 2.0


# 5 ---------------------------------------------------------------------------
def test_calibration_fingerprint_and_baseline(oneway, ws1):
    adj = oneway["adjacency"]
    cal = calibrate_cog2(
        ws1,
        adj,
        [4, 8],
        unit_dim=6,
        env_dim=16,
        min_supports=10,
        max_supports=20,
        seed=0,
    )
    assert set(cal["theta_in"]) == {4, 8} and set(cal["theta_out"]) == {4, 8}
    for k in (4, 8):
        assert 0 < cal["theta_in"][k] < 0.5
        b = cal["baseline"][k]
        assert b["q25"] <= b["median"] <= b["q75"]
        assert len(b["values"]) == cal["diagnostics"][k]["n_supports"]
    assert cal["config"]["fingerprint"]["shift_mode"] == "cross"
    sc = FluxScorer(ws1, unit_dim=6, env_dim=16, calibration=cal)
    r = sc(oneway["truth"]["units"][2])
    assert np.isclose(r["cost"], r["nu"] - r["baseline_nu"])
    assert 0 < r["p_size"] <= 1
    # different data -> refused
    other = FluxWorkspace(
        oneway["feats"][:5000].astype(np.float64),
        lag=1,
        n_null=8,
        seed=1,
        dtype=np.float64,
    )
    with pytest.raises(ValueError):
        FluxScorer(other, unit_dim=6, env_dim=16, calibration=cal)
    with pytest.raises(ValueError):  # different env_dim
        FluxScorer(ws1, unit_dim=6, env_dim=12, calibration=cal)
    with pytest.raises(ValueError):  # different combine rule
        FluxScorer(ws1, unit_dim=6, env_dim=16, calibration=cal, combine="sum")


# 7 ---------------------------------------------------------------------------
def test_pair_workspace_excludes_boundary_pairs(oneway):
    feats = oneway["feats"][:5000].astype(np.float64)
    N = feats.shape[1]
    pairs = np.array([[0, 1], [7, 8], [8, 9], [20, 21]])
    pf = np.random.default_rng(0).normal(size=(5000, 4, 1))
    pws = PairFluxWorkspace(
        pf,
        pairs,
        N,
        residue_feats=feats,
        lag=1,
        n_null=4,
        seed=1,
        dtype=np.float64,
    )
    m = feats.shape[2]
    sup, S, E = pws._cols(list(range(8)))  # residues 0..7
    base = N * m
    assert base + 0 in S  # (0,1) inside
    assert base + 1 not in S and base + 1 not in E  # (7,8) straddles
    assert base + 2 in E and base + 3 in E  # (8,9), (20,21) outside
    assert len(S) == 8 * m + 1


def test_pair_workspace_without_pairs_reproduces_residue_workspace(oneway):
    feats = oneway["feats"][:5000].astype(np.float64)
    N = feats.shape[1]
    ws = FluxWorkspace(feats, lag=1, n_null=4, seed=1, dtype=np.float64)
    pws = PairFluxWorkspace(
        np.zeros((5000, 0, 1)),
        np.zeros((0, 2)),
        N,
        residue_feats=feats,
        lag=1,
        n_null=4,
        seed=1,
        dtype=np.float64,
    )
    u = oneway["truth"]["units"][1]
    a = FluxScorer(ws, unit_dim=6, env_dim=16)(u)
    b = FluxScorer(pws, unit_dim=6, env_dim=16)(u)
    assert a["nu_in"] == pytest.approx(b["nu_in"], abs=1e-8)
    assert np.allclose(a["rho_in"], b["rho_in"], atol=1e-10)
    assert np.allclose(a["rho_out"], b["rho_out"], atol=1e-10)
    with pytest.raises(ValueError):
        FluxScorer(pws, unit_dim=6, env_dim=16, backend="schur")


# 9 ---------------------------------------------------------------------------
def test_channel_count_matches_planted_gears_without_faces():
    """No face residues: the soft count should sit near the planted gear
    count (mean absolute error well under one channel), and full
    conditioning must not count more than reduced conditioning.
    """
    b = planted_system(
        n_units=6,
        unit_size=10,
        m=4,
        latent_dim=5,
        n_frames=8000,
        n_replicas=2,
        face_residues=0,
        decoy_pairs=[(3, 4)],
        seed=0,
    )
    cpl, T = b["truth"]["couplings"], b["truth"]["units"]
    ws = FluxWorkspace(
        b["feats"],
        lag=1,
        segments=b["segments"],
        n_null=16,
        seed=1,
        shift_mode="cross",
    )
    err_full, tot_full, tot_red = [], 0.0, 0.0
    full = FluxScorer(ws, unit_dim=6, env_dim=16)
    red = FluxScorer(
        ws, unit_dim=6, env_dim=16, cond_unit="reduced", cond_env=None
    )
    for a, u in enumerate(T):
        ti = sum(k for s, d, k in cpl if d == a)
        to = sum(k for s, d, k in cpl if s == a)
        r = full(u)
        err_full += [r["nu_in"] - ti, r["nu_out"] - to]
        tot_full += r["nu_in"] + r["nu_out"]
        q = red(u)
        tot_red += q["nu_in"] + q["nu_out"]
    assert np.mean(np.abs(err_full)) < 0.5, err_full
    assert tot_full <= tot_red + 0.3


# 10 --------------------------------------------------------------------------
def test_batched_torch_null_matches_numpy_loop(oneway, ws1):
    """The GPU path (here run with torch on the CPU) gives the same null
    correlations as the per-shift numpy loop.
    """
    pytest.importorskip("torch")
    sc = FluxScorer(ws1, unit_dim=6, env_dim=16)
    u = oneway["truth"]["units"][1]
    try:
        ws1.null_device = "numpy"
        a = sc.parts(u, True)
        ws1.null_device = "cpu"
        b = sc.parts(u, True)
    finally:
        ws1.null_device = None
    assert np.allclose(a["null_in"], b["null_in"], atol=1e-10)
    assert np.allclose(a["null_out"], b["null_out"], atol=1e-10)


# 11 --------------------------------------------------------------------------
def test_replica_offsets_inflate_null_and_centring_removes_them():
    """Replicas sitting in different states (a per-replica offset on every
    feature) blow the shift-null threshold up and hide real channels;
    centring each replica restores both.
    """
    from pdukit.cog.features import center_replicas

    b = planted_system(
        n_units=3,
        unit_size=8,
        m=4,
        couplings=[(0, 1, 1)],
        n_frames=5000,
        n_replicas=3,
        seed=0,
    )
    F = b["feats"].astype(np.float64)
    sd = F.std(axis=0)
    rng = np.random.default_rng(1)
    for a, e in b["segments"]:
        F[a:e] += 2.0 * sd * rng.normal(size=sd.shape)

    def theta_and_listener(X):
        ws = FluxWorkspace(
            X,
            lag=1,
            segments=b["segments"],
            n_null=16,
            seed=1,
            dtype=np.float64,
            shift_mode="cross",
        )
        r = FluxScorer(ws, unit_dim=6, env_dim=16)(b["truth"]["units"][1])
        return r["theta_in"], r["nu_in"]

    th_off, nu_off = theta_and_listener(F.copy())
    th_c, nu_c = theta_and_listener(center_replicas(F.copy(), b["segments"]))
    assert th_off > 3 * th_c, (th_off, th_c)
    assert nu_off < 0.5 and 0.6 < nu_c < 1.6, (nu_off, nu_c)


# 12 --------------------------------------------------------------------------
def test_within_replica_drift_and_highpass():
    """A slow drift inside each replica (random walk) inflates the null even
    after centring; removing a running mean ~10x the dynamics' memory
    restores the threshold and the planted channel, and does not disturb a
    drift-free system.
    """
    from pdukit.cog.features import center_replicas, highpass_replicas

    b = planted_system(
        n_units=3,
        unit_size=8,
        m=4,
        couplings=[(0, 1, 1)],
        n_frames=5000,
        n_replicas=3,
        seed=0,
    )
    F = b["feats"].astype(np.float64)
    sd = F.std(axis=0)
    rng = np.random.default_rng(2)
    G = F.copy()
    for a, e in b["segments"]:
        rw = np.cumsum(rng.normal(size=(e - a,) + sd.shape), axis=0)
        G[a:e] += 1.5 * sd * rw / rw.std(axis=0)

    def run(X):
        ws = FluxWorkspace(
            X,
            lag=1,
            segments=b["segments"],
            n_null=16,
            seed=1,
            dtype=np.float64,
            shift_mode="cross",
        )
        r = FluxScorer(ws, unit_dim=6, env_dim=16)(b["truth"]["units"][1])
        return r["theta_in"], r["nu_in"]

    th_d, _ = run(center_replicas(G.copy(), b["segments"]))
    th_h, nu_h = run(
        center_replicas(
            highpass_replicas(G.copy(), b["segments"], 100), b["segments"]
        )
    )
    th_0, nu_0 = run(
        center_replicas(
            highpass_replicas(F.copy(), b["segments"], 100), b["segments"]
        )
    )
    assert th_d > 4 * th_h, (th_d, th_h)
    assert 0.6 < nu_h < 1.6 and 0.6 < nu_0 < 1.6, (nu_h, nu_0)


# 13 --------------------------------------------------------------------------
def _toy_coords(F=1500, N=30, seed=0):
    rng = np.random.default_rng(seed)
    base = np.cumsum(rng.normal(size=(N, 3)), axis=0) * 2.2
    X = base[None] + 0.3 * rng.normal(size=(F, N, 3))
    # a slow drift of one region, as a replica still relaxing
    X[:, :10] += np.linspace(0, 2.0, F)[:, None, None] * np.array([1.0, 0, 0])
    return X.astype(np.float32)


def test_fit_removes_rigid_motion_and_rse_is_blind_to_it():
    from pdukit.cog.features import fit_coordinates
    from pdukit.numerics.features import rse_features

    X = _toy_coords()
    rng = np.random.default_rng(3)
    Y = X.copy()
    for f in range(len(Y)):  # random rotation + translation
        Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        Q *= np.sign(np.linalg.det(Q))
        Y[f] = Y[f] @ Q.T + rng.normal(size=3) * 20
    fa, _, _ = rse_features(X, cutoff=12.0, n_components=4, max_frames=100)
    fb, _, _ = rse_features(Y, cutoff=12.0, n_components=4, max_frames=100)
    # same features up to the sign of each slow direction
    for i in range(fa.shape[1]):
        for k in range(fa.shape[2]):
            c = np.corrcoef(fa[:, i, k], fb[:, i, k])[0, 1]
            assert abs(c) > 0.999 or np.std(fa[:, i, k]) < 1e-6
    Z = fit_coordinates(Y.copy()).astype(np.float64)
    W = fit_coordinates(X.copy()).astype(np.float64)
    # the two fitted trajectories agree up to ONE global rotation
    a, b = Z.mean(axis=0), W.mean(axis=0)
    U, _, Vt = np.linalg.svd(a.T @ b)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    assert np.abs(Z @ R - W).max() < 0.05


def test_highpass_before_features_removes_drift_from_features():
    from pdukit.cog.features import (
        center_replicas,
        coarse_sizes,
        highpass_replicas,
    )
    from pdukit.numerics.features import rse_features

    X = _toy_coords()
    segs = [(0, 750), (750, 1500)]

    def dfilt(Di, sg):
        highpass_replicas(Di[:, None, :], sg, 50)
        center_replicas(Di[:, None, :], sg)

    raw, _, _ = rse_features(
        X, cutoff=12.0, n_components=4, segments=segs, max_frames=100
    )
    flt, _, _ = rse_features(
        X,
        cutoff=12.0,
        n_components=4,
        segments=segs,
        max_frames=100,
        distance_filter=dfilt,
    )

    def slow_fraction(f):  # variance left in 100-frame block means
        b = f[:1500].reshape(15, 100, *f.shape[1:]).mean(axis=1)
        return float(b.var() / max(f.var(), 1e-12))

    assert slow_fraction(flt) < 0.5 * slow_fraction(raw)
    assert coarse_sizes(2, 40, 6)[0] == 2 and coarse_sizes(2, 40, 6)[-1] == 40
    assert len(coarse_sizes(2, 40, 6)) <= 7


# 14 --------------------------------------------------------------------------
def test_batched_engine_matches_numpy():
    """The GPU engine (run here with torch on the CPU) gives the numpy
    path's spectra and nulls to round-off."""
    pytest.importorskip("torch")
    b = planted_system(
        n_units=3,
        unit_size=8,
        m=4,
        couplings=[(0, 1, 1)],
        n_frames=3000,
        n_replicas=2,
        seed=0,
    )
    ws = FluxWorkspace(
        b["feats"],
        lag=1,
        segments=b["segments"],
        n_null=8,
        seed=1,
        shift_mode="cross",
    )
    ws.null_device = "numpy"
    ref = FluxScorer(ws, unit_dim=6, env_dim=12, batched=False)
    eng = FluxScorer(
        ws, unit_dim=6, env_dim=12, batched=True, engine_device="cpu"
    )
    sups = b["truth"]["units"] + [list(range(4, 12))]
    P = [ref.parts(s, True) for s in sups]
    Q = eng.parts_batch(sups, True)
    for p, q in zip(P, Q):
        for k in ("rho_in", "rho_out", "null_in", "null_out"):
            assert np.allclose(p[k], q[k], atol=1e-8), k


def test_auto_window_is_smallest_near_lowest_threshold():
    from pdukit.cog.scoring.factory import choose_window

    # 13-replica T4L GA scan (job 7304410)
    rows = [
        dict(window_frames=w, theta_in=i, theta_out=o)
        for w, i, o in [
            (10, 0.0110, 0.0118),
            (36, 0.0099, 0.0110),
            (130, 0.0141, 0.0126),
            (471, 0.0187, 0.0119),
            (1701, 0.0200, 0.0147),
            (6142, 0.0217, 0.0146),
        ]
    ]
    w, best = choose_window(rows, 0.15)
    assert w == 10 and abs(best - 0.0110) < 1e-12
    # a tight tolerance picks the true minimum
    assert choose_window(rows, 0.0)[0] == 36
    # a flat curve -> the smallest window
    flat = [
        dict(window_frames=w, theta_in=0.02, theta_out=0.02)
        for w in (50, 10, 200)
    ]
    assert choose_window(flat)[0] == 10


def test_predictive_graph_finds_driver_not_contacts():
    """Residue 2 drives residue 0 (no contact needed): rho[0, 2] is large,
    the reverse and unrelated pairs sit at the null.
    """
    from pdukit.cog.graph import pair_rho, predictive_graph

    rng = np.random.default_rng(0)
    N, m, T = 6, 2, 40000
    X = np.zeros((T, N * m))
    for t in range(1, T):
        X[t] = 0.8 * X[t - 1] + rng.normal(size=N * m)
        X[t, 0:2] += 0.6 * X[t - 1, 4:6]
    X -= X.mean(0)
    a, b = X[:-1], X[1:]
    n = len(a)

    class W:
        pass

    ws = W()
    ws.N, ws.m = N, m
    ws.C00, ws.C01, ws.C11 = a.T @ a / n, a.T @ b / n, b.T @ b / n
    ws.Cs00, ws.Cs01 = [], []
    for _ in range(8):
        s = np.roll(a, int(rng.integers(1000, n - 1000)), 0)
        ws.Cs00.append(s.T @ a / n)
        ws.Cs01.append(s.T @ b / n)
    r = pair_rho(ws.C00, ws.C01, ws.C11, N, m)
    assert r[0, 2] > 0.5 and r[2, 0] < 0.05
    g = predictive_graph(ws, k=2, log=lambda *_: None)
    assert g["adjacency"][0, 2] and g["adjacency"][2, 0]
    assert r[0, 2] > g["theta"] > r[2, 0]


def test_coverage_curve_and_modes_for_coverage():
    from pdukit.numerics.features import coverage_curve, modes_for_coverage

    c = coverage_curve([0.9, 0.3, 0.1, -0.05])  # negatives count as 0
    np.testing.assert_allclose(c[-1], 1.0)
    assert c[0] == pytest.approx(0.81 / 0.91)
    k, per = modes_for_coverage(
        [[0.9, 0.3, 0.1], [0.5, 0.5, 0.5], [1.0, 0.0]], 0.9
    )
    assert per == [2, 3, 1] and k == 2  # median residue, rounded up
    assert coverage_curve([1.0, 1.0], power=1).tolist() == [0.5, 1.0]


def test_rse_auto_modes_equal_fixed_with_the_chosen_number():
    from pdukit.numerics.features import rse_features

    X = _toy_coords()
    segs = [(0, 750), (750, 1500)]
    rep, fix = {}, {}
    auto, _, ea = rse_features(
        X, cutoff=12.0, n_components=-1, coverage=0.99, lag=5,
        segments=segs, max_frames=100, report=rep,
    )
    k = rep["n_components"]
    assert rep["chosen"] and k == auto.shape[2] >= 2
    same, _, ef = rse_features(
        X, cutoff=12.0, n_components=k, coverage=0.99, lag=5,
        segments=segs, max_frames=100, report=fix,
    )
    np.testing.assert_array_equal(auto, same)
    np.testing.assert_allclose(ea, ef)
    assert not fix["chosen"]
    assert fix["coverage_kept"] == rep["coverage_kept"]
    # the median residue reaches the target with k modes
    assert np.median(rep["coverage_kept"]) >= 0.99 - 1e-9
    assert len(rep["modes_for_target"]) == X.shape[1]


def test_rse_settings_are_checked():
    from pdukit.config.config import Config
    from pdukit.errors import ConfigError

    for bad in [("rse-components", "0"), ("rse-coverage", "1.5")]:
        with pytest.raises(ConfigError):
            Config.build(sets=[("outdir", "o"), ("planted", "1"), bad])
    cfg = Config.build(sets=[("outdir", "o"), ("planted", "1")])
    assert cfg.rse_components == -1 and cfg.rse_coverage == 0.9


def test_tica_summary_reports_coverage():
    from pdukit.cog.features import tica_spectrum_summary

    report = {
        "n_components": 2, "chosen": True, "coverage_target": 0.9,
        "coverage_measure": "sum lambda^2",
        "coverage_kept": [0.95, 0.85, 0.92], "modes_for_target": [2, 3, 2],
    }
    ts = tica_spectrum_summary(
        [[0.9, 0.3, 0.1]] * 3, lag=5, kept=2, report=report
    )
    cov = ts["coverage"]
    assert ts["kept"] == 2 and cov["kept_median"] == 0.92
    assert cov["fraction_residues_reaching_target"] == pytest.approx(0.667)
    assert "coverage" not in tica_spectrum_summary([[0.9, 0.1]], lag=5)


def test_seed_groups_add_strongest_partners():
    from pdukit.cog.units.seeds import seed_groups

    rho = np.zeros((6, 6))
    rho[0, 3] = 0.9  # 3 drives 0
    rho[3, 4] = 0.5  # 4 drives 3
    rho[1, 2] = 0.2
    adj = np.ones((6, 6), bool) & ~np.eye(6, dtype=bool)
    assert seed_groups([0], rho, adj, 1) == [[0]]
    assert seed_groups([0], rho, adj, 2) == [[0, 3]]
    assert seed_groups([0], rho, adj, 3) == [[0, 3, 4]]
    # identical starts are kept once; neighbours only when there are any
    assert seed_groups([0, 3], rho, adj, 2) == [[0, 3]]
    chain = np.zeros((6, 6), bool)
    chain[0, 1] = chain[1, 0] = True
    assert seed_groups([0], rho, chain, 2) == [[0, 1]]


def test_rse_max_components_caps_the_automatic_choice():
    from pdukit.numerics.features import rse_features

    X = _toy_coords(F=600, N=20)
    rep = {}
    f, _, _ = rse_features(X, cutoff=12.0, n_components=-1, coverage=0.999,
                           lag=2, max_frames=100, report=rep,
                           max_components=3)
    assert f.shape[2] == rep["n_components"] == 3
    assert rep["modes_for_target_median"] >= 3


def test_seed_fraction_and_new_growth_settings_are_checked():
    from pdukit.config.config import Config
    from pdukit.errors import ConfigError

    base = [("outdir", "o"), ("planted", "1")]
    for bad in [("seed-fraction", "1.5"), ("rse-max-components", "-1"),
                ("start-min-gain", "-0.1"), ("seed-group", "-1")]:
        with pytest.raises(ConfigError):
            Config.build(sets=base + [bad])
    cfg = Config.build(sets=base)
    assert (cfg.seed_fraction, cfg.rse_max_components, cfg.seed_group,
            cfg.start_min_gain) == (0.0, 0, 1, 0.0)
