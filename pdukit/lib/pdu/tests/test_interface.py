"""Planted-channel validation for the flux-dimension diagnostics.

These are constructed consistency checks, not biological validation, in the
same spirit as the v3.0 verification note.  Each asserts a property the
estimator must have if it is measuring what it claims to measure.

Construction
------------
Three latent blocks A, B, C evolve as a stable VAR(1).  A and B are joined by a
SINGLE rank-1 channel in each direction; C is fully decoupled.  Within-block
dynamics is dense.  Residues 0-9 read out block A, 10-19 block B, 20-29 block
C, each through a random mixing matrix plus private observation noise.

What must hold
--------------
1. a planted block reports exactly one channel;
2. a decoupled block reports none;
3. adding a decoupled block to a planted one changes nothing;
4. a support straddling the A/B boundary reports more than one channel;
5. SCALING THE COUPLING UP DOES NOT CHANGE THE CHANNEL COUNT.  This is the
   property that separates the criterion from a weak-coupling criterion and so
   from the iVAMPnets independence assumption;
6. matched random supports report clearly more channels than planted ones;
7. greedy growth never merges across a true block boundary.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from ..interface import (
    FluxWorkspace,
    _bootstrap_ci_rel_width,
    _bootstrap_ci_rel_width_by_support,
    calibrate_null_thresholds,
    contact_graph,
    control_supports,
    decompose,
    detect_length_unit,
    grow_support,
    residue_budget,
    score_support,
)

D = 5
F = 4000
N_PER = 10
M_FEAT = 6

A = list(range(0, 10))
B = list(range(10, 20))
C = list(range(20, 30))

KW = dict(lag=1, ridge=1e-2, unit_dim=8, env_dim=30, n_null=15, seed=1)


def _stable_block(d, rho, rng):
    M = rng.normal(size=(d, d))
    return M / np.abs(np.linalg.eigvals(M)).max() * rho


def simulate(channel_gain=1.0, seed=0):
    """Return (F, 30, M_FEAT) features with a planted rank-1 channel."""
    rng = np.random.default_rng(seed)
    Z = np.zeros((3 * D, 3 * D))
    Z[0:D, 0:D] = _stable_block(D, 0.85, rng)
    Z[D : 2 * D, D : 2 * D] = _stable_block(D, 0.85, rng)
    Z[2 * D :, 2 * D :] = _stable_block(D, 0.85, rng)

    for r0, c0 in ((0, D), (D, 0)):
        u, v = rng.normal(size=D), rng.normal(size=D)
        Z[r0 : r0 + D, c0 : c0 + D] = (
            channel_gain
            * 0.30
            * np.outer(u / np.linalg.norm(u), v / np.linalg.norm(v))
        )

    spec = np.abs(np.linalg.eigvals(Z)).max()
    if spec >= 0.99:
        Z *= 0.98 / spec

    x = np.zeros(3 * D)
    traj = np.empty((F, 3 * D))
    for t in range(F):
        x = Z @ x + rng.normal(scale=0.3, size=3 * D)
        traj[t] = x

    feats = np.empty((F, 3 * N_PER, M_FEAT))
    for b in range(3):
        lat = traj[:, b * D : (b + 1) * D]
        for r in range(N_PER):
            W = rng.normal(size=(D, M_FEAT))
            feats[:, b * N_PER + r, :] = lat @ W + rng.normal(
                scale=0.25, size=(F, M_FEAT)
            )
    return feats


@pytest.fixture(scope="module")
def feats():
    return simulate(channel_gain=1.0, seed=0)


def test_planted_block_reports_one_channel(feats):
    for block in (A, B):
        r = score_support(feats, block, **KW)
        assert r["k_star"] == 1, r["rho"][:3]
        assert 0.9 <= r["k_eff"] <= 1.3


def test_decoupled_block_reports_no_channel(feats):
    r = score_support(feats, C, **KW)
    assert r["k_star"] == 0
    assert not np.isfinite(r["k_eff"])  # undefined, by design


def test_adding_a_decoupled_block_changes_nothing(feats):
    r = score_support(feats, A + C, **KW)
    assert r["k_star"] == 1
    assert 0.9 <= r["k_eff"] <= 1.3


def test_straddling_support_reports_more_channels(feats):
    r = score_support(feats, A[:5] + B[:5], **KW)
    assert r["k_star"] > 1
    assert r["k_eff"] > 1.5


def test_channel_count_is_independent_of_coupling_strength():
    """The load-bearing test: strength must not decide the channel count."""
    seen = []
    for gain in (0.5, 1.0, 2.0, 3.0):
        r = score_support(simulate(channel_gain=gain, seed=0), A, **KW)
        seen.append((gain, r["k_eff"], r["coupling_total"]))
    k_effs = [k for _, k, _ in seen]
    couplings = [c for _, _, c in seen]
    assert max(couplings) / min(couplings) > 3.0, (
        seen
    )  # strength really varied
    assert max(k_effs) - min(k_effs) < 0.3, seen  # channel count did not


def test_random_controls_report_more_channels(feats):
    ctrl = control_supports(
        feats, size=10, n_random=6, n_segment=0, n_ball=0, seed=3
    )
    vals = [score_support(feats, S, **KW)["k_eff"] for S in ctrl["random"]]
    vals = [v for v in vals if np.isfinite(v)]
    planted = score_support(feats, A, **KW)["k_eff"]
    assert np.median(vals) > planted + 1.0, (np.median(vals), planted)


def test_growth_does_not_cross_a_block_boundary(feats):
    N = feats.shape[1]
    adj = np.zeros((N, N), bool)
    for blk in (A, B, C):
        for i in blk:
            for j in blk:
                if i != j:
                    adj[i, j] = True
    for u, v in ((9, 10), (8, 11), (19, 20), (18, 21)):
        adj[u, v] = adj[v, u] = True

    kw = {k: v for k, v in KW.items() if k != "lag"}
    for seed_res, truth in ((2, set(A)), (14, set(B)), (25, set(C))):
        r = grow_support(feats, seed_res, adj, max_size=14, lag=1, **kw)
        support = set(r["support"])
        assert support <= truth, (seed_res, sorted(support))


# ---------------------------------------------------------------------------
# residue budget and coordinate units
# ---------------------------------------------------------------------------


def _block_adjacency(n):
    adj = np.zeros((n, n), bool)
    for blk in (A, B, C):
        for i in blk:
            for j in blk:
                if i != j:
                    adj[i, j] = True
    for u, v in ((9, 10), (8, 11), (19, 20), (18, 21)):
        adj[u, v] = adj[v, u] = True
    return adj


def test_residue_budget_separates_coupled_from_decoupled(feats):
    """Coupled blocks show an outside share; the decoupled one does not."""
    adj = _block_adjacency(feats.shape[1])
    b = residue_budget(
        feats, adjacency=adj, lag=1, ridge=1e-2, env_dim=30, n_null=8, seed=1
    )
    out = np.array(b["outside"])
    res = np.array(b["resolved"])
    assert out[A + B].mean() > 10 * max(out[C].mean(), 1e-6)
    assert res[A + B].mean() > 0.8  # coupled residues resolve
    assert res[C].mean() < 0.3  # decoupled ones do not


def test_residue_budget_is_a_partition(feats):
    adj = _block_adjacency(feats.shape[1])
    b = residue_budget(
        feats, adjacency=adj, lag=1, ridge=1e-2, env_dim=30, n_null=6, seed=1
    )
    total = (
        np.array(b["inside"])
        + np.array(b["outside"])
        + np.array(b["irreducible"])
    )
    assert np.allclose(total, 1.0, atol=1e-9)


def test_length_unit_detection_and_cutoff_invariance():
    """An 8 A cutoff must give the same graph whether coords are A or nm.

    mdCATH stores nanometres while hdf5_layout documents Angstrom, and the
    symptom of getting this wrong is a contact graph where every residue
    contacts every other.
    """
    rng = np.random.default_rng(0)
    # realistic CA fluctuation (~0.4 A rms per coordinate); the reference is a
    # median of per-frame distances, so a physically absurd noise level would
    # inflate the apparent spacing by Jensen's inequality rather than test
    # anything about unit detection.
    ang = rng.normal(size=(50, 30, 3)) * 0.4
    ang[:, :, 0] += np.arange(30) * 3.8
    nm = ang / 10.0
    a_info, n_info = detect_length_unit(ang), detect_length_unit(nm)
    assert a_info["unit"] == "angstrom", a_info
    assert n_info["unit"] == "nanometre", n_info
    assert not a_info["suspect"] and not n_info["suspect"]
    assert abs(a_info["spacing_angstrom"] - 3.8) < 0.4
    a_adj = contact_graph(ang, cutoff=8.0)
    n_adj = contact_graph(nm, cutoff=8.0)
    assert np.array_equal(a_adj, n_adj)
    assert a_adj.sum(axis=1).mean() < 15  # not a fully dense graph


def test_workspace_matches_direct_path(feats):
    """The covariance workspace is the same estimator, not an approximation.

    It exists only for speed -- at 114k frames the direct path PCAs the
    environment inside every call and takes about a minute per candidate.
    k_eff is the primary statistic and must agree; k_star can differ by one
    because the two draw independent shift realisations for the null.
    """
    ws = FluxWorkspace(
        feats,
        lag=1,
        n_null=15,
        seed=1,
        ridge=1e-2,
        dtype=np.float64,
        truncation="pca",
    )
    for S in (A, B, A + C):
        d = score_support(feats, S, **KW)
        w = ws.score(S, unit_dim=8, env_dim=30)
        assert abs(d["rho"][0] - w["rho"][0]) < 0.02, (
            d["rho"][0],
            w["rho"][0],
        )
        assert abs(d["k_eff"] - w["k_eff"]) < 0.25, (d["k_eff"], w["k_eff"])
    d = score_support(feats, C, **KW)
    w = ws.score(C, unit_dim=8, env_dim=30)
    assert d["k_star"] == 0 and w["k_star"] == 0
    assert not np.isfinite(d["k_eff"]) and not np.isfinite(w["k_eff"])


def test_koopman_truncation_recovers_the_planted_structure(feats):
    """The estimator truncates by SLOWNESS, not variance -- it must still find
    the planted channels, and must not blur the coupled/decoupled contrast.
    """
    ws = FluxWorkspace(
        feats,
        lag=1,
        n_null=15,
        seed=1,
        ridge=1e-2,
        dtype=np.float64,
        truncation="koopman",
    )
    a = ws.score(A, unit_dim=8, env_dim=30)
    c = ws.score(C, unit_dim=8, env_dim=30)
    straddle = ws.score(A[:5] + B[:5], unit_dim=8, env_dim=30)
    assert a["k_star"] >= 1
    assert a["k_eff"] < 2.0, a["k_eff"]
    assert c["k_star"] == 0, c["rho"][:3]  # decoupled stays decoupled
    assert straddle["k_eff"] > a["k_eff"], (straddle["k_eff"], a["k_eff"])


def test_koopman_and_pca_truncation_can_disagree(feats):
    """Documented, not asserted equal: they are different reductions.

    Kept as a regression guard -- if these ever become identical, the Koopman
    branch has silently fallen back to the equal-time covariance.
    """
    wk = FluxWorkspace(
        feats, lag=1, n_null=8, seed=1, dtype=np.float64, truncation="koopman"
    )
    wp = FluxWorkspace(
        feats, lag=1, n_null=8, seed=1, dtype=np.float64, truncation="pca"
    )
    bk = wk._koopman_basis(np.arange(60), 8)
    bp = wp._basis(wp.C00[np.ix_(np.arange(60), np.arange(60))], 8)
    # principal angles between the two subspaces: identical would mean the
    # lagged covariance was never used
    from numpy.linalg import qr, svd

    qk, _ = qr(bk)
    qp, _ = qr(bp)
    sv = svd(qk.T @ qp, compute_uv=False)
    assert np.min(sv) < 0.999, "koopman basis collapsed onto the PCA basis"


def test_workspace_refuses_whole_protein(feats):
    """A candidate with no environment has no flux to measure; say so."""
    ws = FluxWorkspace(feats, lag=1, n_null=4, seed=1)
    with pytest.raises(ValueError):
        ws.score(list(range(feats.shape[1])))


def test_koopman_reduction_keeps_slow_not_loud():
    """The reason every truncation had to change from PCA to Koopman.

    Build one SLOW low-amplitude mode and one FAST high-amplitude mode.  Keep a
    single component.  PCA keeps the loud one and discards the dynamics; the
    Koopman (tICA) reduction keeps the slow one.  On the real run the PCA
    compression retained 93% of the variance, which said nothing at all about
    whether the slow modes survived.
    """
    from ..interface import _inv_sqrt

    rng = np.random.default_rng(0)
    F, d, lag = 20000, 20, 20

    def ar1(rho, n):
        z = np.zeros(n)
        for t in range(1, n):
            z[t] = rho * z[t - 1] + rng.normal(scale=np.sqrt(1 - rho**2))
        return z

    slow = ar1(0.995, F) * 0.4
    fast = rng.normal(size=F) * 3.0
    p1 = rng.normal(size=d)
    p1 /= np.linalg.norm(p1)
    p2 = rng.normal(size=d)
    p2 /= np.linalg.norm(p2)
    D = (
        np.outer(slow, p1)
        + np.outer(fast, p2)
        + rng.normal(scale=0.2, size=(F, d))
    )
    D -= D.mean(0)

    A0, A1 = D[:-lag], D[lag:]
    n = len(A0)
    C = (D.T @ D) / F
    C0 = (A0.T @ A0) / n
    C1 = (A0.T @ A1) / n

    w, V = np.linalg.eigh(0.5 * (C + C.T))
    pca = D @ V[:, np.argsort(w)[::-1][:1]]

    L = _inv_sqrt(0.5 * (C0 + C0.T), 1e-2)
    M = 0.5 * (L @ (0.5 * (C1 + C1.T)) @ L)
    M = M + M.T
    w2, U = np.linalg.eigh(M)
    koop = D @ (L @ U[:, np.argsort(w2)[::-1][:1]])

    def corr(x, ref):
        x = (x[:, 0] - x[:, 0].mean()) / x[:, 0].std()
        r = (ref - ref.mean()) / ref.std()
        return abs(float(np.mean(x * r)))

    assert corr(koop, slow) > 0.7, corr(koop, slow)
    assert corr(koop, fast) < 0.2, corr(koop, fast)
    assert corr(pca, fast) > 0.7, corr(pca, fast)
    assert corr(pca, slow) < 0.2, corr(pca, slow)


def test_decomposition_recovers_the_planted_blocks(feats):
    """The whole protein must be covered, and the units must be the real ones.

    Seed-and-grow returns candidates; this asserts the alternating assignment
    of v3.0 sec 10 steps 5-6 returns a COVER: three units matching the three
    planted blocks, every residue assigned, no orphans.
    """
    adj = _block_adjacency(feats.shape[1])
    ws = FluxWorkspace(
        feats, lag=1, n_null=10, seed=1, dtype=np.float64, truncation="koopman"
    )
    r = decompose(
        feats,
        adj,
        ws,
        unit_dim=6,
        env_dim=20,
        n_seeds=6,
        max_size=12,
        max_iter=4,
        seed=0,
    )

    assert r["coverage"] == 1.0, r["orphans"]
    assert r["n_units"] == 3, [u for u in r["units"]]
    found = sorted(sorted(u) for u in r["units"])
    assert found == sorted([sorted(A), sorted(B), sorted(C)]), found
    # the decoupled block must still report no resolved channel
    c_unit = [s for s in r["scores"] if set(s["residues"]) == set(C)][0]
    assert not c_unit["resolved"], c_unit


def test_decomposition_assigns_every_residue_even_when_weak(feats):
    """Coverage is a guarantee: a residue with no positive affinity is still
    placed in its best unit and reported, never silently dropped.
    """
    adj = _block_adjacency(feats.shape[1])
    ws = FluxWorkspace(feats, lag=1, n_null=6, seed=1, dtype=np.float64)
    r = decompose(
        feats,
        adj,
        ws,
        unit_dim=6,
        env_dim=20,
        n_seeds=4,
        max_size=10,
        max_iter=2,
        seed=1,
    )
    assigned = {i for i, m in enumerate(r["membership"]) if m}
    assert assigned == set(range(feats.shape[1]))
    assert r["orphans"] == []


def test_membership_and_unit_lists_agree(feats):
    """Membership was indexed against the PRE-merge units, so after merging and
    dropping small units the two disagreed -- 165 assignments were silently
    inconsistent on the first lysozyme decomposition, which made the boundary
    residue list meaningless.  They must agree exactly.
    """
    adj = _block_adjacency(feats.shape[1])
    ws = FluxWorkspace(feats, lag=1, n_null=6, seed=1, dtype=np.float64)
    r = decompose(
        feats,
        adj,
        ws,
        unit_dim=6,
        env_dim=20,
        n_seeds=5,
        max_size=12,
        max_iter=3,
        n_controls=0,
        seed=0,
    )
    units, mem = r["units"], r["membership"]
    for i, m in enumerate(mem):
        for a in m:
            assert a < len(units), (i, a, len(units))
            assert i in units[a], f"residue {i} claims unit {a} but is absent"
    for a, u in enumerate(units):
        for i in u:
            assert a in mem[i], (
                f"unit {a} contains {i} but membership disagrees"
            )
    assert sum(len(m) for m in mem) == sum(len(u) for u in units)
    assert r["boundary_residues"] == [
        i for i, m in enumerate(mem) if len(m) > 1
    ]


def test_controls_are_disjoint_from_discovered_units(feats):
    """An unfiltered ball control is contaminated with true positives.

    A compact blob dropped at random will sometimes BE a unit, so requiring
    a discovered unit to beat it is testing a drug against a lower dose of
    itself.
    On the first lysozyme decomposition that one effect moved the verdict from
    +1.36 (11/11 units better than random) to +0.07 (6/11 better than the best
    of three).  Controls must be resampled away from the discovered units.
    """
    units = [A, B]
    ctrl = control_supports(
        feats,
        size=10,
        adjacency=_block_adjacency(feats.shape[1]),
        n_random=6,
        n_segment=6,
        n_ball=6,
        seed=5,
        exclude=units,
        max_overlap=0.3,
    )
    ctrl.pop("_rejected_for_overlap", None)
    ctrl.pop("_overlap_median", None)
    ctrl.pop("_overlap_max", None)
    for kind, sets in ctrl.items():
        for S in sets:
            for u in units:
                j = len(set(S) & set(u)) / max(len(set(S) | set(u)), 1)
                assert j <= 0.3, (kind, S, u, j)
    # and every bucket must be populated -- an empty bucket becomes a nan
    # control, which reads as a missing number rather than a broken comparison
    for kind, sets in ctrl.items():
        assert len(sets) > 0, kind


def test_decomposition_reports_both_gaps(feats):
    """Vs random and vs best control answer different questions; conflating
    them is what produced a spurious negative, so both are reported.
    """
    adj = _block_adjacency(feats.shape[1])
    ws = FluxWorkspace(feats, lag=1, n_null=6, seed=1, dtype=np.float64)
    r = decompose(
        feats,
        adj,
        ws,
        unit_dim=6,
        env_dim=20,
        n_seeds=4,
        max_size=12,
        max_iter=2,
        n_controls=4,
        seed=0,
    )
    assert "gap_vs_random" in r and "gap_vs_best_control" in r
    assert r["control_overlap_threshold"] == 0.3


# ---------------------------------------------------------------------------
# C6 redesign: size-conditioned null calibration
# ---------------------------------------------------------------------------

# Recorded from the pre-refactor code (the inline null-threshold computation
# that used to live directly inside `score()` / `FluxWorkspace.score()`) on
# this file's own `feats` fixture, with the same `KW` used everywhere else in
# this file.  This is the guardrail check from the implementation plan: with
# no new arguments supplied, the C6 refactor (splitting `score()` into
# `_project_and_residualise` / `_leading_null_rho`, promoting `_gram` to a
# method, etc.) must not move a single number.
_GOLDEN_SCORE_SUPPORT = {
    "A": {
        "rho_null_p95": 0.10792102871208278,
        "margin": 0.41695911731513013,
        "k_star": 1,
        "k_eff": 1.0,
        "coupling_total": 0.3126922873838445,
        "coupling_resolved": 0.26385221925527413,
    },
    "B": {
        "rho_null_p95": 0.10642779143413597,
        "margin": 0.34705690768304287,
        "k_star": 1,
        "k_eff": 1.0,
        "coupling_total": 0.23610743914689084,
        "coupling_resolved": 0.19432149754385025,
    },
    "C": {
        "rho_null_p95": 0.1102697801925892,
        "margin": -0.014447819159500955,
        "k_star": 0,
        "k_eff": None,
        "coupling_total": 0.03775972139193916,
        "coupling_resolved": 0.0,
    },
    "straddle": {
        "rho_null_p95": 0.1214527791635445,
        "margin": 0.2656781064668693,
        "k_star": 3,
        "k_eff": 2.062844704813243,
        "coupling_total": 0.34656307180545337,
        "coupling_resolved": 0.2671041376850331,
    },
}

_GOLDEN_WORKSPACE_SCORE = {
    "A": {
        "rho_null_p95": 0.1251381338936124,
        "margin": 0.251123399922457,
        "k_star": 1,
        "k_eff": 1.0,
        "coupling_total": 0.18977067292804847,
        "sigma": 0.15598042449558222,
        "nuc_spec": 1.0,
        "coupling_resolved": 0.12591318927524547,
    },
    "B": {
        "rho_null_p95": 0.12873369545184155,
        "margin": 0.2159578865684325,
        "k_star": 1,
        "k_eff": 1.0,
        "coupling_total": 0.16110043793514944,
        "sigma": 0.1385862022640774,
        "nuc_spec": 1.0,
        "coupling_resolved": 0.10223992237095182,
    },
    "C": {
        "rho_null_p95": 0.12726540490430324,
        "margin": -0.00945117918820211,
        "k_star": 0,
        "k_eff": None,
        "coupling_total": 0.05479058812508108,
        "sigma": 0.16598031303053384,
        "nuc_spec": None,
        "coupling_resolved": 0.0,
    },
    "straddle": {
        "rho_null_p95": 0.1310051945172836,
        "margin": 0.08965562469759736,
        "k_star": 4,
        "k_eff": 2.2600994567673562,
        "coupling_total": 0.14779756375571002,
        "sigma": 0.2596765018968282,
        "nuc_spec": 2.2592390951857775,
        "coupling_resolved": 0.05427746889061346,
    },
}


def _assert_matches_golden(result, golden):
    for key, expected in golden.items():
        actual = result[key]
        if expected is None:
            assert actual is None or not np.isfinite(actual), (key, actual)
        elif isinstance(expected, float):
            assert np.isclose(actual, expected, rtol=1e-9, atol=1e-12), (
                key,
                actual,
                expected,
            )
        else:
            assert actual == expected, (key, actual, expected)


def test_c6_refactor_matches_pre_refactor_golden_values(feats):
    """Guardrail regression test (plan sec. 5): with no new arguments, the
    refactor that split the null-threshold computation out of `score()` must
    reproduce the pre-refactor numbers exactly, not just approximately.
    """
    for name, S in (("A", A), ("B", B), ("C", C), ("straddle", A[:5] + B[:5])):
        r = score_support(feats, S, **KW)
        _assert_matches_golden(r, _GOLDEN_SCORE_SUPPORT[name])

    ws = FluxWorkspace(
        feats,
        lag=1,
        n_null=15,
        seed=1,
        ridge=1e-2,
        dtype=np.float64,
        truncation="koopman",
    )
    for name, S in (("A", A), ("B", B), ("C", C), ("straddle", A[:5] + B[:5])):
        r = ws.score(S, unit_dim=8, env_dim=30)
        _assert_matches_golden(r, _GOLDEN_WORKSPACE_SCORE[name])


def test_extend_null_shifts_matches_direct_construction(feats):
    """A workspace built with n_null=k and then extended by n_more must draw
    the same shifts as one built with n_null=k+n_more directly -- there is no
    special case for shifts added later.
    """
    ws_extended = FluxWorkspace(
        feats, lag=1, n_null=6, seed=3, dtype=np.float64
    )
    ws_extended.extend_null_shifts(4)
    ws_direct = FluxWorkspace(
        feats, lag=1, n_null=10, seed=3, dtype=np.float64
    )
    assert ws_extended.n_null == ws_direct.n_null == 10
    for a, b in zip(ws_extended.Cs00, ws_direct.Cs00):
        assert np.array_equal(a, b)
    for a, b in zip(ws_extended.Cs01, ws_direct.Cs01):
        assert np.array_equal(a, b)


def test_calibrate_null_thresholds_converges_and_feeds_score(feats):
    """calibrate_null_thresholds() must return a threshold for every
    requested size, converge under a generous target on test-scale data, and
    feed usable thresholds into `score()`/`decompose()` via `null_thresholds`
    / `null_calibration` -- landing in the same ballpark as the default
    per-candidate path, not necessarily identical (that comparison is
    follow-up validation work, not this test's job).
    """
    adj = _block_adjacency(feats.shape[1])
    ws = FluxWorkspace(
        feats, lag=1, n_null=8, seed=1, dtype=np.float64, truncation="koopman"
    )
    sizes = [len(A), len(A) + len(B[:2])]
    calib = calibrate_null_thresholds(
        ws,
        adj,
        sizes=sizes,
        unit_dim=8,
        env_dim=30,
        target_rel_width=0.3,
        batch_supports=5,
        min_supports=10,
        max_pool_size=200,
        seed=0,
    )

    assert set(calib["theta"]) == set(sizes)
    for k in sizes:
        diag = calib["diagnostics"][k]
        assert diag["converged"] is True
        assert diag["n_supports"] > 0
        assert len(diag["contact_density"]) == diag["n_supports"]
        assert len(diag["effective_dim"]) == diag["n_supports"]
        assert len(diag["per_support"]) == diag["n_supports"]
        assert diag["elapsed_seconds"] > 0.0
    assert calib["config"]["unit_dim"] == 8
    assert calib["config"]["env_dim"] == 30
    assert calib["config"]["n_shifts"] == len(ws.Cs00)

    r_default = ws.score(A, unit_dim=8, env_dim=30)
    r_calibrated = ws.score(
        A, unit_dim=8, env_dim=30, null_thresholds=calib["theta"]
    )
    assert r_calibrated["resolved"] == r_default["resolved"]
    assert abs(r_calibrated["k_eff"] - r_default["k_eff"]) < 1.5

    with pytest.raises(KeyError):
        ws.score(list(range(3)), null_thresholds=calib["theta"])

    d = decompose(
        feats,
        adj,
        ws,
        unit_dim=8,
        env_dim=30,
        n_seeds=4,
        max_size=12,
        max_iter=2,
        n_controls=0,
        seed=0,
        null_calibration=calib,
    )
    assert d["coverage"] == 1.0


def test_calibrate_null_thresholds_per_support_matches_default_null(feats):
    """Each calibration support's own theta_S (95th percentile of its row of
    shift-null correlations) must be exactly the threshold the default
    per-candidate null in `score()` gives that support at the SAME
    unit_dim/env_dim -- which is also what proves the dimensions passed to
    `calibrate_null_thresholds` actually reach the scoring (its old implicit
    default was env_dim=40, which would not match env_dim=30 here).
    """
    adj = _block_adjacency(feats.shape[1])
    ws = FluxWorkspace(
        feats, lag=1, n_null=8, seed=1, dtype=np.float64, truncation="koopman"
    )
    calib = calibrate_null_thresholds(
        ws,
        adj,
        sizes=[4, 7],
        unit_dim=8,
        env_dim=30,
        target_rel_width=0.5,
        batch_supports=3,
        min_supports=3,
        max_pool_size=48,
        seed=2,
    )
    for k in (4, 7):
        rows = calib["diagnostics"][k]["per_support"]
        assert rows
        for entry in rows:
            assert len(entry["support"]) == k
            assert len(entry["leading_rho"]) == len(ws.Cs00)
            r = ws.score(entry["support"], unit_dim=8, env_dim=30)
            assert np.isclose(
                entry["theta_S"], r["rho_null_p95"], rtol=1e-12, atol=1e-14
            )
            assert np.isclose(
                entry["theta_S"], np.percentile(entry["leading_rho"], 95)
            )
        pooled = [v for e in rows for v in e["leading_rho"]]
        assert np.isclose(calib["theta"][k], np.percentile(pooled, 95))

    other = ws.score(rows[0]["support"], unit_dim=8, env_dim=40)
    assert not np.isclose(rows[0]["theta_S"], other["rho_null_p95"])


def test_decompose_rejects_calibration_at_other_dimensions(feats):
    """A table calibrated at one unit_dim/env_dim must not be used to score
    at another: decompose() raises for the full dict, and logs a warning
    (it cannot check) for a bare theta table.
    """
    adj = _block_adjacency(feats.shape[1])
    ws = FluxWorkspace(
        feats, lag=1, n_null=8, seed=1, dtype=np.float64, truncation="koopman"
    )
    calib = calibrate_null_thresholds(
        ws,
        adj,
        sizes=[4],
        unit_dim=8,
        env_dim=30,
        target_rel_width=0.5,
        batch_supports=3,
        min_supports=3,
        max_pool_size=24,
        seed=0,
    )
    dkw = dict(n_seeds=2, max_size=6, max_iter=1, n_controls=0, seed=0)
    with pytest.raises(ValueError, match="env_dim"):
        decompose(
            feats,
            adj,
            ws,
            unit_dim=8,
            env_dim=20,
            null_calibration=calib,
            **dkw,
        )
    with pytest.raises(ValueError, match="unit_dim"):
        decompose(
            feats,
            adj,
            ws,
            unit_dim=6,
            env_dim=30,
            null_calibration=calib,
            **dkw,
        )

    logs = []
    decompose(
        feats,
        adj,
        ws,
        unit_dim=8,
        env_dim=20,
        null_calibration=calib["theta"],
        on_log=logs.append,
        **dkw,
    )
    assert any("bare theta table" in m for m in logs)


def test_support_bootstrap_counts_supports_not_values():
    """When each support carries its own null level, the n_shift values of a
    support are not independent: the support-level bootstrap must give a
    clearly wider interval than resampling values one by one.  When values
    are i.i.d. regardless of support, the two must roughly agree.
    """
    rng = np.random.default_rng(0)
    n_sup, J = 40, 20
    levels = rng.normal(0.35, 0.03, size=(n_sup, 1))
    clustered = levels + rng.normal(0.0, 0.01, size=(n_sup, J))
    iid = rng.normal(0.35, 0.03, size=(n_sup, J))

    p_s, w_s = _bootstrap_ci_rel_width_by_support(
        clustered, rng=np.random.default_rng(1)
    )
    p_v, w_v = _bootstrap_ci_rel_width(
        clustered.ravel(), rng=np.random.default_rng(1)
    )
    assert np.isclose(p_s, p_v)  # same point estimate
    assert w_s > 2.0 * w_v

    _, w_s_iid = _bootstrap_ci_rel_width_by_support(
        iid, rng=np.random.default_rng(1)
    )
    _, w_v_iid = _bootstrap_ci_rel_width(
        iid.ravel(), rng=np.random.default_rng(1)
    )
    assert 0.6 < w_s_iid / w_v_iid < 1.6


def test_decompose_memoise_changes_nothing_but_run_time(feats):
    """The per-call score cache must leave every output of `decompose()`
    unchanged: same units, scores, controls, affinities and trace with the
    cache on and off.  Only the returned `score_cache` counters differ.
    """
    adj = _block_adjacency(feats.shape[1])
    ws = FluxWorkspace(
        feats, lag=1, n_null=10, seed=1, dtype=np.float64, truncation="koopman"
    )
    kw = dict(
        unit_dim=6, env_dim=20, n_seeds=4, max_size=12, max_iter=3, seed=0
    )
    on = decompose(feats, adj, ws, memoise=True, **kw)
    off = decompose(feats, adj, ws, memoise=False, **kw)
    assert on["score_cache"]["hits"] > 0
    assert off["score_cache"]["hits"] == 0
    for key in on:
        if key == "score_cache":
            continue
        assert json.dumps(on[key], sort_keys=True, default=str) == json.dumps(
            off[key], sort_keys=True, default=str
        ), key
