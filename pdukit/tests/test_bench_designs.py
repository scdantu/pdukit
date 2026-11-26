"""Named synthetic designs and the latent-dynamics maths behind them."""

import numpy as np
import pytest

from pdukit.bench.designs import (
    PRESETS,
    DesignName,
    SyntheticSystem,
    make_design,
)
from pdukit.config.schema import SETTINGS
from pdukit.errors import ConfigError
from pdukit.numerics import latent
from pdukit.numerics.geometry import connect_components

# Fingerprints of the systems made by the original scripts
# (analysis/synthetic_suite_100/synth100.py and
# analysis/synthetic_suite_2026-11-21/synth.py, 21 Nov 2026), seed 0:
# frames per replica (None = design default), feats shape, feats[0, :3, 0],
# mean |feats|.  The package versions were bit-identical when moved.
REFERENCE = {
    "units": (50, [100, 100, 4], [0.600774, -0.589953, -2.398223], 0.878816),
    "nocomm": (50, [100, 100, 4], [-1.610795, 0.173681, -0.087671], 0.781627),
    "mixed": (50, [100, 100, 4], [0.970033, -0.193366, 0.452386], 0.914476),
    "medium": (50, [100, 100, 4], [2.16989, -0.001316, -2.085231], 1.197692),
    "tau": (50, [100, 100, 4], [-0.146548, -0.691092, 0.602487], 0.754603),
    "small-units": (
        50, [100, 36, 4], [-0.473133, 0.932232, -0.528345], 0.493602),
    "small-nocomm": (
        50, [100, 36, 4], [-0.04179, 0.497858, -0.50135], 0.45163),
    "small-mixed": (
        None, [6000, 36, 4], [-0.262955, -0.084731, -0.318012], 0.512259),
    "small-medium": (
        None, [6000, 36, 4], [-1.508233, -5.023035, -0.312365], 1.282892),
    "small-hierarchy": (
        50, [100, 30, 4], [-0.307615, -0.379679, -1.513762], 1.024206),
}


def _build(name, seed=0, frames=50):
    return make_design(name, n_frames=frames).build(seed)


@pytest.mark.parametrize("name", sorted(REFERENCE))
def test_design_reproduces_original_scripts(name):
    frames, shape, first, mean_abs = REFERENCE[name]
    system = make_design(name, n_frames=frames).build(0)
    f = system.feats.astype(float)
    assert list(system.feats.shape) == shape
    np.testing.assert_allclose(f[0, :3, 0], first, rtol=1e-5, atol=1e-6)
    assert abs(np.abs(f).mean() - mean_abs) < 1e-5


@pytest.mark.parametrize("name", [d.value for d in PRESETS])
def test_design_answer_is_consistent(name):
    s = _build(name)
    N = s.feats.shape[1]
    assert s.adjacency.shape == (N, N) and s.coords.shape == (N, 3)
    assert not s.adjacency.diagonal().any()
    assert (s.adjacency == s.adjacency.T).all()
    joined = connect_components(s.adjacency.copy(), s.coords)
    assert np.array_equal(joined, s.adjacency), "contact graph connected"
    assert s.truth["contacts"], "contact pairs recorded for scoring"
    assert s.config["design"] == name
    units = s.truth["units"]
    if "mixed" in name or "medium" in name:
        assert units == []
    else:
        members = sorted(i for u in units for i in u)
        assert members == list(range(N)), "units partition the residues"


def test_same_seed_same_system_other_seed_differs():
    a, b, c = _build("units", 3), _build("units", 3), _build("units", 4)
    assert np.array_equal(a.feats, b.feats)
    assert not np.array_equal(a.feats, c.feats)


def test_tau_design_answer_has_parents_and_lifetimes():
    t = _build("tau").truth
    assert len(t["fine_units"]) == 9 and len(t["coarse_units"]) == 7
    assert t["named"]["P1"] == t["named"]["P1a"] + t["named"]["P1b"]
    assert t["lifetimes_ns"]["FAST"] == [0.1, 0.15, 0.2]


def test_nocomm_and_units_share_internal_dynamics_but_not_couplings():
    u, n = _build("units"), _build("nocomm")
    assert u.truth["couplings"] and n.truth["couplings"] == []
    assert u.truth["units"] == n.truth["units"]


def test_make_design_overrides_and_errors():
    h = make_design("small-hierarchy", rho_fast=0.6, n_frames=None)
    assert h.rho_fast == 0.6 and h.n_frames == 6000
    u = make_design("units", rho_fast=0.6)  # not a field: ignored
    assert u == PRESETS[DesignName.UNITS]
    with pytest.raises(ConfigError, match="unknown design 'foo'"):
        make_design("foo")
    with pytest.raises(ConfigError, match="planted_system"):
        make_design("planted")


def test_schema_choices_match_design_names():
    row = next(r for r in SETTINGS if r[1] == "bench-design")
    assert row[5] == [d.value for d in DesignName]


def test_system_as_dict_has_planted_layout():
    d = _build("small-units").as_dict()
    assert set(d) == {
        "feats", "segments", "coords", "adjacency", "truth", "config"
    }
    assert d["feats"].dtype == np.float32
    assert isinstance(_build("small-units"), SyntheticSystem)


def test_normal_block_has_unit_stationary_variance():
    rng = np.random.default_rng(0)
    K, L = latent.normal_block([0.9, 0.5, 0.1], rng)
    # Lyapunov: Sigma = K Sigma K' + L L' holds with Sigma = I
    np.testing.assert_allclose(K @ K.T + L @ L.T, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(
        np.sort(np.linalg.eigvalsh(K)), [0.1, 0.5, 0.9], atol=1e-12
    )


def test_lifetime_to_rho():
    assert latent.lifetime_to_rho(1.0, 0.1) == pytest.approx(np.exp(-0.1))


def test_stabilise_only_when_needed():
    K = np.diag([0.5, 0.99995])
    same, flag = latent.stabilise(K, 0.999, 0.998)
    assert flag and latent.spectral_radius(same) == pytest.approx(0.998)
    K2, flag2 = latent.stabilise(np.diag([0.5, 0.9]), 0.999, 0.998)
    assert not flag2 and np.array_equal(K2, np.diag([0.5, 0.9]))


def test_vector_and_matrix_noise_give_the_same_trajectory():
    K = np.diag([0.8, 0.3])
    a = latent.simulate_var(
        K, np.ones(2), 20, 1, np.random.default_rng(1), 5
    )
    b = latent.simulate_var(
        K, np.eye(2), 20, 1, np.random.default_rng(1), 5
    )
    np.testing.assert_array_equal(a[0], b[0])


def test_add_coupling_is_directed():
    K = np.zeros((4, 4))
    off = np.array([0, 2, 4])
    U = V = np.array([[1.0], [0.0]])
    latent.add_coupling(K, off, source=0, target=1, U=U, V=V, gain=0.5)
    assert K[2, 0] == 0.5 and K[0, 2] == 0


def test_pipeline_loads_a_named_design():
    from pdukit.cog.features import load_data
    from pdukit.config.config import Config

    cfg = Config.build(sets=[
        ("outdir", "out"), ("planted", "1"), ("bench-design", "nocomm"),
        ("bench-seed", "2"), ("design-frames", "40"),
        ("design-replicas", "1"),
    ])
    data = load_data(cfg, {})
    assert data["feats"].shape == (40, 100, 4)
    assert data["meta"]["planted"]["design"] == "nocomm"
    assert data["meta"]["planted"]["seed"] == 2
    assert len(data["truth"]["units"]) == 9


def test_scatter_moves_every_answer_key():
    from pdukit.cog.features import scatter_planted

    b = _build("tau").as_dict()
    s = scatter_planted(b, seed=1)
    p = np.random.default_rng(1).permutation(100)
    for key in ("units", "fine_units", "coarse_units"):
        moved = sorted(int(p[i]) for i in b["truth"][key][0])
        assert s["truth"][key][0] == moved
    assert s["truth"]["named"]["P1"] == sorted(
        int(p[i]) for i in b["truth"]["named"]["P1"])
