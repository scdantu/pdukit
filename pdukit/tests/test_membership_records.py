"""Records for the membership analysis: affinity, blockiness, feature pass."""

from types import SimpleNamespace

import numpy as np
import pytest

from pdukit.cog.stages.prepare import direct_window
from pdukit.cog.units.affinity import residue_affinity
from pdukit.numerics.cohesion import blockiness

N = 24
UNITS = [list(range(0, 8)), list(range(8, 16)), list(range(16, 24))]


def _rho(strength, seed=0):
    rng = np.random.default_rng(seed)
    R = rng.uniform(0.0, 0.1, size=(N, N))
    for u in UNITS:
        R[np.ix_(u, u)] += strength
    np.fill_diagonal(R, 0.0)
    return R


def test_blockiness_separates_blocks_from_a_flat_field():
    blocks = blockiness(_rho(0.5), UNITS, n_random=100)
    flat = blockiness(_rho(0.0), UNITS, n_random=100)
    assert blocks["auc"] > 0.95 and blocks["percentile"] == 100
    assert blocks["ratio"] > 3
    assert abs(flat["auc"] - 0.5) < 0.15 and flat["z"] < 3
    assert flat["n_random"] == 100 and blocks["n_units"] == 3


def test_affinity_without_scorer_points_to_the_right_unit():
    rows, summary = residue_affinity(_rho(0.5), [UNITS[0], UNITS[1]])
    assert len(rows) == N * 2
    r = summary[20]  # in neither core: no strong preference
    assert r["member_of"] == []
    for res in (3, 12):
        s = summary[res]
        own = 0 if res < 8 else 1
        assert s["member_of"] == [own]
        assert s["cohesion_best_unit"] == own
    assert "nu_best_unit" not in summary[3]  # no scorer -> cohesion only


class _FakeScorer:
    """nu_in = number of members outside the first planted unit."""

    def __call__(self, support):
        out = sum(1 for i in support if i >= 8)
        return {"nu_in": float(out), "nu_out": 0.0, "theta_in": 0.1,
                "theta_out": 0.1, "rho_in": [0.2], "rho_out": [0.2]}


def test_affinity_with_a_scorer_uses_nu_in():
    rows, summary = residue_affinity(
        _rho(0.5), [UNITS[0]], scorer=_FakeScorer()
    )
    assert all("delta_nu_in" in r for r in rows)
    inside = [r for r in rows if r["residue"] == 2][0]
    outside = [r for r in rows if r["residue"] == 10][0]
    assert inside["delta_nu_in"] == 0 and outside["delta_nu_in"] == 1
    assert summary[2]["nu_units_positive"] == 0


@pytest.mark.parametrize("hp, before, extra, expected", [
    ("1000", True, {}, 1000),
    ("auto", True, {}, None),
    ("0", True, {}, None),
    ("1000", False, {}, None),
    ("1000", True, {"null_cal_check": True}, None),
    ("1000", True, {"raw_feature_pass": True}, None),
    ("1000", True, {"planted": True}, None),
])
def test_unfiltered_pass_is_skipped_only_when_nothing_needs_it(
        hp, before, extra, expected):
    a = SimpleNamespace(highpass_window=hp, highpass_before_features=before,
                        planted=False, null_cal_check=False,
                        raw_feature_pass=False)
    for k, v in extra.items():
        setattr(a, k, v)
    assert direct_window(a) == expected
