"""Fixed reference global modes (pdu refmodes): the maths pieces."""

import numpy as np

from pdukit.cog.reference_modes import (
    backbone_dihedral_features,
    ca_distance_features,
    dihedral,
)
from pdukit.numerics.tica import explained, reversible_tica, timescale


def test_dihedral_of_known_geometry():
    # trans (180) and a right-handed 90 degree turn
    p0, p1, p2 = (np.array([[1.0, 0, 0]]), np.array([[0.0, 0, 0]]),
                  np.array([[0.0, 1, 0]]))
    trans = dihedral(p0, p1, p2, np.array([[-1.0, 1, 0]]))[0]
    cis = dihedral(p0, p1, p2, np.array([[1.0, 1, 0]]))[0]
    side = dihedral(p0, p1, p2, np.array([[0.0, 1, 1]]))[0]
    assert abs(abs(trans) - np.pi) < 1e-9 and abs(cis) < 1e-9
    assert abs(abs(side) - np.pi / 2) < 1e-9


def test_backbone_features_shape():
    rng = np.random.default_rng(0)
    xyz = rng.normal(size=(5, 3 * 6, 3))
    f = backbone_dihedral_features(xyz)
    assert f.shape == (5, 4 * 5)
    assert np.allclose(f[:, :5] ** 2 + f[:, 5:10] ** 2, 1)


def test_tica_finds_the_slow_process():
    rng = np.random.default_rng(1)
    T = 20000
    slow = np.zeros(T)
    for t in range(1, T):
        slow[t] = 0.99 * slow[t - 1] + rng.normal() * np.sqrt(1 - 0.99 ** 2)
    X = np.c_[slow + 0.1 * rng.normal(size=T), rng.normal(size=(T, 3))]
    lam, V = reversible_tica(X, [(0, T)], 10, 2)
    mode = (X - X.mean(0)) @ V[:, 0]
    assert lam[0] > 0.8 and abs(np.corrcoef(mode, slow)[0, 1]) > 0.95
    assert timescale(lam[0], 10) > 50
    assert explained(slow, X) > 0.95


def test_ca_distances_keep_near_pairs():
    coords = np.zeros((3, 4, 3), np.float32)
    coords[:, :, 0] = [0, 4, 8, 40]
    f = ca_distance_features(coords, cutoff=15, min_seq_sep=2)
    assert f.shape == (3, 1) and np.allclose(f, 8)
