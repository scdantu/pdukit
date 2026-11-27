"""Synthetic coordinates with planted units (pdukit.bench.coords)."""

import json

import numpy as np
import pytest

from pdukit.bench.coords import PlantedMD, build, write_planted_md


def test_build_shapes_truth_and_geometry():
    cfg = PlantedMD(frames=300, replicas=2, spacing=20.0)
    base, reps, truth = build(cfg, seed=3)
    assert base.shape == (60, 3) and len(reps) == 2
    assert reps[0].shape == (300, 60, 3)
    assert truth["units"][1] == list(range(10, 20))
    assert truth["couplings"] == [[0, 1], [1, 2], [3, 4]]
    d = np.linalg.norm(base[:, None] - base[None], axis=-1)
    unit = np.arange(60) // 10
    same = unit[:, None] == unit[None]
    assert d[same].max() < d[~same].min()  # blobs do not interpenetrate


def test_same_seed_same_system():
    cfg = PlantedMD(frames=50, replicas=1)
    a = build(cfg, seed=1)[1][0]
    b = build(cfg, seed=1)[1][0]
    c = build(cfg, seed=2)[1][0]
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_residues_of_a_unit_move_together():
    cfg = PlantedMD(frames=4000, replicas=1, noise=0.05, couplings=())
    base, reps, _ = build(cfg, seed=0)
    disp = np.linalg.norm(reps[0] - base[None], axis=-1)
    corr = np.corrcoef(disp.T)
    unit = np.arange(60) // 10
    same = (unit[:, None] == unit[None]) & ~np.eye(60, dtype=bool)
    assert np.abs(corr[same]).mean() > 3 * np.abs(corr[~same]).mean()


def test_write_planted_md(tmp_path):
    pytest.importorskip("MDAnalysis")
    paths, top, truth = write_planted_md(
        tmp_path, PlantedMD(frames=20, replicas=2), seed=0)
    assert len(paths) == 2 and top.is_file()
    assert json.loads(truth.read_text())["seed"] == 0
