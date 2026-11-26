"""The tutorial preset and demo-data generator stay usable."""

import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve()
ROOT = next(p for p in HERE.parents if (p / "tools").is_dir())


def test_tutorial_ppf_resolves():
    """``pdu check`` accepts tutorial.ppf."""
    ppf = ROOT / "pdukit" / "params" / "tutorial.ppf"
    out = subprocess.run(
        [sys.executable, "-m", "pdukit", "check", "--planted", "-p", str(ppf)],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert out.returncode == 0, out.stderr[-500:]


def test_demo_generator_writes_files(tmp_path):
    """The generator writes a topology and XTC replicas."""
    pytest.importorskip("MDAnalysis")
    subprocess.run(
        [sys.executable, str(ROOT / "tools" / "make_demo_trajectories.py"),
         str(tmp_path), "--frames", "50", "--replicas", "2"],
        check=True, capture_output=True,
    )
    assert (tmp_path / "top.pdb").exists()
    assert (tmp_path / "rep2.xtc").exists()
