"""Run options of the machine, not the science: log file, threads, GPU."""

import sys

import pytest

from pdukit import runlog
from pdukit.cli.arguments import parse_run
from pdukit.cog.runinfo import limit_threads
from pdukit.cog.scoring import backend


def test_log_lines_are_written_to_the_log_file(tmp_path, capsys):
    path = runlog.set_file(tmp_path / "pdu.log")
    try:
        runlog.log("hello")
        runlog.log("!!careful")
    finally:
        runlog.close_file()
    runlog.log("after close")
    text = path.read_text()
    assert "hello" in text and "! careful" in text
    assert "after close" not in text
    assert "after close" in capsys.readouterr().out


def test_no_log_file_without_the_folder(tmp_path):
    assert runlog.set_file(tmp_path / "missing" / "pdu.log") is None


def test_progress_reports_each_step_and_each_pass(capsys):
    p = runlog.Progress("features", step=50)
    for i in range(1, 5):
        p(i, 4)
    for i in range(1, 5):
        p(i, 4)
    lines = capsys.readouterr().out.splitlines()
    assert [ln.split("features")[1] for ln in lines] == [
        ": 1/4 (25%)", ": 2/4 (50%)", ": 4/4 (100%)",
        " (pass 2): 1/4 (25%)", " (pass 2): 2/4 (50%)",
        " (pass 2): 4/4 (100%)",
    ]


def test_nt_default_is_all_cores_and_n_caps_blas():
    threadpoolctl = pytest.importorskip("threadpoolctl")
    n, cores, undo = limit_threads(-1)
    assert n == cores and undo is None
    before = [lib["num_threads"] for lib in threadpoolctl.threadpool_info()]
    n, cores, undo = limit_threads(1)
    try:
        assert n == 1
        info = threadpoolctl.threadpool_info()
        assert all(lib["num_threads"] == 1 for lib in info)
    finally:
        undo()
    after = [lib["num_threads"] for lib in threadpoolctl.threadpool_info()]
    assert after == before


def test_nt_on_the_command_line(tmp_path):
    cfg, _ = parse_run(["--planted", "-o", str(tmp_path), "--nt", "4"])
    assert cfg.nt == 4
    cfg, _ = parse_run(["--planted", "-o", str(tmp_path)])
    assert cfg.nt == -1


def test_device_defaults_to_auto(tmp_path):
    cfg, _ = parse_run(["--planted", "-o", str(tmp_path)])
    assert cfg.device == "auto"


def test_auto_without_torch_is_numpy(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)  # import fails
    assert backend._resolve_backend("auto") == ("numpy", None, "cpu")


def test_unused_gpu_is_reported_with_the_fix(monkeypatch):
    monkeypatch.setattr(backend, "gpu_present", lambda: True)
    monkeypatch.setitem(sys.modules, "torch", None)
    note = backend.gpu_note("auto", "cpu")
    assert note.startswith("!!") and "cu121" in note
    assert backend.gpu_note("cpu", "cpu") is None
    assert backend.gpu_note("auto", "cuda") is None


def test_no_note_on_a_machine_without_gpu(monkeypatch):
    monkeypatch.setattr(backend, "gpu_present", lambda: False)
    assert backend.gpu_note("auto", "cpu") is None
    assert "no CUDA GPU" in backend.gpu_note("cuda", "cpu")


def _args(**kw):
    from types import SimpleNamespace

    base = dict(device="cpu", batched="auto", rse_components=-1,
                no_memory_check=False)
    base.update(kw)
    return SimpleNamespace(**base)


def test_workspace_estimate_matches_what_numpy_allocates():
    import tracemalloc

    import numpy as np

    from pdukit.cog.scoring.factory import workspace_bytes
    from pdukit.cog.scoring.workspace import FluxWorkspace

    F, N, K, n_null = 2000, 100, 4, 16
    feats = np.random.default_rng(0).normal(size=(F, N, K)).astype(
        np.float32)
    tracemalloc.start()
    ws = FluxWorkspace(feats, lag=1, n_null=n_null, seed=0, device="cpu")
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    est = workspace_bytes(F, N * K, n_null)
    assert ws.D == N * K
    assert 0.7 < peak / est < 1.4


def test_tester_case_is_stopped_with_settings_that_fit(monkeypatch):
    import numpy as np

    from pdukit.cog.scoring import factory
    from pdukit.errors import ResourceError

    # 400 residues, K = 20, 64 shifts on a 64 GB machine: ~67 GB needed
    feats = np.lib.stride_tricks.as_strided(
        np.zeros(1, np.float32), shape=(60000, 400, 20), strides=(0, 0, 0))
    monkeypatch.setattr(factory, "available_memory", lambda: 60e9)
    with pytest.raises(ResourceError) as err:
        factory.check_memory(_args(), feats, 64, "the workspace")
    msg = str(err.value)
    assert "rse-max-components=" in msg and "n-null=" in msg
    k = int(msg.split("rse-max-components=")[1].split()[0])
    assert factory.workspace_bytes(60000, 400 * k, 64) <= 0.85 * 60e9
    assert 10 <= k < 20


def test_check_passes_when_it_fits_or_is_switched_off(monkeypatch):
    import numpy as np

    from pdukit.cog.scoring import factory

    feats = np.zeros((100, 10, 4), np.float32)
    monkeypatch.setattr(factory, "available_memory", lambda: 1e12)
    assert factory.check_memory(_args(), feats, 64, "the workspace") > 0
    monkeypatch.setattr(factory, "available_memory", lambda: 1.0)
    factory.check_memory(_args(no_memory_check=True), feats, 64, "x")
    monkeypatch.setattr(factory, "available_memory", lambda: None)
    factory.check_memory(_args(), feats, 64, "x")  # unknown: no stop


def test_available_memory_is_known_here():
    from pdukit.cog.runinfo import available_memory

    free = available_memory()
    assert free is None or free > 0


@pytest.mark.parametrize("name, slow", [
    ("NVIDIA GeForce RTX 3070", True),
    ("NVIDIA RTX A6000", True),
    ("NVIDIA RTX A3000 Laptop GPU", True),
    ("AMD Instinct MI250X", False),
    ("NVIDIA GH200 120GB", False),
    ("NVIDIA A100-SXM4-80GB", False),
    (None, False),
])
def test_slow_double_precision_gpus_are_named(name, slow):
    note = backend.fp64_note(name)
    assert (note is not None) == slow
    if slow:
        assert note.startswith("!!") and "device=cpu" in note


def test_options_guide_names_real_settings_and_every_group():
    from pdukit.config import SETTINGS, guide

    keys = {row[1] for row in SETTINGS}
    assert [k for k, _ in guide.START_HERE if k not in keys] == []
    assert {row[0] for row in SETTINGS} == set(guide.GROUP_INTRO)


def test_a_shipped_preset_can_be_named_without_its_path(tmp_path,
                                                        monkeypatch):
    from pdukit.cli.arguments import shipped_preset

    monkeypatch.chdir(tmp_path)
    for name in ("workstation", "workstation.ppf"):
        assert shipped_preset(name).endswith("params/workstation.ppf")
    assert shipped_preset("t4l_final_tau1ns").endswith(
        "final/t4l_final_tau1ns.ppf")
    (tmp_path / "workstation.ppf").write_text("lag = 3\n")
    assert shipped_preset("workstation.ppf") == "workstation.ppf"
    assert shipped_preset("nothing-like-this") == "nothing-like-this"
    cfg, _ = parse_run(["-p", "workstation", "--planted", "-o", "o"])
    assert cfg.device == "cpu" and cfg.rse_max_components == 12
