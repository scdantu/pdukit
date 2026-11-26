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
