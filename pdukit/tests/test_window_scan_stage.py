"""The high-pass window scan runs only when it chooses the window."""

from types import SimpleNamespace

import pytest

from pdukit.cog.stages import prepare


def _state(hp, check=False, never=False, planted=False):
    args = SimpleNamespace(
        highpass_window=hp, null_cal_check=check, no_null_calibration=never,
        planted=planted, frame_ps=20.0, null_cal_tol=0.15,
        null_cal_target=0.06, verify_lags=None,
    )
    return SimpleNamespace(args=args, results={}, feats=None, segments=None,
                           adj=None, timings={}, save=lambda: None)


@pytest.fixture
def scans(monkeypatch):
    calls = []

    class FakeScan:
        def __init__(self, args):
            pass

        def run(self, *_):
            calls.append(1)
            row = {"window_frames": 200, "theta_in": 0.05, "theta_out": 0.05}
            return {"rows": [row], "chosen_window": 200, "theta_min": 0.05,
                    "any_below_target": True}

    monkeypatch.setattr(prepare, "WindowScan", FakeScan)
    return calls


@pytest.mark.parametrize("hp", ["0", "100"])
def test_fixed_or_no_window_skips_the_scan(scans, hp, capsys):
    s = _state(hp)
    prepare.ScanWindowStage().run(s)
    assert scans == [] and s.hp == int(hp)
    assert "scan skipped" in capsys.readouterr().out


def test_auto_runs_the_scan_and_takes_its_window(scans):
    s = _state("auto")
    prepare.ScanWindowStage().run(s)
    assert scans == [1] and s.hp == 200


def test_check_runs_the_scan_with_a_fixed_window(scans):
    s = _state("100", check=True)
    prepare.ScanWindowStage().run(s)
    assert scans == [1] and s.hp == 100


def test_auto_without_the_scan_is_an_error(scans):
    with pytest.raises(SystemExit, match="auto needs"):
        prepare.ScanWindowStage().run(_state("auto", never=True))
