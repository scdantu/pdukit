"""Tests for the phase-tagged progress log."""

import re

import pytest

from pdukit import runlog
from pdukit.runlog import PHASES, log, set_level, set_phase

LINE = re.compile(r"^\[\s*\d+\.\d\] [A-Z]{3} [ !] \S")


def test_phase_codes_are_three_letters():
    assert all(re.fullmatch(r"[A-Z]{3}", code) for code in PHASES)


def test_line_format_and_phase(capsys):
    set_level("info")
    set_phase("CAL")
    log("size   6  theta_in 0.2001")
    out = capsys.readouterr().out.strip()
    assert LINE.match(out)
    assert " CAL " in out
    set_phase("INI")


def test_warning_marker_and_phase_override(capsys):
    set_level("info")
    log("!! something odd", phase="COH")
    out = capsys.readouterr().out.strip()
    assert LINE.match(out) and " COH !" in out
    assert "!!" not in out


def test_debug_hidden_unless_requested(capsys):
    set_level("info")
    log.debug("per-step trace")
    assert capsys.readouterr().out == ""
    set_level("debug")
    log.debug("per-step trace")
    assert "per-step trace" in capsys.readouterr().out
    set_level("info")


def test_quiet_keeps_warnings_only(capsys):
    set_level("quiet")
    log("routine")
    log("!! warning")
    out = capsys.readouterr().out
    set_level("info")
    assert "routine" not in out and "warning" in out


def test_unknown_phase_rejected():
    with pytest.raises(KeyError):
        set_phase("XYZ")
    assert runlog.get_phase() in PHASES
