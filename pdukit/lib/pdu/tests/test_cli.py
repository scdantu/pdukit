"""Tests of the command line and the .ppf parameter file."""

import pytest

from ....cli import main
from ..cog2_cli import parse_args
from ..ppf import PpfError, read_ppf

BASE = ["--planted", "-o", "out"]


def _science(args):
    skip = {"params", "_ppf"}
    return {k: v for k, v in vars(args).items() if k not in skip}


def test_file_overrides_default_and_command_line_overrides_file(tmp_path):
    ppf = tmp_path / "a.ppf"
    ppf.write_text("lag = 7   ; frames\nfaces = yes\nverify-lags = none\n")
    args = parse_args(BASE + ["-p", str(ppf)])
    assert args.lag == 7 and args.faces is True and args.verify_lags == []
    args = parse_args(BASE + ["-p", str(ppf), "--lag", "9"])
    assert args.lag == 9 and args.faces is True


def test_resolved_parameters_reproduce_the_run(tmp_path):
    args = parse_args(BASE + ["--lag", "11", "--faces", "--stab-null", "7"])
    resolved = tmp_path / "pdu_out.ppf"
    resolved.write_text(args._ppf)
    again = parse_args(BASE + ["-p", str(resolved)])
    assert _science(again) == _science(args)


def test_keys_ignore_case_dash_and_underscore(tmp_path):
    ppf = tmp_path / "a.ppf"
    ppf.write_text("Highpass_Window = 40\nMAX-UNIT = 12\n")
    args = parse_args(BASE + ["-p", str(ppf)])
    assert args.highpass_window == 40 and args.max_unit == 12


def test_unknown_key_is_an_error_with_a_suggestion(tmp_path, capsys):
    ppf = tmp_path / "a.ppf"
    ppf.write_text("frames-per-replcia = 5\n")
    with pytest.raises(SystemExit):
        parse_args(BASE + ["-p", str(ppf)])
    assert "frames-per-replica" in capsys.readouterr().err


def test_bad_lines_are_reported_with_their_line_number(tmp_path):
    ppf = tmp_path / "a.ppf"
    ppf.write_text("lag = 3\nno equals sign\n")
    with pytest.raises(PpfError, match=":2:"):
        read_ppf(ppf)
    ppf.write_text("lag = 3\nlag = 4\n")
    with pytest.raises(PpfError, match="already set"):
        read_ppf(ppf)


def test_flags_take_yes_no_values(tmp_path):
    ppf = tmp_path / "a.ppf"
    ppf.write_text("faces = no\nstability = 1\nno-null-calibration = on\n")
    args = parse_args(BASE + ["-p", str(ppf)])
    assert not args.faces and args.stability
    assert args.no_null_calibration is True


def test_n_holdout_excludes_the_explicit_lists():
    with pytest.raises(SystemExit):
        parse_args(BASE + ["--n-holdout", "2", "--holdout-replicas", "1"])


def test_pdu_command_dispatch(capsys):
    assert main([]) == 0
    assert "summarise" in capsys.readouterr().out
    assert main(["check", "--planted", "--lag", "3"]) == 0
    assert "lag" in capsys.readouterr().out
    assert main(["nonsense"]) == 2
