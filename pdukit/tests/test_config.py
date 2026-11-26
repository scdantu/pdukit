"""Config: layers, checks, source tracking and the resolved parameter file."""

import subprocess
import sys
from pathlib import Path

import pytest

from pdukit.config import SETTINGS, Config, ConfigError
from pdukit.config.schema import RETIRED

ROOT = Path(__file__).resolve().parents[2]
PARAMS = ROOT / "pdukit" / "params"
PRESETS = sorted(PARAMS.rglob("*.ppf"))
FILE_SETS = [("xtc", "a.xtc b.xtc"), ("xtc-topology", "top.pdb"),
             ("outdir", "out")]


def test_schema_has_one_row_per_setting():
    """Keys and destination names are unique, and every row has help text."""
    assert len(SETTINGS) == len({r[1] for r in SETTINGS})
    assert len(SETTINGS) == len({r[2] for r in SETTINGS})
    assert all(r[8] for r in SETTINGS)


@pytest.mark.parametrize("ppf", PRESETS, ids=lambda p: p.name)
def test_presets_resolve_and_round_trip(ppf, tmp_path):
    """A preset loads, and the ppf it resolves to gives the same values."""
    cfg = Config.build(ppf=str(ppf), sets=FILE_SETS)
    resolved = tmp_path / "pdu_out.ppf"
    resolved.write_text(cfg.to_ppf())
    again = Config.build(ppf=str(resolved), sets=FILE_SETS)
    assert again.to_dict() == cfg.to_dict()


def test_cli_overrides_ppf_and_tracks_source():
    """--set beats the file; the source of each value is recorded."""
    ppf = PARAMS / "tutorial.ppf"
    cfg = Config.build(ppf=str(ppf), sets=FILE_SETS + ["lag=33"])
    assert cfg["lag"] == 33 and cfg.source("lag") == "cli"
    assert cfg.source("dec-seeds").startswith("ppf:")
    assert cfg.dec_seeds == cfg["dec-seeds"] == cfg["dec_seeds"]


def test_errors_are_clear():
    """Unknown keys suggest a fix; bad values name the setting."""
    with pytest.raises(ConfigError, match="did you mean dec-seeds"):
        Config.build(sets=FILE_SETS + ["dec-seed=3"])
    with pytest.raises(ConfigError, match="not a valid int"):
        Config.build(sets=FILE_SETS + ["lag=abc"])
    with pytest.raises(ConfigError, match="output directory"):
        Config.build(sets=[("planted", "1")])


def test_assign_copy_and_override():
    """cfg.KEY = value is checked; copy() is independent; override undoes."""
    cfg = Config.build(sets=FILE_SETS)
    cfg.lag = 5
    assert cfg.lag == 5 and cfg.source("lag") == "user"
    other = cfg.copy()
    other.lag = 9
    assert cfg.lag == 5 and other.lag == 9
    alias = cfg
    alias.lag = 7
    assert cfg.lag == 7
    with pytest.raises(AttributeError, match="did you mean"):
        cfg.lagg = 1
    with pytest.raises(ConfigError):
        cfg.backend = "nonsense"
    with pytest.raises(ConfigError):
        cfg.lag = "abc"
    cfg["dec-seeds"] = 4
    assert cfg.dec_seeds == 4
    before = cfg.shift_mode
    with cfg.override(shift_mode="within"):
        assert cfg.shift_mode == "within"
    assert cfg.shift_mode == before


def test_from_ppf_with_changes():
    """Config.from_ppf(path, **changes) loads and edits in one call."""
    cfg = Config.from_ppf(PARAMS / "tutorial.ppf", lag=11)
    assert cfg.lag == 11 and cfg.source("lag") == "user"


def test_run_command_line(tmp_path):
    """pdu run accepts the 7 arguments and old flags (with a warning)."""
    ppf = PARAMS / "tutorial.ppf"
    cmd = [sys.executable, "-m", "pdukit", "run", "-p", str(ppf), "-f",
           "a.xtc", "-s", "t.pdb", "-o", str(tmp_path), "--set", "lag=7",
           "--dec-seeds", "5", "--check"]
    out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                         check=False)
    assert out.returncode == 0, out.stderr[-400:]
    assert "lag" in out.stdout and "<- cli" in out.stdout
    assert "deprecated" in out.stderr


def test_retired_names_are_not_current_settings():
    assert RETIRED
    assert not RETIRED & {r[1] for r in SETTINGS}


def test_shipped_parameter_files_use_no_retired_setting():
    for ppf in PRESETS:
        keys = {
            line.split("=")[0].strip().replace("_", "-").lower()
            for line in ppf.read_text().splitlines()
            if "=" in line and not line.lstrip().startswith(";")
        }
        assert not keys & RETIRED, f"{ppf.name}: {sorted(keys & RETIRED)}"


def test_a_retired_setting_is_ignored_with_a_warning(tmp_path, capsys):
    ppf = tmp_path / "old.ppf"
    ppf.write_text("lag = 7\nstab-null = 7\nmerge_mode = reearn\n")
    cfg = Config.build(
        ppf=str(ppf), require_outdir=False, require_input=False
    )
    err = capsys.readouterr().err
    assert cfg.lag == 7
    assert "stab-null" in err and "merge-mode" in err and "retired" in err
    assert "stab_null" not in cfg.to_ppf()
    assert cfg.to_ppf() == Config.build(
        sets=["lag=7"], require_outdir=False, require_input=False
    ).to_ppf()


def test_a_misspelt_setting_is_still_an_error(tmp_path):
    ppf = tmp_path / "bad.ppf"
    ppf.write_text("lagg = 7\n")
    with pytest.raises(ConfigError, match="unknown setting"):
        Config.build(ppf=str(ppf), require_outdir=False, require_input=False)


@pytest.mark.parametrize("value", ["mc", "merge-mc"])
def test_archived_postprocess_modes_are_refused_with_a_reason(value):
    with pytest.raises(ConfigError, match="archived"):
        Config.build(
            sets=[f"incoming-postprocess={value}"],
            require_outdir=False,
            require_input=False,
        )


def test_postprocess_none_and_split_are_accepted():
    for value in ("none", "split"):
        cfg = Config.build(
            sets=[f"incoming-postprocess={value}"],
            require_outdir=False,
            require_input=False,
        )
        assert cfg.incoming_postprocess == value
