"""Documentation that is generated from the code must match the code."""

from pathlib import Path

import pytest

from pdukit.config import Config
from pdukit.config import reference as ref
from pdukit.io.ppf import read_ppf


def test_every_option_has_help_text():
    assert ref.options_without_help() == []


def test_default_ppf_is_current():
    path = ref.PARAMS / "default.ppf"
    assert path.read_text() == ref.render_default_ppf(), (
        "run: python -m pdukit.config.reference --write"
    )


def test_default_ppf_reproduces_the_built_in_defaults():
    path = ref.PARAMS / "default.ppf"
    assert read_ppf(path)  # not an empty file
    files = [("outdir", "out"), ("planted", "1")]
    from_file = Config.build(ppf=str(path), sets=files).to_dict()
    built_in = Config.build(sets=files).to_dict()
    assert {k: v for k, v in from_file.items() if built_in[k] != v} == {}


def test_options_reference_is_current():
    path = ref.REPO / "docs" / "OPTIONS.md"
    if not path.exists():  # installed package without the docs
        pytest.skip("docs/OPTIONS.md not present")
    assert path.read_text() == ref.render_markdown(), (
        "run: python -m pdukit.config.reference --write"
    )


@pytest.mark.parametrize(
    "preset", sorted(p.name for p in ref.PARAMS.glob("*.ppf"))
)
def test_shipped_presets_load(preset):
    """Every preset parses and names only options that exist."""
    path = Path(ref.PARAMS) / preset
    assert read_ppf(path)
    Config.build(ppf=str(path), sets=[("outdir", "out"), ("planted", "1")])
