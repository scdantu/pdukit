"""The version written in the README and CITATION.cff matches the tag."""

import re
from pathlib import Path

import pdukit

ROOT = Path(__file__).resolve().parents[2]


def _text(name):
    path = ROOT / name
    return path.read_text() if path.is_file() else None


def test_readme_and_citation_agree():
    """README title and CITATION.cff name the same release."""
    readme, cff = _text("README.md"), _text("CITATION.cff")
    if readme is None or cff is None:
        return  # running from an installed copy
    in_readme = re.search(r"^# .*\(v([^)]+)\)", readme, re.M).group(1)
    in_cff = re.search(r"^version:\s*(\S+)", cff, re.M).group(1)
    assert in_readme == in_cff


def test_installed_version_matches_release_tag():
    """On a tagged commit the package version is the one in the README."""
    readme = _text("README.md")
    version = pdukit.__version__
    if readme is None or "dev" in version or version == "unknown":
        return  # untagged commit or not installed: nothing to compare
    in_readme = re.search(r"^# .*\(v([^)]+)\)", readme, re.M).group(1)
    assert version.split("+")[0] == in_readme
