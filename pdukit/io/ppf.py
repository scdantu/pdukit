"""Parameter files (``.ppf``): ``key = value``, as in GROMACS ``.mdp``.

    ; a comment (';' or '#' starts one, anywhere on a line)
    lag             = 25        ; frames
    highpass-window = 500
    faces           = 0         ; flags: 0/1, yes/no, true/false, on/off
    xtc             = rep1.xtc rep2.xtc rep3.xtc   ; lists: separated by spaces
    xtc-topology    =           ; an empty value is ignored

This module only splits the text into ``(key, text, line)`` entries.  What the
keys mean, their types and defaults live in ``pdukit.config``.
"""

from __future__ import annotations

import re
from pathlib import Path

from pdukit.errors import PpfError


def normalise_key(key):
    """Return the canonical form of a key: lower case, dashes only."""
    return re.sub(r"[\s_]+", "-", key.strip().lower())


class PpfReader:
    """Read parameter files."""

    @staticmethod
    def read(path):
        """Return the ``(key, text, line_number)`` entries of a file."""
        entries = []
        seen = {}
        with open(path) as fh:
            for number, line in enumerate(fh, 1):
                text = re.split(r"[;#]", line, maxsplit=1)[0].strip()
                if not text:
                    continue
                if "=" not in text:
                    raise PpfError(f"{path}:{number}: expected 'key = value'")
                key, value = text.split("=", 1)
                key = normalise_key(key)
                if not re.fullmatch(r"[a-z][a-z0-9-]*", key):
                    raise PpfError(f"{path}:{number}: bad key {key!r}")
                if key in seen:
                    raise PpfError(
                        f"{path}:{number}: {key!r} already set on line "
                        f"{seen[key]}"
                    )
                seen[key] = number
                entries.append((key, value.strip(), number))
        return entries


class PpfWriter:
    """Write parameter files."""

    @staticmethod
    def write(path, text):
        """Write the resolved settings ``text`` to ``path``."""
        Path(path).write_text(text)


def read_ppf(path):
    """Return the entries of a file (same as ``PpfReader.read``)."""
    return PpfReader.read(path)
