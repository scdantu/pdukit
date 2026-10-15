"""PDU parameter files (``.ppf``): ``key = value``, as in GROMACS .mdp.

    ; a comment (';' or '#' starts one, anywhere on a line)
    lag            = 25        ; frames
    highpass-window = 500
    faces          = 0         ; flags: 0/1, yes/no, true/false, on/off
    verify-lags    = none
    xtc-topology   =           ; an empty value is ignored

A key is the name of a command-line option without ``--``; case, ``-`` and
``_`` do not matter.  Precedence is: command line, then parameter file, then
the built-in default.  Unknown keys are errors (with a suggestion), so a typo
cannot silently fall back to a default.
"""

from __future__ import annotations

import argparse
import difflib
import re

TRUE = {"1", "yes", "true", "on", "y", "t"}
FALSE = {"0", "no", "false", "off", "n", "f"}

# options that describe a particular run (files, machine), not the science
RUN_SPECIFIC = {
    "params",
    "outdir",
    "xtc",
    "xtc-topology",
    "feature-cache",
    "device",
}


class PpfError(ValueError):
    """A parameter file could not be read."""


def normalise_key(key):
    """Return the canonical form of a key: lower case, dashes only."""
    return re.sub(r"[\s_]+", "-", key.strip().lower())


def read_ppf(path):
    """Return the ``(key, value, line_number)`` entries of a file."""
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
                    f"{path}:{number}: {key!r} already set on line {seen[key]}"
                )
            seen[key] = number
            entries.append((key, value.strip(), number))
    return entries


def option_table(parser):
    """Map each long option name (without ``--``) to its argparse action."""
    table = {}
    for action in parser._actions:
        if isinstance(action, argparse._HelpAction):
            continue
        for option in action.option_strings:
            if option.startswith("--"):
                table[normalise_key(option[2:])] = action
    return table


def _to_bool(text, where):
    word = text.strip().lower()
    if word in TRUE:
        return True
    if word in FALSE:
        return False
    raise PpfError(f"{where}: {text!r} is not a yes/no value")


def _convert(parser, action, text, where):
    """Convert the text of a value as argparse would convert the option."""
    try:
        if action.nargs in ("+", "*"):
            return [parser._get_value(action, word) for word in text.split()]
        value = parser._get_value(action, text)
        parser._check_value(action, value)
        return value
    except argparse.ArgumentError as exc:
        raise PpfError(f"{where}: {exc}") from None


def ppf_defaults(parser, path):
    """Return ``{dest: value}`` for the entries of a parameter file."""
    table = option_table(parser)
    out = {}
    for key, text, number in read_ppf(path):
        where = f"{path}:{number}: {key}"
        action = table.get(key)
        if action is None:
            near = difflib.get_close_matches(key, table, n=3, cutoff=0.6)
            hint = f" (did you mean {', '.join(near)}?)" if near else ""
            raise PpfError(f"{path}:{number}: unknown parameter {key!r}{hint}")
        if text == "":
            continue
        if isinstance(action, argparse._StoreTrueAction):
            if _to_bool(text, where):
                out[action.dest] = True
        elif isinstance(action, argparse._StoreFalseAction):
            if _to_bool(text, where):
                out[action.dest] = False
        else:
            out[action.dest] = _convert(parser, action, text, where)
    return out


def _format_value(action, value):
    if isinstance(action, argparse._StoreTrueAction):
        return "1" if value else "0"
    if isinstance(action, argparse._StoreFalseAction):
        return "0" if value else "1"
    if isinstance(value, (list, tuple)):
        return " ".join(str(v) for v in value)
    return str(value)


def format_ppf(parser, args, exclude=RUN_SPECIFIC, header=""):
    """Return the resolved settings of ``args`` as parameter-file text."""
    lines = [header.rstrip("\n")] if header else []
    for group in parser._action_groups:
        if group.title == "planted benchmark" and not getattr(
            args, "planted", False
        ):
            continue
        rows = []
        for action in group._group_actions:
            if isinstance(action, argparse._HelpAction):
                continue
            options = [o for o in action.option_strings if o.startswith("--")]
            if not options:
                continue
            key = normalise_key(options[0][2:])
            value = getattr(args, action.dest, None)
            if key in exclude or value is None:
                continue
            text = _format_value(action, value)
            note = action.help or ""
            if "%(" in note:
                note = note % {"default": action.default}
            note = re.sub(r"\s+", " ", note).strip()
            note = re.split(r"(?<=[.;:])\s", note, maxsplit=1)[0]
            if len(note) > 70:
                note = note[:67].rsplit(" ", 1)[0] + " ..."
            rows.append((key, text, note))
        if not rows:
            continue
        lines.append("")
        lines.append(f"; ---- {group.title} ----")
        width = max(len(k) for k, _, _ in rows)
        for key, text, note in rows:
            tail = f"  ; {note}" if note else ""
            lines.append(f"{key:<{width}} = {text}{tail}".rstrip())
    return "\n".join(lines) + "\n"
