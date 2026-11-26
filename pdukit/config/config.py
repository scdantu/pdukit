"""Build the settings of a run as ONE flat, read-only dictionary.

    cfg = Config.build(ppf="t4l.ppf", sets=["lag=20"],
                       files={"xtc": [...], "xtc-topology": "top.pdb", ...})
    cfg["dec-seeds"]      # or cfg.dec_seeds; both give the typed value

Layers, later ones win:

1. the default in the schema (`pdukit.config.schema`);
2. a parameter file (read by `pdukit.io.ppf.PpfReader`);
3. the command line (``-f``, ``-s``, ``-o``, ``--set KEY=VALUE``, ...).

Every value remembers where it came from (``cfg.source("lag")`` gives
``"default"``, ``"ppf:12"`` or ``"cli"``).  Values are converted and checked
once, here; unknown keys are errors with a "did you mean" hint.  The result is
handed to the pipeline, which only reads it.
"""

from __future__ import annotations

import difflib
import math
import re
import sys
from collections.abc import Mapping
from contextlib import contextmanager
from enum import Enum

from pdukit.config.schema import ARCHIVED_CHOICES, RETIRED, SETTINGS
from pdukit.errors import ConfigError, PpfError
from pdukit.io.ppf import PpfReader, normalise_key

TRUE = {"1", "yes", "true", "on", "y", "t"}
FALSE = {"0", "no", "false", "off", "n", "f"}

_ROWS = {row[1]: row for row in SETTINGS}  # key -> schema row
_BY_DEST = {row[2]: row for row in SETTINGS}  # dest -> schema row


def _to_bool(text, where):
    word = text.strip().lower()
    if word in TRUE:
        return True
    if word in FALSE:
        return False
    raise ConfigError(f"{where}: {text!r} is not a yes/no value")


def _convert(row, text, where):
    """Return the typed value of ``text``; ``None`` means 'leave default'."""
    _, _, _, kind, _, choices, is_list, _, _ = row
    if kind == "flag":
        return True if _to_bool(text, where) else None
    if kind == "noflag":
        return False if _to_bool(text, where) else None
    if is_list:
        return text.split()
    try:
        value = {"int": int, "float": float, "str": str}[kind](text)
    except ValueError:
        raise ConfigError(
            f"{where}: {text!r} is not a valid {kind} value"
        ) from None
    if choices and value not in choices:
        gone = ARCHIVED_CHOICES.get(row[1], {}).get(value)
        note = (
            f" ({value!r} was {gone}; see the git tag "
            "archive/pre-incoming-only)"
            if gone
            else ""
        )
        raise ConfigError(
            f"{where}: {value!r} is not one of "
            f"{', '.join(map(str, choices))}{note}"
        )
    return value


def _suggest(key):
    near = difflib.get_close_matches(key, _ROWS, n=3, cutoff=0.6)
    return f" (did you mean {', '.join(near)}?)" if near else ""


class Config(Mapping):
    """The settings of a run: every parameter, readable and assignable.

        cfg = Config.from_ppf("t4l.ppf", lag=20)
        cfg.lag                  # read   (also cfg["lag"], cfg["dec-seeds"])
        cfg.lag = 30             # change; type and choices are checked
        cfg2 = cfg.copy()        # an independent copy (cfg2 = cfg is NOT)
        with cfg.override(shift_mode="within"): ...   # temporary change

    Unknown names raise with a "did you mean" hint.  Combinations of
    settings are checked by ``cfg.validate()`` (done again when a run starts).
    """

    def __init__(self, values, raw, sources):
        """Wrap the typed ``values`` (by dest), the raw text and sources."""
        object.__setattr__(self, "_d", dict(values))
        object.__setattr__(self, "_raw", dict(raw))
        object.__setattr__(self, "_src", dict(sources))

    def _dest(self, key):
        dest = str(key).replace("-", "_")
        if dest in self._d:
            return dest
        near = difflib.get_close_matches(dest, list(self._d), n=3, cutoff=0.6)
        hint = f" (did you mean {', '.join(near)}?)" if near else ""
        raise KeyError(f"unknown setting {key!r}{hint}")

    def __getitem__(self, key):
        """Return the value of a setting."""
        return self._d[self._dest(key)]

    def __iter__(self):
        """Iterate over the setting names (dest form)."""
        return iter(self._d)

    def __len__(self):
        """Return the number of settings."""
        return len(self._d)

    def __getattr__(self, name):
        """Return a setting as an attribute (``cfg.lag``)."""
        if name == "_ppf":
            return self.to_ppf()
        if name.startswith("__") or name not in self._d:
            raise AttributeError(name)
        return self._d[name]

    def __setattr__(self, name, value):
        """Change one setting (checked); ``cfg.lag = 30``."""
        try:
            self._set(name, value, "user")
        except KeyError as exc:
            raise AttributeError(exc.args[0]) from None

    def __setitem__(self, key, value):
        """Change one setting: ``cfg["dec-seeds"] = 10``."""
        self._set(key, value, "user")

    def _set(self, key, value, source):
        dest = self._dest(key)
        row = _BY_DEST[dest]
        kind, choices, is_list = row[3], row[5], row[6]
        if isinstance(value, Enum):
            value = value.value
        if isinstance(value, str) and kind != "str" and not is_list:
            typed = _convert(row, value, key)
            value = row[4] if typed is None else typed
        elif kind in ("int", "float") and (
            isinstance(value, bool) or not isinstance(value, (int, float))
        ):
            raise ConfigError(f"{key}: {value!r} is not a number")
        if choices and value not in choices:
            raise ConfigError(
                f"{key}: {value!r} is not one of "
                f"{', '.join(map(str, choices))}"
            )
        tmp = dict(self._raw)
        tmp[dest] = value
        Config._normalise(tmp)
        self._raw[dest] = value
        self._d[dest] = tmp[dest]
        self._src[dest] = source

    def copy(self):
        """Return an independent copy (``cfg2 = cfg.copy()``)."""
        import copy

        return Config(copy.deepcopy(self._d), copy.deepcopy(self._raw),
                      dict(self._src))

    def validate(self, require_outdir=False, require_input=False):
        """Check combinations of settings; raise `ConfigError`."""
        Config._validate(self._d, require_outdir, require_input)
        return self

    def update(self, **changes):
        """Change several settings at once: ``cfg.update(lag=20, tau=5)``."""
        for key, value in changes.items():
            self._set(key, value, "user")
        return self

    @classmethod
    def from_ppf(cls, path=None, **changes):
        """Load a parameter file (or defaults) and apply ``changes``."""
        cfg = cls.build(ppf=path, require_outdir=False, require_input=False)
        return cfg.update(**changes)

    def to_dict(self):
        """Return a plain copy of the settings, by dest name."""
        return dict(self._d)

    def source(self, key):
        """Return where a value came from: default, ``ppf:LINE`` or cli."""
        return self._src.get(self._dest(key), "default")

    @contextmanager
    def override(self, **changes):
        """Change settings inside a ``with`` block, then restore them."""
        old = {k: self[k] for k in changes}
        for k, v in changes.items():
            self._d[self._dest(k)] = v
        try:
            yield self
        finally:
            for k, v in old.items():
                self._d[self._dest(k)] = v

    def to_ppf(
        self, header="; resolved parameters of this run (pdu run -p)",
        sources=False,
    ):
        """Return the settings as parameter-file text (``pdu_out.ppf``)."""
        lines = [header.rstrip("\n")] if header else []
        groups = []
        for row in SETTINGS:
            if not groups or groups[-1][0] != row[0]:
                groups.append((row[0], []))
            groups[-1][1].append(row)
        planted = bool(self._d.get("planted"))
        for title, rows_ in groups:
            if title == "planted benchmark" and not planted:
                continue
            out = []
            for _, key, dest, kind, _, _, _, run_specific, help_ in rows_:
                value = self._raw.get(dest)
                if run_specific or value is None:
                    continue
                if kind == "flag":
                    text = "1" if value else "0"
                elif kind == "noflag":
                    text = "0" if value else "1"
                elif isinstance(value, (list, tuple)):
                    text = " ".join(str(v) for v in value)
                else:
                    text = str(value)
                note = re.split(r"(?<=[.;:])\s", help_, maxsplit=1)[0]
                if len(note) > 70:
                    note = note[:67].rsplit(" ", 1)[0] + " ..."
                if sources and self._src.get(dest, "default") != "default":
                    note = f"<- {self._src[dest]}"
                out.append((key, text, note))
            if not out:
                continue
            lines.append("")
            lines.append(f"; ---- {title} ----")
            width = max(len(k) for k, _, _ in out)
            for key, text, note in out:
                tail = f"  ; {note}" if note else ""
                lines.append(f"{key:<{width}} = {text}{tail}".rstrip())
        return "\n".join(lines) + "\n"


    # ---- building -------------------------------------------------------

    @staticmethod
    def defaults():
        """Return ``({dest: value}, {dest: 'default'})`` from the schema."""
        return {row[2]: row[4] for row in SETTINGS}

    @staticmethod
    def build(ppf=None, sets=(), require_outdir=True, require_input=True):
        """Return the checked `Options` of a run.

        ``ppf`` is a parameter file or ``None``.  ``sets`` is a list of
        ``"key=value"`` strings or ``(key, text)`` pairs, applied after the
        file (the command line wins).
        """
        values = Config.defaults()
        sources = {}
        if ppf:
            try:
                entries = PpfReader.read(ppf)
            except (OSError, PpfError) as exc:
                raise ConfigError(str(exc)) from None
            retired = []
            for key, text, number in entries:
                if key in RETIRED:
                    retired.append(key)
                    continue
                Config._apply(values, sources, key, text, f"ppf:{number}",
                              f"{ppf}:{number}: {key}")
            if retired:
                print(f"pdu: warning: {ppf}: {len(retired)} retired "
                      f"setting(s) ignored: {', '.join(sorted(retired))}",
                      file=sys.stderr)
        for item in sets:
            if isinstance(item, str):
                if "=" not in item:
                    raise ConfigError(
                        f"--set expects KEY=VALUE, got {item!r}"
                    )
                key, text = item.split("=", 1)
            else:
                key, text = item
            key = normalise_key(key)
            Config._apply(values, sources, key, text.strip(), "cli",
                          f"--set {key}")
        raw = dict(values)
        Config._normalise(values)
        Config._validate(values, require_outdir, require_input)
        return Config(values, raw, sources)

    @staticmethod
    def _apply(values, sources, key, text, source, where):
        if key in RETIRED:
            print(f"pdu: warning: {where}: retired setting, ignored",
                  file=sys.stderr)
            return
        row = _ROWS.get(key)
        if row is None:
            raise ConfigError(f"{where}: unknown setting {key!r}"
                              f"{_suggest(key)}")
        if text == "":
            return
        value = _convert(row, text, where)
        if value is not None:
            values[row[2]] = value
            sources[row[2]] = source

    @staticmethod
    def _normalise(v):
        """Turn text settings into the forms the pipeline uses."""
        try:
            text = str(v["recovery_channel_ranks"])
            ranks = [int(k) for k in text.split(",") if k.strip()]
            if any(k < 1 for k in ranks):
                raise ValueError
        except ValueError:
            raise ConfigError(
                "recovery-channel-ranks must be comma-separated positive "
                "integers"
            ) from None
        v["recovery_channel_ranks"] = ranks
        text = str(v["verify_lags"]).strip().lower()
        v["verify_lags"] = (
            [] if text in ("", "none", "0", "off")
            else [int(x) for x in str(v["verify_lags"]).split(",") if x]
        )
        if v["cond_env"] == "reduced":
            v["cond_env"] = None
        hw = str(v["highpass_window"]).lower()
        v["highpass_window"] = "auto" if hw == "auto" else int(hw)

    @staticmethod
    def _validate(a, require_outdir, require_input):
        """Check combinations of settings; raise `ConfigError`."""
        def fail(msg):
            raise ConfigError(msg)

        if (not math.isfinite(a["incoming_min_gain"])
                or a["incoming_min_gain"] < 0):
            fail("incoming-min-gain must be finite and >= 0")
        if (not math.isfinite(a["split_markov_time"])
                or a["split_markov_time"] <= 0
                or not math.isfinite(a["split_alpha"])
                or not 0 < a["split_alpha"] < 1
                or a["split_null"] < 1 or a["split_max_depth"] < 0):
            fail("invalid split settings: require positive time/null count, "
                 "0 < alpha < 1, depth >= 0")
        if (a["recovery_modes"] < 1 or a["recovery_local_dim"] < 0
                or a["recovery_max_combinations"] < 1):
            fail("recovery requires modes/combinations >= 1 and "
                 "local-dim >= 0")
        if not (math.isfinite(a["start_min_gain"])
                and a["start_min_gain"] >= 0):
            fail("start-min-gain must be finite and >= 0")
        if not 0 <= a["seed_fraction"] <= 1:
            fail(f"seed-fraction must be in [0, 1], not {a['seed_fraction']}")
        if a["rse_max_components"] < 0:
            fail("rse-max-components must be >= 0 (0 = no limit)")
        if a["seed_group"] < 0:
            fail(f"seed-group must be >= 0 (0 = fill unit-dim), not "
                 f"{a['seed_group']}")
        if a["rse_components"] != -1 and a["rse_components"] < 1:
            fail(f"rse-components must be -1 (choose from rse-coverage) or "
                 f">= 1, not {a['rse_components']}")
        if not (math.isfinite(a["rse_coverage"])
                and 0 < a["rse_coverage"] <= 1):
            fail(f"rse-coverage must be in (0, 1], not {a['rse_coverage']}")
        if a["incoming_starts"] and a["grow_objective"] != "incoming":
            fail("incoming-starts requires grow-objective = incoming")
        if require_outdir and not a["outdir"]:
            fail("give an output directory (-o)")
        if (require_input and not a["planted"]
                and not (a["xtc_topology"] and a["xtc"])):
            fail("give trajectories and a topology (-f, -s), or planted = 1")
        if a["pair_features"] and a["planted"]:
            fail("pair-features needs coordinates (MD data)")
        if a["pair_features"] and a["backend"] == "schur":
            fail("pair-features cannot be used with backend = schur")
        if a["n_holdout"] is not None and (
            a["discovery_replicas"] is not None
            or a["holdout_replicas"] is not None
        ):
            fail("n-holdout cannot be combined with discovery-replicas "
                 "or holdout-replicas")


Options = Config  # old name, kept for one release
