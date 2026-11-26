"""Phase-tagged progress log shared by the whole pipeline.

Every line is ``[ elapsed] PHS  message``: the elapsed seconds, a
three-letter code for the pipeline phase that is running, then a short
message.  Warnings use ``PHS !!``.  Per-step traces (one line per growth
step, per split node, ...) are ``debug`` and hidden unless
``--log-level debug`` is given.

The current phase is global state set with :func:`set_phase`, so code deep
in a phase (growth, split, ports) is tagged without being told which phase
it belongs to.  A message may still override it with ``phase=``.
"""

import time

PHASES = {
    "INI": "initialisation",
    "DAT": "data loading",
    "REP": "replica split and overlap",
    "WIN": "high-pass window scan",
    "PRE": "feature preprocessing",
    "WSP": "lagged-pair workspace",
    "GRA": "predictive graph",
    "CAL": "calibration",
    "GAT": "gate",
    "COH": "cohesion scoring and split",
    "GRO": "growth",
    "MRG": "merge of grown candidates",
    "SPL": "recursive split on cohesion",
    "POR": "ports and bridges",
    "REF": "refinement",
    "SCR": "fixed-set scoring",
    "FAC": "faces",
    "UNI": "unit report",
    "RES": "resolve",
    "LED": "ledger",
    "STB": "stability",
    "DEF": "PDU definition",
    "VER": "verification",
    "CMP": "comparison with COG 1",
    "MOD": "global-mode recovery",
    "OUT": "output",
}

LEVELS = {"quiet": 30, "info": 20, "debug": 10}
_T0 = time.perf_counter()
_STATE = {"phase": "INI", "level": LEVELS["info"]}


def set_phase(code):
    """Make ``code`` (a key of :data:`PHASES`) the current phase."""
    if code not in PHASES:
        raise KeyError(f"unknown log phase {code!r}")
    _STATE["phase"] = code


def get_phase():
    """Return the current phase code."""
    return _STATE["phase"]


def set_level(name):
    """Choose verbosity: ``quiet`` (warnings), ``info`` or ``debug``."""
    _STATE["level"] = LEVELS[name]


def log(msg, level="info", phase=None):
    """Print one phase-tagged, time-stamped line (flushed)."""
    msg = str(msg)
    num = LEVELS[level] if level in LEVELS else LEVELS["info"]
    mark = " "
    if msg.startswith("!!"):
        msg, num, mark = msg[2:], 30, "!"
    if num < _STATE["level"]:
        return
    code = phase or _STATE["phase"]
    stamp = time.perf_counter() - _T0
    print(f"[{stamp:7.1f}] {code} {mark} {msg.strip()}", flush=True)


def debug(msg, phase=None):
    """Log a per-step trace shown only with ``--log-level debug``."""
    log(msg, level="debug", phase=phase)


log.debug = debug
