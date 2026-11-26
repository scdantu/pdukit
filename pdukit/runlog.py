"""Phase-tagged progress log shared by the whole pipeline.

Every line is ``[ elapsed] PHS  message``: the elapsed seconds, a
three-letter code for the pipeline phase that is running, then a short
message.  Warnings use ``PHS !!``.  Per-step traces (one line per growth
step, per split node, ...) are ``debug`` and hidden unless
``--log-level debug`` is given.

The current phase is global state set with :func:`set_phase`, so code deep
in a phase (growth, split, ports) is tagged without being told which phase
it belongs to.  A message may still override it with ``phase=``.

Every line also goes to the run's log file (``<outdir>/pdu.log``) once
:func:`set_file` has opened it, so a run in the background leaves a log.
"""

import time
from pathlib import Path

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
_STATE = {"phase": "INI", "level": LEVELS["info"], "file": None}


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


def set_file(path):
    """Also write every log line to ``path`` (replaced if it exists).

    Does nothing when the folder of ``path`` does not exist.  Returns the
    path written to, or None.
    """
    close_file()
    path = Path(path)
    if not path.parent.is_dir():
        return None
    _STATE["file"] = open(path, "w", encoding="utf-8")  # noqa: SIM115
    return path


def close_file():
    """Stop writing the log file (the screen log goes on)."""
    fh = _STATE["file"]
    _STATE["file"] = None
    if fh is not None:
        fh.close()


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
    line = f"[{stamp:7.1f}] {code} {mark} {msg.strip()}"
    print(line, flush=True)
    fh = _STATE["file"]
    if fh is not None:
        fh.write(line + "\n")
        fh.flush()


def debug(msg, phase=None):
    """Log a per-step trace shown only with ``--log-level debug``."""
    log(msg, level="debug", phase=phase)


log.debug = debug


class Progress:
    """Log ``label: done/total`` each time another ``step`` percent is done.

    Called as ``progress(done, total)``; a new pass starts when ``done``
    goes back to the start, and its lines say ``pass 2``, ``pass 3``, ...
    """

    def __init__(self, label, step=10):
        """Report ``label`` every ``step`` percent."""
        self.label, self.step = label, max(1, int(step))
        self.passes, self._last, self._done = 0, -1, None

    def __call__(self, done, total):
        """Record that ``done`` of ``total`` items are finished."""
        if self._done is None or done <= self._done:
            self.passes += 1
            self._last = -1
        self._done = done
        pct = 100 * done // max(total, 1)
        mark = pct // self.step
        if mark > self._last:
            self._last = mark
            tag = f" (pass {self.passes})" if self.passes > 1 else ""
            log(f"{self.label}{tag}: {done}/{total} ({pct}%)")
