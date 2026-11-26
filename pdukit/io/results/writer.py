"""ResultsWriter: every file a run writes into its output directory.

``results.json`` is rewritten after every stage, so a job that runs out
of time keeps what it finished.  The other files are written when their
stage finishes:

    results.json            everything recorded so far (settings, data,
                            calibration, units, ...), with the
                            ``schema_version`` of its layout
    pdu_out.ppf             the resolved settings, as a parameter file
    calibration.json        thresholds and the size-matched baseline
    bootstrap_rho.npz       bootstrap draws of the pair coupling
    incoming_*.csv / .json  growth, merge, split and port tables
    global_mode_*.csv / ... mode recovery tables and report

The tables are written by the functions in `results.stage_files`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pdukit.io.ppf import PpfWriter
from pdukit.io.results import SCHEMA_VERSION, stage_files
from pdukit.io.results.jsonable import jsonable as _jsonable


class ResultsWriter:
    """Writes the files of one run into the directory ``out``."""

    def __init__(self, out):
        """Remember the output directory (it must exist)."""
        self.out = Path(out)

    def save(self, results):
        """Rewrite ``results.json`` with everything recorded so far."""
        data = {"schema_version": SCHEMA_VERSION, **results}
        with (self.out / "results.json").open("w") as fh:
            json.dump(_jsonable(data), fh, indent=1, allow_nan=True)

    def write_ppf(self, text):
        """Write the resolved settings as ``pdu_out.ppf``."""
        PpfWriter.write(self.out / "pdu_out.ppf", text)

    def write_calibration(self, cal):
        """Write the calibration record as ``calibration.json``."""
        with (self.out / "calibration.json").open("w") as fh:
            json.dump(_jsonable(cal), fh, indent=1, allow_nan=True)

    def write_bootstrap(self, boot):
        """Save the bootstrap draws so port checks can be redone offline."""
        np.savez_compressed(
            self.out / "bootstrap_rho.npz",
            rho=np.asarray(boot, dtype=np.float32),
        )

    def write_incoming_tables(self, result):
        """Growth, candidate and proposal tables of the incoming growth."""
        stage_files.write_incoming_tables(self.out, result)

    def write_merge(self, merge):
        """Merge record and its tables."""
        stage_files.write_merge(self.out, merge)

    def write_split(self, result):
        """Split record and its tables."""
        stage_files.write_split(self.out, result)

    def write_classes(self, classes):
        """Core, port and bridge classes of the residues."""
        stage_files.write_classes(self.out, classes)

    def write_mode_recovery(self, report):
        """Global-mode recovery report and tables."""
        stage_files.write_mode_recovery(self.out, report)
