#!/usr/bin/env python3
"""COG end-to-end run: calibrate, discover, check cohesion, verify.

    # planted benchmark (known answer; minutes on one CPU)
    python -m pdukit.lib.pdu.cog2_run --planted --outdir OUT

    # lysozyme, three replicas: discover on R1+R2, verify on R3
    python -m pdukit.lib.pdu.cog2_run --xtc-topology GA.tpr \
        --xtc R0001.xtc R0002.xtc R0003.xtc --outdir OUT --device cuda

The pipeline is `cog2_pipeline.Cog2Run` (one method per stage; see its module
docstring); the command line is `cog2_cli`.  This module is the entry point
and keeps the names that earlier code imported from it.

Outputs in --outdir: results.json and the CSV tables (units.csv, ...).
"""

from __future__ import annotations

import sys

from .cog2_cli import parse_args
from .cog2_data import (
    center_replicas,
    coarse_sizes,
    fit_coordinates,
    highpass_replicas,
)
from .cog2_pipeline import Cog2Run
from .cog2_scoring import choose_window

__all__ = [
    "center_replicas",
    "choose_window",
    "coarse_sizes",
    "fit_coordinates",
    "highpass_replicas",
    "main",
]


def main(argv=None):
    """Parse the command line and run the pipeline; return the exit code."""
    args = parse_args(argv, description=__doc__.split("\n\n")[0])
    return Cog2Run(args).run()


if __name__ == "__main__":
    sys.exit(main())
