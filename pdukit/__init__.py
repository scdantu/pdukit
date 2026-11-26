"""pdukit -- COG protein dynamical unit (PDU) detection from MD.

config/    settings: the schema and `Config`, one object with every setting
model/     `Protein`, `Residue`, `Unit` and `Ensemble` (what the user holds)
runner.py  `Cog`: run COG on an ensemble
result.py  `Result`: tables, figures, 3D view of a finished run
io/        reading and writing files: trajectories, mdCATH, ppf, results
numerics/  pure maths: geometry, the reference kernel, residue features
cog/       the phases of COG: pipeline, stages, scoring, calibration, units
analysis/  tables, comparison, summary
numerics/  (also) the confidence scores RCS/UCS
viz/       figures and PyMOL scripts (pdu plot)
cli/       the ``pdu`` command, one module per task
bench/     planted systems with a known answer
runlog.py  phase-tagged progress log
hpc/       example Slurm scripts
params/    parameter files for the example runs
tests/     the test suite

Four objects are enough for a notebook::

    import pdukit as pdu
    ens = pdu.Ensemble(["rep1.xtc", "rep2.xtc"], "top.pdb")
    cfg = pdu.Config.from_ppf("params.ppf", lag=20)
    result = pdu.Cog(cfg).run(ens, out="runs/x")
    result.protein, result.table("units"), result.view()
"""

from importlib.metadata import PackageNotFoundError, version

from pdukit.config.config import Config
from pdukit.errors import ConfigError, PduError, PpfError
from pdukit.model import Ensemble, Protein, Residue, ResidueClass, Unit
from pdukit.result import Result
from pdukit.runner import Cog

try:  # written by setuptools-scm when the package is installed
    from pdukit._version import version as __version__
except ImportError:
    try:
        __version__ = version("pdukit")
    except PackageNotFoundError:  # running from a checkout, not installed
        __version__ = "unknown"


def demo(folder=None, **kwargs):
    """Return the synthetic 4-unit `Ensemble` (needs MDAnalysis)."""
    return Ensemble.demo(folder, **kwargs)


__all__ = [
    "Cog", "Config", "ConfigError", "Ensemble", "PduError", "PpfError",
    "Protein", "Residue", "ResidueClass", "Result", "Unit", "demo",
]
