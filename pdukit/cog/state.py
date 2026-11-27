"""The state of one pipeline run: options, data, workspace, units, results."""

from __future__ import annotations

from pathlib import Path

from pdukit.cog.runinfo import _environment
from pdukit.cog.scoring import scorer as flux_scorer
from pdukit.cog.scoring.scorer import STATISTIC, VERSION
from pdukit.io.results.writer import ResultsWriter


def _settings_dict(args):
    """Return the settings as a plain dict (Options or argparse Namespace)."""
    if hasattr(args, "to_dict"):
        return args.to_dict()
    return {k: v for k, v in vars(args).items() if not k.startswith("_")}


class RunState:
    """Everything the stages of one run read and write.

    Built from the settings; each stage fills in its part (data, replica
    split, workspace, graph, calibration, units) and ``save`` rewrites
    ``results.json`` with what is recorded so far.
    """

    def __init__(self, args):
        """Prepare the output directory and the results record."""
        self.args = args
        flux_scorer.RHO_MIN = float(args.rho_min or 0.0)
        self.out = Path(args.outdir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.writer = ResultsWriter(self.out)
        self.timings = {}
        repo_root = Path(__file__).resolve().parents[2]
        self.results = {
            "version": VERSION,
            "statistic": STATISTIC,
            "args": _settings_dict(args),
            "environment": _environment(repo_root),
            "timings": self.timings,
        }
        ppf = getattr(args, "_ppf", None)
        if ppf:
            self.writer.write_ppf(ppf)
        # data
        self.source = None
        self.data = None
        self.feats = None
        self.segments = None
        self.adj = None
        self.spectra = None
        self.rse_report = None
        self.n_res = 0
        self.pair_all = None
        # replica split and preprocessing
        self.n_rep = 0
        self.disc = []
        self.hold = []
        self.hp = 0
        self.features_window = None  # window the loaded features carry
        self.f_disc = self.s_disc = self.p_disc = None
        # workspace, graph and calibration
        self.ws = None
        self.schur = None
        self.search_adj = None
        self.base_adj = None
        self.rho_all = None
        self.theta_pair = None
        self.engine_probe = None
        self.cal = None
        self.add_gate = None
        self.protect = None
        self.coh = None
        self.grow_coh = False
        # discovery
        self.scorer = None
        self.scorer_eff = None
        self.dec = None
        self.units = []
        self.face_list = []
        # data used by the lag scan (held-out data if there is any)
        self.lag_data = None

    def save(self):
        """Rewrite ``results.json`` with everything recorded so far."""
        self.writer.save(self.results)
