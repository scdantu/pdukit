"""Cog: run COG on an ensemble and get a Result.

    cfg = Config.from_ppf("t4l.ppf", lag=20)
    result = Cog(cfg).run(ensemble, out="runs/t4l")

The settings are copied at the start of ``run``, so the same ``Cog`` can be
run again with other changes (``cog.run(ens, lag=30)``).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from pdukit.config.config import Config
from pdukit.errors import PduError


class Cog:
    """The COG runner: settings in, `Result` out."""

    def __init__(self, config=None):
        """Use ``config`` (default: all defaults)."""
        self.config = config if config is not None else Config.from_ppf()

    def run(self, ensemble, out=None, **changes):
        """Run the pipeline on ``ensemble``; return the `Result`.

        ``out`` is the run folder; if omitted a temporary one is used (the
        files then vanish with the session unless you ``result.save(path)``).
        ``changes`` are settings for this run only (``lag=30``).
        """
        from pdukit.cog.pipeline import Pipeline
        from pdukit.cog.state import RunState
        from pdukit.result import Result

        cfg = self.config.copy()
        cfg.update(**ensemble.settings(), **changes)
        out = (
            Path(out) if out else Path(tempfile.mkdtemp(prefix="pdukit_run_"))
        )
        cfg.outdir = str(out)
        cfg.validate(require_outdir=True, require_input=True)
        code = Pipeline(RunState(cfg)).run()
        if code:
            raise PduError(f"the run stopped with exit code {code}")
        return Result.load(out, topology=ensemble.topology)
