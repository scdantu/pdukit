"""The runner: the stages of a run, in order, on one run state."""

from __future__ import annotations

from pdukit.cog.scoring.scorer import VERSION
from pdukit.cog.stages.discover import DiscoverUnitsStage, ReportModesStage
from pdukit.cog.stages.prepare import (
    LoadFeaturesStage,
    PreprocessStage,
    ScanWindowStage,
    SplitReplicasStage,
)
from pdukit.cog.stages.search import (
    BuildGraphStage,
    BuildWorkspaceStage,
    CalibrateStage,
)
from pdukit.runlog import PHASES, log, set_level, set_phase

STAGES = (
    LoadFeaturesStage,
    SplitReplicasStage,
    ScanWindowStage,
    PreprocessStage,
    BuildWorkspaceStage,
    BuildGraphStage,
    CalibrateStage,
    DiscoverUnitsStage,
    ReportModesStage,
)


class Pipeline:
    """Runs the stages in order and writes ``results.json`` as it goes.

    ``Pipeline(RunState(settings)).run()`` is a whole run.  ``stages`` is
    a sequence of stage classes (default: all nine, in the order of
    ``STAGES``).
    """

    def __init__(self, state, stages=STAGES):
        """Hold the run state and make one instance of each stage."""
        self.state = state
        self.stages = [stage() for stage in stages]

    @staticmethod
    def _stage(code, func):
        """Run one pipeline stage with its log phase code set."""
        assert code in PHASES
        set_phase(code)
        return func()

    def run(self):
        """Run every stage; return the process exit code."""
        state = self.state
        args = state.args
        if args.score_sets or args.grow_objective != "incoming":
            raise SystemExit(
                "pdu run: only grow-objective=incoming is available; the "
                "channels and cohesion modes and --score-sets are archived "
                "(git tag archive/pre-incoming-only)"
            )
        set_level(getattr(args, "log_level", "info"))
        set_phase("INI")
        log(f"{VERSION} run -> {state.out}")
        state.save()
        for stage in self.stages:
            self._stage(stage.code, lambda: stage.run(state))
            if stage.code == "CAL" and args.calibrate_only:
                log("--calibrate-only: stopping after calibration")
                return 0
        state.save()
        return 0
