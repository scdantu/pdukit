"""Every implementation still fits the interface it is meant to fit."""

import inspect

import pytest

from pdukit.cog.pipeline import STAGES
from pdukit.cog.protocols import Scorer, Stage, Workspace
from pdukit.cog.scoring.pair import PairFluxWorkspace
from pdukit.cog.scoring.scorer import FluxScorer
from pdukit.cog.scoring.scorer_masked import MaskedScorer
from pdukit.cog.scoring.workspace import FluxWorkspace
from pdukit.cog.scoring.workspace_schur import SchurWorkspace
from pdukit.cog.units.cohesion import CohesionScorer


def _methods(protocol):
    return [
        name
        for name, member in vars(protocol).items()
        if callable(member) and not name.startswith("_") or name == "__call__"
    ]


def _fits(cls, protocol, name):
    """cls.name accepts every argument that protocol.name takes."""
    want = dict(inspect.signature(getattr(protocol, name)).parameters)
    have = dict(inspect.signature(getattr(cls, name)).parameters)
    want.pop("self")
    have.pop("self")
    kw = [p for p, v in want.items() if v.kind is not v.POSITIONAL_ONLY]
    pos = [p for p, v in want.items() if v.kind is v.POSITIONAL_ONLY]
    missing = [p for p in kw if p not in have]
    assert not missing, f"{cls.__name__}.{name} lacks {missing}"
    assert kw == [p for p in have if p in kw], (
        f"{cls.__name__}.{name} takes {kw} in a different order"
    )
    room = [
        v
        for v in have.values()
        if v.kind in (v.POSITIONAL_ONLY, v.POSITIONAL_OR_KEYWORD)
    ]
    assert len(room) >= len(pos), f"{cls.__name__}.{name} takes too few"


@pytest.mark.parametrize("stage", STAGES, ids=lambda c: c.__name__)
def test_stage(stage):
    assert isinstance(stage.code, str) and len(stage.code) == 3
    _fits(stage, Stage, "run")


def test_stage_codes_are_unique():
    assert len({s.code for s in STAGES}) == len(STAGES)


@pytest.mark.parametrize(
    "cls",
    [FluxWorkspace, SchurWorkspace, PairFluxWorkspace],
    ids=lambda c: c.__name__,
)
def test_workspace(cls):
    for name in _methods(Workspace):
        _fits(cls, Workspace, name)


@pytest.mark.parametrize(
    "cls", [FluxScorer, MaskedScorer, CohesionScorer], ids=lambda c: c.__name__
)
def test_scorer(cls):
    for name in _methods(Scorer):
        _fits(cls, Scorer, name)
