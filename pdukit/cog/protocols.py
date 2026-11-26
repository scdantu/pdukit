"""The three interfaces that have more than one implementation.

Nothing else in pdukit has an interface: a class gets one only when
several versions of it exist and callers must not care which they hold.

``Stage``      the nine steps of a run (``pdukit.cog.stages``)
``Workspace``  scores one candidate unit as a flux spectrum: ``FluxWorkspace``,
               ``SchurWorkspace`` (whole environment) and
               ``PairFluxWorkspace`` (pair-owned features)
``Scorer``     scores candidate units for the search: ``FluxScorer``,
               ``MaskedScorer`` (leaves residues out) and ``CohesionScorer``

These are structural types: a class satisfies one by having the methods, it
does not inherit from it.  They add no behaviour; ``tests/test_protocols.py``
checks that every implementation still fits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from pdukit.cog.state import RunState


class Stage(Protocol):
    """One step of a run: reads and writes the run state."""

    code: str

    def run(self, state: RunState) -> None:
        """Advance ``state`` by this stage."""


class Workspace(Protocol):
    """Flux spectrum of one candidate unit against its environment."""

    def score(
        self,
        support: Any,
        unit_dim: int = ...,
        env_dim: int | None = ...,
        null_thresholds: Any = ...,
    ) -> Any:
        """Return the flux spectrum of ``support``.

        ``SchurWorkspace`` accepts ``env_dim`` and ignores it (its
        environment is whole).
        """


class Scorer(Protocol):
    """Scores candidate supports; each result carries a ``cost``."""

    def __call__(self, support: Any, /) -> Any:
        """Score one support."""

    def score_batch(self, supports: Any, /) -> Any:
        """Score many supports at once, one result each, in order."""
