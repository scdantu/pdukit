"""The protein as COG sees it: residues, units and their labels.

``Protein`` is the object people hold: it starts as a bare chain read from a
topology and gets its COG properties (unit, label, confidence, coupling)
filled in when a run finishes.  This package imports nothing from the
pipeline, only from `pdukit.numerics` (and, lazily, MDAnalysis).
"""

from pdukit.model.ensemble import Ensemble
from pdukit.model.enums import (
    Backend,
    Phase,
    ResidueClass,
    SearchGraph,
    Seeding,
    ShiftMode,
    SplitMethod,
    SplitTest,
)
from pdukit.model.protein import Protein, Residue, Unit

__all__ = [
    "Backend", "Ensemble", "Phase", "Protein", "Residue", "ResidueClass",
    "SearchGraph", "Seeding", "ShiftMode", "SplitMethod", "SplitTest", "Unit",
]
