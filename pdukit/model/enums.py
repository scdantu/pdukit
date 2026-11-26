"""Names used as values: enums instead of loose strings."""

from __future__ import annotations

from enum import Enum


class Choice(str, Enum):
    """A fixed list of words; compares equal to the plain word."""

    def __str__(self):
        """Return the plain word."""
        return self.value


class ResidueClass(Choice):
    """What a residue is to the units (``str`` mixin: compares to text)."""

    CORE = "core"        # belongs to one unit's core
    PORT = "port"        # attached to one unit, couples to the outside
    BRIDGE = "bridge"    # sits between two or more units
    SHARED = "shared"    # in the core of more than one unit
    LOOSE = "loose"      # in no unit
    UNASSIGNED = "unassigned"  # before a run


class LogLevel(Choice):
    """How much a run prints (``log-level``)."""

    QUIET = "quiet"
    INFO = "info"
    DEBUG = "debug"


class ShiftMode(Choice):
    """How time-shift nulls are made (``shift-mode``)."""

    AUTO = "auto"
    GLOBAL = "global"
    WITHIN = "within"
    CROSS = "cross"


class Backend(Choice):
    """The scoring engine (``backend``)."""

    FLUX = "flux"
    SCHUR = "schur"


class Seeding(Choice):
    """How growth seeds are chosen (``seeding``)."""

    RANDOM = "random"
    SPREAD = "spread"


class SearchGraph(Choice):
    """The graph growth searches on (``search-graph``)."""

    PREDICTIVE = "predictive"
    CONTACT = "contact"
    UNION = "union"


class SplitMethod(Choice):
    """How a merged unit is cut (``split-method``)."""

    RECURSIVE_SPECTRAL = "recursive-spectral"
    DIRECTED_MODULARITY = "directed-modularity"
    MARKOV_STABILITY = "markov-stability"


class SplitTest(Choice):
    """Which test accepts a cut (``split-test``)."""

    BOTH = "both"
    GRAPH_NULL = "graph-null"
    BOOTSTRAP = "bootstrap"
    NONE = "none"


class Phase(Choice):
    """Phase codes in the run log (descriptions: ``runlog.PHASES``)."""

    INI = "INI"
    DAT = "DAT"
    REP = "REP"
    WIN = "WIN"
    PRE = "PRE"
    WSP = "WSP"
    GRA = "GRA"
    CAL = "CAL"
    GAT = "GAT"
    COH = "COH"
    GRO = "GRO"
    MRG = "MRG"
    SPL = "SPL"
    POR = "POR"
    REF = "REF"
    SCR = "SCR"
    FAC = "FAC"
    UNI = "UNI"
    RES = "RES"
    LED = "LED"
    STB = "STB"
    DEF = "DEF"
    VER = "VER"
    CMP = "CMP"
    MOD = "MOD"
    OUT = "OUT"
