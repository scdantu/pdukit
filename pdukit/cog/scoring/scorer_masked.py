"""Cores and faces, and a per-residue ledger over the COG phases.

Cores and faces (agreed 12 Oct 2026)
------------------------------------
A face residue sits where two units meet and moves with both (in the watch
picture: the tooth where two gears mesh).  Kept in one unit, it hands that
unit a copy of its neighbour's motion = an extra channel.  So:

  * core score  (drives J): the unit's core C, scored with the face residues
                 F removed from BOTH sides (neither unit nor environment);
  * unit score  (reported): C plus its own faces, scored as usual.

A residue r of unit u is a face when both hold:
  (a) leave-one-out: nu(C) - nu(C without r) >= delta_min  (it carries
      channels the rest of the core does not), and
  (b) two-sided coupling: its predictive coupling to a second unit v exceeds
      the pair null theta and coupling(v) / coupling(u) >= balance_min.
Faces are taken out, the cores re-scored with faces masked, and the test is
repeated until no new face appears (max_rounds).  Each face is attached to
every unit it couples to, with shares.

Ledger
------
One row per residue per phase: unit, unit status, leave-one-out change of
the unit's nu, coupling to the best / second unit and their balance, label
(core / face / unassigned).  Recording only: it never changes the result.
"""

from __future__ import annotations


class MaskedScorer:
    """A FluxScorer whose every score removes `exclude` residues from the
    environment (and from the support).  Everything else is delegated.
    """

    def __init__(self, scorer, exclude):
        """Wrap a scorer so that every score leaves out ``exclude``."""
        self._s = scorer
        self.exclude = sorted(set(int(i) for i in (exclude or [])))

    def __getattr__(self, name):
        """Forward unknown attributes to the wrapped scorer."""
        return getattr(self._s, name)

    def _clean(self, g):
        return [int(i) for i in g if int(i) not in set(self.exclude)]

    def score_batch(self, supports, excludes=None):
        """Score many supports with the excluded residues removed."""
        sup = [self._clean(g) for g in supports]
        return self._s.score_batch(sup, excludes=[self.exclude] * len(sup))

    def score(self, support):
        """Score one support with the excluded residues removed."""
        return self.score_batch([support])[0]

    def __call__(self, support):
        """Score one support (the ``decompose(scorer=...)`` interface)."""
        return self.score(support)


