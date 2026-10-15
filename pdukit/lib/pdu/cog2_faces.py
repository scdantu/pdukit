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

import numpy as np


class MaskedScorer:
    """A Cog2Scorer whose every score removes `exclude` residues from the
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


def coupling_matrix(rho, units, exclude_self=True):
    """c[r, a] = strongest predictive link (either direction) between
    residue r and any member of unit a (r itself excluded).
    """
    rho = np.asarray(rho, float)
    sym = np.maximum(rho, rho.T)
    np.fill_diagonal(sym, 0.0)
    N = sym.shape[0]
    c = np.zeros((N, len(units)))
    for a, u in enumerate(units):
        if len(u):
            c[:, a] = sym[:, list(u)].max(axis=1)
    return c


def leave_one_out(scorer, units, min_size=3, key="nu"):
    """delta[r] = score(unit) - score(unit without r), score = nu or cost,
    for every r in a unit of size > min_size.  One batched call per unit.
    delta < 0: the residue makes its unit more closed (a core residue);
    delta >= 0: keeping it does not lower the unit's cost (face signal).
    """
    out = {}
    for a, u in enumerate(units):
        u = sorted(int(x) for x in u)
        if len(u) <= min_size:
            continue
        res = scorer.score_batch([u] + [[x for x in u if x != r] for r in u])
        v0 = float(res[0][key])
        for r, rr in zip(u, res[1:]):
            out[r] = v0 - float(rr[key])
    return out


def find_faces(
    scorer,
    units,
    rho,
    theta,
    *,
    delta_min=0.0,
    balance_min=0.8,
    min_size=3,
    max_rounds=3,
    log=print,
):
    """Face rule (12 Oct): keeping the residue does NOT lower its
    unit's cost (cost(unit) - cost(unit without it) >= delta_min, default
    0), AND it couples to a second unit beyond the pair null with a high
    balance (coupling to the second unit / to its own >= balance_min,
    default 0.8).  Faces are then left out of the core score (J).
    """
    units = [sorted(int(x) for x in u) for u in units]
    faces = {}  # residue -> home unit index
    history = []
    for rnd in range(int(max_rounds)):
        F = set(faces)
        cores = [[x for x in u if x not in F] for u in units]
        ms = MaskedScorer(scorer, F)
        loo = leave_one_out(ms, cores, min_size=min_size, key="cost")
        c = coupling_matrix(rho, cores)
        new = []
        for a, u in enumerate(cores):
            for r in u:
                d = loo.get(r)
                if d is None or d < delta_min:
                    continue
                own = c[r, a]
                others = [(c[r, b], b) for b in range(len(cores)) if b != a]
                if not others:
                    continue
                c2, b = max(others)
                if c2 > theta and own > 0 and c2 / own >= balance_min:
                    new.append((r, a, b, d, c2 / own))
        history.append(
            {
                "round": rnd + 1,
                "new_faces": [
                    {
                        "residue": r,
                        "home": a,
                        "other": b,
                        "loo_delta": round(d, 3),
                        "balance": round(bal, 3),
                    }
                    for r, a, b, d, bal in new
                ],
            }
        )
        if not new:
            break
        for r, a, *_ in new:
            faces[r] = a
        log(
            f"faces round {rnd + 1}: {len(new)} new face residues "
            f"({len(faces)} in total)"
        )
    F = set(faces)
    cores = [[x for x in u if x not in F] for u in units]
    c = coupling_matrix(rho, cores)
    face_list = []
    for r, a in sorted(faces.items()):
        tops = [
            b
            for b in range(len(cores))
            if b == a
            or (
                c[r, b] > theta
                and c[r, a] > 0
                and c[r, b] / c[r, a] >= balance_min
            )
        ]
        w = np.array([c[r, b] for b in tops])
        face_list.append(
            {
                "residue": int(r),
                "units": [int(b) for b in tops],
                "shares": [
                    round(float(x), 3) for x in w / max(w.sum(), 1e-12)
                ],
                "home": int(a),
            }
        )
    return {
        "cores": cores,
        "faces": face_list,
        "face_set": sorted(F),
        "history": history,
        "delta_min": delta_min,
        "balance_min": balance_min,
        "theta": float(theta),
    }


def ledger_rows(
    phase, N, units, statuses, rho, theta, loo=None, faces=None, offset=0
):
    """Ledger rows for one phase."""
    c = coupling_matrix(rho, units) if len(units) else np.zeros((N, 0))
    owner = {}
    for a, u in enumerate(units):
        for x in u:
            owner.setdefault(int(x), a)
    fset = {f["residue"]: f for f in (faces or [])}
    rows = []
    for r in range(N):
        order = np.argsort(c[r])[::-1] if c.shape[1] else []
        b1 = int(order[0]) if len(order) else -1
        b2 = int(order[1]) if len(order) > 1 else -1
        c1 = float(c[r, b1]) if b1 >= 0 else 0.0
        c2 = float(c[r, b2]) if b2 >= 0 else 0.0
        a = owner.get(r, -1)
        if r in fset:
            label = "face"
        elif a < 0:
            label = "unassigned"
        else:
            label = "core" if statuses[a] != "unresolved" else "unresolved"
        rows.append(
            {
                "phase": phase,
                "residue": r + offset,
                "index": r,
                "unit": a,
                "unit_status": statuses[a] if a >= 0 else "",
                "label": label,
                "face_units": " ".join(map(str, fset[r]["units"]))
                if r in fset
                else "",
                "loo_delta_cost": (
                    round(loo[r], 3) if loo and r in loo else ""
                ),
                "best_unit": b1,
                "best_coupling": round(c1, 3),
                "second_unit": b2,
                "second_coupling": round(c2, 3),
                "balance": round(c2 / c1, 3) if c1 > 0 else "",
                "couples_beyond_null": int(np.sum(c[r] > theta))
                if c.shape[1]
                else 0,
            }
        )
    return rows


def ledger_summary(rows, N):
    """Per residue across phases: units visited, moves, final label."""
    by = {}
    for r in rows:
        by.setdefault(r["index"], []).append(r)
    out = []
    for i in range(N):
        rs = by.get(i, [])
        units = [x["unit"] for x in rs]
        moves = sum(1 for p, q in zip(units, units[1:]) if p != q)
        out.append(
            {
                "index": i,
                "phases": [x["phase"] for x in rs],
                "units": units,
                "labels": [x["label"] for x in rs],
                "moves": moves,
                "final_label": rs[-1]["label"] if rs else "",
            }
        )
    return out
