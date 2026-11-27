"""Every residue's affinity to every unit.

For residue r and unit core S_k the change when r is in rather than out:

    r outside S_k:  X(S_k + r) - X(S_k)
    r inside  S_k:  X(S_k)     - X(S_k - r)

for X = nu_in, nu_out (COG's channel counts; a NEGATIVE change in nu_in
means r belongs: the unit hears less from outside with it) and X =
cohesion (predictive cohesion of the set; positive = r belongs).  The rho
ratios of the port step (drives, driven) are added for every unit.

The table answers: is a residue's preference spread over many units
(0.2, 0.2, 0.2, ...) or concentrated on one, and does that change with the
lag?  It costs about residues x units extra scores.
"""

from __future__ import annotations

import numpy as np

from pdukit.numerics.cohesion import cohesion


def _scores(scorer, supports):
    from pdukit.cog.units.ports import _score

    return _score(scorer, supports)


def residue_affinity(rho, units, scorer=None):
    """Return (rows, summary): one row per residue x unit, one per residue.

    ``rho`` is the pair coupling (rho[i, j]: j predicts i), ``units`` the
    unit cores, ``scorer`` the run's scorer (None: cohesion and rho ratios
    only).
    """
    R = np.array(rho, float)
    N = R.shape[0]
    Rn = R.copy()
    np.fill_diagonal(Rn, np.nan)
    cores = [sorted({int(x) for x in u}) for u in units]
    pairs = []  # (r, k, with, without)
    for k, S in enumerate(cores):
        Sset = set(S)
        for r in range(N):
            if r in Sset:
                if len(S) - 1 < 2:
                    continue
                pairs.append((r, k, S, [i for i in S if i != r]))
            else:
                pairs.append((r, k, sorted(S + [r]), S))
    supports, index = [], {}
    for _, _, w, wo in pairs:
        for s in (w, wo):
            key = tuple(s)
            if key not in index:
                index[key] = len(supports)
                supports.append(list(s))
    meas = _scores(scorer, supports)
    coh = [cohesion(R, s)["cohesion"] for s in supports]
    rows = []
    for r, k, w, wo in pairs:
        a, b = index[tuple(w)], index[tuple(wo)]
        S = cores[k]
        rest = [i for i in range(N) if i not in S and i != r]
        drives = np.nanmean(Rn[S, r]) / max(np.nanmean(Rn[rest, r]), 1e-12)
        driven = np.nanmean(Rn[r, S]) / max(np.nanmean(Rn[r, rest]), 1e-12)
        row = dict(
            residue=r,
            unit=k,
            member=r in S,
            drives=float(drives),
            driven=float(driven),
            delta_cohesion=float(coh[a] - coh[b]),
        )
        if meas[a] is not None and meas[b] is not None:
            row["delta_nu_in"] = meas[a]["nu_in"] - meas[b]["nu_in"]
            row["delta_nu_out"] = meas[a]["nu_out"] - meas[b]["nu_out"]
        rows.append(row)
    return rows, _summary(rows, N, len(cores))


def _spread(aff):
    """Best and second unit index, and the effective number of units.

    effective units = exp(entropy) of the positive affinities normalised
    to sum 1: 1 = one unit, n = spread evenly over n units, 0 = none.
    """
    order = np.argsort(-aff)
    pos = np.clip(aff, 0, None)
    eff = 0.0
    if pos.sum() > 0:
        p = pos[pos > 0] / pos.sum()
        eff = float(np.exp(-(p * np.log(p)).sum()))
    return order, eff


def _summary(rows, N, n_units):
    """Per residue: best and second unit by nu_in and by cohesion.

    Affinity is -delta_nu_in (the unit hears less from outside with the
    residue) and delta_cohesion (the set is more cohesive with it).
    """
    has_nu = bool(rows) and "delta_nu_in" in rows[0]
    measures = (("nu", "delta_nu_in", -1.0),) if has_nu else ()
    measures += (("cohesion", "delta_cohesion", 1.0),)
    out = []
    for r in range(N):
        mine = [x for x in rows if x["residue"] == r]
        row = dict(residue=r,
                   member_of=[int(x["unit"]) for x in mine if x["member"]])
        for tag, key, sign in measures:
            if not mine:
                continue
            aff = sign * np.array([x[key] for x in mine], float)
            order, eff = _spread(aff)
            row.update({
                f"{tag}_best_unit": int(mine[order[0]]["unit"]),
                f"{tag}_best": float(aff[order[0]]),
                f"{tag}_second_unit": (
                    int(mine[order[1]]["unit"]) if len(order) > 1 else None
                ),
                f"{tag}_second": (
                    float(aff[order[1]]) if len(order) > 1 else None
                ),
                f"{tag}_units_positive": int((aff > 0).sum()),
                f"{tag}_effective_units": eff,
            })
        out.append(row)
    return out
