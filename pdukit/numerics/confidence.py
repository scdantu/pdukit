"""Residue and unit confidence scores: RCS and UCS (0-100).

The growth stage starts many independent seeds.  Each seed that ends with an
eligible support is a re-detection of some unit.  A seed *supports* a final
unit when at least ``min_overlap`` of its residues lie inside that unit
(core + ports); the same overlap rule the merge uses.

* **RCS** (residue confidence score): the percentage of the seeds supporting
  a unit that contain the residue.  100 = every supporting seed found it,
  low = only a few did.  Residues in no unit score 0.
* **UCS** (unit confidence score): the median RCS of the unit's core
  residues (ports are scored too, but do not define the unit).

Both are reproducibility fractions, not probabilities of being "true": they
say how stable the result is under the random choice of seed.  ``n_seeds``
is the number of supporting seeds; with fewer than ``LOW_EVIDENCE`` the
score is flagged because a fraction of one or two seeds says little.
"""

import numpy as np

LOW_EVIDENCE = 3
BANDS = (
    (90, "very high"),
    (70, "high"),
    (50, "low"),
    (0, "very low"),
)


def band(score):
    """Return the pLDDT-style band name for a 0-100 score."""
    for floor, name in BANDS:
        if score >= floor:
            return name
    return "very low"


def _units(results):
    """Return ``{unit: (core set, extended set)}`` for the final units."""
    ports = {
        u["unit"]: u
        for u in (results.get("incoming_ports") or {}).get("units", [])
    }
    out = {}
    for u in results.get("units_final") or results.get("units_split") or []:
        p = ports.get(u["unit"], {})
        core = set(p.get("core", u["residues"]))
        out[u["unit"]] = (core, core | set(p.get("ports", [])))
    return out


def seed_assignment(results, min_overlap=None):
    """Return ``{run index: unit or None}`` for every growth seed.

    ``None`` for seeds that ended below the minimum size (``eligible`` is
    false) or whose support overlaps no unit by at least ``min_overlap``.
    """
    if min_overlap is None:
        min_overlap = float(
            (results.get("args") or {}).get("merge_overlap", 0.5)
        )
    units = _units(results)
    runs = (results.get("incoming_growth") or {}).get("runs") or []
    out = {}
    for idx, run in enumerate(runs):
        out[idx] = None
        if not run.get("eligible"):
            continue
        support = set(run["final"])
        best, best_frac = None, 0.0
        for k, (_, ext) in units.items():
            frac = len(support & ext) / len(support)
            if frac > best_frac:
                best, best_frac = k, frac
        if best is not None and best_frac >= min_overlap:
            out[idx] = best
    return out


def assign_seeds(results, min_overlap=None):
    """Return ``{unit: [seed final supports]}`` for the supporting seeds."""
    runs = (results.get("incoming_growth") or {}).get("runs") or []
    out = {k: [] for k in _units(results)}
    for idx, unit in seed_assignment(results, min_overlap).items():
        if unit is not None:
            out[unit].append(sorted(runs[idx]["final"]))
    return out


def residue_scores(results, min_overlap=None):
    """Return ``{residue: {unit, rcs, n_seeds, role}}`` for unit residues."""
    units = _units(results)
    seeds = assign_seeds(results, min_overlap)
    out = {}
    for k, (core, ext) in units.items():
        n = len(seeds[k])
        for i in sorted(ext):
            hits = sum(1 for s in seeds[k] if i in s)
            out[i] = {
                "unit": k,
                "rcs": 100.0 * hits / n if n else 0.0,
                "n_seeds": n,
                "role": "core" if i in core else "port",
            }
    return out


def unit_scores(results, min_overlap=None):
    """Return one dict per final unit: ``unit, size, ucs, n_seeds, band``."""
    res = residue_scores(results, min_overlap)
    rows = []
    for k, (core, ext) in _units(results).items():
        vals = [res[i]["rcs"] for i in core if i in res]
        n = res[next(iter(core))]["n_seeds"] if core else 0
        ucs = float(np.median(vals)) if vals else 0.0
        rows.append(
            {
                "unit": k,
                "size": len(core),
                "n_ports": len(ext) - len(core),
                "ucs": ucs,
                "band": band(ucs),
                "n_seeds": n,
                "low_evidence": n < LOW_EVIDENCE,
            }
        )
    return rows


def rcs_vector(results, n_res, min_overlap=None):
    """Return the RCS of every residue index (0 outside units)."""
    vec = np.zeros(n_res)
    for i, d in residue_scores(results, min_overlap).items():
        if 0 <= i < n_res:
            vec[i] = d["rcs"]
    return vec
