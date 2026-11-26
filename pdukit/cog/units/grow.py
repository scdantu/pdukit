"""Strict incoming-only candidate growth; no J subtraction or output penalty.

Every accepted addition must decrease nu_in. Existing theta, features and
soft-count definitions are supplied by FluxScorer without modification.
"""

from __future__ import annotations

import math

import numpy as np


def _measurement(result):
    """Keep the two directions separate and omit the legacy cost/J."""
    row = {}
    for key in ("nu_in", "nu_out", "theta_in", "theta_out"):
        value = float(result[key])
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"invalid {key}: {value}")
        row[key] = value
    for direction in ("in", "out"):
        rho = [float(v) for v in result[f"rho_{direction}"]]
        if any(not math.isfinite(v) or v < 0 for v in rho):
            raise ValueError(f"invalid rho_{direction}")
        row[f"rho_{direction}"] = rho
        power = np.maximum(np.square(rho) - row[f"theta_{direction}"] ** 2, 0)
        total = float(power.sum())
        row[f"keff_{direction}"] = (
            total * total / float(np.square(power).sum())
            if total > 0
            else None
        )
    return row


def grow_incoming(
    scorer,
    adjacency,
    starts,
    *,
    max_size=120,
    min_size=3,
    min_gain=0.0,
    start_min_gain=0.0,
    numerical_tol=1e-12,
    on_log=None,
):
    """Grow each supplied start by the best strictly improving addition.

    `start_min_gain` applies to the first addition only: a start must lower
    nu_in by more than it to grow at all (noise-level first steps stop).
    `starts` is a sequence of nonempty residue lists, normally singletons.
    All graph-frontier additions are evaluated, with residue index breaking
    exact ties. No forced bootstrap to min_size and no uphill/flat moves.
    Short stopped candidates remain in `runs`, but not in `units`.
    Identical final supports are deduplicated; overlaps are preserved.
    """
    graph = np.asarray(adjacency, dtype=bool)
    if graph.ndim != 2 or graph.shape[0] != graph.shape[1]:
        raise ValueError("adjacency must be square")
    n = len(graph)
    if (
        not math.isfinite(min_gain)
        or min_gain < 0
        or not math.isfinite(numerical_tol)
        or numerical_tol < 0
    ):
        raise ValueError(
            "gain and numerical tolerance must be finite and >= 0"
        )
    if min_size < 1 or max_size < min_size or max_size >= n:
        raise ValueError("require 1 <= min_size <= max_size < residue count")
    cache = {}

    def scores(supports):
        keys = [tuple(sorted(s)) for s in supports]
        missing = list(dict.fromkeys(k for k in keys if k not in cache))
        if missing:
            values = (
                scorer.score_batch([list(k) for k in missing])
                if hasattr(scorer, "score_batch")
                else [scorer(list(k)) for k in missing]
            )
            if len(values) != len(missing):
                raise ValueError("scorer returned wrong number of results")
            for key, value in zip(missing, values):
                cache[key] = _measurement(value)
        return [cache[k] for k in keys]

    log = on_log if on_log is not None else (lambda *_: None)
    debug = getattr(on_log, "debug", None) or (lambda *_: None)
    log(f"incoming growth: {len(starts)} starts, max size {max_size}")
    runs, units, seen = [], [], set()
    for run_id, initial in enumerate(starts):
        support = sorted(set(int(r) for r in initial))
        if not support or support[0] < 0 or support[-1] >= n:
            raise ValueError("start must contain valid residue indices")
        if len(support) > max_size:
            raise ValueError("start exceeds max_size")
        current = scores([support])[0]
        debug(
            f"seed {run_id + 1}/{len(starts)} res "
            f"{' '.join(map(str, support))}  nu_in {current['nu_in']:.3f}  "
            f"nu_out {current['nu_out']:.3f}"
        )
        history = [
            {
                "step": 0,
                "added_residue": None,
                "residues": support.copy(),
                **current,
            }
        ]
        proposals = []
        while len(support) < max_size:
            frontier = sorted(
                set(np.flatnonzero(graph[support].any(axis=0))) - set(support)
            )
            if not frontier:
                reason = "no_frontier"
                break
            values = scores([support + [int(r)] for r in frontier])
            best = min(
                range(len(frontier)),
                key=lambda i: (values[i]["nu_in"], int(frontier[i])),
            )
            gain = current["nu_in"] - values[best]["nu_in"]
            needed = min_gain if len(history) > 1 else max(
                min_gain, start_min_gain
            )
            accept = gain > max(needed, numerical_tol)
            for i, (residue, value) in enumerate(zip(frontier, values)):
                proposals.append(
                    {
                        "step": len(history),
                        "parent_residues": support.copy(),
                        "candidate_residue": int(residue),
                        "parent_nu_in": current["nu_in"],
                        "gain_in": current["nu_in"] - value["nu_in"],
                        "accepted": bool(accept and i == best),
                        **value,
                    }
                )
            if not accept:
                reason = "no_incoming_improvement"
                debug(
                    f"stop at size {len(support)}: best +res "
                    f"{int(frontier[best])} changes nu_in by {-gain:+.3f} "
                    f"({len(frontier)} frontier)"
                )
                break
            residue = int(frontier[best])
            support = sorted(support + [residue])
            current = values[best]
            debug(
                f"size {len(support):3d} +res {residue:4d}  nu_in "
                f"{current['nu_in']:.3f} (gain {gain:.3f})  nu_out "
                f"{current['nu_out']:.3f}  frontier {len(frontier)}"
            )
            history.append(
                {
                    "step": len(history),
                    "added_residue": residue,
                    "residues": support.copy(),
                    **current,
                }
            )
        else:
            reason = "max_size"
        eligible = len(support) >= min_size
        key = tuple(support)
        note = ""
        if not eligible:
            note = "  (< min size)"
        elif key in seen:
            note = "  (duplicate)"
        report = log if eligible and key not in seen else debug
        report(
            f"seed {run_id + 1}/{len(starts)}  size {len(support):3d}  "
            f"nu_in {current['nu_in']:.3f}  nu_out {current['nu_out']:.3f}  "
            f"{reason}{note}  res {' '.join(map(str, support))}"
        )
        if eligible and key not in seen:
            seen.add(key)
            units.append(
                {
                    "unit": len(units),
                    "residues": support.copy(),
                    "size": len(support),
                    "source_run": run_id,
                    "stop_reason": reason,
                    **current,
                }
            )
        runs.append(
            {
                "run": run_id,
                "initial": history[0]["residues"],
                "final": support.copy(),
                "eligible": eligible,
                "stop_reason": reason,
                "history": history,
                "proposals": proposals,
            }
        )
    log(
        f"growth done: {len(units)} candidates (>= {min_size} residues) from "
        f"{len(starts)} starts, {len(cache)} supports scored "
        f"(--log-level debug shows every seed and step)"
    )
    return {
        "objective": "nu_in",
        "reference_subtracted": False,
        "outgoing_used_for_selection": False,
        "min_gain": min_gain,
        "start_min_gain": start_min_gain,
        "numerical_tol": numerical_tol,
        "units": units,
        "runs": runs,
        "n_scored": len(cache),
    }

