"""Core / port / bridge classification after incoming growth and splitting,
and scoring of fixed residue sets.

Incoming-only growth keeps a unit's CORE: residues whose addition does not
raise the unit's incoming channel count.  Many residues left out of every
core still move with one particular unit; they were excluded because adding
them lets outside information in.  They are the unit's PORTS (entry points of
its incoming channels).  Residues that prefer no single unit are BRIDGES
between units.

Classification of a residue r outside every core S_k
  drives  d_k = mean rho[S_k, r] / mean rho[rest, r]   (r predicts the unit)
  driven  l_k = mean rho[r, S_k] / mean rho[r, rest]   (the unit predicts r)
  rho[i, j]: j's present predicts i's future beyond i's own present.
  * no unit with d_k > 1 and l_k > 1             -> bridge
  * otherwise try those units, strongest preference (d_k * l_k) first, in
    guarded rounds (see classify_residues): r is a port of the first unit
    where adding it lowers cohesion neither against the current unit nor
    against the core by more than `z` bootstrap sd (block-bootstrap rho)
  * preferred unit(s) exist but every test fails in every round -> loose
Residues in two or more cores are reported as shared.
Each core is scored again together with its ports (the extended unit), so
incoming/outgoing channels are reported for core and for core + ports.
"""

from __future__ import annotations

import numpy as np

from pdukit.cog.units.grow import _measurement
from pdukit.numerics.cohesion import cohesion


def _score(scorer, supports):
    if scorer is None or not supports:
        return [None] * len(supports)
    values = (
        scorer.score_batch(supports)
        if hasattr(scorer, "score_batch")
        else [scorer(s) for s in supports]
    )
    return [_measurement(v) for v in values]


def _strength(meas):
    """Sum the excess squared canonical correlation above the noise level."""
    out = {}
    for d in ("in", "out"):
        rho = np.asarray(meas[f"rho_{d}"], float)
        out[f"strength_{d}"] = float(
            np.maximum(rho**2 - meas[f"theta_{d}"] ** 2, 0).sum()
        )
    return out


def _sd(values):
    values = [v for v in values if np.isfinite(v)]
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def classify_residues(
    rho,
    units,
    *,
    bootstrap=None,
    z=1.645,
    scorer=None,
    on_log=None,
    max_rounds=50,
    guard=True,
):
    """Classify every residue as core, shared, port, bridge or loose.

    Ports are added by guarded rounds (18 Oct 2026).  Each unit starts as
    its core.  Candidates are visited strongest preference first; a residue
    joins the first preferred unit for which both hold:
      (a) step:  cohesion(S + r) - cohesion(S)    >= -z * sd_boot
      (b) guard: cohesion(S + r) - cohesion(core) >= -z * sd_boot
    sd_boot is the standard deviation of the same difference over the
    block-bootstrap rho matrices (paired: both terms on the same resample).
    Rounds repeat until no residue joins, so a residue refused early can
    join once the unit has grown towards it, while the guard stops the unit
    drifting away from its core.  Preferences (drives, driven) are measured
    against the cores and do not change between rounds.  guard=False drops
    test (b) (ablation only).
    """
    log = on_log if on_log is not None else (lambda *_: None)
    guard_on = guard
    R = np.array(rho, float)
    N = R.shape[0]
    np.fill_diagonal(R, np.nan)
    boots = [np.array(b, float) for b in (bootstrap or [])]
    mats = [R] + boots
    cores = [sorted(set(int(x) for x in u)) for u in units]
    owners = {}
    for k, S in enumerate(cores):
        for r in S:
            owners.setdefault(r, []).append(k)

    def coh_all(S):
        return [cohesion(M, S)["cohesion"] for M in mats]

    core_coh = [coh_all(S) for S in cores]
    rows, todo = {}, []
    for r in range(N):
        if r in owners:
            rows[r] = dict(
                residue=r,
                cls="shared" if len(owners[r]) > 1 else "core",
                unit=" ".join(map(str, owners[r])),
            )
            continue
        prefs = []
        for k, S in enumerate(cores):
            rest = [i for i in range(N) if i not in S and i != r]
            d = np.nanmean(R[S, r]) / max(np.nanmean(R[rest, r]), 1e-12)
            lo = np.nanmean(R[r, S]) / max(np.nanmean(R[r, rest]), 1e-12)
            prefs.append((k, float(d), float(lo)))
        ranked = sorted(prefs, key=lambda t: -(t[1] * t[2]))
        row = dict(
            residue=r,
            best_units=" ".join(str(t[0]) for t in ranked[:2]),
            best_drives=" ".join(f"{t[1]:.3f}" for t in ranked[:2]),
            best_driven=" ".join(f"{t[2]:.3f}" for t in ranked[:2]),
        )
        cand = [t for t in ranked if t[1] > 1 and t[2] > 1]
        row.update(cls="bridge" if not cand else "loose", unit="")
        rows[r] = row
        if cand:
            todo.append((r, cand))
    ext = [list(S) for S in cores]
    cur = [list(c) for c in core_coh]
    joined, n_rounds = set(), 0
    for rnd in range(1, max_rounds + 1):
        n_rounds, added = rnd, 0
        for r, cand in todo:
            if r in joined:
                continue
            for i, (k, d, lo) in enumerate(cand):
                new = coh_all(ext[k] + [r])
                step = new[0] - cur[k][0]
                guard = new[0] - core_coh[k][0]
                step_sd = _sd([n - c for n, c in zip(new[1:], cur[k][1:])])
                guard_sd = _sd(
                    [n - c for n, c in zip(new[1:], core_coh[k][1:])]
                )
                ok_step = bool(step >= -z * step_sd)
                ok_guard = bool(guard >= -z * guard_sd) or not guard_on
                if i == 0 or (ok_step and ok_guard):
                    rows[r].update(
                        tested_unit=k,
                        drives=d,
                        driven=lo,
                        delta_cohesion=step,
                        delta_cohesion_sd=step_sd,
                        guard_delta=guard,
                        guard_sd=guard_sd,
                        failed=""
                        if ok_step and ok_guard
                        else ("step" if not ok_step else "")
                        + ("+" if not (ok_step or ok_guard) else "")
                        + ("guard" if not ok_guard else ""),
                        round=rnd,
                    )
                if ok_step and ok_guard:
                    rows[r].update(cls="port", unit=str(k))
                    ext[k] = sorted(ext[k] + [r])
                    cur[k] = new
                    joined.add(r)
                    added += 1
                    break
        log(f"round {rnd}: +{added} ports ({len(joined)} total)")
        if added == 0:
            break
    rows = [rows[r] for r in range(N)]
    ports = {k: sorted(set(ext[k]) - set(S)) for k, S in enumerate(cores)}
    extended = [sorted(e) for e in ext]
    core_scores = _score(scorer, cores)
    ext_scores = _score(scorer, extended)
    unit_rows = []
    for k, S in enumerate(cores):
        diff = [e - c for e, c in zip(cur[k][1:], core_coh[k][1:])]
        u = dict(
            unit=k,
            core_size=len(S),
            n_ports=len(ports[k]),
            extended_size=len(extended[k]),
            core=S,
            ports=ports[k],
            extended=extended[k],
            cohesion_core=core_coh[k][0],
            cohesion_extended=cur[k][0],
            cohesion_change=cur[k][0] - core_coh[k][0],
            cohesion_change_sd=_sd(diff),
        )
        for tag, m in (("core", core_scores[k]), ("extended", ext_scores[k])):
            if m is None:
                continue
            u.update(
                {
                    f"nu_in_{tag}": m["nu_in"],
                    f"nu_out_{tag}": m["nu_out"],
                    f"keff_in_{tag}": m["keff_in"],
                    f"keff_out_{tag}": m["keff_out"],
                }
            )
            u.update({f"{key}_{tag}": v for key, v in _strength(m).items()})
        unit_rows.append(u)
    counts = {
        c: sum(row["cls"] == c for row in rows)
        for c in ("core", "shared", "port", "bridge", "loose")
    }
    log(f"classes {counts}")
    for u in unit_rows:

        def fmt(key):
            return "n/a" if key not in u else f"{u[key]:.2f}"

        log(
            f"unit {u['unit']}  core {u['core_size']} + ports "
            f"{u['n_ports']} = {u['extended_size']}  nu_in "
            f"{fmt('nu_in_core')}>{fmt('nu_in_extended')}  nu_out "
            f"{fmt('nu_out_core')}>{fmt('nu_out_extended')}  cohesion "
            f"{u['cohesion_core']:.3f}>{u['cohesion_extended']:.3f} "
            f"({u['cohesion_change']:+.3f}, sd {u['cohesion_change_sd']:.3f})"
        )
    bridges = [row["residue"] for row in rows if row["cls"] == "bridge"]
    log(f"bridges {bridges}")
    return dict(
        rule="port: prefers unit both ways; guarded rounds: step and "
        f"core-guard cohesion drops each within {z} bootstrap sd; "
        "bridge: prefers no unit both ways",
        z=z,
        guard=guard_on,
        bootstrap_samples=len(boots),
        rounds=n_rounds,
        counts=counts,
        residues=rows,
        units=unit_rows,
    )


def score_sets(sets, scorer, rho=None, on_log=None):
    """Incoming/outgoing channels, strength and cohesion of fixed sets."""
    log = on_log if on_log is not None else (lambda *_: None)
    meas = _score(scorer, [s["residues"] for s in sets])
    rows = []
    for s, m in zip(sets, meas):
        row = dict(
            name=s["name"],
            size=len(s["residues"]),
            mode_unit=s["mode_unit"],
            residues=s["residues"],
            **m,
            **_strength(m),
        )
        if rho is not None and len(s["residues"]) >= 2:
            row.update(cohesion(rho, s["residues"]))
        rows.append(row)
        log(
            f"  set {s['name']} ({row['size']}): nu_in {m['nu_in']:.3f} "
            f"(strength {row['strength_in']:.4f}), nu_out {m['nu_out']:.3f} "
            f"(strength {row['strength_out']:.4f}), cohesion "
            f"{row.get('cohesion', float('nan')):.3f}"
        )
    return rows


