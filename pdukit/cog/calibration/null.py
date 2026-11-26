"""Checks on the null calibration table: which workspace it belongs to, whether
it fits the run, and filling gaps in its theta table.
"""

from __future__ import annotations

import numpy as np


def workspace_fingerprint(ws):
    """Return what a null table depends on beyond unit_dim and env_dim.

    That is the data (how many lagged pairs, which segments, a numerical
    signature of the covariances) and the estimator settings.  Stored in
    every calibration's config and checked by `check_null_calibration`.
    """
    return {
        "n_pairs": int(ws.n),
        "N": int(ws.N),
        "m": int(ws.m),
        "D": int(ws.D),
        "lag": int(ws.lag),
        "ridge": float(ws.ridge),
        "truncation": str(ws.truncation),
        "reversible": bool(ws.reversible),
        "shift_mode": str(getattr(ws, "shift_mode", "global")),
        "segments": [[int(a), int(b)] for a, b in ws.segments],
        "trace_C00": float(np.trace(ws.C00)),
        "trace_C01": float(np.trace(ws.C01)),
    }


def check_null_calibration(calibration, ws, unit_dim, env_dim, *, rtol=1e-4):
    """Raise ValueError if `calibration` was not made on this workspace's
    data and settings.  `calibration` is the full dict a calibration returns.
    Calibrations made before fingerprints existed are checked on
    unit_dim/env_dim only.
    """
    config = calibration.get("config") or {}
    problems = []
    for key, want in (("unit_dim", unit_dim), ("env_dim", env_dim)):
        if (
            key in config
            and config[key] is not None
            and want is not None
            and int(config[key]) != int(want)
        ):
            problems.append(f"{key}: table {config[key]} vs run {want}")
    fp = config.get("fingerprint")
    if fp:
        now = workspace_fingerprint(ws)
        for key, got in fp.items():
            if key not in now:
                continue
            want = now[key]
            if isinstance(got, float) or isinstance(want, float):
                if not np.isclose(
                    float(got), float(want), rtol=rtol, atol=1e-12
                ):
                    problems.append(f"{key}: table {got} vs run {want}")
            elif got != want:
                problems.append(f"{key}: table {got} vs run {want}")
    if problems:
        raise ValueError(
            "null calibration does not match this workspace -- "
            "a null threshold is only valid for the data and "
            "settings it was calibrated on: " + "; ".join(problems)
        )


def complete_theta_table(theta, max_size, *, mode="interpolate"):
    """Return a {size: theta} table covering every size 1..max_size.

    A size missing from the table used to make `score()` raise KeyError,
    which `decompose()` silently turned into an infinite cost -- so the
    calibrated sizes, not the data, decided which unit sizes could exist.
    'interpolate' fills gaps linearly and holds the end values beyond the
    calibrated range (theta is close to flat in size); 'strict' raises.
    Returns (table, filled_sizes).
    """
    th = {int(k): float(v) for k, v in theta.items()}
    if not th:
        raise ValueError("empty null table")
    ks = sorted(th)
    vals = [th[k] for k in ks]
    out, filled = {}, []
    for k in range(1, int(max_size) + 1):
        if k in th:
            out[k] = th[k]
            continue
        if mode == "strict":
            raise ValueError(
                f"null table has no entry for size {k}; "
                f"calibrated sizes are {ks}"
            )
        out[k] = float(np.interp(k, ks, vals))
        filled.append(k)
    return out, filled
