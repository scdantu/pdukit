"""Full-environment flux spectrum via the whole-protein precision matrix
(the "Schur" estimator), and its size-conditioned null calibration.

Built from the prototype in ``Claude outputs/schur_prototype.py`` (left
unchanged), split the way `interface.py` splits `FluxWorkspace`: a
once-per-protein precompute, a per-support step and a per-shift step, so the
calibration below and `SchurWorkspace.score()` run exactly the same code.
`interface.py` is not modified.

THE STATISTIC.  Same flux spectrum as `FluxWorkspace.score()` -- canonical
correlations between the unit's future residualised on the unit's present,
and the environment's present residualised on the unit's present -- with the
unit still reduced to its `unit_dim` slowest (Koopman) directions, but the
environment kept WHOLE instead of reduced to `env_dim` slow modes.

THE TRICK.  With one global ridge r = ridge * mean diagonal of C00,

    B = C00 + r I            G = B^-1        (once per protein)

and for a candidate with feature columns S and environment columns E the
block-inverse (Schur complement) identity gives, for any X,

    B_EE^-1 X_ES = (G X)_ES - G_ES (G_SS)^-1 (G X)_SS

so the only per-candidate inverse is G_SS (order |S| * m).  Residualising
the environment on the unit's present is folded in with a Woodbury identity
whose inverse is only unit_dim x unit_dim.  The products G X are precomputed
once for X = C01 and for every shift replica's Cs00[j], Cs01[j].

THE SHIFT NULL.  A circular time shift changes only the cross-covariance
between the shifted environment and the unshifted unit (rows E, columns S of
Cs00[j] / Cs01[j]); the environment's own covariance B_EE is unchanged, as
`FluxWorkspace` already assumes.  So each shift needs only its precomputed
G Cs00[j], G Cs01[j]; nothing is shifted inside G.  `brute()` computes the
same quantities with explicit e x e algebra and is the reference the tests
compare against.
"""

from __future__ import annotations

import time

import numpy as np

from .interface import (
    _bootstrap_ci_rel_width,
    _bootstrap_ci_rel_width_two_way,
    _grow_ball_support,
    _inv_sqrt,
    _support_descriptors,
    workspace_fingerprint,
)

STATISTIC = "schur_full_env"
MIN_SHIFTS = 8


# ---------------------------------------------------------------------------
# pure array functions -- shared with null_calibration_schur_cpu
# ---------------------------------------------------------------------------


def _global_ridge(C00, ridge):
    D = C00.shape[0]
    return float(ridge) * max(float(np.trace(C00)) / D, 1e-30)


def _schur_precompute_np(C00, C01, Cs00, Cs01, ridge):
    """Once per protein: (r, B, G, G C01, G Cs00[j], G Cs01[j]), stacked."""
    C00 = np.asarray(C00, dtype=np.float64)
    D = C00.shape[0]
    r = _global_ridge(C00, ridge)
    B = C00 + r * np.eye(D)
    G = np.linalg.inv(B)
    GC01 = G @ np.asarray(C01, dtype=np.float64)
    GCs00 = np.stack([G @ np.asarray(X, dtype=np.float64) for X in Cs00])
    GCs01 = np.stack([G @ np.asarray(X, dtype=np.float64) for X in Cs01])
    return r, B, G, GC01, GCs00, GCs01


def _schur_unit_np(C00, C01, C11, G, S, E, Vu, ridge):
    """Per support: everything the shifts reuse."""
    C00_SS = C00[np.ix_(S, S)]
    Chh = Vu.T @ C00_SS @ Vu
    Cyy = Vu.T @ C11[np.ix_(S, S)] @ Vu
    Chy = Vu.T @ C01[np.ix_(S, S)] @ Vu
    Ih = _inv_sqrt(Chh, ridge)
    Ih = Ih @ Ih  # ridge-regularised Chh^-1
    Ryy = Cyy - Chy.T @ Ih @ Chy
    return {
        "Vu": Vu,
        "Ih": Ih,
        "Ih_inv": np.linalg.inv(Ih),
        "Chy": Chy,
        "Cyy": Cyy,
        "Ryy": Ryy,
        "Phi": Ih @ Chy,
        "Lyy": _inv_sqrt(Ryy, ridge),
        "GssInv": np.linalg.inv(G[np.ix_(S, S)]),
        "GES": G[np.ix_(E, S)],
    }


def _binv_ee_vu(GX, S, E, u):
    """B_EE^-1 X_ES Vu = (GX)_ES Vu - G_ES G_SS^-1 (GX)_SS Vu."""
    Vu = u["Vu"]
    return GX[np.ix_(E, S)] @ Vu - u["GES"] @ (
        u["GssInv"] @ (GX[np.ix_(S, S)] @ Vu)
    )


def _schur_transfer(Ch, Cy, bh, by, u):
    """T = Rzy' Rzz^-1 Rzy with Rzz = B_EE - Ch Ih Ch', Rzy = Cy - Ch Ih Chy,
    via Woodbury (only a unit_dim x unit_dim inverse).
    """
    hh, hy, yy = Ch.T @ bh, Ch.T @ by, Cy.T @ by
    Phi = u["Phi"]
    RBR = yy - hy.T @ Phi - Phi.T @ hy + Phi.T @ hh @ Phi
    RBC = hy.T - Phi.T @ hh
    Wm = np.linalg.inv(u["Ih_inv"] - hh)
    return RBR + RBC @ Wm @ RBC.T


def _schur_spectrum(T, u):
    """Canonical correlations, descending, from T and the unit's Ryy."""
    L = u["Lyy"]
    M = L @ T @ L
    w = np.linalg.eigvalsh(0.5 * (M + M.T))[::-1]
    return np.sqrt(np.clip(w, 0.0, 1.0))


def _schur_null_rho(X00_j, X01_j, GX00_j, GX01_j, S, E, u):
    """Leading canonical correlation against one shift replica."""
    Vu = u["Vu"]
    Ch = X00_j[np.ix_(E, S)] @ Vu
    Cy = X01_j[np.ix_(E, S)] @ Vu
    bh = _binv_ee_vu(GX00_j, S, E, u)
    by = _binv_ee_vu(GX01_j, S, E, u)
    return float(_schur_spectrum(_schur_transfer(Ch, Cy, bh, by, u), u)[0])


# ---------------------------------------------------------------------------
# workspace
# ---------------------------------------------------------------------------


class SchurWorkspace:
    """Full-environment scoring on top of an existing `FluxWorkspace`.

    The constructor does the once-per-protein work (G and the G X products,
    timed as `precompute_seconds`); everything per candidate is small.
    """

    def __init__(self, ws):
        """Precompute the covariance blocks of the workspace."""
        t0 = time.perf_counter()
        self.ws = ws
        self.ridge = float(ws.ridge)
        (self.r, self.B, self.G, self.GC01, self.GCs00, self.GCs01) = (
            _schur_precompute_np(ws.C00, ws.C01, ws.Cs00, ws.Cs01, ws.ridge)
        )
        # References to the workspace's own shift matrices (no stacked copy:
        # with 20 shifts at D = 1312 the copies alone were ~0.5 GB).
        self.Cs00 = [np.asarray(X, dtype=np.float64) for X in ws.Cs00]
        self.Cs01 = [np.asarray(X, dtype=np.float64) for X in ws.Cs01]
        self.n_shift = int(len(self.Cs00))
        self.precompute_seconds = time.perf_counter() - t0

    # -- steps ------------------------------------------------------------
    def _project_and_residualise(self, support, unit_dim=8):
        """Per support: (support, S, E, unit quantities)."""
        ws = self.ws
        support, S, E = ws._cols(support)
        if E.size == 0:
            raise ValueError(
                "candidate covers the whole protein: no environment"
            )
        Vu = ws._koopman_basis(S, unit_dim, ws.reversible)
        u = _schur_unit_np(
            ws.C00, ws.C01, ws.C11, self.G, S, E, Vu, self.ridge
        )
        return support, S, E, u

    def _leading_null_rho(self, S, E, u, j):
        """Per shift: leading null correlation against shift replica j."""
        return _schur_null_rho(
            self.Cs00[j], self.Cs01[j], self.GCs00[j], self.GCs01[j], S, E, u
        )

    def _real_spectrum(self, S, E, u):
        ws, Vu = self.ws, u["Vu"]
        Ch = ws.C00[np.ix_(E, S)] @ Vu
        Cy = ws.C01[np.ix_(E, S)] @ Vu
        bh = -u["GES"] @ (u["GssInv"] @ Vu)  # B_EE^-1 B_ES Vu
        by = _binv_ee_vu(self.GC01, S, E, u)
        return _schur_spectrum(_schur_transfer(Ch, Cy, bh, by, u), u)

    def null_rows(self, support, unit_dim=8):
        """One leading null correlation per shift for `support`."""
        _, S, E, u = self._project_and_residualise(support, unit_dim)
        return np.array(
            [self._leading_null_rho(S, E, u, j) for j in range(self.n_shift)]
        )

    # -- public -----------------------------------------------------------
    def score(self, support, unit_dim=8, env_dim=None, null_thresholds=None):
        """Flux spectrum for one candidate, with the same output fields as
        `FluxWorkspace.score()`.  `env_dim` is accepted (so a caller written
        for FluxWorkspace can pass it) and ignored: the environment is whole.

        With `null_thresholds` None the threshold is this candidate's own
        shift null (95th percentile over the shifts); with a {size: theta}
        table it is looked up, and a size missing from the table is an
        error.
        """
        support, S, E, u = self._project_and_residualise(support, unit_dim)
        rho = self._real_spectrum(S, E, u)
        if null_thresholds is None:
            leading = [
                self._leading_null_rho(S, E, u, j) for j in range(self.n_shift)
            ]
            thr = float(np.percentile(leading, 95)) if leading else 0.0
        else:
            size = int(support.size)
            if size not in null_thresholds:
                raise KeyError(
                    f"null_thresholds has no entry for support size {size}"
                )
            thr = float(null_thresholds[size])

        rho2 = rho**2
        total = float(rho2.sum())
        k_star = int(np.sum(rho > thr))
        rho2c = np.maximum(rho2 - thr**2, 0.0)
        totc = float(rho2c.sum())
        if totc > 0:
            k_eff = float(totc**2 / np.sum(rho2c**2))
            curve = (np.cumsum(rho2c) / totc).tolist()
        else:
            k_eff, curve = float("nan"), []
        Cyy, Chy, Ih = u["Cyy"], u["Chy"], u["Ih"]
        tr_yy = float(np.trace(Cyy))
        sigma = (
            (float(np.trace(Chy.T @ Ih @ Chy)) / tr_yy)
            if tr_yy > 0
            else float("nan")
        )
        rhoc = np.sqrt(rho2c)
        nuc_spec = (
            float(rhoc.sum() / rhoc[0])
            if rhoc.size and rhoc[0] > 0
            else float("nan")
        )
        return {
            "support": support.tolist(),
            "size": int(support.size),
            "n_pairs": self.ws.n,
            "lag": self.ws.lag,
            "rho": rho.tolist(),
            "rho_null_p95": thr,
            "resolved": bool(k_star > 0),
            "margin": float(rho[0] - thr) if rho.size else 0.0,
            "low_power": bool(rho.size == 0 or (rho[0] - thr) <= 0.0),
            "k_star": k_star,
            "k_eff": k_eff,
            "coupling_total": total,
            "sigma": sigma,
            "nuc_spec": nuc_spec,
            "coupling_resolved": totc,
            "delta_int_curve": curve,
            "unit_dim": int(u["Vu"].shape[1]),
            "env_dim": None,
            "statistic": STATISTIC,
        }

    # -- reference ----------------------------------------------------------
    def _brute_parts(self, support, unit_dim):
        ws = self.ws
        support, S, E = ws._cols(support)
        Vu = ws._koopman_basis(S, unit_dim, ws.reversible)
        Chh = Vu.T @ ws.C00[np.ix_(S, S)] @ Vu
        Cyy = Vu.T @ ws.C11[np.ix_(S, S)] @ Vu
        Chy = Vu.T @ ws.C01[np.ix_(S, S)] @ Vu
        Ih = _inv_sqrt(Chh, self.ridge)
        Ih = Ih @ Ih
        Ryy = Cyy - Chy.T @ Ih @ Chy
        BEE = self.B[np.ix_(E, E)]

        def one(X00, X01):
            Ch = X00[np.ix_(E, S)] @ Vu
            Cy = X01[np.ix_(E, S)] @ Vu
            Rzz = BEE - Ch @ Ih @ Ch.T
            Rzy = Cy - Ch @ Ih @ Chy
            T = Rzy.T @ np.linalg.solve(Rzz, Rzy)
            L = _inv_sqrt(Ryy, self.ridge)
            w = np.linalg.eigvalsh(0.5 * (L @ T @ L + (L @ T @ L).T))[::-1]
            return np.sqrt(np.clip(w, 0.0, 1.0))

        return one

    def brute(self, support, unit_dim=8):
        """Return the same estimator with explicit e x e algebra.

        The prototype's reference: (real spectrum, 95th percentile of the
        shift null).
        """
        one = self._brute_parts(support, unit_dim)
        ws = self.ws
        rho = one(ws.C00, ws.C01)
        lead = [one(a, b)[0] for a, b in zip(ws.Cs00, ws.Cs01)]
        return rho, float(np.percentile(lead, 95))

    def brute_null_rows(self, support, unit_dim=8):
        """Explicit e x e leading null correlation for every shift."""
        one = self._brute_parts(support, unit_dim)
        return np.array(
            [one(a, b)[0] for a, b in zip(self.ws.Cs00, self.ws.Cs01)]
        )


# ---------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------


def calibrate_null_thresholds_schur(
    schur_ws,
    adjacency,
    sizes,
    *,
    unit_dim,
    target_rel_width=0.02,
    batch_supports=10,
    min_supports=50,
    max_pool_size=5000,
    seed=0,
    on_log=None,
):
    """Calibrate the size-conditioned null of the Schur statistic.

    For the Schur (full-environment) statistic -- the same machinery as
    `interface.calibrate_null_thresholds` (ball supports on the contact
    graph, the workspace's shift replicas,
    theta_k = 95th percentile of the pooled leading null correlations,
    support-level bootstrap stopping after at least `min_supports` supports,
    cap at `max_pool_size` pooled values), with each value computed by
    `SchurWorkspace`.  Same random stream and same return shape:

      {"theta", "diagnostics" (per size: n_pairs, n_supports, rel_ci_width,
       rel_ci_width_naive, converged, stopped_by, elapsed_seconds,
       contact_density, effective_dim, per_support), "config",
       "elapsed_seconds"}

    Differences from the interface version: fewer than 8 shifts is an error
    (shifts added afterwards would need their own G products, so none are
    added silently), and each calibration support's real spectrum is not
    computed (the calibration never uses it).  ``config["env_dim"]`` is None.
    """
    log = on_log if on_log is not None else (lambda *_: None)
    t_start = time.perf_counter()
    rng = np.random.default_rng(seed)
    adjacency = np.asarray(adjacency, dtype=bool)
    N = adjacency.shape[0]
    n_shift = schur_ws.n_shift
    if n_shift < MIN_SHIFTS:
        raise ValueError(
            f"the Schur calibration needs at least {MIN_SHIFTS} "
            f"shift replicas; the workspace has {n_shift}"
        )
    if int(min_supports) * n_shift > int(max_pool_size):
        raise ValueError(
            f"min_supports={min_supports} x {n_shift} shifts exceeds "
            f"max_pool_size={max_pool_size}: no size could ever be declared "
            f"converged"
        )

    def sample_support(k, tries=40):
        s = _grow_ball_support(N, k, adjacency, rng)
        for _ in range(tries - 1):
            if len(s) == k:
                break
            s = _grow_ball_support(N, k, adjacency, rng)
        return s

    C00 = schur_ws.ws.C00
    theta, diagnostics = {}, {}
    for k in sizes:
        k = int(k)
        t_size = time.perf_counter()
        pool, densities, eff_dims, per_support, support_rows = (
            [],
            [],
            [],
            [],
            [],
        )
        n_supports = 0
        rel_width, converged, stopped_by = float("inf"), False, None

        while True:
            for _ in range(batch_supports):
                support = sample_support(k)
                support_, S, E, u = schur_ws._project_and_residualise(
                    support, unit_dim
                )
                density, eff_dim = _support_descriptors(
                    support_, S, adjacency, C00
                )
                densities.append(density)
                eff_dims.append(eff_dim)
                n_supports += 1
                row = [
                    schur_ws._leading_null_rho(S, E, u, j)
                    for j in range(n_shift)
                ]
                pool.extend(row)
                support_rows.append(row)
                per_support.append(
                    {
                        "support": [int(i) for i in support_],
                        "leading_rho": [float(v) for v in row],
                        "theta_S": float(np.percentile(row, 95)),
                        "contact_density": float(density),
                        "effective_dim": float(eff_dim),
                    }
                )

            if n_supports >= min_supports:
                _, rel_width = _bootstrap_ci_rel_width_two_way(
                    support_rows, rng=rng
                )
            if n_supports >= min_supports and rel_width < target_rel_width:
                converged, stopped_by = True, "converged"
                break
            if len(pool) >= max_pool_size:
                stopped_by = "max_pool_size"
                log(
                    f"calibrate_null_thresholds_schur: size {k} hit "
                    f"max_pool_size ({max_pool_size}) before converging "
                    f"(rel CI width {rel_width:.4f} >= {target_rel_width})"
                )
                break

        _, rel_width_naive = _bootstrap_ci_rel_width(pool, rng=rng)
        theta[k] = float(np.percentile(pool, 95))
        diagnostics[k] = {
            "n_pairs": len(pool),
            "n_supports": n_supports,
            "rel_ci_width": float(rel_width),
            "rel_ci_width_naive": float(rel_width_naive),
            "converged": bool(converged),
            "stopped_by": stopped_by,
            "elapsed_seconds": time.perf_counter() - t_size,
            "contact_density": densities,
            "effective_dim": eff_dims,
            "per_support": per_support,
        }
        log(
            f"calibrate_null_thresholds_schur: size {k} theta={theta[k]:.4f} "
            f"n_pairs={len(pool)} n_supports={n_supports} "
            f"rel_ci_width={rel_width:.4f} (naive {rel_width_naive:.4f}) "
            f"converged={converged} "
            f"({diagnostics[k]['elapsed_seconds']:.1f}s)"
        )

    ws = schur_ws.ws
    config = {
        "statistic": STATISTIC,
        "unit_dim": int(unit_dim),
        "env_dim": None,
        "ridge": float(schur_ws.ridge),
        "global_ridge": float(schur_ws.r),
        "n_shifts": int(n_shift),
        "lag": int(ws.lag),
        "truncation": ws.truncation,
        "reversible": bool(ws.reversible),
        "target_rel_width": float(target_rel_width),
        "batch_supports": int(batch_supports),
        "min_supports": int(min_supports),
        "max_pool_size": int(max_pool_size),
        "seed": int(seed),
        "bootstrap": "two_way",
        "precompute_seconds": float(schur_ws.precompute_seconds),
        "fingerprint": workspace_fingerprint(ws),
    }
    return {
        "theta": theta,
        "diagnostics": diagnostics,
        "config": config,
        "elapsed_seconds": time.perf_counter() - t_start,
    }
