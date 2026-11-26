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

from pdukit.numerics.kernel import _inv_sqrt

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

# ---------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------


