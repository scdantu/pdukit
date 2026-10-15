"""Varun's PDU definition, scored for given residue sets (linear estimators).

Reference: "A Stable, End-to-End Mathematical Definition of a Protein
Dynamical Unit" (V. Thakre), sections 4-6 and 9.  A residue set S is a PDU at
lag tau if

    Q(S) = r_eta(S) / d_pred(S)  <=  epsilon,     0 < d_pred < inf.

This module only SCORES sets; it does not search.  Everything is linear and
built from the workspace covariances (C00, C01, C11 and the shifted Cs00,
Cs01), plus a frame subsample for the box-counting estimate.

Numerator: interface width r_eta (Varun 9.1)
-------------------------------------------
    K = C10 (C00 + lam I)^-1            (all features, unit + environment)
    B = K[S future, R present]          environment -> unit
    C = K[R future, S present]          unit -> environment
    B~ = W_S B W_R^-1,  C~ = W_R C W_S^-1   with W = C00_block^(-1/2)
    nu_eta(M) = sum_i s_i^2 / (s_i^2 + eta^2)       (soft rank, Varun eq. 5)
    r = max(nu(B~), nu(C~))
Whole blocks are used: no 8 / 20 truncation.  eta (one per direction) is the
95th percentile of the LEADING singular value when the source block's
present is displaced in time (the workspace's shifts) -- the same rule for
every candidate and every random set.

Denominator: predictive dimension d_pred (Varun 5 and 9.2), linear versions
--------------------------------------------------------------------------
Only the unit's own features.  The unit's present-to-future singular values
s_k and directions (VAMP / canonical correlation of x_t^S with x_{t+tau}^S)
give, in the linear model, the forward and backward predictive coordinates of
a state x:  J(x) = [s * U' W0 x,  s * V' W1 x].
  d_soft  = sum_k s_k^2 / (s_k^2 + eta_d^2)        (1a; eta_d = 95th pct of
                                                    the leading singular value
                                                    with the unit's present
                                                    displaced in time)
  d_box   = slope of log N(rho) vs log(1/rho) for the cloud J(x_t) of
            subsampled frames, N = greedy covering number, radii a declared
            geometric range relative to the median pairwise distance (1b)
For a linear Gaussian unit the two agree up to the choice of scale.

Score and null (Varun 9.3)
--------------------------
Q_soft = r / d_soft, Q_box = r / d_box.  For each size, M random sets of the
same size (grown like the baseline regions) are scored the same way;
p = (1 + #{Q_random <= Q}) / (M + 1)  (small p = unusually narrow).
"""

from __future__ import annotations

import time

import numpy as np


def _cols(sup, m):
    sup = np.asarray(sorted(int(i) for i in sup))
    return (sup[:, None] * m + np.arange(m)[None, :]).ravel()


def _sym_pow(M, p, rel_ridge=1e-8):
    M = 0.5 * (M + M.T)
    w, V = np.linalg.eigh(M)
    w = np.maximum(w, rel_ridge * max(float(w.max()), 1e-300))
    return (V * w**p) @ V.T


def soft_rank(s, eta):
    """Varun eq.

    (5), literal: every singular value counts s^2/(s^2+eta^2), so a noise floor
    of many values just below eta adds up.
    """
    s = np.asarray(s, float)
    if eta <= 0:
        return float(np.sum(s > 0))
    return float(np.sum(s**2 / (s**2 + eta**2)))


def edge_rank(s, eta):
    """Soft count that is zero at and below eta: sum max(0, 1 - eta^2/s^2)
    (the COG 2.1 form).  Same singular values, same eta.
    """
    s = np.asarray(s, float)
    if eta <= 0:
        return float(np.sum(s > 0))
    s = s[s > eta]
    return float(np.sum(1.0 - eta**2 / s**2))


class PduDef:
    """Scores residue sets with Varun's linear estimators on a workspace."""

    def __init__(self, ws, *, lam_rel=1e-3, n_shifts=32, percentile=95.0):
        """Hold the lagged covariances and the shift nulls of the workspace."""
        self.ws = ws
        self.N, self.m = int(ws.N), int(ws.m)
        self.D = self.N * self.m
        self.C00 = np.asarray(ws.C00, float)[: self.D, : self.D]
        self.C01 = np.asarray(ws.C01, float)[: self.D, : self.D]
        self.C11 = np.asarray(ws.C11, float)[: self.D, : self.D]
        self.lam = float(lam_rel) * float(np.mean(np.diag(self.C00)))
        k = min(int(n_shifts), len(getattr(ws, "Cs00", [])))
        self.Cs00 = [
            np.asarray(ws.Cs00[j], float)[: self.D, : self.D] for j in range(k)
        ]
        self.Cs01 = [
            np.asarray(ws.Cs01[j], float)[: self.D, : self.D] for j in range(k)
        ]
        self.pct = float(percentile)

    # -- interface width ------------------------------------------------
    def _w(self, S, R):
        return (
            _sym_pow(self.C00[np.ix_(S, S)], -0.5),
            _sym_pow(self.C00[np.ix_(R, R)], -0.5),
            _sym_pow(self.C00[np.ix_(S, S)], 0.5),
            _sym_pow(self.C00[np.ix_(R, R)], 0.5),
        )

    def _blocks(self, S, R, C00, C10, W):
        """B~ and C~ singular values for given (possibly shifted) moments."""
        A = C00 + self.lam * np.eye(C00.shape[0])
        K = np.linalg.solve(A.T, C10.T).T  # C10 A^-1
        WS, WR, WSi, WRi = W
        Bt = WS @ K[np.ix_(S, R)] @ WRi
        Ct = WR @ K[np.ix_(R, S)] @ WSi
        return (
            np.linalg.svd(Bt, compute_uv=False),
            np.linalg.svd(Ct, compute_uv=False),
        )

    def _shifted(self, j, src):
        """Moments with the present of the columns `src` displaced by
        shift j (future and the other present block unchanged).
        """
        C00 = self.C00.copy()
        C10 = self.C01.T.copy()
        X00, X01 = self.Cs00[j], self.Cs01[j]
        oth = np.setdiff1d(np.arange(self.D), src)
        # cov(shifted src present, other present) and with every future
        C00[np.ix_(src, oth)] = X00[np.ix_(src, oth)]
        C00[np.ix_(oth, src)] = X00[np.ix_(src, oth)].T
        C10[:, src] = X01[src, :].T
        return C00, C10

    def interface(self, sup):
        """Return the interface rank and soft channel count of a support."""
        S = _cols(sup, self.m)
        R = np.setdiff1d(np.arange(self.D), S)
        W = self._w(S, R)
        sB, sC = self._blocks(S, R, self.C00, self.C01.T, W)
        nB, nC = [], []
        for j in range(len(self.Cs00)):
            C00, C10 = self._shifted(j, R)  # environment displaced
            b, _ = self._blocks(S, R, C00, C10, W)
            nB.append(b[0] if b.size else 0.0)
            C00, C10 = self._shifted(j, S)  # unit displaced
            _, c = self._blocks(S, R, C00, C10, W)
            nC.append(c[0] if c.size else 0.0)
        etaB = float(np.percentile(nB, self.pct)) if nB else 0.0
        etaC = float(np.percentile(nC, self.pct)) if nC else 0.0
        rB, rC = edge_rank(sB, etaB), edge_rank(sC, etaC)
        lB, lC = soft_rank(sB, etaB), soft_rank(sC, etaC)
        return {
            "r": max(rB, rC),
            "r_in": rB,
            "r_out": rC,
            "r_lit": max(lB, lC),
            "r_in_lit": lB,
            "r_out_lit": lC,
            "n_in_above_eta": int(np.sum(sB > etaB)),
            "n_out_above_eta": int(np.sum(sC > etaC)),
            "eta_in": etaB,
            "eta_out": etaC,
            "sv_in": [round(float(x), 4) for x in sB[:12]],
            "sv_out": [round(float(x), 4) for x in sC[:12]],
        }

    # -- predictive dimension ------------------------------------------
    def own_spectrum(self, sup):
        """Return the support's own predictive spectrum and its null level."""
        S = _cols(sup, self.m)
        W0 = _sym_pow(self.C00[np.ix_(S, S)] + self.lam * np.eye(len(S)), -0.5)
        W1 = _sym_pow(self.C11[np.ix_(S, S)] + self.lam * np.eye(len(S)), -0.5)
        M = W0 @ self.C01[np.ix_(S, S)] @ W1
        U, s, Vt = np.linalg.svd(M)
        null = []
        for j in range(len(self.Cs00)):
            Mj = W0 @ self.Cs01[j][np.ix_(S, S)] @ W1
            null.append(float(np.linalg.svd(Mj, compute_uv=False)[0]))
        eta = float(np.percentile(null, self.pct)) if null else 0.0
        return S, W0, W1, U, s, Vt, eta

    def dimension(
        self, sup, frames=None, *, radii=None, n_frames=2000, rng=None
    ):
        """Return the predictive dimension of a support."""
        S, W0, W1, U, s, Vt, eta = self.own_spectrum(sup)
        out = {
            "d_soft": edge_rank(s, eta),
            "d_soft_lit": soft_rank(s, eta),
            "eta_d": eta,
            "s_own": [round(float(x), 4) for x in s[:16]],
            "n_above_eta": int(np.sum(s > eta)),
        }
        if frames is None:
            return out
        rng = rng if rng is not None else np.random.default_rng(0)
        F = frames.shape[0]
        idx = rng.choice(F, min(int(n_frames), F), replace=False)
        X = frames.reshape(F, -1)[idx][:, S].astype(float)
        X = X - np.asarray(self.ws.m0, float)[S][None, :]
        keep = s > eta
        if keep.sum() == 0:
            out.update({"d_box": 0.0, "box_table": []})
            return out
        fw = (X @ W0 @ U[:, keep]) * s[keep]
        bw = (X @ W1 @ Vt.T[:, keep]) * s[keep]
        J = np.hstack([fw, bw])
        # reference scale: median pairwise distance on a small subsample
        sub = J[: min(500, len(J))]
        dd = np.sqrt(((sub[:, None, :] - sub[None, :, :]) ** 2).sum(-1))
        ref = float(np.median(dd[np.triu_indices(len(sub), 1)]))
        J = J / max(ref, 1e-12)
        radii = radii or [0.8, 0.6, 0.45, 0.34, 0.25, 0.19, 0.14]
        table = []
        for rho in radii:
            n = _cover_count(J, rho)
            table.append((float(rho), int(n)))
        x = np.log(1.0 / np.array([r for r, _ in table]))
        y = np.log(np.array([n for _, n in table], float))
        ok = np.array([n < len(J) / 10 for _, n in table])
        if ok.sum() >= 3:
            slope = float(np.polyfit(x[ok], y[ok], 1)[0])
        else:
            slope = float("nan")
        out.update(
            {
                "d_box": slope,
                "box_table": table,
                "box_points": int(ok.sum()),
                "box_dims": int(J.shape[1]),
            }
        )
        return out


def _cover_count(J, rho):
    """Greedy covering: number of balls of radius rho (centred on points)
    needed to cover the cloud J.
    """
    left = np.ones(len(J), bool)
    order = np.arange(len(J))
    n = 0
    r2 = rho * rho
    for i in order:
        if not left[i]:
            continue
        n += 1
        d2 = ((J - J[i]) ** 2).sum(1)
        left &= d2 > r2
        if not left.any():
            break
    return n


def score_sets(pdd, sets, frames, *, n_frames=3000, seed=0):
    """Score each set: interface rank, dimension and their ratio Q."""
    rng = np.random.default_rng(seed)
    out = []
    for sup in sets:
        sup = sorted(int(x) for x in sup)
        it = pdd.interface(sup)
        dm = pdd.dimension(sup, frames, n_frames=n_frames, rng=rng)
        rec = {"size": len(sup), **it, **dm}
        rec["Q_soft"] = (
            (it["r"] / dm["d_soft"]) if dm["d_soft"] > 0 else float("nan")
        )
        rec["Q_lit"] = (
            (it["r_lit"] / dm["d_soft_lit"])
            if dm["d_soft_lit"] > 0
            else float("nan")
        )
        db = dm.get("d_box", float("nan"))
        rec["Q_box"] = (
            (it["r"] / db) if (db == db and db > 0) else float("nan")
        )
        out.append(rec)
    return out


def run_pdudef(
    ws,
    units,
    adjacency,
    frames,
    *,
    n_random=20,
    n_shifts=32,
    n_frames=3000,
    lam_rel=1e-3,
    seed=0,
    statuses=None,
    log=print,
):
    """Score the units and random controls with the PDU definition."""
    from .cog2_scoring import ball_support as _ball_support

    t0 = time.perf_counter()
    pdd = PduDef(ws, lam_rel=lam_rel, n_shifts=n_shifts)
    N = pdd.N
    rng = np.random.default_rng(seed)
    rows = score_sets(pdd, units, frames, n_frames=n_frames, seed=seed)
    nulls = {}
    for k in sorted({r["size"] for r in rows}):
        if not 2 <= k < N:
            continue
        sets = []
        for _ in range(int(n_random)):
            s_ = _ball_support(N, k, adjacency, rng)
            if len(s_) == k:
                sets.append(s_)
        nulls[k] = score_sets(
            pdd, sets, frames, n_frames=n_frames, seed=seed + k
        )
    for i, r in enumerate(rows):
        r["unit"] = i
        r["residues"] = sorted(int(x) for x in units[i])
        if statuses is not None:
            r["cog_status"] = statuses[i]
        nl = nulls.get(r["size"], [])
        for q in ("Q_soft", "Q_lit", "Q_box", "r", "r_lit", "d_soft"):
            vals = [x[q] for x in nl if x[q] == x[q]]
            r[f"random_{q}_median"] = (
                round(float(np.median(vals)), 3) if vals else None
            )
        for q in ("Q_soft", "Q_lit", "Q_box"):
            vals = [x[q] for x in nl if x[q] == x[q]]
            r[f"p_{q}"] = (
                round((1 + sum(v <= r[q] for v in vals)) / (len(vals) + 1), 3)
                if vals and r[q] == r[q]
                else None
            )
        log(
            f"PDU definition, unit {i} (size {r['size']}"
            + (f", COG {r.get('cog_status')}" if statuses else "")
            + f"): r {r['r']:.2f} (in {r['r_in']:.2f}, out {r['r_out']:.2f};"
            f" literal eq.5 {r['r_lit']:.2f}); d {r['d_soft']:.2f} (literal "
            f"{r['d_soft_lit']:.2f}, box {r.get('d_box', float('nan')):.2f})"
            f"; Q {r['Q_soft']:.3f} (random {r['random_Q_soft_median']}, "
            f"p {r['p_Q_soft']}); Q literal {r['Q_lit']:.3f} (random "
            f"{r['random_Q_lit_median']}, p {r['p_Q_lit']}); Q box "
            f"{r['Q_box']:.3f} (p {r['p_Q_box']})"
        )
    return {
        "settings": {
            "lam_rel": lam_rel,
            "n_shifts": len(pdd.Cs00),
            "n_random": n_random,
            "n_frames": n_frames,
            "eta_rule": "95th pct of leading null singular "
            "value (time-shifted source)",
            "random_sets": "grown like the baseline regions",
        },
        "units": rows,
        "random": {
            str(k): [
                {
                    q: (round(float(x[q]), 4) if x[q] == x[q] else None)
                    for q in (
                        "r",
                        "r_lit",
                        "d_soft",
                        "d_soft_lit",
                        "d_box",
                        "Q_soft",
                        "Q_lit",
                        "Q_box",
                    )
                }
                for x in v
            ]
            for k, v in nulls.items()
        },
        "seconds": round(time.perf_counter() - t0, 1),
    }
