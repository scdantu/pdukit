"""Predictive (dynamical) neighbour graph for the COG 2.1 search.

The PDU criterion is dynamical only: how many independent channels connect a
residue set's present/future to the rest of the protein.  Spatial contact is
not part of it.  The search, however, needs a neighbourhood to propose
candidates from; until now that was the contact graph, which made units
spatially contiguous by construction.  This module builds the neighbourhood
from dynamics instead:

  rho[i, j] = leading partial canonical correlation between residue j's
              present and residue i's future, given residue i's own present
              (does j predict i beyond what i already knows about itself?)

computed for every ordered pair from the workspace covariances (C00, C01,
C11), with a shift null from the workspace's displaced-source covariances
(Cs00, Cs01).  Residues i and j are linked if either direction exceeds the
null threshold; each residue keeps its `k` strongest partners (union over
both ends) so the search frontier stays bounded.

The graph only decides WHERE the search looks.  Whether a set is a unit is
still decided by the PDU cost (channels to the whole rest of the protein).
"""

from __future__ import annotations

import numpy as np


def _isqrt_batch(S, rel_ridge=1e-6):
    S = 0.5 * (S + np.swapaxes(S, -1, -2))
    w, V = np.linalg.eigh(S)
    floor = rel_ridge * np.maximum(w.max(-1, keepdims=True), 1e-300)
    w = np.maximum(w, floor)
    return (V * (w**-0.5)[..., None, :]) @ np.swapaxes(V, -1, -2)


def _inv_batch(S, rel_ridge=1e-6):
    S = 0.5 * (S + np.swapaxes(S, -1, -2))
    w, V = np.linalg.eigh(S)
    floor = rel_ridge * np.maximum(w.max(-1, keepdims=True), 1e-300)
    w = np.maximum(w, floor)
    return (V * (1.0 / w)[..., None, :]) @ np.swapaxes(V, -1, -2)


def pair_rho(C00, C01, C11, N, m, X00=None, X01=None):
    """rho[i, j]: j's present -> i's future given i's present.

    C00, C01, C11 : (D, D) with D >= N*m (residue i owns columns i*m..).
    X00, X01      : optional displaced-source versions of C00 / C01
                    (rows = displaced source) for the null.
    """
    D = N * m
    C00 = np.asarray(C00)[:D, :D]
    C01 = np.asarray(C01)[:D, :D]
    C11 = np.asarray(C11)[:D, :D]
    S00 = C00 if X00 is None else np.asarray(X00)[:D, :D]
    S01 = C01 if X01 is None else np.asarray(X01)[:D, :D]
    B = C00.reshape(N, m, N, m)  # B[a, :, b, :] = C00[a, b]
    rho = np.zeros((N, N))
    # per-source residual covariance of j's present given i's present needs
    # C00[j, j], C00[j, i], C00[i, j]
    Cjj = np.stack([B[j, :, j, :] for j in range(N)])  # (N, m, m)
    for i in range(N):
        sl = slice(i * m, (i + 1) * m)
        Cii = C00[sl, sl]
        Ai = _inv_batch(Cii[None])[0]  # (m, m)
        Bii = C01[sl, sl]  # x_i, y_i
        Syy = C11[sl, sl] - Bii.T @ Ai @ Bii
        Wy = _isqrt_batch(Syy[None])[0]  # (m, m)
        # source blocks for all j (displaced in the null)
        Xji = S00[:, sl].reshape(N, m, m)  # x_j(s), x_i
        Xjy = S01[:, sl].reshape(N, m, m)  # x_j(s), y_i
        # cov(y_i, x_j | x_i) = C01[j,i]^T - C01[i,i]^T Ai C00[i,j]
        cross = np.swapaxes(Xjy, 1, 2) - (Bii.T @ Ai)[None] @ np.swapaxes(
            Xji, 1, 2
        )  # (N, m, m)
        Sxx = Cjj - Xji @ Ai[None] @ np.swapaxes(Xji, 1, 2)  # (N, m, m)
        M = Wy[None] @ cross @ _isqrt_batch(Sxx)
        s = np.linalg.svd(M, compute_uv=False)[:, 0]
        s[i] = 0.0
        rho[i] = np.clip(s, 0.0, 1.0)
    return rho


def predictive_graph(
    ws, k=12, null_shifts=16, percentile=95.0, contact=None, log=print
):
    """Dynamical neighbour graph from a FluxWorkspace `ws`.

    Returns dict with 'adjacency' (N x N bool, symmetric), 'rho' (directed,
    rho[i, j] = j predicts i), 'theta' (null threshold), stats.
    """
    N, m = int(ws.N), int(ws.m)
    rho = pair_rho(ws.C00, ws.C01, ws.C11, N, m)
    null = []
    n_sh = min(int(null_shifts), len(getattr(ws, "Cs00", [])))
    for s in range(n_sh):
        r0 = pair_rho(
            ws.C00, ws.C01, ws.C11, N, m, X00=ws.Cs00[s], X01=ws.Cs01[s]
        )
        null.append(r0[~np.eye(N, dtype=bool)])
    theta = (
        float(np.percentile(np.concatenate(null), percentile)) if null else 0.0
    )
    strength = np.maximum(rho, rho.T)
    np.fill_diagonal(strength, 0.0)
    sig = strength > theta
    A = np.zeros((N, N), bool)
    for i in range(N):
        cand = np.where(sig[i])[0]
        if cand.size:
            top = cand[np.argsort(strength[i, cand])[::-1][:k]]
            A[i, top] = True
    lonely = [i for i in range(N) if not sig[i].any()]
    for i in lonely:  # no partner beyond the null: keep its
        top = np.argsort(strength[i])[::-1][:k]  # strongest ones so the
        A[i, top] = True  # search can still place it (flagged)
    A = A | A.T
    np.fill_diagonal(A, False)
    out = {
        "adjacency": A,
        "rho": rho,
        "theta": theta,
        "k": int(k),
        "null_shifts": n_sh,
        "significant_pairs_fraction": float(sig[np.triu_indices(N, 1)].mean()),
        "mean_degree": float(A.sum(1).mean()),
        "no_significant_partner": [int(i) for i in lonely],
    }
    if contact is not None:
        C = np.asarray(contact, bool)
        e = A[np.triu_indices(N, 1)]
        c = C[np.triu_indices(N, 1)]
        out["edges_that_are_contacts"] = float((e & c).sum() / max(e.sum(), 1))
        out["contacts_that_are_edges"] = float((e & c).sum() / max(c.sum(), 1))
    log(
        f"predictive graph: null threshold {theta:.3f} ({n_sh} shifts); "
        f"{100 * out['significant_pairs_fraction']:.0f}% of residue pairs "
        f"predict each other beyond it; keeping the {k} strongest partners "
        f"-> mean degree {out['mean_degree']:.1f}"
        + (
            f"; {100 * out['edges_that_are_contacts']:.0f}% of edges are "
            f"spatial contacts"
            if contact is not None
            else ""
        )
        + (
            f"; {len(lonely)} residues with no partner beyond the null "
            "(linked to their strongest ones)"
            if lonely
            else ""
        )
    )
    return out
