"""Pair-owned features: a residue-pair feature belongs to the pair.

See the module notes in scorer.py (PAIR-OWNED FEATURES).
"""

from __future__ import annotations

import numpy as np

from pdukit.cog.scoring.workspace import FluxWorkspace

# ---------------------------------------------------------------------------
# pair-owned features (Stage 4)
# ---------------------------------------------------------------------------


class PairFluxWorkspace(FluxWorkspace):
    """FluxWorkspace whose feature columns are owned by residues OR by
    residue pairs.

    pair_feats (F, P, mp) with `pairs` (P, 2) residue indices; optional
    residue_feats (F, N, m).  For a candidate support S, a column is a unit
    column if ALL its owners are in S, an environment column if NONE is, and
    is excluded otherwise (a pair straddling the boundary IS the boundary,
    and counting it on either side would put the interface inside one of
    them).  Everything else -- covariances, shifts, Koopman bases -- is the
    parent's code on the concatenated columns.

    Not usable with the Schur (full-environment) backend: its block-inverse
    identity needs unit and environment columns to cover every column.
    """

    def __init__(
        self, pair_feats, pairs, n_residues, residue_feats=None, **kw
    ):
        """Build the workspace from pair-owned (and residue) features."""
        blocks, owners = [], []
        if residue_feats is not None:
            R = np.asarray(residue_feats)
            F, N, m = R.shape
            if N != int(n_residues):
                raise ValueError("residue_feats has the wrong residue count")
            blocks.append(R.reshape(F, N * m))
            owners += [(i,) for i in range(N) for _ in range(m)]
        Pf = np.asarray(pair_feats)
        if Pf.ndim == 2:
            Pf = Pf[:, :, None]
        F, P, mp = Pf.shape
        pairs = np.asarray(pairs, dtype=int).reshape(-1, 2)
        if pairs.shape[0] != P:
            raise ValueError("pairs and pair_feats disagree on P")
        blocks.append(Pf.reshape(F, P * mp))
        owners += [(int(a), int(b)) for a, b in pairs for _ in range(mp)]
        flat = np.concatenate(blocks, axis=1)
        super().__init__(flat[:, :, None], **kw)
        self.N = int(n_residues)
        self.col_owners = owners
        own = np.zeros((self.D, self.N), dtype=bool)
        for c, o in enumerate(owners):
            own[c, list(o)] = True
        self._own = own
        self._n_own = own.sum(axis=1)
        self.pairs = pairs
        self.has_boundary_columns = True

    def _cols(self, support):
        support = np.asarray(sorted(set(int(i) for i in support)), dtype=int)
        if support.size == 0 or support.min() < 0 or support.max() >= self.N:
            raise ValueError("support out of range")
        mask = np.zeros(self.N, dtype=bool)
        mask[support] = True
        n_in = self._own[:, mask].sum(axis=1)
        S = np.where(n_in == self._n_own)[0]
        E = np.where(n_in == 0)[0]
        if S.size == 0:
            raise ValueError("support owns no feature columns")
        return support, S, E


def pair_distance_features(
    ca_coords,
    *,
    cutoff=8.0,
    min_seq_sep=3,
    dmed=None,
    max_frames=200,
    standardise=True,
    dtype=np.float32,
):
    """Residue-pair distances for every pair in contact (median distance
    below `cutoff`, |i - j| >= min_seq_sep).  Returns (feats (F, P, 1),
    pairs (P, 2)).  Distance units are whatever ca_coords uses; pass
    `cutoff` in the same units (or a `dmed` from `median_distance_matrix`).
    """
    X = np.asarray(ca_coords)
    F, N, _ = X.shape
    if dmed is None:
        idx = np.linspace(0, F - 1, min(F, max_frames)).astype(int)
        dmed = np.median(
            np.linalg.norm(X[idx][:, :, None] - X[idx][:, None], axis=-1),
            axis=0,
        )
    ii, jj = np.where(np.triu(dmed < cutoff, k=int(min_seq_sep)))
    pairs = np.stack([ii, jj], axis=1)
    out = np.empty((F, len(pairs), 1), dtype=dtype)
    step = 2048
    for a in range(0, F, step):
        b = min(F, a + step)
        d = np.linalg.norm(X[a:b, ii] - X[a:b, jj], axis=-1)
        out[a:b, :, 0] = d
    if standardise:
        mu = out.mean(axis=0, keepdims=True)
        sd = out.std(axis=0, keepdims=True)
        out = (out - mu) / np.where(sd > 0, sd, 1.0)
    return out.astype(dtype), pairs
