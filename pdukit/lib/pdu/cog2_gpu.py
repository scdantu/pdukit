"""COG 2.1 batched scoring on the GPU.

Scores MANY candidate supports of the same size in one go -- the 300 ball
supports of one calibration size, or the ~12 frontier candidates of one
growth step -- with every covariance kept on the device.  The arithmetic is
`cog2._Direction` / `Cog2Scorer.parts` (both directions, whole-block
conditioning, per-shift null), rearranged so nothing large leaves the GPU:

  * Each block's slow (Koopman) basis is found from ONE Cholesky factor of
    its ridged covariance, C + rI = L L', as the top eigenvectors of
    L^-1 A L^-T (A = symmetrised lagged covariance).  The current code finds
    the same basis via an inverse square root; the two differ only by an
    invertible change of coordinates inside the k-dimensional slow
    subspace, and canonical correlations are invariant to that, so every
    spectrum is unchanged (tests check this to 1e-8).
  * The same factor gives (C + rI)^-1 for the whole-block conditioning by
    triangular solves, so no D x D inverse is ever formed.
  * Shifts are processed in chunks, all supports of the batch at once.

env_basis
  'exact'  the environment's own slow basis (an e x e eigenproblem per
           candidate) -- identical numbers to the numpy path.
  'ritz'   the "one inverse" / Schur idea: the environment's slow basis is
           approximated inside the span of the GLOBAL slowest `ritz_dim`
           directions restricted to the environment (Rayleigh-Ritz), and
           (C_EE + rI)^-1 is applied through the block inverse of one
           global inverse.  No e x e factorisation per candidate at all.
           An approximation: the run records it, and it should be checked
           against 'exact' on one data set before it is relied on.
"""

from __future__ import annotations

import numpy as np


def _isqrt(C, ridge):
    """Batched ridge-regularised inverse square root (see interface)."""
    import torch

    C = 0.5 * (C + C.transpose(-1, -2))
    p = C.shape[-1]
    tr = torch.diagonal(C, dim1=-2, dim2=-1).sum(-1) / max(p, 1)
    scale = torch.clamp(tr, min=1e-30)[..., None, None]
    eye = torch.eye(p, dtype=C.dtype, device=C.device)
    w, V = torch.linalg.eigh(C + ridge * scale * eye)
    wmax = torch.clamp(w.max(dim=-1, keepdim=True).values, min=1e-30)
    w = torch.maximum(w, 1e-14 * wmax)
    return (V / torch.sqrt(w)[..., None, :]) @ V.transpose(-1, -2)


class BatchedEngine:
    """GPU-resident covariances of one FluxWorkspace, and batched `parts`."""

    def __init__(
        self,
        ws,
        *,
        unit_dim,
        env_dim,
        device=None,
        env_basis="exact",
        ritz_dim=None,
        shift_chunk=None,
        reversible=True,
    ):
        """Move the workspace to the device and prepare the shift batches."""
        import torch

        self.th = torch
        if device is None:
            device = (
                "cuda"
                if (
                    getattr(ws, "backend", "") == "torch"
                    and getattr(ws, "device", "") == "cuda"
                )
                else "cpu"
            )
        if env_basis not in ("exact", "ritz"):
            raise ValueError("env_basis must be 'exact' or 'ritz'")
        if not getattr(ws, "reversible", True) or not reversible:
            raise ValueError(
                "the batched engine implements the reversible "
                "(tICA) basis only"
            )
        self.ws, self.dev = ws, device
        self.unit_dim, self.env_dim = int(unit_dim), int(env_dim)
        self.ridge = float(ws.ridge)
        self.env_basis = env_basis
        f64 = torch.float64

        def tensor(a):
            return torch.as_tensor(
                np.asarray(a, dtype=np.float64), device=device, dtype=f64
            )

        self.C00 = tensor(ws.C00)
        self.C01 = tensor(ws.C01)
        self.C11 = tensor(ws.C11)
        self.A = 0.5 * (self.C01 + self.C01.T)
        self._n_shift_cached = -1
        self.shift_chunk = shift_chunk
        if env_basis == "ritz":
            D = self.C00.shape[0]
            r = self.ridge * float(torch.trace(self.C00)) / D
            self.r_glob = r
            Cr = self.C00 + r * torch.eye(D, dtype=f64, device=device)
            self.G = torch.linalg.inv(Cr)
            self.G = 0.5 * (self.G + self.G.T)
            K = int(ritz_dim or 4 * self.env_dim)
            L = torch.linalg.cholesky(Cr)
            Y = torch.linalg.solve_triangular(L, self.A, upper=False)
            M = torch.linalg.solve_triangular(L, Y.T, upper=False)
            w, Z = torch.linalg.eigh(0.5 * (M + M.T))
            Z = Z[:, torch.argsort(w, descending=True)[:K]]
            self.W = torch.linalg.solve_triangular(L.T, Z, upper=True)

    # -- shifts ---------------------------------------------------------------
    def _shifts(self):
        th = self.th
        n = len(self.ws.Cs00)
        if n != self._n_shift_cached:
            self.X00 = th.stack(
                [th.as_tensor(np.asarray(X, np.float64)) for X in self.ws.Cs00]
            ).to(self.dev)
            self.X01 = th.stack(
                [th.as_tensor(np.asarray(X, np.float64)) for X in self.ws.Cs01]
            ).to(self.dev)
            self._n_shift_cached = n
        return self.X00, self.X01

    # -- building blocks
    # --------------------------------------------------------
    @staticmethod
    def _g(M, R, C):
        """M[R_b, C_b] for a batch of row/column index tensors -> (B, r, c)."""
        return M[R[:, :, None], C[:, None, :]]

    def _chol(self, C):
        th = self.th
        p = C.shape[-1]
        tr = th.diagonal(C, dim1=-2, dim2=-1).sum(-1) / p
        r = self.ridge * th.clamp(tr, min=1e-30)
        eye = th.eye(p, dtype=C.dtype, device=C.device)
        return th.linalg.cholesky(
            0.5 * (C + C.transpose(-1, -2)) + r[:, None, None] * eye
        )

    def _slow(self, L, A, k):
        """Top-k generalized eigenvectors of A v = lam (C + rI) v."""
        th = self.th
        Y = th.linalg.solve_triangular(L, A, upper=False)
        M = th.linalg.solve_triangular(L, Y.transpose(-1, -2), upper=False)
        w, Z = th.linalg.eigh(0.5 * (M + M.transpose(-1, -2)))
        idx = th.argsort(w, dim=-1, descending=True)[:, :k]
        Z = th.gather(Z, 2, idx[:, None, :].expand(-1, Z.shape[1], -1))
        return th.linalg.solve_triangular(L.transpose(-1, -2), Z, upper=True)

    def _solve(self, L, X):
        """(C + rI)^-1 X from the Cholesky factor."""
        return self.th.cholesky_solve(X, L)

    def _env_solve_ritz(self, S, E, X):
        """(C_EE + r I)^-1 X via the block inverse of the global G."""
        th = self.th
        G = self.G
        GEE = self._g(G, E, E)
        GES = self._g(G, E, S)
        GSS = self._g(G, S, S)
        return GEE @ X - GES @ th.linalg.solve(GSS, GES.transpose(1, 2) @ X)

    # -- the batched equivalent of Cog2Scorer.parts
    # --------------------------------
    def parts(self, S_list, E_list, with_null):
        """S_list, E_list: equal-length column-index arrays (all S the same
        length, all E the same length).  Returns a list of dicts with
        rho_in, rho_out, null_in, null_out, sigma (numpy).
        """
        th = self.th
        dev = self.dev
        S = th.as_tensor(np.stack(S_list), device=dev)
        E = th.as_tensor(np.stack(E_list), device=dev)
        B, s = S.shape
        e = E.shape[1]
        ku, ke = min(self.unit_dim, s), min(self.env_dim, e)
        g = self._g
        C00, C01, C11, A = self.C00, self.C01, self.C11, self.A

        # unit block: factor, slow basis, conditioning inverse
        C00_SS = g(C00, S, S)
        L_S = self._chol(C00_SS)
        Vu = self._slow(L_S, g(A, S, S), ku)  # (B, s, ku)
        # environment block
        if self.env_basis == "exact":
            C00_EE = g(C00, E, E)
            L_E = self._chol(C00_EE)
            Ve = self._slow(L_E, g(A, E, E), ke)  # (B, e, ke)
            envsolve = lambda X: self._solve(L_E, X)  # noqa: E731
        else:
            WE = self.W[E]  # (B, e, K)
            AW = g(A, E, E) @ WE
            CW = g(C00, E, E) @ WE + self.r_glob * WE
            Ak = WE.transpose(1, 2) @ AW
            Ck = WE.transpose(1, 2) @ CW
            Lk = th.linalg.cholesky(0.5 * (Ck + Ck.transpose(1, 2)))
            Ve = WE @ self._slow(Lk, 0.5 * (Ak + Ak.transpose(1, 2)), ke)
            envsolve = lambda X: self._env_solve_ritz(S, E, X)  # noqa: E731

        out = [dict() for _ in range(B)]
        T = lambda M: M.transpose(-1, -2)  # noqa: E731

        # ---------------- outgoing: unit present -> env future -------------
        C01_EE = g(C01, E, E)
        Cff_o = T(Ve) @ g(C11, E, E) @ Ve
        Cpf_o = C01_EE @ Ve  # (B, e, ke)
        IpCpf_o = envsolve(Cpf_o)
        Rff_o = Cff_o - T(Cpf_o) @ IpCpf_o
        Lff_o = _isqrt(Rff_o, self.ridge)
        Css_o = T(Vu) @ C00_SS @ Vu

        def spec_out(X00_SE, X01_SE):
            Csp = T(Vu) @ X00_SE  # (B, ku, e)
            Csf = T(Vu) @ X01_SE @ Ve
            IpCspT = envsolve(T(Csp))
            Rss = Css_o - Csp @ IpCspT
            Rsf = Csf - Csp @ IpCpf_o
            M = Lff_o @ T(Rsf) @ _isqrt(Rss, self.ridge)
            return th.clamp(th.linalg.svdvals(M), 0.0, 1.0)

        rho_out = spec_out(g(C00, S, E), g(C01, S, E))

        # ---------------- incoming: env present -> unit future -------------
        Cff_i = T(Vu) @ g(C11, S, S) @ Vu
        Cpf_i = g(C01, S, S) @ Vu
        IpCpf_i = self._solve(L_S, Cpf_i)
        Rff_i = Cff_i - T(Cpf_i) @ IpCpf_i
        Lff_i = _isqrt(Rff_i, self.ridge)
        Css_i = T(Ve) @ g(C00, E, E) @ Ve
        sigma = th.diagonal(T(Cpf_i) @ IpCpf_i, dim1=-2, dim2=-1).sum(
            -1
        ) / th.clamp(th.diagonal(Cff_i, dim1=-2, dim2=-1).sum(-1), min=1e-300)

        def spec_in(X00_ES, X01_ES):
            Csp = T(Ve) @ X00_ES  # (B, ke, s)
            Csf = T(Ve) @ X01_ES @ Vu
            Rss = Css_i - Csp @ self._solve(L_S, T(Csp))
            Rsf = Csf - Csp @ IpCpf_i
            M = Lff_i @ T(Rsf) @ _isqrt(Rss, self.ridge)
            return th.clamp(th.linalg.svdvals(M), 0.0, 1.0)

        rho_in = spec_in(g(C00, E, S), g(C01, E, S))

        null_in = null_out = None
        if with_null:
            X00, X01 = self._shifts()
            n = X00.shape[0]
            # chunk the shifts so a (chunk, B, s, e) gather stays < ~1 GB
            per = max(B * s * e * 8, 1)
            ch = self.shift_chunk or max(1, min(n, int(1.0e9 // per)))
            ni, no = [], []
            for j0 in range(0, n, ch):
                for j in range(j0, min(n, j0 + ch)):
                    no.append(spec_out(g(X00[j], S, E), g(X01[j], S, E))[:, 0])
                    ni.append(spec_in(g(X00[j], E, S), g(X01[j], E, S))[:, 0])
            null_out = th.stack(no, dim=1).cpu().numpy()  # (B, n)
            null_in = th.stack(ni, dim=1).cpu().numpy()

        ri, ro = rho_in.cpu().numpy(), rho_out.cpu().numpy()
        sg = sigma.cpu().numpy()
        for b in range(B):
            out[b] = {
                "rho_in": ri[b],
                "rho_out": ro[b],
                "null_in": None if null_in is None else null_in[b],
                "null_out": None if null_out is None else null_out[b],
                "sigma": float(sg[b]),
            }
        return out
