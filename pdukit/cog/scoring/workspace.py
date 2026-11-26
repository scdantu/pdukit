"""The flux workspace: everything needed to score a support set, built once
from the features and reused for every candidate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pdukit.cog.scoring.backend import _resolve_backend
from pdukit.numerics.kernel import _inv_sqrt


@dataclass
class FluxResult:
    """Boundary-flux spectrum for one candidate support."""

    support: list  # residue indices
    size: int
    n_pairs: int  # usable lagged pairs
    lag: int
    rho: list  # canonical correlations, descending
    rho_null_p95: float  # 95th pct of leading rho under shift null
    k_star: int  # channels above the null
    k_eff: float  # participation ratio of rho^2 (continuous)
    coupling_total: float  # sum rho^2 -- the MAGNITUDE of the flux
    delta_int_curve: list  # cumulative fraction of flux by channel
    unit_dim: int
    env_dim: int


class FluxWorkspace:
    """Precomputed lagged covariances, so scoring a candidate never touches
    the trajectory again.

    WHY THIS EXISTS.  The direct implementation PCAs the environment inside
    every call, which costs O(n q^2).  At mdCATH size (n ~ 2500) that is a
    second; on 114,000 frames of lysozyme it is a minute per candidate, and a
    single growth run needs hundreds of candidates.  Every quantity the
    estimator needs is a function of three D x D matrices, so those are built
    once in a couple of passes and each candidate afterwards is submatrix
    algebra independent of n.

    The mathematics is unchanged -- same residualisation, same canonical
    correlations, same shift null.  `test_workspace_matches_direct_path`
    asserts the two agree.

    Notation, with X0 the resolved features at t and X1 at t+tau, both centred:

        C00 = X0'X0/n     C01 = X0'X1/n     C11 = X1'X1/n

    For a candidate with feature columns S and environment columns E, writing
    Vu, Ve for the unit and environment PCA bases:

        Chh = Vu' C00[S,S] Vu    Cyy = Vu' C11[S,S] Vu    Chy = Vu' C01[S,S] Vu
        Czz = Ve' C00[E,E] Ve    Chz = Vu' C00[S,E] Ve    Czy = Ve' C01[E,S] Vu

    and residualising the unit's future and the environment on the unit's own
    present gives, with no approximation,

        Ryy = Cyy - Chy' Chh^-1 Chy
        Rzz = Czz - Chz' Chh^-1 Chz
        Rzy = Czy - Chz' Chh^-1 Chy

    whose whitened singular values are the flux spectrum.

    The shift null needs the environment displaced in time against a fixed
    unit, so it additionally stores, per shift j,

        Cs00[j] = X0_shifted' X0 / n      Cs01[j] = X0_shifted' X1 / n

    The shifted environment keeps its own covariance (a circular shift is a
    permutation of rows), so Czz and both PCA bases are reused unchanged.

    GPU.  Pass ``device="cuda"`` (or "auto") to run through PyTorch.  Both
    phases benefit: the build is a handful of (n x D)' (n x D) matmuls, and
    each candidate is an eigh plus an svd on matrices of order D.  The
    accumulation runs in float32 and the D x D algebra in float64, which keeps
    the conditioning of the whitening step intact on a GH200.  Without torch,
    or without a visible GPU, it falls back to numpy -- and records that in
    ``backend``, because a silent fallback to CPU on a GPU job would be worse
    than a slow one you know about.
    """

    def __init__(
        self,
        feats,
        lag=1,
        segments=None,
        n_null=8,
        seed=0,
        ridge=1e-2,
        dtype=np.float32,
        on_progress=None,
        device=None,
        chunk=8192,
        reversible=True,
        truncation="koopman",
        shift_mode="global",
    ):
        """Prepare the covariances of the replicas for scoring supports."""
        A = np.asarray(feats)
        if A.ndim != 3:
            raise ValueError("feats must be (F, N, m)")
        self.F, self.N, self.m = A.shape
        self.D = self.N * self.m
        self.lag = int(lag)
        self.ridge = float(ridge)
        self.n_null = int(n_null)
        self.reversible = bool(reversible)
        if truncation not in ("koopman", "pca"):
            raise ValueError("truncation must be 'koopman' or 'pca'")
        # 'pca' exists only so the equal-time path can be compared against the
        # direct implementation; 'koopman' is the estimator.
        self.truncation = truncation
        # How the null displaces the environment in time (COG 2.1 fix):
        #   global  circular shift over the concatenated trajectory, as
        #           before -- can wrap across replica seams; kept as the
        #           default so earlier results reproduce exactly
        #   within  circular shift inside each replica (segment)
        #   cross   environment taken from a DIFFERENT replica, at a random
        #           offset -- exactly independent of the unit, no residual
        #           slow correlation from a short shift
        if shift_mode not in ("global", "within", "cross"):
            raise ValueError(
                "shift_mode must be 'global', 'within' or 'cross'"
            )
        self.shift_mode = shift_mode
        self.backend, self._torch, self.device = _resolve_backend(device)

        # A view when feats is already the right dtype and C-contiguous, so the
        # trajectory is not duplicated just to be reshaped.  Kept as an
        # attribute (not a local) so `extend_null_shifts` can build further
        # shift pairs later without re-deriving it.
        self._flat = np.ascontiguousarray(A, dtype=dtype).reshape(
            self.F, self.D
        )

        if segments is None:
            segments = [(0, self.F)]
        self.segments = [tuple(map(int, sg)) for sg in segments]
        usable = [(a, b) for a, b in self.segments if b - a > self.lag]
        if not usable:
            raise ValueError(
                "no usable lagged pairs; lag too large for segments"
            )
        self.n = int(sum(b - a - self.lag for a, b in usable))
        self.rows0 = np.concatenate(
            [np.arange(a, b - self.lag) for a, b in usable]
        )
        self.rows1 = np.concatenate(
            [np.arange(a + self.lag, b) for a, b in usable]
        )

        self._step = max(1024, int(chunk))
        total_ticks = 1 + 2 * self.n_null
        tick = [0]

        def _tick():
            tick[0] += 1
            if on_progress is not None:
                on_progress(tick[0], total_ticks)

        S00, s0, _ = self._gram(self.rows0, self.rows0)
        S01, _, s1 = self._gram(self.rows0, self.rows1)
        S11, _, _ = self._gram(self.rows1, self.rows1)
        self.m0 = s0 / self.n
        self.m1 = s1 / self.n
        self.C00 = S00 / self.n - np.outer(self.m0, self.m0)
        self.C01 = S01 / self.n - np.outer(self.m0, self.m1)
        self.C11 = S11 / self.n - np.outer(self.m1, self.m1)
        _tick()

        # Kept so `extend_null_shifts` can continue the same draw sequence:
        # a workspace built with n_null=k and then extended by n_more matches
        # one built with n_null=k+n_more directly, because both draw from the
        # same np.random.default_rng(seed) stream in the same order.
        self._rng = np.random.default_rng(seed)
        self._usable = usable
        if self.shift_mode == "cross" and len(usable) < 2:
            raise ValueError(
                "shift_mode='cross' needs at least two segments (replicas)"
            )
        self.Cs00, self.Cs01 = [], []
        for _ in range(self.n_null):
            rs = self._shifted_rows(self._rng)
            Ss0, ms, _ = self._gram(rs, self.rows0)
            self.Cs00.append(Ss0 / self.n - np.outer(ms / self.n, self.m0))
            _tick()
            Ss1, ms1, _ = self._gram(rs, self.rows1)
            self.Cs01.append(Ss1 / self.n - np.outer(ms1 / self.n, self.m1))
            _tick()

    def extend_null_shifts(self, n_more, seed=None):
        """Append `n_more` more (Cs00, Cs01) shift pairs to the shared null
        pool, using the same construction `__init__` uses for its own.

        With `seed` left as `None` this continues the SAME rng stream
        `__init__` started, so a workspace built with `n_null=k` and then
        extended by `n_more` here draws identically to one built with
        `n_null=k + n_more` directly -- there is no special case for shifts
        added later.  Pass an explicit `seed` to draw an independent batch
        instead.
        """
        rng = self._rng if seed is None else np.random.default_rng(seed)
        for _ in range(int(n_more)):
            rs = self._shifted_rows(rng)
            Ss0, ms, _ = self._gram(rs, self.rows0)
            self.Cs00.append(Ss0 / self.n - np.outer(ms / self.n, self.m0))
            Ss1, ms1, _ = self._gram(rs, self.rows1)
            self.Cs01.append(Ss1 / self.n - np.outer(ms1 / self.n, self.m1))
        self.n_null = len(self.Cs00)

    # -- helpers ----------------------------------------------------------
    def _shifted_rows(self, rng):
        """Row indices of the displaced environment, one per lagged pair.

        'global' draws exactly as before (one integer per shift), so a
        workspace built with the default reproduces earlier results.
        """
        if self.shift_mode == "global":
            min_shift = max(10 * self.lag, self.F // 10, 5)
            sh = int(
                rng.integers(min_shift, max(min_shift + 1, self.F - min_shift))
            )
            return (self.rows0 - sh) % self.F
        segs = [(int(a), int(b)) for a, b in self._usable]
        out = np.empty_like(self.rows0)
        pos = 0
        if self.shift_mode == "within":
            u = float(rng.random())
            for a, b in segs:
                L = b - a
                n_seg = L - self.lag
                ms = max(10 * self.lag, L // 10, 5)
                span = max(L - 2 * ms, 1)
                sh = ms + int(u * span)
                r = self.rows0[pos : pos + n_seg]
                out[pos : pos + n_seg] = a + ((r - a - sh) % L)
                pos += n_seg
            return out
        # cross: a random derangement of the segments plus a random offset
        k = len(segs)
        while True:
            perm = rng.permutation(k)
            if np.all(perm != np.arange(k)):
                break
        u = float(rng.random())
        for i, (a, b) in enumerate(segs):
            n_seg = (b - a) - self.lag
            ta, tb = segs[int(perm[i])]
            Lt = tb - ta
            off = int(u * Lt)
            r = self.rows0[pos : pos + n_seg]
            out[pos : pos + n_seg] = ta + ((r - a + off) % Lt)
            pos += n_seg
        return out

    def _gram(self, a_rows, b_rows):
        """(sum a'b, sum a, sum b) over paired row-index arrays, chunked.

        Never materialises a full (n x D) copy: at 114k frames that is
        450 MB per copy and there would be four of them live at once.
        """
        S = np.zeros((self.D, self.D), dtype=np.float64)
        sa = np.zeros(self.D, dtype=np.float64)
        sb = np.zeros(self.D, dtype=np.float64)
        use_torch = self.backend == "torch"
        if use_torch:
            th = self._torch
            dev = th.device(self.device)
            T = th.zeros((self.D, self.D), dtype=th.float32, device=dev)
        for c0 in range(0, a_rows.size, self._step):
            c1 = min(c0 + self._step, a_rows.size)
            Ac = self._flat[a_rows[c0:c1]]
            Bc = self._flat[b_rows[c0:c1]]
            if use_torch:
                ta = self._torch.as_tensor(Ac, device=dev)
                tb = self._torch.as_tensor(Bc, device=dev)
                T += ta.T @ tb
            else:
                S += Ac.T.astype(np.float64) @ Bc.astype(np.float64)
            sa += Ac.sum(axis=0, dtype=np.float64)
            sb += Bc.sum(axis=0, dtype=np.float64)
        if use_torch:
            # Apple MPS has no float64 at all, so the cast to double must
            # happen after the tensor is back on the host, never on device.
            S = T.cpu().numpy().astype(np.float64)
        return S, sa, sb

    def _cols(self, support):
        support = np.asarray(sorted(set(int(i) for i in support)), dtype=int)
        if support.size == 0 or support.min() < 0 or support.max() >= self.N:
            raise ValueError("support out of range")
        mask = np.zeros(self.N, dtype=bool)
        mask[support] = True
        col = np.repeat(mask, self.m)
        return support, np.where(col)[0], np.where(~col)[0]

    def _basis(self, C, k):
        """Top-k eigenvectors of a symmetric matrix (variance directions).

        Retained only for the equal-time fallback; the estimator uses
        `_koopman_basis`.  See that method for why.
        """
        n = C.shape[0]
        k = int(min(k, n))
        if k < n // 4:
            try:
                from scipy.linalg import eigh as _seigh

                w, V = _seigh(
                    C,
                    subset_by_index=[n - k, n - 1],
                    check_finite=False,
                    driver="evx",
                )
                return np.ascontiguousarray(V[:, ::-1])
            except Exception:  # noqa: BLE001
                pass
        w, V = np.linalg.eigh(C)
        return np.ascontiguousarray(V[:, np.argsort(w)[::-1][:k]])

    def _koopman_basis(self, idx, k, reversible=True):
        """Top-k SLOW directions of a block -- tICA/VAMP, not PCA.

        WHY THIS REPLACED PCA.  Truncating with the equal-time covariance keeps
        the directions of largest AMPLITUDE.  Slowness and amplitude are
        different things: a slow collective mode can be small and be discarded
        in the first few components, while a fast high-amplitude local rattle
        is kept.  Doing that before a lagged-covariance estimator throws away
        exactly the dynamics the estimator exists to find -- the classic reason
        `koopman.py` whitens with C(0) and then diagonalises C(tau) rather than
        running PCA first.

        Reversible branch (the default, matching `koopman.tica`): whiten the
        block with its equal-time covariance, symmetrise the lagged covariance,
        and take the leading eigenvectors --

            L = C00^-1/2 ,  A = L (C01 + C01') L / 2 ,  V = L U_k

        Non-reversible branch (VAMP), for driven or non-equilibrium data:

            M = C00^-1/2 C01 C11^-1/2 ,  V = C00^-1/2 U_k  from the SVD of M

        Returns the (block_dim, k) map from block features to slow coordinates.
        """
        C0 = self.C00[np.ix_(idx, idx)]
        C1 = self.C01[np.ix_(idx, idx)]
        n = C0.shape[0]
        k = int(min(k, n))

        # GPU path.  This is the per-candidate hot spot once the truncation is
        # Koopman -- two inverse square roots plus a decomposition on a matrix
        # of order D -- so on CUDA it belongs on the device.  Needs float64, so
        # CUDA only; Apple MPS has no double precision and stays on the host.
        if self.backend == "torch" and self.device == "cuda" and n > 256:
            th = self._torch
            dev = th.device("cuda")
            t0 = th.as_tensor(C0, device=dev, dtype=th.float64)
            t1 = th.as_tensor(C1, device=dev, dtype=th.float64)
            t0 = 0.5 * (t0 + t0.T)
            scale = th.clamp(th.trace(t0) / max(n, 1), min=1e-30)
            t0 = t0 + self.ridge * scale * th.eye(
                n, device=dev, dtype=th.float64
            )
            w0, V0 = th.linalg.eigh(t0)
            w0 = th.clamp(w0, min=1e-14 * float(w0.max()))
            L = (V0 / th.sqrt(w0)) @ V0.T
            if reversible:
                A = L @ (0.5 * (t1 + t1.T)) @ L
                A = 0.5 * (A + A.T)
                w, U = th.linalg.eigh(A)
                U = U[:, th.argsort(w, descending=True)[:k]]
            else:
                tt = th.as_tensor(
                    self.C11[np.ix_(idx, idx)], device=dev, dtype=th.float64
                )
                tt = 0.5 * (tt + tt.T)
                st = th.clamp(th.trace(tt) / max(n, 1), min=1e-30)
                tt = tt + self.ridge * st * th.eye(
                    n, device=dev, dtype=th.float64
                )
                wt, Vt = th.linalg.eigh(tt)
                wt = th.clamp(wt, min=1e-14 * float(wt.max()))
                Ly = (Vt / th.sqrt(wt)) @ Vt.T
                U, _, _ = th.linalg.svd(L @ t1 @ Ly, full_matrices=False)
                U = U[:, :k]
            return np.ascontiguousarray((L @ U).cpu().numpy())

        L = _inv_sqrt(C0, self.ridge)
        if reversible:
            A = L @ (0.5 * (C1 + C1.T)) @ L
            A = 0.5 * (A + A.T)
            if k < n // 4:
                try:
                    from scipy.linalg import eigh as _seigh

                    w, U = _seigh(
                        A,
                        subset_by_index=[n - k, n - 1],
                        check_finite=False,
                        driver="evx",
                    )
                    U = np.ascontiguousarray(U[:, ::-1])
                except Exception:  # noqa: BLE001
                    w, U = np.linalg.eigh(A)
                    U = np.ascontiguousarray(U[:, np.argsort(w)[::-1][:k]])
            else:
                w, U = np.linalg.eigh(A)
                U = np.ascontiguousarray(U[:, np.argsort(w)[::-1][:k]])
        else:
            Ly = _inv_sqrt(self.C11[np.ix_(idx, idx)], self.ridge)
            M = L @ C1 @ Ly
            U, sv, _ = np.linalg.svd(M, full_matrices=False)
            U = np.ascontiguousarray(U[:, :k])
        return np.ascontiguousarray(L @ U)

    def _spectrum(self, Ryy, Rzz, Rzy):
        M = _inv_sqrt(Ryy, self.ridge) @ Rzy.T @ _inv_sqrt(Rzz, self.ridge)
        return np.clip(np.linalg.svd(M, compute_uv=False), 0.0, 1.0)

    def _project_and_residualise(self, support, unit_dim=8, env_dim=40):
        """Unit/environment bases, residualised covariances, and the real
        (non-null) canonical-correlation spectrum for one candidate.

        Everything `score()` needs before it either recomputes a per-
        candidate shift null (the default) or looks one up in a pre-
        calibrated table (`null_thresholds`).  Split out so
        `calibrate_null_thresholds` can share this exact computation instead
        of duplicating it.
        """
        support, S, E = self._cols(support)
        if E.size == 0:
            raise ValueError(
                "candidate covers the whole protein: no environment"
            )

        if self.truncation == "koopman":
            Vu = self._koopman_basis(S, unit_dim, self.reversible)
            Ve = self._koopman_basis(E, env_dim, self.reversible)
        else:
            Vu = self._basis(self.C00[np.ix_(S, S)], unit_dim)
            Ve = self._basis(self.C00[np.ix_(E, E)], env_dim)

        Chh = Vu.T @ self.C00[np.ix_(S, S)] @ Vu
        Cyy = Vu.T @ self.C11[np.ix_(S, S)] @ Vu
        Chy = Vu.T @ self.C01[np.ix_(S, S)] @ Vu
        Czz = Ve.T @ self.C00[np.ix_(E, E)] @ Ve
        Chz = Vu.T @ self.C00[np.ix_(S, E)] @ Ve

        Ih = _inv_sqrt(Chh, self.ridge)
        Ih = Ih @ Ih  # ridge-regularised Chh^-1
        Ryy = Cyy - Chy.T @ Ih @ Chy
        Rzz = Czz - Chz.T @ Ih @ Chz

        Czy = Ve.T @ self.C01[np.ix_(E, S)] @ Vu
        rho = self._spectrum(Ryy, Rzz, Czy - Chz.T @ Ih @ Chy)

        return support, S, E, Vu, Ve, Ih, Ryy, Chy, Czz, Cyy, rho

    def _leading_null_rho(
        self, Ih, Ryy, Chy, Vu, Ve, Czz, S, E, Cs00_j, Cs01_j
    ):
        """Leading canonical correlation against one shift-null replica,
        for a candidate whose real-spectrum quantities are already computed
        (by `_project_and_residualise`).
        """
        Chz_n = Vu.T @ Cs00_j[np.ix_(E, S)].T @ Ve
        Czy_n = Ve.T @ Cs01_j[np.ix_(E, S)] @ Vu
        Rzz_n = Czz - Chz_n.T @ Ih @ Chz_n
        return float(self._spectrum(Ryy, Rzz_n, Czy_n - Chz_n.T @ Ih @ Chy)[0])

    # -- public -----------------------------------------------------------
    def score(self, support, unit_dim=8, env_dim=40, null_thresholds=None):
        """Flux spectrum for one candidate; same output shape as
        `flux_spectrum`.

        With `null_thresholds` left as `None` (the default), the null
        threshold is recomputed for this one candidate from the workspace's
        own shift replicas, exactly as before.  Passing a `{size: theta}`
        mapping -- e.g. the `"theta"` table from `calibrate_null_thresholds`
        -- skips that per-candidate computation entirely and looks up
        `thr = null_thresholds[support.size]` instead; a size missing from
        the table is an error rather than a silent fall-back, since mixing
        the two threshold designs in one run would be silent and wrong.
        """
        support, S, E, Vu, Ve, Ih, Ryy, Chy, Czz, Cyy, rho = (
            self._project_and_residualise(support, unit_dim, env_dim)
        )

        if null_thresholds is None:
            leading = [
                self._leading_null_rho(
                    Ih, Ryy, Chy, Vu, Ve, Czz, S, E, Cs00, Cs01
                )
                for Cs00, Cs01 in zip(self.Cs00, self.Cs01)
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

        # SELF-PREDICTION (v3.1 item 1).  Eq. (13) residualises the unit's
        # future by subtracting the part its own present explains; sigma is
        # that subtracted fraction -- how much of the unit's future the unit
        # itself accounts for.  k_eff is monotone non-increasing in |S| and
        # floors at 1, so it carries no size information; sigma is the term
        # that might.  Reported always, used only when cohesion_lambda > 0.
        tr_yy = float(np.trace(Cyy))
        sigma = (
            (float(np.trace(Chy.T @ Ih @ Chy)) / tr_yy)
            if tr_yy > 0
            else float("nan")
        )

        # NUCLEAR-TO-SPECTRAL RATIO (v3.1 item 6) on the null-corrected
        # amplitudes.  k_eff is a participation ratio, so (0.8, 0.2) reads
        # 1.12 and a weak second channel is nearly free to a greedy search;
        # this reads 1.25 and charges for it.
        rhoc = np.sqrt(rho2c)
        nuc_spec = (
            float(rhoc.sum() / rhoc[0])
            if rhoc.size and rhoc[0] > 0
            else float("nan")
        )

        return {
            "support": support.tolist(),
            "size": int(support.size),
            "n_pairs": self.n,
            "lag": self.lag,
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
            "unit_dim": int(Vu.shape[1]),
            "env_dim": int(Ve.shape[1]),
        }
