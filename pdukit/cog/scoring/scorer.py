"""COG 2.1: two-way interface spectra, soft channel count, matched baseline,
cohesion check and pair-owned features.

Plain-language summary (the watch).  A unit is a wheel-train that talks to
the rest of the watch through a few gears.  COG v1.0 counted only the gears
that DRIVE the unit (environment -> unit, "incoming").  A unit can also DRIVE
the rest (unit -> environment, "outgoing"); a wheel that only pushes looked
perfectly closed to v1.0.  COG 2.1 counts both, compares the count with what
an ordinary compact blob of the same size shows, and then asks whether the
unit hangs together inside (cohesion) or is two wheels glued by geography.

THE TWO DIRECTIONS (block model K = [[A, B], [C, D]], unit first).

  incoming  r_in  ~ rank B : canonical correlations between the unit's future
            and the environment's present, both residualised on the unit's
            own present.  Identical to `FluxWorkspace.score()`.
  outgoing  r_out ~ rank C : canonical correlations between the
            ENVIRONMENT's future and the unit's present, both residualised on
            the environment's own present.

Both answer the same question (computed by `FluxScorer`, which replaced
the generic function `directional_spectrum` in 1.0.2a):
"how many independent channels does the source's present carry about the
target's future, beyond what the target already knows about itself".  The
shift null displaces the SOURCE in time (rows of the workspace's Cs00/Cs01),
so it is the same construction in both directions.

SOFT CHANNEL COUNT.  With null threshold theta and canonical correlations
rho_i, each channel counts

        nu_i = max(rho_i^2 - theta^2, 0) / (max(rho_i^2 - theta^2, 0) +
        theta^2) = max(0, 1 - theta^2 / rho_i^2)

i.e. one minus the fraction of the channel's squared correlation that noise
alone could produce.  nu = sum_i nu_i.  It equals the hard count k* when the
channels are far above noise, goes smoothly to 0 at the noise edge, and has
no free scale: the scale is the null itself.  (`eta` overrides theta^2 as the
soft scale if wanted.)

COST.  nu(S) = max(nu_in, nu_out) by default ("the unit is as open as its
more open side"), and

        cost(S) = nu(S) - median nu over compact ball supports of size |S|

so a candidate is rewarded for being MORE closed than an ordinary compact set
of its size, not for being small.  The baseline pools ball supports of sizes
|S| +- `baseline_window` (default 2) to keep its sampling noise out of the
growth curve.  `p_size` is the fraction of those ball
supports that are at least as closed (small p = unusually closed).

CALIBRATION (`calibrate_cog2`) estimates, per size, theta_in, theta_out and the
ball baseline, with the two-way (supports x shifts) bootstrap stopping rule
and a workspace fingerprint; `FluxScorer` refuses a calibration made on other
data or settings.

COHESION (`cohesion_check`, Stage 3).  Bisect the unit by spectral
clustering of its internal residue coupling, and measure the channel count
across the internal cut (both directions, own shift null, conditioned on the
environment's slow modes so a common outside driver does not count).  A real
unit's halves talk to each other at least as much as the unit talks to the
outside; two independent wheels glued together show ~0 channels across the
cut.

PAIR-OWNED FEATURES (`PairFluxWorkspace`, Stage 4).  A residue-pair feature
(e.g. a contact distance) belongs to the PAIR.  For a candidate unit S it is
a unit feature if both residues are in S, an environment feature if neither
is, and EXCLUDED if the pair straddles the boundary (it is the boundary).

Approximation to note.  With shift_mode 'within' or 'cross' the displaced
source's own covariance is not exactly the unshifted one (a different
multiset of frames); the code, like `FluxWorkspace`, reuses the unshifted
covariance.  With long replicas the difference is sampling noise.
"""

from __future__ import annotations

import numpy as np

from pdukit.cog.calibration.null import (
    check_null_calibration,
    complete_theta_table,
)
from pdukit.numerics.kernel import _inv_sqrt

STATISTIC = "cog2_two_way_soft_count"


VERSION = "COG 2.1"


# ---------------------------------------------------------------------------
# soft channel count
# ---------------------------------------------------------------------------

# Minimum channel strength (effect-size floor).  A channel counts only if
# its correlation is above max(noise threshold, RHO_MIN).  0 = off.  Set
# once per run by the pipeline (rho-min); applies to units, baseline regions,
# calibration and the cohesion cut alike.
RHO_MIN = 0.0


def soft_count(rho, theta, eta=None):
    """Nu = sum_i s_i^2 / (s_i^2 + eta), s_i^2 = max(rho_i^2 - theta^2, 0),
    eta = theta^2 by default (so nu_i = 1 - theta^2/rho_i^2 above the edge).
    theta is raised to RHO_MIN when that is larger.
    """
    rho = np.asarray(rho, dtype=np.float64)
    th2 = max(float(theta), float(RHO_MIN)) ** 2
    s2 = np.maximum(rho**2 - th2, 0.0)
    eta = th2 if eta is None else float(eta)
    if eta <= 0.0:
        return float(np.sum(s2 > 0.0))
    return float(np.sum(s2 / (s2 + eta)))


def _combine(a, b, how):
    if how == "max":
        return max(a, b)
    if how == "sum":
        return a + b
    if how == "in":
        return a
    if how == "out":
        return b
    raise ValueError("combine must be 'max', 'sum', 'in' or 'out'")


# ---------------------------------------------------------------------------
# one directional spectrum, real and shift-null
# ---------------------------------------------------------------------------


def _block_diag(*Ms):
    rows = sum(M.shape[0] for M in Ms)
    cols = sum(M.shape[1] for M in Ms)
    out = np.zeros((rows, cols))
    r = c = 0
    for M in Ms:
        out[r : r + M.shape[0], c : c + M.shape[1]] = M
        r += M.shape[0]
        c += M.shape[1]
    return out


def _null_device(ws):
    """Device for the batched shift-null path, or None for the numpy loop.

    CUDA when the workspace runs on it (Maria's null calibration was ~6x
    faster on the GPU for exactly this loop).  `ws.null_device` overrides
    ('cpu' runs the batched torch path on the CPU -- used by the tests;
    'numpy' forces the loop).
    """
    forced = getattr(ws, "null_device", None)
    if forced == "numpy":
        return None
    try:
        import torch  # noqa: F401
    except ImportError:
        return None
    if forced is not None:
        return forced
    if (
        getattr(ws, "backend", "numpy") == "torch"
        and getattr(ws, "device", "cpu") == "cuda"
    ):
        return "cuda"
    return None


def _shift_stack(ws, dev):
    """All shift matrices stacked on `dev` in float64, cached on the
    workspace and rebuilt if the shift pool grows.
    """
    import torch

    key = (dev, len(ws.Cs00), id(ws.Cs00[-1]) if ws.Cs00 else 0)
    cache = getattr(ws, "_cog2_shift_stack", None)
    if cache is None or cache[0] != key:
        X00 = torch.stack(
            [torch.as_tensor(np.asarray(X, dtype=np.float64)) for X in ws.Cs00]
        ).to(dev)
        X01 = torch.stack(
            [torch.as_tensor(np.asarray(X, dtype=np.float64)) for X in ws.Cs01]
        ).to(dev)
        cache = (key, X00, X01)
        ws._cog2_shift_stack = cache
    return cache[1], cache[2]


def _inv_sqrt_torch(C, ridge):
    """Batched mirror of interface._inv_sqrt (last two dims)."""
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


def _inv_sqrt_big(ws, C):
    """interface._inv_sqrt, on the GPU when the matrix is large and the
    workspace is on CUDA (the whole-environment conditioning block is of
    order D).
    """
    if C.shape[0] > 256 and _null_device(ws) == "cuda":
        import torch

        t = torch.as_tensor(np.asarray(C, dtype=np.float64), device="cuda")
        return _inv_sqrt_torch(t, ws.ridge).cpu().numpy()
    return _inv_sqrt(C, ws.ridge)


class FluxDirection:
    """Everything about (target, conditioning) that the source's real and
    shifted versions share.

    target T (cols, basis Vt), extra conditioning Z (cols, basis Vz, may be
    empty), source Src (cols, basis Vs).  The conditioning set at time t is
    P = T u Z with basis blockdiag(Vc, Vz), where Vc is the basis the
    TARGET'S PRESENT is described by when it is subtracted.

    THE RULER LEAK (COG 2.1 fix).  With Vc = Vt -- the target's present
    summarised by the same few slow modes as its future -- whatever the
    summary leaves out is not subtracted, and any source correlated with the
    left-out part looks like an extra channel.  On the planted benchmark this
    read a unit with no planted gear as ~0.8 channels.  `Vc` lets the
    target's present be kept whole (identity) or wider than its future.
    """

    def __init__(self, ws, T, Vt, Src, Vs, Z=None, Vz=None, Vc=None):
        """Set up the blocks (target, source, optional covariates)."""
        r = ws.ridge
        C00, C01, C11 = ws.C00, ws.C01, ws.C11
        self.ws, self.T, self.Src, self.Vt, self.Vs = ws, T, Src, Vt, Vs
        self._vp_identity = False
        if Vc is None:
            Vc = Vt
        elif isinstance(Vc, str) and Vc == "full":
            Vc = np.eye(len(T))
            self._vp_identity = Z is None or not len(Z)
        if Z is not None and len(Z):
            self.P = np.concatenate([T, Z])
            self.Vp = _block_diag(Vc, Vz)
        else:
            self.P, self.Vp = T, Vc
        Cpp = self.Vp.T @ C00[np.ix_(self.P, self.P)] @ self.Vp
        Cff = Vt.T @ C11[np.ix_(T, T)] @ Vt
        self.Cpf = self.Vp.T @ C01[np.ix_(self.P, T)] @ Vt
        Ip = _inv_sqrt_big(ws, Cpp)
        self.Ip = Ip @ Ip
        self.Rff = Cff - self.Cpf.T @ self.Ip @ self.Cpf
        self.Lff = _inv_sqrt(self.Rff, r)
        self.Css = Vs.T @ C00[np.ix_(Src, Src)] @ Vs
        # self-prediction of the target (sigma, as in FluxWorkspace.score)
        tr = float(np.trace(Cff))
        self.sigma = (
            float(np.trace(self.Cpf.T @ self.Ip @ self.Cpf)) / tr
            if tr > 0
            else float("nan")
        )

    def _spec(self, Csp, Csf):
        Rss = self.Css - Csp @ self.Ip @ Csp.T
        Rsf = Csf - Csp @ self.Ip @ self.Cpf
        M = self.Lff @ Rsf.T @ _inv_sqrt(Rss, self.ws.ridge)
        return np.clip(np.linalg.svd(M, compute_uv=False), 0.0, 1.0)

    def real(self):
        """Return the singular values of the real (unshifted) data."""
        ws = self.ws  # noqa: F841
        Csp = self.Vs.T @ ws.C00[np.ix_(self.Src, self.P)] @ self.Vp
        Csf = self.Vs.T @ ws.C01[np.ix_(self.Src, self.T)] @ self.Vt
        return self._spec(Csp, Csf)

    def null_leading(self, j):
        """Leading correlation with the SOURCE displaced by shift j."""
        X00, X01 = self.ws.Cs00[j], self.ws.Cs01[j]
        Csp = self.Vs.T @ X00[np.ix_(self.Src, self.P)] @ self.Vp
        Csf = self.Vs.T @ X01[np.ix_(self.Src, self.T)] @ self.Vt
        s = self._spec(Csp, Csf)
        return float(s[0]) if s.size else 0.0

    def null_rows(self):
        """Return the leading correlation under every time-shift null."""
        dev = _null_device(self.ws)
        if dev is not None and len(self.ws.Cs00):
            return self._null_rows_torch(dev)
        return np.array(
            [self.null_leading(j) for j in range(len(self.ws.Cs00))]
        )

    def _null_rows_torch(self, dev):
        """All shifts at once: the same algebra as `null_leading`, batched
        over the shift index on `dev`.
        """
        import torch

        f64 = torch.float64
        X00, X01 = _shift_stack(self.ws, dev)

        def t(a):
            return torch.as_tensor(
                np.asarray(a, dtype=np.float64), device=dev, dtype=f64
            )

        Src = torch.as_tensor(self.Src, device=dev)
        P = torch.as_tensor(self.P, device=dev)
        T = torch.as_tensor(self.T, device=dev)
        Vs, Vt = t(self.Vs), t(self.Vt)
        A = X00.index_select(1, Src).index_select(2, P)  # (n, s, p)
        B = X01.index_select(1, Src).index_select(2, T)  # (n, s, t)
        Csp = Vs.T @ A
        if not self._vp_identity:
            Csp = Csp @ t(self.Vp)
        Csf = Vs.T @ B @ Vt
        Ip, Cpf = t(self.Ip), t(self.Cpf)
        Rss = t(self.Css) - Csp @ Ip @ Csp.transpose(1, 2)
        Rsf = Csf - Csp @ Ip @ Cpf
        M = (
            t(self.Lff)
            @ Rsf.transpose(1, 2)
            @ _inv_sqrt_torch(Rss, self.ws.ridge)
        )
        sv = torch.clamp(torch.linalg.svdvals(M), 0.0, 1.0)
        return sv[:, 0].cpu().numpy()


# ---------------------------------------------------------------------------
# the scorer
# ---------------------------------------------------------------------------


class FluxScorer:
    """support -> result dict, for `interface.decompose(scorer=..., cost_fn=
    FluxScorer.cost_fn)`.

    backend 'flux'  : both directions with the environment reduced to its
                      `env_dim` slowest modes (FluxWorkspace).
            'schur' : incoming direction with the WHOLE environment
                      (`schur.SchurWorkspace`); outgoing still reduced -- the
                      outgoing target is the environment's future, and keeping
                      all of it is not cheap.  Pass `schur=` to reuse one.

    calibration None: each candidate gets its own shift null (95th percentile
    over the workspace's shifts, per direction) and cost = nu (no baseline).
    With a `calibrate_cog2` result: thresholds and baseline are looked up by
    size (gaps interpolated), after checking the fingerprint.
    """

    def __init__(
        self,
        ws,
        *,
        unit_dim=8,
        env_dim=20,
        calibration=None,
        backend="flux",
        schur=None,
        combine="max",
        eta=None,
        null_fill="interpolate",
        percentile=95.0,
        baseline_window=2,
        cond_unit="full",
        cond_env="full",
        batched=None,
        engine_device=None,
        env_basis="exact",
        batch_size=32,
        engine=None,
    ):
        """Attach the workspace, null tables and size-matched calibration."""
        if backend not in ("flux", "schur"):
            raise ValueError("backend must be 'flux' or 'schur'")
        self.ws, self.unit_dim, self.env_dim = ws, int(unit_dim), int(env_dim)
        self.backend, self.combine, self.eta = backend, combine, eta
        # How the target's own present is described when it is subtracted
        # (see _Direction, "the ruler leak"):
        #   cond_unit  'reduced' = unit_dim slow modes (v1.0), 'full' = every
        #              unit feature (incoming direction)
        #   cond_env   None = env_dim slow modes (v1.0), an int = that many,
        #              'full' = every environment feature (outgoing)
        if cond_unit not in ("reduced", "full"):
            raise ValueError("cond_unit must be 'reduced' or 'full'")
        if not (
            cond_env is None
            or cond_env == "full"
            or (isinstance(cond_env, int) and cond_env > 0)
        ):
            raise ValueError("cond_env must be None, 'full' or a positive int")
        self.cond_unit, self.cond_env = cond_unit, cond_env
        # Batched GPU engine (cog2_gpu): used when asked (batched=True) or,
        # by default, when the workspace runs on CUDA and the settings are
        # the ones the engine implements (flux backend, whole-block
        # conditioning).  Same numbers as the numpy path.
        self.engine = None
        self.batch_size = int(batch_size)
        self.env_basis = env_basis
        can = (
            backend == "flux"
            and cond_unit == "full"
            and cond_env == "full"
            and ws.reversible
            and not getattr(ws, "has_boundary_columns", False)
            and ws.truncation == "koopman"
        )
        if engine is not None:
            if engine.ws is not ws:
                raise ValueError("engine belongs to another workspace")
            self.engine = engine
            self.env_basis = engine.env_basis
            batched = False
        elif batched is None:
            batched = can and _null_device(ws) == "cuda"
        if batched:
            if not can:
                raise ValueError(
                    "batched scoring needs backend='flux', "
                    "cond_unit='full', cond_env='full', a "
                    "reversible Koopman residue workspace"
                )
            from pdukit.cog.scoring.engine_torch import TorchEngine

            self.engine = TorchEngine(
                ws,
                unit_dim=unit_dim,
                env_dim=env_dim,
                device=engine_device,
                env_basis=env_basis,
            )
        elif env_basis != "exact" and self.engine is None:
            raise ValueError("env_basis='ritz' needs the batched engine")
        self.percentile = float(percentile)
        _combine(0.0, 0.0, combine)  # validates
        if backend == "schur":
            if getattr(ws, "has_boundary_columns", False):
                raise ValueError(
                    "the Schur backend needs unit and "
                    "environment columns to cover every column; "
                    "PairFluxWorkspace excludes boundary pairs"
                )
            if schur is None:
                from pdukit.cog.scoring.workspace_schur import SchurWorkspace

                schur = SchurWorkspace(ws)
        self.schur = schur
        self.calibration = calibration
        self.theta_in = self.theta_out = self.base_med = None
        if calibration is not None:
            cfg = calibration.get("config", {})
            if cfg.get("statistic") not in (None, STATISTIC):
                raise ValueError(f"calibration is for {cfg.get('statistic')}")
            if cfg.get("backend", backend) != backend:
                raise ValueError(
                    f"calibration backend {cfg.get('backend')} "
                    f"!= scorer backend {backend}"
                )
            if cfg.get("combine", combine) != combine:
                raise ValueError(
                    f"calibration combine {cfg.get('combine')} "
                    f"!= scorer combine {combine}"
                )
            for key, val in (
                ("cond_unit", cond_unit),
                ("cond_env", cond_env),
                ("env_basis", env_basis),
            ):
                if cfg.get(key, val) != val:
                    raise ValueError(
                        f"calibration {key} {cfg.get(key)} != "
                        f"scorer {key} {val}"
                    )
            check_null_calibration(calibration, ws, unit_dim, env_dim)
            Nm1 = ws.N - 1
            self.theta_in, _ = complete_theta_table(
                calibration["theta_in"], Nm1, mode=null_fill
            )
            self.theta_out, _ = complete_theta_table(
                calibration["theta_out"], Nm1, mode=null_fill
            )
            base = calibration["baseline"]
            raw = {
                int(k): np.asarray(v["values"], dtype=float)
                for k, v in base.items()
            }
            # Pool the ball values over sizes k-w..k+w.  A per-size median
            # from a few dozen balls jumps by ~1 channel between neighbouring
            # sizes (seen: 2.60 at 29, 1.07 at 30), and growth then follows
            # the jumps instead of the data.  The baseline changes slowly
            # with size, so pooling neighbours costs little bias.
            w = int(baseline_window)
            ks = sorted(raw)
            self._base_vals = {}
            for k in ks:
                pool = [raw[j] for j in ks if abs(j - k) <= w]
                self._base_vals[k] = np.concatenate(pool)
            self.base_med, _ = complete_theta_table(
                {k: float(np.median(v)) for k, v in self._base_vals.items()},
                Nm1,
                mode=null_fill,
            )
            self._base_sizes = np.array(ks)

    # cost for decompose(cost_fn=...)
    @staticmethod
    def cost_fn(result):
        """Return the cost of a scored result (``decompose`` cost_fn)."""
        return float(result["cost"])

    def __call__(self, support):
        """Score a support (the ``decompose(scorer=...)`` interface)."""
        return self.score(support)

    # -- parts --------------------------------------------------------------
    def _cols(self, support, exclude=None):
        """(support, unit columns, environment columns).

        `exclude`: residues removed from the environment as well (e.g. face
        residues when a core is scored): they are neither unit nor environment.
        """
        support, S, E = self.ws._cols(support)
        if exclude:
            ex = sorted(set(int(i) for i in exclude) - set(support.tolist()))
            if ex:
                E = np.setdiff1d(E, self.ws._cols(ex)[1])
        return support, S, E

    def parts_batch(self, supports, with_null, excludes=None):
        """`parts` for many supports; on the batched engine, supports whose
        unit and environment have the same number of columns are scored in
        one GPU call (chunks of `batch_size`).  `excludes`: one residue set
        per support (or None) removed from that support's environment.
        """
        if excludes is None:
            excludes = [None] * len(supports)
        if self.engine is None:
            return [
                self.parts(s_, with_null, exclude=x_)
                for s_, x_ in zip(supports, excludes)
            ]
        ws = self.ws  # noqa: F841
        cols = []
        for sup, x_ in zip(supports, excludes):
            support, S, E = self._cols(sup, x_)
            if E.size == 0:
                raise ValueError(
                    "candidate covers the whole protein: no environment"
                )
            cols.append((support, S, E))
        out = [None] * len(supports)
        groups = {}
        for i, (_, S, E) in enumerate(cols):
            groups.setdefault((S.size, E.size), []).append(i)
        for idx in groups.values():
            for c0 in range(0, len(idx), self.batch_size):
                chunk = idx[c0 : c0 + self.batch_size]
                res = self.engine.parts(
                    [cols[i][1] for i in chunk],
                    [cols[i][2] for i in chunk],
                    with_null,
                )
                for i, r in zip(chunk, res):
                    r.update(
                        {
                            "support": cols[i][0],
                            "S": cols[i][1],
                            "E": cols[i][2],
                        }
                    )
                    out[i] = r
        return out

    def score_batch(self, supports, excludes=None):
        """`score` for many supports at once (batched when the engine is on).

        Returns a list of result dicts.
        """
        cal = self.calibration is not None
        return [
            self._finish(p)
            for p in self.parts_batch(
                supports, with_null=not cal, excludes=excludes
            )
        ]

    def parts(self, support, with_null, exclude=None):
        """Real spectra (and, if asked, per-shift null leading correlations)
        for both directions.  Used by `score` and by `calibrate_cog2`.
        """
        if self.engine is not None:
            return self.parts_batch([support], with_null, excludes=[exclude])[
                0
            ]
        ws = self.ws  # noqa: F841
        support, S, E = self._cols(support, exclude)
        if E.size == 0:
            raise ValueError(
                "candidate covers the whole protein: no environment"
            )
        Vu = ws._koopman_basis(S, self.unit_dim, ws.reversible)
        Ve = ws._koopman_basis(E, self.env_dim, ws.reversible)
        out = {"support": support, "S": S, "E": E}
        Vc_u = "full" if self.cond_unit == "full" else None
        if self.cond_env is None:
            Vc_e = None
        elif self.cond_env == "full":
            Vc_e = "full"
        else:
            Vc_e = ws._koopman_basis(E, int(self.cond_env), ws.reversible)
        # unit present -> env future
        d_out = FluxDirection(ws, E, Ve, S, Vu, Vc=Vc_e)
        out["rho_out"] = d_out.real()
        out["null_out"] = d_out.null_rows() if with_null else None
        if self.backend == "flux":
            # env present -> unit future
            d_in = FluxDirection(ws, S, Vu, E, Ve, Vc=Vc_u)
            out["rho_in"] = d_in.real()
            out["null_in"] = d_in.null_rows() if with_null else None
            out["sigma"] = d_in.sigma
        else:
            sw = self.schur
            _, S2, E2, u = sw._project_and_residualise(support, self.unit_dim)
            out["rho_in"] = sw._real_spectrum(S2, E2, u)
            out["null_in"] = (
                np.array(
                    [
                        sw._leading_null_rho(S2, E2, u, j)
                        for j in range(sw.n_shift)
                    ]
                )
                if with_null
                else None
            )
            Cyy, Chy, Ih = u["Cyy"], u["Chy"], u["Ih"]
            tr = float(np.trace(Cyy))
            out["sigma"] = (
                float(np.trace(Chy.T @ Ih @ Chy)) / tr
                if tr > 0
                else float("nan")
            )
        return out

    def _p_size(self, size, nu):
        if self.calibration is None:
            return float("nan")
        k = int(self._base_sizes[np.argmin(np.abs(self._base_sizes - size))])
        vals = self._base_vals[k]
        return float((1 + np.sum(vals <= nu + 1e-12)) / (vals.size + 1))

    # -- public ---------------------------------------------------------------
    def score(self, support):
        """Score a support: closure cost, status and the channel strength."""
        cal = self.calibration is not None
        return self._finish(self.parts(support, with_null=not cal))

    def _finish(self, p):
        cal = self.calibration is not None
        size = int(p["support"].size)
        if cal:
            th_in, th_out = self.theta_in[size], self.theta_out[size]
        else:
            th_in = float(np.percentile(p["null_in"], self.percentile))
            th_out = float(np.percentile(p["null_out"], self.percentile))
        rin, rout = p["rho_in"], p["rho_out"]
        nu_in = soft_count(rin, th_in, self.eta)
        nu_out = soft_count(rout, th_out, self.eta)
        nu = _combine(nu_in, nu_out, self.combine)
        base = self.base_med[size] if cal else 0.0
        kin, kout = int(np.sum(rin > th_in)), int(np.sum(rout > th_out))

        def keff(rho, th):
            c = np.maximum(np.asarray(rho) ** 2 - th**2, 0.0)
            t = float(c.sum())
            return float(t * t / np.sum(c * c)) if t > 0 else float("nan")

        m_in = float(rin[0] - th_in) if rin.size else 0.0
        m_out = float(rout[0] - th_out) if rout.size else 0.0
        # The v1.0 fields, filled from the more open direction, so every
        # existing consumer of a score dict (reports, gaps) keeps working.
        dom_in = nu_in >= nu_out
        rho_d, th_d = (rin, th_in) if dom_in else (rout, th_out)
        rc = np.sqrt(np.maximum(np.asarray(rho_d) ** 2 - th_d**2, 0.0))
        return {
            "support": p["support"].tolist(),
            "size": size,
            "n_pairs": self.ws.n,
            "lag": self.ws.lag,
            "statistic": STATISTIC,
            "backend": self.backend,
            "rho_in": rin.tolist(),
            "rho_out": rout.tolist(),
            "theta_in": float(th_in),
            "theta_out": float(th_out),
            "nu_in": nu_in,
            "nu_out": nu_out,
            "nu": float(nu),
            "k_star_in": kin,
            "k_star_out": kout,
            "k_eff_in": keff(rin, th_in),
            "k_eff_out": keff(rout, th_out),
            "margin_in": m_in,
            "margin_out": m_out,
            "baseline_nu": float(base),
            "p_size": self._p_size(size, nu),
            "cost": float(nu - base),
            # v1.0-compatible fields
            "rho": rho_d.tolist(),
            "rho_null_p95": float(th_d),
            "resolved": bool(kin + kout > 0),
            "k_star": max(kin, kout),
            "k_eff": keff(rho_d, th_d),
            "margin": max(m_in, m_out),
            "low_power": bool(max(m_in, m_out) <= 0.0),
            "sigma": float(p["sigma"]),
            "nuc_spec": (
                float(rc.sum() / rc[0])
                if rc.size and rc[0] > 0
                else float("nan")
            ),
            "unit_dim": self.unit_dim,
            "env_dim": self.env_dim,
        }
