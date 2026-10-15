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

Both are one function, `directional_spectrum(ws, target, source, ...)`:
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
and a workspace fingerprint; `Cog2Scorer` refuses a calibration made on other
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

import time

import numpy as np

from .interface import (
    FluxWorkspace,
    _bootstrap_ci_rel_width_two_way,
    _grow_ball_support,
    _inv_sqrt,
    _variance_components,
    check_null_calibration,
    complete_theta_table,
    workspace_fingerprint,
)

STATISTIC = "cog2_two_way_soft_count"
VERSION = "COG 2.1"


# ---------------------------------------------------------------------------
# soft channel count
# ---------------------------------------------------------------------------

# Minimum channel strength (effect-size floor).  A channel counts only if
# its correlation is above max(noise threshold, RHO_MIN).  0 = off.  Set
# once per run by cog2_run (--rho-min); applies to units, baseline regions,
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


class _Direction:
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


def directional_spectrum(
    ws,
    target,
    source,
    *,
    target_dim=8,
    source_dim=20,
    condition=None,
    condition_dim=20,
    with_null=False,
):
    """Canonical correlations between the TARGET's future and the SOURCE's
    present, both residualised on the target's present (and on `condition`'s
    present, if given).  `target`, `source`, `condition` are COLUMN index
    arrays of the workspace.  Bases are the slow (Koopman) directions of each
    block, as in `FluxWorkspace`.  Returns (rho, null_rows or None).
    """
    T = np.asarray(target, dtype=int)
    Src = np.asarray(source, dtype=int)
    Vt = ws._koopman_basis(T, target_dim, ws.reversible)
    Vs = ws._koopman_basis(Src, source_dim, ws.reversible)
    Z = Vz = None
    if condition is not None and len(condition):
        Z = np.asarray(condition, dtype=int)
        Vz = ws._koopman_basis(Z, condition_dim, ws.reversible)
    d = _Direction(ws, T, Vt, Src, Vs, Z, Vz)
    return d.real(), (d.null_rows() if with_null else None)


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


def column_groups(ws, columns):
    """Group workspace columns by owner (residue, or residue pair for a
    PairFluxWorkspace).  Returns (owners list, list of column arrays).
    """
    columns = np.asarray(columns, dtype=int)
    if hasattr(ws, "col_owners"):
        keys = [ws.col_owners[c] for c in columns]
    else:
        keys = [(int(c) // ws.m,) for c in columns]
    groups = {}
    for c, k in zip(columns, keys):
        groups.setdefault(k, []).append(int(c))
    owners = sorted(groups)
    return owners, [np.array(groups[k]) for k in owners]


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


# ---------------------------------------------------------------------------
# the scorer
# ---------------------------------------------------------------------------


class Cog2Scorer:
    """support -> result dict, for `interface.decompose(scorer=..., cost_fn=
    Cog2Scorer.cost_fn)`.

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
            from .cog2_gpu import BatchedEngine

            self.engine = BatchedEngine(
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
                from .schur import SchurWorkspace

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
        d_out = _Direction(ws, E, Ve, S, Vu, Vc=Vc_e)
        out["rho_out"] = d_out.real()
        out["null_out"] = d_out.null_rows() if with_null else None
        if self.backend == "flux":
            # env present -> unit future
            d_in = _Direction(ws, S, Vu, E, Ve, Vc=Vc_u)
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


# ---------------------------------------------------------------------------
# calibration: theta_in, theta_out and the ball baseline, per size
# ---------------------------------------------------------------------------


def _ball(N, k, adjacency, rng, tries=40):
    s = _grow_ball_support(N, k, adjacency, rng)
    for _ in range(tries - 1):
        if len(s) == k:
            break
        s = _grow_ball_support(N, k, adjacency, rng)
    return sorted(int(i) for i in s)


def calibrate_cog2(
    ws,
    adjacency,
    sizes,
    *,
    unit_dim=8,
    env_dim=20,
    backend="flux",
    schur=None,
    combine="max",
    eta=None,
    percentile=95.0,
    cond_unit="full",
    cond_env="full",
    target_rel_width=0.03,
    batch_supports=10,
    min_supports=40,
    max_supports=400,
    min_shifts=8,
    seed=0,
    on_log=None,
    batched=None,
    env_basis="exact",
    engine_device=None,
    scorer=None,
):
    """Per support size k: theta_in[k], theta_out[k] (the `percentile` of the
    leading shift-null correlation, pooled over compact ball supports and
    shifts) and the baseline distribution of nu over the same balls.

    Stopping: after `min_supports`, stop when the two-way (supports x shifts)
    bootstrap 95% CI of BOTH thetas is narrower than `target_rel_width`
    (relative), or at `max_supports`.  Returns a dict with theta_in,
    theta_out, baseline {k: median, q25, q75, frac_open, values},
    diagnostics and config (with the workspace fingerprint).
    """
    log = on_log if on_log is not None else (lambda *_: None)
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    adjacency = np.asarray(adjacency, dtype=bool)
    N = adjacency.shape[0]
    if len(ws.Cs00) < max(8, int(min_shifts)):
        ws.extend_null_shifts(max(8, int(min_shifts)) - len(ws.Cs00))
    if scorer is None:
        scorer = Cog2Scorer(
            ws,
            unit_dim=unit_dim,
            env_dim=env_dim,
            backend=backend,
            schur=schur,
            combine=combine,
            eta=eta,
            cond_unit=cond_unit,
            cond_env=cond_env,
            batched=batched,
            env_basis=env_basis,
            engine_device=engine_device,
        )
    theta_in, theta_out, baseline, diagnostics = {}, {}, {}, {}
    for k in sorted({int(s) for s in sizes}):
        if not 1 <= k < N:
            continue
        tk = time.perf_counter()
        rows_in, rows_out, real_in, real_out, sups = [], [], [], [], []
        w_in = w_out = float("inf")
        stopped = "max_supports"
        while len(sups) < max_supports:
            batch = [
                s_
                for s_ in (
                    _ball(N, k, adjacency, rng) for _ in range(batch_supports)
                )
                if len(s_) == k
            ]
            try:
                parts_list = scorer.parts_batch(batch, with_null=True)
            except (ValueError, np.linalg.LinAlgError):
                parts_list = []
                for s_ in batch:
                    try:
                        parts_list.append(scorer.parts(s_, with_null=True))
                    except (ValueError, np.linalg.LinAlgError):
                        parts_list.append(None)
            for sup, p in zip(batch, parts_list):
                if p is None:
                    continue
                sups.append(sup)
                rows_in.append(p["null_in"])
                rows_out.append(p["null_out"])
                real_in.append(p["rho_in"])
                real_out.append(p["rho_out"])
            if not sups:
                break
            if len(sups) >= min_supports:
                _, w_in = _bootstrap_ci_rel_width_two_way(
                    rows_in, percentile=percentile, n_boot=500, rng=rng
                )
                _, w_out = _bootstrap_ci_rel_width_two_way(
                    rows_out, percentile=percentile, n_boot=500, rng=rng
                )
                if max(w_in, w_out) < target_rel_width:
                    stopped = "converged"
                    break
        if not sups:
            log(f"calibrate_cog2: size {k}: no ball support of that size")
            continue
        th_i = float(np.percentile(np.asarray(rows_in), percentile))
        th_o = float(np.percentile(np.asarray(rows_out), percentile))
        nu_i = np.array([soft_count(r, th_i, eta) for r in real_in])
        nu_o = np.array([soft_count(r, th_o, eta) for r in real_out])
        nu = np.array([_combine(a, b, combine) for a, b in zip(nu_i, nu_o)])
        theta_in[k], theta_out[k] = th_i, th_o
        baseline[k] = {
            "median": float(np.median(nu)),
            "q25": float(np.percentile(nu, 25)),
            "q75": float(np.percentile(nu, 75)),
            "median_nu_in": float(np.median(nu_i)),
            "median_nu_out": float(np.median(nu_o)),
            "frac_open": float(np.mean(nu > 0)),
            "values": [round(float(v), 5) for v in nu],
        }
        diagnostics[k] = {
            "n_supports": len(sups),
            "stopped_by": stopped,
            "rel_ci_width_in": float(w_in),
            "rel_ci_width_out": float(w_out),
            "variance_components_in": _variance_components(rows_in),
            "variance_components_out": _variance_components(rows_out),
            "elapsed_seconds": time.perf_counter() - tk,
            "supports": sups,
        }
        log(
            f"calibrate_cog2: size {k:3d}  theta_in {th_i:.4f}  theta_out "
            f"{th_o:.4f}  baseline nu median {baseline[k]['median']:.2f} "
            f"[{baseline[k]['q25']:.2f}, {baseline[k]['q75']:.2f}]  "
            f"{len(sups)} supports, {stopped} "
            f"(CI {w_in:.3f}/{w_out:.3f}, "
            f"{diagnostics[k]['elapsed_seconds']:.1f}s)"
        )
    config = {
        "statistic": STATISTIC,
        "version": VERSION,
        "unit_dim": int(unit_dim),
        "env_dim": int(env_dim),
        "backend": backend,
        "combine": combine,
        "eta": eta,
        "cond_unit": cond_unit,
        "cond_env": cond_env,
        "env_basis": scorer.env_basis,
        "batched": scorer.engine is not None,
        "percentile": float(percentile),
        "n_shifts": int(len(ws.Cs00)),
        "target_rel_width": float(target_rel_width),
        "min_supports": int(min_supports),
        "max_supports": int(max_supports),
        "seed": int(seed),
        "fingerprint": workspace_fingerprint(ws),
    }
    return {
        "theta_in": theta_in,
        "theta_out": theta_out,
        "baseline": baseline,
        "diagnostics": diagnostics,
        "config": config,
        "elapsed_seconds": time.perf_counter() - t0,
    }


# ---------------------------------------------------------------------------
# cohesion (Stage 3)
# ---------------------------------------------------------------------------


def _corr_blocks(ws, cols):
    """Standardised equal-time and lagged covariances on `cols`."""
    sd0 = np.sqrt(np.maximum(np.diag(ws.C00)[cols], 1e-30))
    sd1 = np.sqrt(np.maximum(np.diag(ws.C11)[cols], 1e-30))
    R0 = ws.C00[np.ix_(cols, cols)] / np.outer(sd0, sd0)
    R1 = ws.C01[np.ix_(cols, cols)] / np.outer(sd0, sd1)
    return R0, R1


def internal_coupling_graph(ws, support):
    """Owner-level (residue or pair) coupling weights inside a support:
    w_gh = mean over the two groups' columns of R0^2 + R1^2 + R1'^2, the
    squared equal-time and lagged correlations both ways.
    """
    support, S, _ = ws._cols(support)
    owners, groups = column_groups(ws, S)
    pos = {c: i for i, c in enumerate(S)}
    R0, R1 = _corr_blocks(ws, S)
    Q = R0**2 + R1**2 + R1.T**2
    n = len(groups)
    W = np.zeros((n, n))
    for a in range(n):
        ia = [pos[c] for c in groups[a]]
        for b in range(a + 1, n):
            ib = [pos[c] for c in groups[b]]
            W[a, b] = W[b, a] = float(Q[np.ix_(ia, ib)].mean())
    return owners, groups, W


def spectral_bisection(W):
    """Two-way split by the Fiedler vector of the normalised Laplacian;
    falls back to a median split if the sign split leaves a side empty.
    """
    n = W.shape[0]
    if n < 2:
        raise ValueError("need at least two groups to bisect")
    d = W.sum(axis=1)
    dinv = 1.0 / np.sqrt(np.maximum(d, 1e-30))
    L = np.eye(n) - dinv[:, None] * W * dinv[None, :]
    w, V = np.linalg.eigh(0.5 * (L + L.T))
    f = V[:, 1] * dinv
    side = f > 0
    if side.all() or not side.any():
        side = f > np.median(f)
    if side.all() or not side.any():
        side = np.arange(n) < n // 2
    return np.where(~side)[0], np.where(side)[0]


def cohesion_check(
    scorer,
    support,
    *,
    cut_dim=None,
    condition_dim=None,
    percentile=95.0,
    result=None,
):
    """Decide whether a unit is one wheel or two glued together.

    1. Bisect the support by spectral clustering of its internal coupling.
    2. Measure channels across the cut in BOTH directions (half A present ->
       half B future, and B -> A), each conditioned on the receiving half's
       present AND the environment's slow modes, with its own shift null.
    3. Compare with the unit's own boundary count nu (from `scorer`).

    Verdicts: 'split'    -- nu_cut < 0.5: the halves are dynamically
                            independent given the outside; not one unit.
              'cohesive' -- nu_cut >= nu_boundary: inside talks at least as
                            much as the boundary does.
              'weak'     -- in between.
    `cohesion_margin` = nu_cut - nu_boundary.
    """
    ws = scorer.ws
    cut_dim = int(cut_dim or scorer.unit_dim)
    condition_dim = int(condition_dim or scorer.env_dim)
    support, S, E = ws._cols(support)
    owners, groups, W = internal_coupling_graph(ws, support)
    if len(groups) < 2:
        return {"verdict": "too_small", "support": support.tolist()}
    ia, ib = spectral_bisection(W)
    A = np.concatenate([groups[i] for i in ia])
    B = np.concatenate([groups[i] for i in ib])
    VA = ws._koopman_basis(A, cut_dim, ws.reversible)
    VB = ws._koopman_basis(B, cut_dim, ws.reversible)
    Z = Vz = None
    if E.size:
        Z = E
        Vz = ws._koopman_basis(E, condition_dim, ws.reversible)
    dAB = _Direction(ws, B, VB, A, VA, Z, Vz)  # A present -> B future
    dBA = _Direction(ws, A, VA, B, VB, Z, Vz)  # B present -> A future
    rAB, rBA = dAB.real(), dBA.real()
    tAB = float(np.percentile(dAB.null_rows(), percentile))
    tBA = float(np.percentile(dBA.null_rows(), percentile))
    nAB = soft_count(rAB, tAB, scorer.eta)
    nBA = soft_count(rBA, tBA, scorer.eta)
    nu_cut = max(nAB, nBA)
    if result is None:
        result = scorer.score(support)
    nu_b = float(result["nu"])
    if nu_cut < 0.5:
        verdict = "split"
    elif nu_cut >= nu_b:
        verdict = "cohesive"
    else:
        verdict = "weak"

    def res(ix):
        return sorted(
            {int(r) for i in ix for r in owners[i]} & set(support.tolist())
        )

    return {
        "support": support.tolist(),
        "half_a": res(ia),
        "half_b": res(ib),
        "rho_a_to_b": rAB.tolist(),
        "rho_b_to_a": rBA.tolist(),
        "theta_a_to_b": tAB,
        "theta_b_to_a": tBA,
        "nu_a_to_b": nAB,
        "nu_b_to_a": nBA,
        "nu_cut": nu_cut,
        "nu_boundary": nu_b,
        "cohesion_margin": float(nu_cut - nu_b),
        "verdict": verdict,
    }


def split_incoherent(scorer, units, *, min_size=3, max_depth=3, on_log=None):
    """Replace every unit whose cohesion verdict is 'split' by its two
    halves, recursively (up to `max_depth` levels, halves of at least
    `min_size` residues).

    WHY.  The cost rewards a set for being more closed than a compact ball
    of its size.  Two neighbouring units that do not talk to each other are,
    together, just as closed as each alone -- and larger -- so the growth
    phase happily returns their union (seen on the planted benchmark: an
    uncoupled decoy pair came out as one 16-residue unit).  Nothing in the
    assignment loop can split a unit; this is the split move.  A union of
    two units joined by a gear is NOT split: at this resolution it is one
    closed unit, and the report says which units it contains only through
    the cohesion halves.

    Returns (units, events) where events records every split.
    """
    log = on_log if on_log is not None else (lambda *_: None)
    out, events = [], []
    todo = [(sorted(map(int, u)), 0) for u in units]
    while todo:
        u, depth = todo.pop(0)
        if depth >= max_depth or len(u) < 2 * min_size:
            out.append(u)
            continue
        try:
            c = cohesion_check(scorer, u)
        except (ValueError, np.linalg.LinAlgError):
            out.append(u)
            continue
        if (
            c.get("verdict") == "split"
            and len(c["half_a"]) >= min_size
            and len(c["half_b"]) >= min_size
        ):
            events.append(
                {
                    "unit": u,
                    "half_a": c["half_a"],
                    "half_b": c["half_b"],
                    "nu_cut": c["nu_cut"],
                    "depth": depth,
                }
            )
            log(
                f"cohesion split: {len(u)} residues -> {len(c['half_a'])} + "
                f"{len(c['half_b'])} (nu across cut {c['nu_cut']:.2f})"
            )
            todo.append((c["half_a"], depth + 1))
            todo.append((c["half_b"], depth + 1))
        else:
            out.append(u)
    return out, events
