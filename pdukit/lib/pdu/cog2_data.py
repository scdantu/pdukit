"""Data loading and trajectory preprocessing for the COG pipeline.

Planted benchmark systems or MD replicas are turned into per-residue
features (with an on-disk cache); the helpers here also centre, high-pass,
trim and slice the replicas before the workspace is built.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from .null_calibration_run import _jsonable, log


def scatter_planted(b, seed=0):
    """Move every planted residue to a random position of the same fold:
    dynamics unchanged, spatial layout (adjacency) unrelated to the units.
    A contact-based search cannot find these units; a dynamics-only one
    should.
    """
    feats = b["feats"]
    N = feats.shape[1]
    p = np.random.default_rng(seed).permutation(N)  # old r -> new p[r]
    f2 = np.empty_like(feats)
    f2[:, p] = feats
    T = dict(b["truth"])

    def mp(u):
        return sorted(int(p[x]) for x in u)

    T["units"] = [mp(u) for u in T["units"]]
    if "home_units" in T:
        T["home_units"] = [mp(u) for u in T["home_units"]]
    T["face_residues"] = mp(T.get("face_residues", []))
    if "face_owners" in T:
        T["face_owners"] = {
            int(p[int(k)]): v for k, v in T["face_owners"].items()
        }
    out = dict(b)
    out["feats"], out["truth"] = f2, T
    return out


def load_data(a, timings, src=None):
    """-> dict(feats, segments, adjacency, coords or None, truth or None,
    meta).  MD data come from `src` (a FeatureSource), unfiltered.
    """
    t = time.perf_counter()
    if a.planted:
        from .benchmark import planted_system

        decoys = []
        for p in (a.decoy_pairs or "").split(","):
            if p.strip():
                x, y = p.split("-")
                decoys.append((int(x), int(y)))
        b = planted_system(
            n_units=a.n_units,
            unit_size=a.unit_size,
            m=a.planted_m,
            latent_dim=a.latent_dim,
            n_frames=a.frames_per_replica,
            n_replicas=a.n_replicas,
            coupling_gain=a.coupling_gain,
            face_residues=a.face_residues,
            decoy_pairs=decoys,
            discontinuous=a.discontinuous,
            seed=a.bench_seed,
        )
        timings["data_seconds"] = time.perf_counter() - t
        log(
            f"planted system: {b['feats'].shape}, couplings "
            f"{b['truth']['couplings']}, faces {b['truth']['face_residues']}, "
            f"decoys {b['truth']['decoy_pairs']}"
        )
        if a.scatter:
            b = scatter_planted(b, seed=a.bench_seed + 1000)
            log(
                "planted units scattered over the fold: residue positions "
                "permuted, so units are not spatially contiguous"
            )
        return {
            "feats": b["feats"],
            "segments": b["segments"],
            "adjacency": b["adjacency"],
            "coords": None,
            "truth": b["truth"],
            "meta": {"planted": b["config"], "scatter": bool(a.scatter)},
        }

    return src.get(window=None)


def file_signature(path):
    """Cache signature of one file, or of every part of a multi-part
    replica ('a.xtc,b.xtc').
    """
    parts = [p_ for p_ in str(path).split(",") if p_]
    out = []
    for p_ in parts:
        st = os.stat(p_)
        out.append(
            [str(Path(p_).resolve()), int(st.st_size), int(st.st_mtime)]
        )
    return out[0] if len(out) == 1 else out


class FeatureSource:
    """RSE features for MD data, with an on-disk cache (default on).

    The cache key covers the input files (path, size, modification time) and
    every setting that changes the features: frames, stride, RSE cutoff and
    dimension, sequence separation, contact cutoff, lag, ridge, rigid-body
    fit and the distance-level highpass window.  `get(window)` returns the
    features with that window applied to the raw distances before the
    reduction (None = unfiltered).  The trajectory is read only if some
    requested feature set is not cached yet.

    Features are built on ALL replicas together: the RSE reduction knows
    nothing about units, so this does not leak the answer into the held-out
    check, and it keeps all replicas' features in the same coordinates.
    """

    VERSION = "rse-v2"

    def __init__(self, a, timings):
        """Prepare the feature cache and remember the run options."""
        self.a, self.timings = a, timings
        self._coords = self._segments = self._meta = self._dmed = None
        if a.no_feature_cache:
            self.cache_dir = None
        else:
            self.cache_dir = (
                Path(a.feature_cache)
                if a.feature_cache
                else Path(a.outdir).resolve().parent / "_feature_cache"
            )
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, window):
        a = self.a
        d = {
            "version": self.VERSION,
            "topology": file_signature(a.xtc_topology),
            "xtc": [file_signature(x) for x in a.xtc],
            "frames": a.frames,
            "stride": a.stride,
            "rse_cutoff": a.rse_cutoff,
            "rse_components": a.rse_components,
            "min_seq_sep": a.min_seq_sep,
            "contact_cutoff": a.contact_cutoff,
            "lag": a.lag,
            "ridge": a.ridge,
            "fit": bool(a.fit),
            "distance_highpass": int(window or 0),
        }
        return hashlib.sha1(
            json.dumps(d, sort_keys=True).encode()
        ).hexdigest()[:16], d

    def coords(self):
        """Return aligned coordinates, segments, metadata, median distances."""
        if self._coords is None:
            from .flux_pilot import load_xtc_segments
            from .interface import median_distance_matrix

            t = time.perf_counter()
            a = self.a
            coords, segments, meta = load_xtc_segments(
                a.xtc_topology,
                a.xtc,
                stride=a.stride,
                max_frames=a.frames or None,
                verbose=True,
            )
            self.timings["load_seconds"] = time.perf_counter() - t
            log(
                f"loaded {coords.shape[0]} frames x {coords.shape[1]} "
                f"residues in {self.timings['load_seconds']:.1f}s"
            )
            dts = [float(m_.get("dt_ps", 0.0)) for m_ in meta]
            good = [d_ for d_ in dts if d_ > 0]
            if good and (max(good) - min(good)) > 0.01 * max(good):
                raise SystemExit(
                    "replicas are saved at different intervals (dt in ps: "
                    + ", ".join(f"{d_:g}" for d_ in dts)
                    + "); a lag in "
                    "frames would mean different times -- re-stride them "
                    "to a common interval first"
                )
            self._dmed = median_distance_matrix(coords)
            if a.fit:
                t = time.perf_counter()
                fit_coordinates(coords, segments)
                log(
                    f"removed overall rotation and translation (Kabsch onto "
                    f"the average structure, {time.perf_counter() - t:.1f}s)"
                )
            self._coords = coords
            self._segments = [tuple(map(int, x)) for x in segments]
            self._meta = meta
        return self._coords, self._segments, self._meta, self._dmed

    def get(self, window=None):
        """Return the features, segments and contact graph (cached on disk)."""
        from .interface import contact_graph, rse_features

        a = self.a
        key, desc = self._key(window)
        path = (
            (self.cache_dir / f"feats_{key}.npz") if self.cache_dir else None
        )
        if path is not None and path.exists():
            t = time.perf_counter()
            z = np.load(path, allow_pickle=False)
            out = {
                "feats": z["feats"],
                "adjacency": z["adjacency"],
                "segments": [tuple(map(int, x)) for x in z["segments"]],
                "dmed": z["dmed"],
                "coords": None,
                "truth": None,
                "tica_spectra": (
                    z["tica_spectra"] if "tica_spectra" in z.files else None
                ),
                "meta": {
                    "replicas": json.loads(str(z["meta"])),
                    "feature_cache": str(path),
                    "cache_hit": True,
                },
            }
            log(
                f"features from cache {path.name} "
                f"({time.perf_counter() - t:.1f}s)"
            )
            return out
        coords, segments, meta, dmed = self.coords()
        t = time.perf_counter()
        dfilt = None
        if window:
            w = int(window)

            def dfilt(Di, segs):
                highpass_replicas(Di[:, None, :], segs, w)
                center_replicas(Di[:, None, :], segs)

        spectra = []
        feats, _, _ = rse_features(
            coords,
            cutoff=a.rse_cutoff,
            n_components=a.rse_components,
            min_seq_sep=a.min_seq_sep,
            dmed=dmed,
            lag=a.lag,
            segments=segments,
            koopman=True,
            ridge=a.ridge,
            distance_filter=dfilt,
            spectra=spectra,
        )
        L_ = max((len(x) for x in spectra), default=0)
        spectra = (
            np.array(
                [x + [np.nan] * (L_ - len(x)) for x in spectra], dtype=float
            )
            if L_
            else None
        )
        adj = np.asarray(
            contact_graph(
                coords, cutoff=a.contact_cutoff, dmed=dmed, units="auto"
            ),
            dtype=bool,
        )
        self.timings[f"features_seconds_w{int(window or 0)}"] = (
            time.perf_counter() - t
        )
        log(
            f"RSE features {feats.shape}"
            + (f" (distances highpassed at {window} frames)" if window else "")
            + f", contact graph mean degree {adj.sum(1).mean():.1f} "
            f"({time.perf_counter() - t:.1f}s)"
        )
        if path is not None:
            tmp = path.with_suffix(".tmp.npz")
            extra = {} if spectra is None else {"tica_spectra": spectra}
            np.savez(
                tmp,
                feats=feats,
                adjacency=adj,
                segments=np.asarray(segments, dtype=np.int64),
                dmed=dmed,
                meta=json.dumps(_jsonable(meta)),
                key=json.dumps(desc),
                **extra,
            )
            os.replace(tmp, path)
            log(f"features cached as {path}")
        return {
            "feats": feats,
            "adjacency": adj,
            "segments": segments,
            "dmed": dmed,
            "coords": None,
            "truth": None,
            "tica_spectra": spectra,
            "meta": {
                "replicas": meta,
                "feature_cache": str(path) if path else None,
                "cache_hit": False,
            },
        }


def take_replicas(arr, segments, idx):
    """Rows of `arr` belonging to the chosen replicas, with re-based segments.

    Returns (array, segments).
    """
    parts, segs, cur = [], [], 0
    for i in idx:
        a, b = segments[i]
        parts.append(arr[a:b])
        segs.append((cur, cur + (b - a)))
        cur += b - a
    return np.concatenate(parts, axis=0), segs


def replica_overlap(
    feats, segments, lag=1, n_tics=5, n_bins=40, ridge=1e-2, chunk=8192
):
    """Measure whether the replicas visit the same conformations.

    Pooled tICA on all replicas (the slow coordinates of the whole data
    set), every frame projected onto the first `n_tics`, and for each pair
    of replicas and each tIC the Bhattacharyya overlap of their histograms
    (1 = identical distribution, 0 = no frames in common), plus the shift of
    their means in units of the pooled spread.

    WHY.  Holding a replica out only tests the units if that replica visits
    the same states.  If replicas sit in different basins, a unit that fails
    on the held-out replica may be state-dependent rather than wrong -- and
    that is itself a finding, but it must be measured, not assumed.
    """
    from .interface import _inv_sqrt

    F, N, m = feats.shape
    X = feats.reshape(F, N * m)
    D = X.shape[1]
    mu = np.zeros(D)
    for a in range(0, F, chunk):
        mu += X[a : a + chunk].sum(axis=0, dtype=np.float64)
    mu /= F
    C0 = np.zeros((D, D))
    C1 = np.zeros((D, D))
    n = 0
    for a_, b_ in segments:
        for c in range(a_, b_ - lag, chunk):
            d = min(c + chunk, b_ - lag)
            x0 = X[c:d].astype(np.float64) - mu
            x1 = X[c + lag : d + lag].astype(np.float64) - mu
            C0 += x0.T @ x0
            C1 += x0.T @ x1
            n += d - c
    C0 /= n
    C1 /= n
    L = _inv_sqrt(C0, ridge)
    A = L @ (0.5 * (C1 + C1.T)) @ L
    w, U = np.linalg.eigh(0.5 * (A + A.T))
    order = np.argsort(w)[::-1][:n_tics]
    V = L @ U[:, order]
    proj = np.empty((F, len(order)))
    for a in range(0, F, chunk):
        proj[a : a + chunk] = (X[a : a + chunk].astype(np.float64) - mu) @ V
    sd = proj.std(axis=0)
    lo, hi = proj.min(axis=0), proj.max(axis=0)
    R = len(segments)
    hist = [
        [
            np.histogram(proj[a_:b_, k], bins=n_bins, range=(lo[k], hi[k]))[0]
            / max(b_ - a_, 1)
            for k in range(len(order))
        ]
        for a_, b_ in segments
    ]
    pairs = []
    for i in range(R):
        for j in range(i + 1, R):
            bc = [
                float(np.sum(np.sqrt(hist[i][k] * hist[j][k])))
                for k in range(len(order))
            ]
            mi = proj[segments[i][0] : segments[i][1]].mean(axis=0)
            mj = proj[segments[j][0] : segments[j][1]].mean(axis=0)
            pairs.append(
                {
                    "replicas": [i, j],
                    "overlap_per_tic": [round(x, 3) for x in bc],
                    "min_overlap": round(min(bc), 3),
                    "mean_shift_sd": [
                        round(float(x), 2) for x in np.abs(mi - mj) / sd
                    ],
                }
            )

    # Integrated autocorrelation time of each tIC WITHIN each replica (each
    # replica centred first, so between-replica offsets do not count), in
    # frames: tau_int = 1 + 2 * sum of the ACF up to its first zero.  The
    # number of independent samples a replica holds is ~ length / tau_int.
    def tau_int(x):
        x = x - x.mean()
        nfft = 1 << int(np.ceil(np.log2(2 * len(x))))
        f = np.fft.rfft(x, nfft)
        acf = np.fft.irfft(f * np.conj(f), nfft)[: len(x)]
        acf = acf / acf[0] if acf[0] > 0 else acf
        neg = np.where(acf <= 0)[0]
        cut = int(neg[0]) if neg.size else len(x)
        return float(1.0 + 2.0 * acf[1:cut].sum())

    taus = [
        [round(tau_int(proj[a_:b_, k]), 1) for k in range(len(order))]
        for a_, b_ in segments
    ]
    return {
        "n_tics": len(order),
        "tic_eigenvalues": [round(float(x), 4) for x in w[order]],
        "tau_int_frames": taus,
        "independent_samples_per_replica": [
            [round((b_ - a_) / max(t, 1.0), 1) for t in row]
            for (a_, b_), row in zip(segments, taus)
        ],
        "pairs": pairs,
        "worst_overlap": min(
            (p["min_overlap"] for p in pairs), default=float("nan")
        ),
    }


def center_replicas(arr, segments):
    """Subtract each replica's own mean, in place.

    WHY (found 12 Oct on lysozyme, reproduced on a planted system).  When
    replicas sit in different states (lysozyme: histogram overlap 0.02 on
    the slowest tIC), every feature carries a per-replica OFFSET.  Across
    the pooled data that offset is a three-level step shared by unit and
    environment -- a "slow mode" with three independent samples.
      * A null that keeps unit and environment in the SAME replica
        (within) or swaps replicas (cross) reproduces it at random strength
        and the threshold explodes: planted system theta 0.04 -> 0.38, every
        real channel hidden (lysozyme GA: theta_in 0.36).
      * The v1.0 global shift partly scrambles it, so its threshold stays
        low, but the REAL spectrum keeps it: every planted unit gained ~2
        spurious channels.
    Centring each replica removes the offsets; on the planted system the
    threshold returns to 0.044 and every channel count is exact.  Price:
    nothing slower than one replica can be claimed -- which is already true,
    since the replicas do not mix.
    """
    for a, b in segments:
        blk = arr[a:b]
        blk -= blk.mean(axis=0, dtype=np.float64).astype(blk.dtype)
    return arr


def highpass_replicas(arr, segments, window):
    """Subtract a centred running mean of `window` frames from every feature,
    within each replica, in place (the window shrinks at the ends).

    WHY.  Centring removes a replica's constant offset but not a DRIFT --
    e.g. a replica still relaxing away from its starting structure.  Two
    unrelated drifting series correlate strongly by chance (the classic
    spurious-regression problem), so drift inflates the shift-null threshold
    the same way offsets did.  After this, motions slower than about
    `window` frames are treated as unsampled and removed; nothing slower
    than the window can be claimed.
    """
    w = int(window)
    if w <= 1:
        return arr
    F, N, m = arr.shape
    for a, b in segments:
        L = b - a
        idx = np.arange(L)
        lo = np.clip(idx - w // 2, 0, L)
        hi = np.clip(idx + w - w // 2, 0, L)
        cnt = (hi - lo).astype(np.float64)[:, None]
        for i in range(N):
            x = arr[a:b, i, :].astype(np.float64)
            c = np.vstack([np.zeros((1, m)), np.cumsum(x, axis=0)])
            arr[a:b, i, :] = (x - (c[hi] - c[lo]) / cnt).astype(arr.dtype)
    return arr


def fit_coordinates(coords, segments=None, n_iter=2, chunk=8192):
    """Remove overall rotation and translation: superpose every frame onto
    the average structure (Kabsch, iterated `n_iter` times so the reference
    is the average of fitted frames), in place.

    The RSE and pair features are inter-residue DISTANCES and so already
    blind to rigid-body motion (tests/test_cog2.py checks this); fitting
    changes them only through floating-point rounding.  It is here for any
    coordinate-based feature, and so the claim can be made explicitly.
    """
    X = coords
    F = X.shape[0]
    ref = X[0].astype(np.float64)
    ref = ref - ref.mean(axis=0)
    for _ in range(int(n_iter)):
        acc = np.zeros_like(ref)
        for a in range(0, F, chunk):
            b = min(a + chunk, F)
            Y = X[a:b].astype(np.float64)
            Y -= Y.mean(axis=1, keepdims=True)
            H = np.einsum("fni,nj->fij", Y, ref)
            U, _, Vt = np.linalg.svd(H)
            d = np.sign(np.linalg.det(np.einsum("fij,fjk->fik", U, Vt)))
            U[:, :, -1] *= d[:, None]
            R = np.einsum("fij,fjk->fik", U, Vt)
            Y = np.einsum("fni,fij->fnj", Y, R)
            X[a:b] = Y.astype(X.dtype)
            acc += Y.sum(axis=0)
        ref = acc / F
        ref -= ref.mean(axis=0)
    return X


def coarse_sizes(lo, hi, n):
    """About n sizes spread geometrically over [lo, hi], ends included."""
    lo, hi, n = int(lo), int(hi), int(n)
    if n <= 1 or hi <= lo:
        return [lo]
    g = np.unique(np.round(np.geomspace(lo, hi, n)).astype(int))
    return sorted(set(g.tolist()) | {lo, hi})


def skip_start(arr, segments, n_skip):
    """Drop the first `n_skip` frames of every replica (equilibration).

    Returns (array, new segments).
    """
    k = int(n_skip)
    if k <= 0:
        return arr, segments
    parts, segs, cur = [], [], 0
    for a, b in segments:
        if b - a <= k:
            raise SystemExit(f"--skip-frames {k} removes a whole replica")
        parts.append(arr[a + k : b])
        segs.append((cur, cur + (b - a - k)))
        cur += b - a - k
    return np.concatenate(parts, axis=0), segs


def make_blocks(segments, block_frames):
    """Cut every replica into consecutive blocks of exactly `block_frames`
    frames (the remainder of each replica is dropped).  Returns a list of
    (replica index, start, end).
    """
    out = []
    L = int(block_frames)
    for i, (a, b) in enumerate(segments):
        for k in range((b - a) // L):
            out.append((i, a + k * L, a + (k + 1) * L))
    return out


def take_blocks(arr, blocks):
    """Rows of the chosen blocks, each block its own segment."""
    parts, segs, cur = [], [], 0
    for _, a, b in blocks:
        parts.append(arr[a:b])
        segs.append((cur, cur + (b - a)))
        cur += b - a
    return np.concatenate(parts, axis=0), segs


def block_frames(a, total_frames):
    """Return the block length in frames for the block check."""
    if a.frame_ps:
        return max(1, int(round(a.block_us * 1e6 / a.frame_ps)))
    return int(a.block_frames or max(100, total_frames // 8))


def subset_by_fraction(arr, segments, frac, rng):
    """Rows of whole replicas in random order until `frac` of all frames
    is reached (the last one truncated).  Returns (array, segments).
    """
    total = sum(b - a for a, b in segments)
    want = max(1, int(round(frac * total)))
    parts, segs, cur = [], [], 0
    for i in rng.permutation(len(segments)):
        a, b = segments[i]
        take = min(b - a, want - cur)
        if take <= 0:
            break
        parts.append(arr[a : a + take])
        segs.append((cur, cur + take))
        cur += take
    return np.concatenate(parts, axis=0), segs


def tica_spectrum_summary(spectra, lag, frame_ps=0.0, kept=8):
    """Decay of each residue's slow-mode spectrum (lag autocorrelations of
    the residue's own tICA components), to judge how many components carry
    signal.  Implied timescale t_k = -lag / ln(lambda_k).
    """
    S = np.asarray(spectra, dtype=float)
    med = np.nanmedian(S, axis=0)
    q25 = np.nanpercentile(S, 25, axis=0)
    q75 = np.nanpercentile(S, 75, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ts = np.where((med > 0) & (med < 1), -lag / np.log(med), np.nan)
    frac = np.nanmean(S > 0.5, axis=0)
    return {
        "median_eigenvalue": [round(float(x), 4) for x in med],
        "q25": [round(float(x), 4) for x in q25],
        "q75": [round(float(x), 4) for x in q75],
        "median_timescale_frames": [round(float(x), 2) for x in ts],
        "median_timescale_ps": (
            [round(float(x * frame_ps), 1) for x in ts] if frame_ps else None
        ),
        "fraction_residues_above_0.5": [round(float(x), 3) for x in frac],
        "kept": int(kept),
        "strong_beyond_kept": int(np.sum(med[kept:] > 0.5)),
        "per_residue": [[round(float(x), 4) for x in r] for r in S],
    }
