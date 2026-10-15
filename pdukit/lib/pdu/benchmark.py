"""COG 2.1 Stage 1: planted benchmark systems with a known answer.

A decomposition method can only be trusted on proteins after it has been
shown to recover units that were put there on purpose.  This module builds
synthetic "proteins" whose dynamical units, couplings and boundary residues
are known exactly, in the same (frames, residues, features) layout the real
pipeline uses, so every COG version can be run on them unchanged.

CONSTRUCTION (a linear stochastic watch: wheels + gears).

  * Each planted unit alpha has a private latent state z_alpha of dimension
    `latent_dim`, with dense internal dynamics A_alpha (every residue of the
    unit feels every internal mode -- the unit is cohesive by construction).
  * Units are joined by COUPLINGS (src -> dst, rank r): a rank-r block
    K[dst, src] = g * U V'.  Direction matters: a coupling 0 -> 1 lets unit 0
    drive unit 1, not the reverse.  This is the gear between two wheels.
  * The whole latent state evolves as a stable VAR(1),
        z_{t+1} = K z_t + noise.
  * Residue i of unit alpha READS OUT its unit's latent state through its own
    random mixing matrix plus private observation noise:
        x_i(t) = W_i z_alpha(t) + e_i(t)       (m features per residue)
  * MULTI-COLOUR FACE residues (`face_residues` > 0) sit on the boundary of
    two coupled units and read out BOTH: half their features from one unit,
    half from the other.  They belong to both units in the truth.
  * Optional `decoy_pairs` place two units side by side in space with NO
    coupling between them -- a compact blob that is not a unit.  A method
    that merges them is rewarding spatial compactness, not dynamics.

GEOMETRY.  Units are placed as small balls along a chain in 3D; residues get
coordinates inside their unit's ball, and the contact graph is built from
distances, so neighbouring units touch at their faces exactly as domains do.
With `discontinuous=True` each unit's residues are split into two sequence
segments far apart in the chain, as in lysozyme's discontinuous domains --
spatially compact, sequence-discontinuous.

THE ANSWER (`truth`).  Units as residue lists (face residues listed in both),
couplings with direction and rank, face residues, decoy pairs, and K itself.

Nothing here is used by the estimators; it is a test bed.
"""

from __future__ import annotations

import numpy as np


def _stable_internal(d, rho_range, rng):
    """Dense internal dynamics with spectral radius drawn from rho_range.

    A random (non-normal) matrix rescaled, as in test_interface.simulate, so
    internal modes mix every latent coordinate (cohesive unit).
    """
    M = rng.normal(size=(d, d))
    rho = float(rng.uniform(*rho_range))
    return M / np.abs(np.linalg.eigvals(M)).max() * rho


def _unit_vectors(d, r, rng):
    Q, _ = np.linalg.qr(rng.normal(size=(d, r)))
    return Q[:, :r]


def _connect_components(adj, coords):
    """Join disconnected components of the contact graph at their closest
    residue pair, so ball growth can reach every residue.
    """
    N = adj.shape[0]
    while True:
        seen = np.zeros(N, dtype=bool)
        comp = []
        for s in range(N):
            if seen[s]:
                continue
            stack, members = [s], []
            seen[s] = True
            while stack:
                u = stack.pop()
                members.append(u)
                for v in np.where(adj[u])[0]:
                    if not seen[v]:
                        seen[v] = True
                        stack.append(int(v))
            comp.append(members)
        if len(comp) <= 1:
            return adj
        a, b = comp[0], [i for c in comp[1:] for i in c]
        d = np.linalg.norm(
            coords[a][:, None, :] - coords[b][None, :, :], axis=-1
        )
        i, j = np.unravel_index(int(np.argmin(d)), d.shape)
        adj[a[i], b[j]] = adj[b[j], a[i]] = True


def default_couplings(n_units, rank=1):
    """Return a ring of directed rank-`rank` gears.

    The direction alternates, plus one two-way link, so the benchmark has
    units that only drive, units that only listen, and units that do both.
    """
    cpl = []
    for a in range(n_units - 1):
        if a % 2 == 0:
            cpl.append((a, a + 1, rank))
        else:
            cpl.append((a + 1, a, rank))
    if n_units >= 2:
        cpl.append((1, 0, rank))  # 0 <-> 1 is the two-way link
    return cpl


def planted_system(
    n_units=4,
    unit_size=8,
    m=4,
    latent_dim=5,
    n_frames=6000,
    n_replicas=3,
    couplings=None,
    coupling_gain=0.35,
    internal_rho=(0.80, 0.92),
    noise=0.3,
    obs_noise=0.25,
    face_residues=0,
    decoy_pairs=(),
    discontinuous=False,
    unit_radius=5.0,
    unit_spacing=9.0,
    contact_cutoff=8.0,
    burn_in=500,
    seed=0,
    dtype=np.float32,
):
    """Build a planted system.

    Returns a dict:

      feats     (n_replicas * n_frames, N, m) residue features
      segments  [(start, end)] one per replica
      coords    (N, 3) residue positions (one static structure)
      adjacency (N, N) bool contact graph
      truth     {"units": [...], "couplings": [(src, dst, rank)],
                 "face_residues": [...], "decoy_pairs": [...],
                 "K": latent transition matrix, "latent_dim": int,
                 "unit_of_latent": [...]}
      config    every argument, for the record

    `couplings` is a list of (src, dst, rank); default `default_couplings`.
    `decoy_pairs` is a list of (a, b) unit pairs that must be UNCOUPLED: any
    coupling between them is removed, and they are placed adjacent in space.
    """
    rng = np.random.default_rng(seed)
    n_units, unit_size, m = int(n_units), int(unit_size), int(m)
    d = int(latent_dim)
    if couplings is None:
        couplings = default_couplings(n_units)
    decoy = [tuple(sorted(map(int, p))) for p in decoy_pairs]
    couplings = [
        (int(s), int(t), int(r))
        for s, t, r in couplings
        if tuple(sorted((int(s), int(t)))) not in decoy
    ]
    for s, t, r in couplings:
        if not (0 <= s < n_units and 0 <= t < n_units) or s == t:
            raise ValueError(f"bad coupling {(s, t, r)}")
        if not 1 <= r <= d:
            raise ValueError(f"coupling rank {r} must be in 1..latent_dim")

    # --- latent dynamics -----------------------------------------------------
    DL = n_units * d
    K = np.zeros((DL, DL))
    for a in range(n_units):
        K[a * d : (a + 1) * d, a * d : (a + 1) * d] = _stable_internal(
            d, internal_rho, rng
        )
    for s, t, r in couplings:
        U, V = _unit_vectors(d, r, rng), _unit_vectors(d, r, rng)
        K[t * d : (t + 1) * d, s * d : (s + 1) * d] += coupling_gain * U @ V.T
    spec = float(np.abs(np.linalg.eigvals(K)).max())
    if spec >= 0.98:
        K *= 0.97 / spec

    # --- residue layout
    # --------------------------------------------------------
    N = n_units * unit_size
    order = np.arange(N)  # noqa: F841
    if discontinuous:
        # unit a gets two half-segments: one in the first half of the chain,
        # one in the second, in the same unit order -- far apart in sequence.
        h1 = unit_size // 2
        h2 = unit_size - h1
        first = [list(range(a * h1, (a + 1) * h1)) for a in range(n_units)]
        off = n_units * h1
        second = [
            list(range(off + a * h2, off + (a + 1) * h2))
            for a in range(n_units)
        ]
        units = [sorted(first[a] + second[a]) for a in range(n_units)]
    else:
        units = [
            list(range(a * unit_size, (a + 1) * unit_size))
            for a in range(n_units)
        ]
    owner = np.empty(N, dtype=int)
    for a, u in enumerate(units):
        owner[u] = a

    # --- geometry: unit centres on a chain, decoy pairs made adjacent
    # ----------
    centres = np.zeros((n_units, 3))
    for a in range(1, n_units):
        step = rng.normal(size=3)
        step[0] = abs(step[0]) + 1.0
        centres[a] = centres[a - 1] + unit_spacing * step / np.linalg.norm(
            step
        )
    coords = np.zeros((N, 3))
    for i in range(N):
        v = rng.normal(size=3)
        v *= unit_radius * rng.uniform(0.2, 1.0) ** (1 / 3) / np.linalg.norm(v)
        coords[i] = centres[owner[i]] + v
    dist = np.linalg.norm(coords[:, None] - coords[None], axis=-1)
    adj = (dist < contact_cutoff) & ~np.eye(N, dtype=bool)
    for a, b in decoy:
        # guarantee the decoy pair touches, so a compact ball can swallow both
        ia, ib = units[a], units[b]
        dd = dist[np.ix_(ia, ib)]
        for k in np.argsort(dd, axis=None)[:3]:
            i, j = np.unravel_index(int(k), dd.shape)
            adj[ia[i], ib[j]] = adj[ib[j], ia[i]] = True
    adj = _connect_components(adj, coords)

    # --- multi-colour face residues
    # ---------------------------------------------
    faces = {}
    if face_residues:
        for s, t, _ in couplings:
            # residues of unit t nearest unit s's centre
            cand = [i for i in units[t] if i not in faces]
            dc = np.linalg.norm(coords[cand] - centres[s], axis=1)
            for i in [cand[k] for k in np.argsort(dc)[: int(face_residues)]]:
                faces[int(i)] = (int(t), int(s))  # (home unit, second unit)

    # --- simulate
    # ---------------------------------------------------------------
    W = rng.normal(size=(N, m, d)) / np.sqrt(d)
    W2 = rng.normal(size=(N, m, d)) / np.sqrt(d)
    feats = np.empty((n_replicas * n_frames, N, m), dtype=dtype)
    segments = []
    for rep in range(int(n_replicas)):
        z = np.zeros(DL)
        traj = np.empty((n_frames, DL))
        for t in range(-int(burn_in), n_frames):
            z = K @ z + rng.normal(scale=noise, size=DL)
            if t >= 0:
                traj[t] = z
        out = np.empty((n_frames, N, m))
        for i in range(N):
            a = owner[i]
            za = traj[:, a * d : (a + 1) * d]
            x = za @ W[i].T
            if i in faces:
                b = faces[i][1]
                zb = traj[:, b * d : (b + 1) * d]
                half = m // 2
                x[:, half:] = (zb @ W2[i].T)[:, half:]
            out[:, i, :] = x + rng.normal(scale=obs_noise, size=(n_frames, m))
        a0 = rep * n_frames
        feats[a0 : a0 + n_frames] = out.astype(dtype)
        segments.append((a0, a0 + n_frames))

    truth_units = [
        sorted(set(u) | {i for i, (h, b) in faces.items() if b == a})
        for a, u in enumerate(units)
    ]
    truth = {
        "units": truth_units,
        "home_units": [sorted(u) for u in units],
        "couplings": couplings,
        "face_residues": sorted(faces),
        "face_owners": {int(i): list(v) for i, v in faces.items()},
        "decoy_pairs": [list(p) for p in decoy],
        "K": K,
        "latent_dim": d,
        "unit_of_latent": [a for a in range(n_units) for _ in range(d)],
    }
    config = dict(
        n_units=n_units,
        unit_size=unit_size,
        m=m,
        latent_dim=d,
        n_frames=n_frames,
        n_replicas=n_replicas,
        couplings=couplings,
        coupling_gain=coupling_gain,
        internal_rho=tuple(internal_rho),
        noise=noise,
        obs_noise=obs_noise,
        face_residues=face_residues,
        decoy_pairs=[list(p) for p in decoy],
        discontinuous=discontinuous,
        seed=seed,
        spectral_radius=float(np.abs(np.linalg.eigvals(K)).max()),
    )
    return {
        "feats": feats,
        "segments": segments,
        "coords": coords,
        "adjacency": adj,
        "truth": truth,
        "config": config,
    }


def match_units(found, truth_units, *, threshold=0.67):
    """Compare a decomposition with the planted answer.

    For every TRUE unit, the best Jaccard overlap with any found unit
    (recall side); for every FOUND unit, the best overlap with any true unit
    (precision side).  A true unit counts as recovered when its best overlap
    reaches `threshold`.  Also reports how many found units straddle two true
    units (each overlapping by >= 25 % of the found unit) -- the merge error.
    The default threshold is 0.67: at 0.5 a merged 20-residue unit counted
    as recovering both of the 10-residue units inside it.
    """
    F = [set(map(int, u)) for u in found]
    T = [set(map(int, u)) for u in truth_units]

    def jac(a, b):
        return len(a & b) / max(len(a | b), 1)

    rec = [max((jac(t, f) for f in F), default=0.0) for t in T]
    prec = [max((jac(f, t) for t in T), default=0.0) for f in F]
    merged = 0
    for f in F:
        big = [t for t in T if len(f & t) >= 0.25 * max(len(f), 1)]
        merged += int(len(big) >= 2)
    return {
        "recall_jaccard": [round(x, 4) for x in rec],
        "precision_jaccard": [round(x, 4) for x in prec],
        "mean_recall": float(np.mean(rec)) if rec else 0.0,
        "mean_precision": float(np.mean(prec)) if prec else 0.0,
        "n_recovered": int(sum(x >= threshold for x in rec)),
        "n_true": len(T),
        "n_found": len(F),
        "n_merged": int(merged),
        "threshold": float(threshold),
    }


def face_recovery(boundary_residues, truth):
    """Fraction of planted multi-colour face residues that the method
    reported as boundary (shared) residues, and how many reported boundary
    residues are not faces.
    """
    faces = set(truth.get("face_residues", []))
    got = set(map(int, boundary_residues))
    return {
        "n_faces": len(faces),
        "found": len(faces & got),
        "fraction": (len(faces & got) / len(faces)) if faces else float("nan"),
        "extra_boundary": len(got - faces),
    }
