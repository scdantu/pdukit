"""Linear latent dynamics for synthetic systems with a known answer.

A synthetic protein is a set of hidden ("latent") coordinates z that evolve
as a stable linear autoregressive process,

    z(t + 1) = K z(t) + L xi(t),      xi ~ N(0, I),

and residues that read some of those coordinates out through random mixing
matrices plus observation noise.  A block of K that only touches the
coordinates of one unit is that unit's internal dynamics; an off-diagonal
block K[target, source] is a directed coupling (source drives target).

Every function takes the random generator ``rng`` explicitly and draws from
it in a fixed order, so a design built twice with the same seed is the same
system.
"""

from __future__ import annotations

import numpy as np


def lifetime_to_rho(lifetime, dt):
    """Per-frame autocorrelation of a mode with the given lifetime.

    rho = exp(-dt / lifetime); ``lifetime`` and ``dt`` in the same unit.
    """
    return float(np.exp(-float(dt) / float(lifetime)))


def normal_block(rhos, rng):
    """Build internal dynamics with given eigenvalues and unit variance.

    Returns ``(K, L)``: K = Q diag(rhos) Q' (a normal matrix, random
    orientation Q) and the noise matrix L = Q diag(sqrt(1 - rhos^2)), which
    makes every mode's stationary variance 1.
    """
    rhos = np.asarray(rhos, float)
    Q, _ = np.linalg.qr(rng.normal(size=(len(rhos), len(rhos))))
    return Q @ np.diag(rhos) @ Q.T, Q @ np.diag(np.sqrt(1 - rhos**2))


def block_diagonal(blocks):
    """Place ``(K_b, L_b)`` blocks on the diagonal.

    Returns ``(K, L, offsets)``; block b occupies rows and columns
    ``offsets[b]:offsets[b + 1]``.
    """
    sizes = [b[0].shape[0] for b in blocks]
    off = np.concatenate([[0], np.cumsum(sizes)]).astype(int)
    D = int(off[-1])
    K, L = np.zeros((D, D)), np.zeros((D, D))
    for k, (Kb, Lb) in enumerate(blocks):
        K[off[k] : off[k + 1], off[k] : off[k + 1]] = Kb
        L[off[k] : off[k + 1], off[k] : off[k + 1]] = Lb
    return K, L, off


def orthonormal_columns(n, r, rng):
    """Random n x r matrix with orthonormal columns (QR of a Gaussian)."""
    return np.linalg.qr(rng.normal(size=(n, r)))[0]


def unit_vector(n, rng):
    """Random unit vector of length n, as an n x 1 matrix."""
    v = rng.normal(size=n)
    return (v / np.linalg.norm(v))[:, None]


def add_coupling(K, offsets, source, target, U, V, gain):
    """Add the directed coupling K[target, source] += gain U V'.

    ``U`` spans the target block, ``V`` the source block (same rank).
    """
    t0, t1 = offsets[target], offsets[target + 1]
    s0, s1 = offsets[source], offsets[source + 1]
    K[t0:t1, s0:s1] += gain * U @ V.T
    return K


def spectral_radius(K):
    """Largest absolute eigenvalue of K."""
    return float(np.abs(np.linalg.eigvals(K)).max())


def stabilise(K, limit, target):
    """Rescale K to spectral radius ``target`` if it reaches ``limit``.

    Returns ``(K, rescaled)``.  A rescale shortens every lifetime, so the
    designs report it.
    """
    sr = spectral_radius(K)
    if sr >= limit:
        return K * (target / sr), True
    return K, False


def simulate_var(K, noise, n_frames, n_replicas, rng, burn_in):
    """Simulate z(t + 1) = K z(t) + noise xi(t) for each replica.

    ``noise`` is a matrix (multiplies xi) or a vector (scales xi element by
    element).  Each replica starts at zero and discards ``burn_in`` frames.
    Returns a list of (n_frames, D) arrays.
    """
    D = K.shape[0]
    matrix = np.ndim(noise) == 2
    out = []
    for _ in range(int(n_replicas)):
        z = np.zeros(D)
        traj = np.empty((int(n_frames), D))
        for t in range(-int(burn_in), int(n_frames)):
            xi = rng.normal(size=D)
            z = K @ z + (noise @ xi if matrix else xi * noise)
            if t >= 0:
                traj[t] = z
        out.append(traj)
    return out


def read_out(
    trajs,
    residue_blocks,
    offsets,
    m,
    obs_noise,
    rng,
    *,
    mix_all=False,
    share_weight=True,
):
    """Turn latent trajectories into per-residue features.

    Residue i reads the latent blocks ``residue_blocks[i]`` through random
    (m x block) matrices; the weights are drawn once, residue by residue,
    then Gaussian observation noise is added frame by frame.

    mix_all       every residue reads ALL latent coordinates (no units)
    share_weight  divide each weight by sqrt(number of blocks read), so a
                  residue reading two blocks has the same variance as one
                  reading a single block

    Returns a list of (frames, N, m) arrays, one per replica.
    """
    N = len(residue_blocks)
    D = int(offsets[-1])
    W = []
    for blocks in residue_blocks:
        if mix_all:
            w = rng.normal(size=(m, D)) / np.sqrt(D)
            W.append([(slice(0, D), w)])
            continue
        row = []
        for b in blocks:
            nb = int(offsets[b + 1] - offsets[b])
            scale = np.sqrt(nb * len(blocks)) if share_weight else np.sqrt(nb)
            w = rng.normal(size=(m, nb)) / scale
            row.append((slice(offsets[b], offsets[b + 1]), w))
        W.append(row)
    feats = []
    for traj in trajs:
        x = np.empty((len(traj), N, m))
        for i in range(N):
            x[:, i] = sum(traj[:, sl] @ w.T for sl, w in W[i])
        feats.append(x + rng.normal(scale=obs_noise, size=x.shape))
    return feats
