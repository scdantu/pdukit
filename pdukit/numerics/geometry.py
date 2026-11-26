"""Distances, length unit and contact graph of a set of coordinates."""

from __future__ import annotations

import numpy as np


def median_distance_matrix(ca_coords, max_frames=200, seed=0):
    """(N, N) median pairwise CA distance taken ACROSS frames.

    Why not the distance matrix of the mean structure?  Because mdCATH
    trajectories are neither superposed nor guaranteed unwrapped.  Averaging
    coordinates that rotate, translate or jump across a periodic boundary
    produces a reference structure that never existed, and both neighbour
    selection and the contact graph then rest on it.  The observed symptom on
    5bqpD00 is a median consecutive-CA spacing of 4.98 A where the polypeptide
    value is 3.8 A.

    Distances are internal, so taking them per frame and reducing with a MEDIAN
    across frames is invariant to rotation and translation, and robust to a
    minority of boundary-corrupted frames.  Frames are subsampled evenly to
    keep the (frames, N, N) intermediate small.
    """
    X = np.asarray(ca_coords)
    F, N, _ = X.shape
    if F > max_frames:
        idx = np.linspace(0, F - 1, int(max_frames)).astype(int)
        X = X[idx]
    X = np.asarray(X, dtype=np.float32)
    d = np.empty((X.shape[0], N, N), dtype=np.float32)
    for k in range(X.shape[0]):  # (N,N,3) at a time, not (F,N,N,3)
        diff = X[k][:, None, :] - X[k][None, :, :]
        d[k] = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))
    return np.median(d, axis=0).astype(np.float64)


def reference_spacing(dmed):
    """Median distance between residues adjacent in sequence, from `dmed`."""
    if dmed.shape[0] < 2:
        return float("nan")
    return float(np.median(np.diag(dmed, k=1)))


def detect_length_unit(ca_coords, dmed=None, max_frames=200):
    """Infer whether CA coordinates are in Angstrom or nanometre.

    Consecutive CA atoms sit 3.8 A = 0.38 nm apart, one of the most reliable
    numbers in structural biology, so the median over the chain identifies the
    unit unambiguously.  The spacing is taken from `median_distance_matrix`,
    not from a mean structure -- see that function for why.

    This is not defensive padding.  mdCATH is GROMACS-derived and stores
    nanometres, while `hdf5_layout.py` documents Angstrom as an assumption, so
    every absolute-distance cutoff in the package is otherwise wrong by 10x.
    The symptom is a contact graph in which every residue contacts every other.

    Returns a dict with 'unit', the observed 'median_ca_spacing', the
    'to_angstrom' factor, and 'spacing_angstrom' -- the spacing expressed in
    Angstrom.  If that last value is far from 3.8, the atom selection or the
    trajectory is suspect and 'suspect' is set.
    """
    X = np.asarray(ca_coords, dtype=np.float64)
    if dmed is None:
        dmed = median_distance_matrix(X, max_frames=max_frames)
    step = reference_spacing(dmed)
    if not np.isfinite(step):
        return {
            "unit": "unknown",
            "median_ca_spacing": step,
            "to_angstrom": 1.0,
            "spacing_angstrom": step,
            "suspect": True,
        }
    if 2.5 <= step <= 5.5:
        unit, to_ang = "angstrom", 1.0
    elif 0.25 <= step <= 0.55:
        unit, to_ang = "nanometre", 10.0
    else:
        unit, to_ang = "unknown", 1.0
    ang = step * to_ang
    return {
        "unit": unit,
        "median_ca_spacing": step,
        "to_angstrom": to_ang,
        "spacing_angstrom": ang,
        "suspect": bool(unit == "unknown" or not (3.2 <= ang <= 4.4)),
    }


def contact_graph(
    ca_coords, cutoff=8.0, dmed=None, units="auto", max_frames=200
):
    """Boolean (N, N) contact adjacency from the MEDIAN inter-residue distance.

    ``cutoff`` is always in ANGSTROM.  With ``units='auto'`` the coordinates'
    own unit is detected and the cutoff converted, so the same call is correct
    for Angstrom and nanometre inputs.  Pass 'angstrom' or 'nanometre' to force
    it, or 'none' to compare raw numbers.

    Growth follows SPATIAL neighbours, not sequence neighbours, because v3.0 D1
    allows supports discontinuous in sequence.
    """
    X = np.asarray(ca_coords, dtype=np.float64)
    if dmed is None:
        dmed = median_distance_matrix(X, max_frames=max_frames)
    if units == "auto":
        scale = 1.0 / detect_length_unit(X, dmed=dmed)["to_angstrom"]
    elif units == "nanometre":
        scale = 0.1
    elif units in ("angstrom", "none"):
        scale = 1.0
    else:
        raise ValueError(f"unknown units {units!r}")
    adj = dmed <= float(cutoff) * scale
    np.fill_diagonal(adj, False)
    return adj


def contact_adjacency(coords, cutoff):
    """Boolean (N, N) contacts: distance below ``cutoff``, no self-contacts."""
    X = np.asarray(coords, float)
    dist = np.linalg.norm(X[:, None] - X[None], axis=-1)
    return (dist < cutoff) & ~np.eye(len(X), dtype=bool)


def connect_components(adj, coords):
    """Join disconnected parts of a contact graph at their closest pair.

    Repeats until the graph is connected, so a search that grows along
    contacts can reach every residue.  ``adj`` is changed in place and
    returned.
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


def ball_layout(groups, rng, spacing=11.0, radius=None, cutoff=8.0):
    """Place each residue group as a ball; ball centres on a 3D walk.

    Each step of the walk has length ``spacing`` and avoids earlier centres
    (up to 200 tries).  A group of n residues gets radius
    ``radius or 2.2 n^(1/3)``.  Returns ``(coords, adjacency)`` with the
    contact graph connected.
    """
    n = len(groups)
    centres = np.zeros((n, 3))
    for a in range(1, n):
        for _ in range(200):
            step = rng.normal(size=3)
            step /= np.linalg.norm(step)
            c = centres[a - 1] + spacing * step
            far = np.linalg.norm(centres[:a] - c, axis=1) > 0.9 * spacing
            if np.all(far):
                break
        centres[a] = c
    N = sum(len(g) for g in groups)
    coords = np.zeros((N, 3))
    for a, g in enumerate(groups):
        r = radius or 2.2 * len(g) ** (1 / 3)
        for i in g:
            v = rng.normal(size=3)
            shell = r * rng.uniform(0.2, 1) ** (1 / 3)
            coords[i] = centres[a] + v / np.linalg.norm(v) * shell
    adj = connect_components(contact_adjacency(coords, cutoff), coords)
    return coords, adj


def lattice_layout(dims, spacing, rng, jitter=0.3, cutoff=8.0):
    """Residues on a jittered 3D lattice; returns ``(coords, adjacency)``."""
    g = np.stack(
        np.meshgrid(*[np.arange(k) for k in dims], indexing="ij"), -1
    ).reshape(-1, 3)
    coords = g * spacing + rng.normal(scale=jitter, size=g.shape)
    adj = connect_components(contact_adjacency(coords, cutoff), coords)
    return coords, adj
