"""Global modes from fixed references, compared with what the PDUs rebuild.

    pdu refmodes RUN -f rep1.xtc rep2.xtc -s top.pdb -o tables/

The run's own global-mode recovery compares the units with a tICA on the
same residue features, so its target moves with K and the window.  This
tool grades the units against references that do not depend on them:

  ca-distance  tICA of all C-alpha distances within 15 A (|i - j| >= 2)
  dihedral     tICA of sin/cos of the backbone phi/psi angles (needs N,
               CA and C atoms in the topology; skipped otherwise)
  features     tICA of the run's own residue features (the current one)

Each reference gets the run's frames, stride, high-pass window and lag.
The PDU side: every unit contributes its ``local_dim`` slowest
coordinates (as in the run); stacked, they give the PDU coordinates Y(t).
"Channels on" = tICA of Y (units coupled); "channels off" = each unit's own
tICA modes, pooled.  For every reference mode r_j(t):

  coverage   R^2 of r_j from all PDU coordinates Y (can the units
             represent the mode at all?)
  on / off   R^2 of r_j from the top n assembled modes with channels on /
             off, and the best single assembled mode (corr^2) with its
             implied timescale next to the reference timescale
Written to reference_modes.csv.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pdukit.io.tables import CsvWriter
from pdukit.numerics.tica import explained, reversible_tica, timescale


def dihedral(p0, p1, p2, p3):
    """Dihedral angles (radians) of four (..., 3) point arrays."""
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1n = b1 / np.linalg.norm(b1, axis=-1, keepdims=True)
    v = b0 - np.sum(b0 * b1n, -1, keepdims=True) * b1n
    w = b2 - np.sum(b2 * b1n, -1, keepdims=True) * b1n
    x = np.sum(v * w, -1)
    y = np.sum(np.cross(b1n, v) * w, -1)
    return np.arctan2(y, x)


def backbone_dihedral_features(xyz):
    """sin/cos of phi (residues 2..n) and psi (1..n-1) from N, CA, C.

    ``xyz`` is (frames, 3 n, 3) with atoms ordered N, CA, C per residue.
    Returns (frames, 4 (n - 1)).
    """
    n = xyz.shape[1] // 3
    N, CA, C = xyz[:, 0::3], xyz[:, 1::3], xyz[:, 2::3]
    phi = dihedral(C[:, :-1], N[:, 1:], CA[:, 1:], C[:, 1:])
    psi = dihedral(N[:, :-1], CA[:, :-1], C[:, :-1], N[:, 1:])
    assert phi.shape[1] == psi.shape[1] == n - 1
    return np.concatenate(
        [np.sin(phi), np.cos(phi), np.sin(psi), np.cos(psi)], axis=1
    )


def ca_distance_features(coords, cutoff=15.0, min_seq_sep=2):
    """All C-alpha distances whose median is within ``cutoff``."""
    from pdukit.numerics.geometry import median_distance_matrix

    dmed = median_distance_matrix(coords)
    N = coords.shape[1]
    i, j = np.triu_indices(N, int(min_seq_sep))
    keep = dmed[i, j] <= cutoff
    i, j = i[keep], j[keep]
    diff = coords[:, i, :] - coords[:, j, :]
    return np.sqrt(np.einsum("fpk,fpk->fp", diff, diff)).astype(np.float32)


def _prepare(X, segments, window):
    from pdukit.cog.features import center_replicas, highpass_replicas

    X = np.array(X, dtype=np.float32, copy=True)
    shape = X.shape
    X3 = X.reshape(shape[0], -1, 1) if X.ndim == 2 else X
    if window:
        highpass_replicas(X3, segments, int(window))
    center_replicas(X3, segments)
    return X3.reshape(shape[0], -1)


class ReferenceModes:
    """Compare a finished run's units with fixed reference global modes."""

    def __init__(self, run, xtc, top, n_modes=10, local_dim=8):
        """Rebuild the run's features and coordinates from its settings."""
        from pdukit.config.config import Config
        from pdukit.io.results.reader import ResultsReader

        self.run = Path(run)
        self.results = ResultsReader.load(run)
        sets = [("outdir", str(self.run)), ("xtc", " ".join(map(str, xtc))),
                ("xtc-topology", str(top))]
        self.cfg = Config.build(ppf=str(self.run / "pdu_out.ppf"), sets=sets)
        self.units = [sorted(set(u["residues"]))
                      for u in self.results.get("units_final", [])]
        self.n_modes = int(n_modes)
        self.local_dim = int(local_dim)
        a = self.cfg
        self.lag = int(a.lag)
        hp = str(a.highpass_window or "0")
        self.window = int(hp) if hp.isdigit() else int(
            self.results.get("null_calibration", {}).get("chosen_window", 0))
        self.top, self.xtc = str(top), [str(x) for x in xtc]

    def features(self):
        """Return the run's residue features (frames, residues, K)."""
        from pdukit.cog.features import FeatureSource

        a = self.cfg
        src = FeatureSource(a, {})
        before = bool(a.highpass_before_features and self.window)
        fd = src.get(window=self.window if before else None)
        F = fd["feats"]
        self.segments = [tuple(map(int, s)) for s in fd["segments"]]
        X = _prepare(F.reshape(F.shape[0], -1), self.segments,
                     0 if before else self.window)
        self._source = src
        return X.reshape(F.shape)

    def pdu_coordinates(self, feats):
        """Stacked slow coordinates of every unit, and each unit's columns."""
        F, N, K = feats.shape
        flat = feats.reshape(F, N * K)
        blocks, owner = [], []
        for u, res in enumerate(self.units):
            cols = np.concatenate([np.arange(r * K, (r + 1) * K) for r in res])
            _, V = reversible_tica(flat[:, cols], self.segments, self.lag,
                                   min(self.local_dim, len(cols)))
            Y = (flat[:, cols] - flat[:, cols].mean(0)) @ V
            blocks.append(Y)
            owner += [u] * Y.shape[1]
        return np.concatenate(blocks, axis=1), np.array(owner)

    def assembled(self, Y, owner):
        """Mode time series with channels on and off, with eigenvalues."""
        lam_on, V = reversible_tica(Y, self.segments, self.lag, self.n_modes)
        on = (Y - Y.mean(0)) @ V
        off_l, off_m = [], []
        for u in np.unique(owner):
            cols = np.where(owner == u)[0]
            lam, W = reversible_tica(Y[:, cols], self.segments, self.lag,
                                     len(cols))
            off_l.append(lam)
            off_m.append((Y[:, cols] - Y[:, cols].mean(0)) @ W)
        lam_off = np.concatenate(off_l)
        modes_off = np.concatenate(off_m, axis=1)
        order = np.argsort(lam_off)[::-1][: self.n_modes]
        return (lam_on, on), (lam_off[order], modes_off[:, order])

    def references(self, feats):
        """Return reference mode time series by name: (eigenvalues, modes)."""
        out = {}
        flat = feats.reshape(feats.shape[0], -1)
        lam, V = reversible_tica(flat, self.segments, self.lag, self.n_modes)
        out["features"] = (lam, (flat - flat.mean(0)) @ V)
        coords, _, _, _ = self._source.coords()
        X = _prepare(ca_distance_features(coords), self.segments, self.window)
        lam, V = reversible_tica(X, self.segments, self.lag, self.n_modes)
        out["ca-distance"] = (lam, (X - X.mean(0)) @ V)
        bb = self._backbone()
        if bb is not None:
            X = _prepare(backbone_dihedral_features(bb), self.segments,
                         self.window)
            lam, V = reversible_tica(X, self.segments, self.lag, self.n_modes)
            out["dihedral"] = (lam, (X - X.mean(0)) @ V)
        return out

    def _backbone(self):
        from pdukit.io.trajectory.mdanalysis import load_xtc_segments

        a = self.cfg
        sel = "protein and (name N or name CA or name C)"
        try:
            import MDAnalysis

            names = list(MDAnalysis.Universe(self.top).select_atoms(sel).names)
        except Exception:  # noqa: BLE001 -- unreadable topology
            return None
        if len(names) < 6 or len(names) % 3 or any(
            names[i:i + 3] != ["N", "CA", "C"]
            for i in range(0, len(names), 3)
        ):
            return None  # no complete N, CA, C backbone (e.g. CA-only)
        try:
            xyz, segs, _ = load_xtc_segments(
                self.top, self.xtc, stride=a.stride,
                max_frames=a.frames or None, verbose=False,
                atom_selection=sel,
            )
        except Exception:  # noqa: BLE001 -- no backbone atoms
            return None
        if xyz.shape[1] % 3 or [tuple(s) for s in segs] != self.segments:
            return None
        return xyz

    def rows(self):
        """One row per reference and reference mode."""
        feats = self.features()
        Y, owner = self.pdu_coordinates(feats)
        (lam_on, on), (lam_off, off) = self.assembled(Y, owner)
        out = []
        for name, (lam_r, ref) in self.references(feats).items():
            for j in range(ref.shape[1]):
                r = ref[:, j]
                row = dict(
                    reference=name, mode=j,
                    reference_timescale_frames=timescale(lam_r[j], self.lag),
                    coverage_r2=round(explained(r, Y), 4),
                )
                for tag, lam, M in (("on", lam_on, on), ("off", lam_off, off)):
                    c = np.array([abs(np.corrcoef(r, M[:, k])[0, 1])
                                  for k in range(M.shape[1])])
                    k = int(np.nanargmax(c))
                    row[f"{tag}_r2"] = round(explained(r, M), 4)
                    row[f"{tag}_best_corr2"] = round(float(c[k] ** 2), 4)
                    row[f"{tag}_best_timescale_frames"] = timescale(
                        lam[k], self.lag)
                out.append(row)
        return out


def main(argv=None):
    """Compare a run's units with fixed reference modes (``pdu refmodes``)."""
    ap = argparse.ArgumentParser(
        prog="pdu refmodes", description=__doc__.split("\n\n")[0]
    )
    ap.add_argument("run", help="finished run folder (pdu_out.ppf, results)")
    ap.add_argument("-f", "--traj", nargs="+", required=True)
    ap.add_argument("-s", "--top", required=True)
    ap.add_argument("-o", "--outdir", required=True)
    ap.add_argument("--n-modes", type=int, default=10)
    a = ap.parse_args(argv)
    rm = ReferenceModes(a.run, a.traj, a.top, n_modes=a.n_modes)
    rows = rm.rows()
    path = CsvWriter(Path(a.outdir) / "reference_modes.csv").write(
        rows, note="R^2 of each reference mode from the PDU coordinates "
        "(coverage) and from the top assembled modes, channels on / off")
    print("wrote", path)
    for name in sorted({r["reference"] for r in rows}):
        sel = [r for r in rows if r["reference"] == name]
        m = {k: np.mean([r[k] for r in sel])
             for k in ("coverage_r2", "on_r2", "off_r2")}
        print(f"{name:12s} coverage {m['coverage_r2']:.2f}  on "
              f"{m['on_r2']:.2f}  off {m['off_r2']:.2f}  (mean R^2)")
    return 0
