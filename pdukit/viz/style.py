"""Shared look and helpers for the ``pdu plot`` figures (needs matplotlib)."""

import numpy as np

from pdukit.viz.pymol import BAND_COLORS, UNIT_COLORS, read_ca, residue_map

AA3 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}  # fmt: skip
GREY = "#BDBDBD"
DARK = "#333333"
BRIDGE = "#B8318C"
PORT_ALPHA = 0.45


def pyplot():
    """Import matplotlib with the non-interactive backend, or explain."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "pdu plot needs matplotlib: pip install 'pdukit[plot]'"
        ) from exc
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 7.5,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.6,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    return plt


def unit_color(k):
    """Return the hex colour of unit ``k``."""
    return UNIT_COLORS[k % len(UNIT_COLORS)]


def band_color(name):
    """Return the hex colour of a confidence band."""
    return BAND_COLORS[name]


def panel_label(ax, text, x=-0.12, y=1.04):
    """Write a bold lower-case panel letter."""
    ax.text(x, y, text, transform=ax.transAxes, fontsize=8, fontweight="bold")


def labels_from_pdb(pdb, n_res):
    """Return residue labels like ``F114`` (or ``None`` without a PDB)."""
    if not pdb:
        return None
    rmap = residue_map(pdb, n_res)
    return [f"{AA3.get(rmap[i][3], 'X')}{rmap[i][1]}" for i in range(n_res)]


def coords_2d(pdb, n_res):
    """Return an ``(n_res, 2)`` layout: CA PCA from a PDB, else a circle."""
    if pdb:
        xyz = np.array([c[2] for c in read_ca(pdb)], dtype=float)
        if len(xyz) == n_res:
            xyz = xyz - xyz.mean(axis=0)
            u, s, vt = np.linalg.svd(xyz, full_matrices=False)
            return xyz @ vt[:2].T
    ang = 2 * np.pi * np.arange(n_res) / max(n_res, 1)
    return np.c_[np.cos(ang), np.sin(ang)] * 10.0


def save(fig, path):
    """Save ``fig`` to ``path`` (format from the extension)."""
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"wrote {path}")
