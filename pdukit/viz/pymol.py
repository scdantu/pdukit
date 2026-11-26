"""PyMOL script showing the units and their coupling network.

``pdu plot pml RUN_DIR --pdb protein.pdb`` writes one ``.pml`` with two
scenes (``F1`` / ``F2`` switch between them in PyMOL):

* ``units_cartoon``: cartoon, every unit in its own colour; residues
  that belong to no unit stay grey.
* ``units_detail``: faint cartoon; each unit's core as cyan sticks, its
  ports as spheres, bridge residues as spheres, a transparent surface
  per unit in the unit colour, and the significant couplings between
  residues as cylinders.

Residue index ``i`` of ``results.json`` is the ``i``-th residue of the
topology the run was made from; its CA atom in the PDB gives the real chain
and number, so no offset has to be guessed.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from pdukit.io.results.reader import ResultsReader
from pdukit.numerics import confidence

UNIT_COLORS = [
    "#3B5BA9",
    "#B07D00",
    "#2E9E5B",
    "#7B4FA0",
    "#D1495B",
    "#00897B",
    "#A0522D",
    "#6B6BD6",
]
OVERFLOW_COLOR = "#999999"
ROLE_COLORS = {
    "core": "#2A9DB8",
    "port": "#C0703A",
    "bridge": "#B8318C",
    "loose": "#BDBDBD",
}
LINK_COLOR = "#555555"
# pLDDT-style confidence colours, by RCS band
BAND_COLORS = {
    "very high": "#0053D6",
    "high": "#65CBF3",
    "low": "#FFDB13",
    "very low": "#FF7D45",
}


def _rgb(hexcolor):
    h = hexcolor.lstrip("#")
    return [round(int(h[i : i + 2], 16) / 255, 4) for i in (0, 2, 4)]


def read_ca(pdb):
    """Return ``[(chain, resi, xyz, resname)]`` for the CA atoms, in order.

    ``chain`` is empty for a PDB without chain identifiers.

    Only the first model is read; alternate locations other than blank/A
    are skipped.  ``resi`` is a string such as ``"52"`` or ``"52a"``.
    """
    out = []
    for line in Path(pdb).read_text().splitlines():
        if line.startswith("ENDMDL"):
            break
        if not line.startswith(("ATOM", "HETATM")):
            continue
        if line[12:16].strip() != "CA" or line[16] not in " A":
            continue
        resi = line[22:26].strip() + line[26].strip().lower()
        xyz = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
        out.append((line[21].strip(), resi, xyz, line[17:20].strip()))
    return out


def residue_map(pdb, n_res, first_residue=None):
    """Map residue index -> ``(chain, resi, xyz, resname)`` from the CA atoms."""
    ca = read_ca(pdb)
    if first_residue is not None:
        chain = ca[0][0] if ca else ""
        ca = [(chain, str(first_residue + i), c[2]) for i, c in enumerate(ca)]
    if len(ca) != n_res:
        raise SystemExit(
            f"{pdb} has {len(ca)} CA atoms but the run has {n_res} residues; "
            "use the PDB the run was made from (or --first-residue N to "
            "number a matching chain from N)"
        )
    return dict(enumerate(ca))


def _sel(indices, rmap):
    """PyMOL selection text for residue indices (empty -> ``none``)."""
    by_chain = {}
    for i in sorted(set(indices)):
        chain, resi = rmap[i][:2]
        by_chain.setdefault(chain, []).append(resi)
    parts = [
        (f"(chain {c} and " if c else "(") + f"resi {'+'.join(r)})"
        for c, r in by_chain.items()
    ]
    return " or ".join(parts) if parts else "none"


def unit_table(results):
    """Return ``[{unit, core, ports}]`` from the final units and ports."""
    ports = {
        u["unit"]: u
        for u in (results.get("incoming_ports") or {}).get("units", [])
    }
    units = []
    for u in results.get("units_final") or results.get("units_split") or []:
        p = ports.get(u["unit"], {})
        units.append(
            {
                "unit": u["unit"],
                "core": list(p.get("core", u["residues"])),
                "ports": list(p.get("ports", [])),
            }
        )
    return units


def role_table(results):
    """Return ``{residue index: class}`` (core/port/bridge/loose/shared)."""
    rows = (results.get("incoming_ports") or {}).get("residues") or []
    return {int(r["residue"]): r["cls"] for r in rows}


def significant_edges(
    results, max_partners=3, min_seq_sep=4, top_frac=0.3, min_rho=None
):
    """Return ``[(i, j, strength)]`` of significant couplings, i < j.

    A coupling is significant when the predictive-graph ``rho`` is above its
    null threshold ``theta``.  The strength of an undirected pair is the
    larger of the two directions.  Each residue keeps its ``max_partners``
    strongest partners (0 = keep all); pairs closer than ``min_seq_sep`` in
    sequence are dropped; finally only the strongest ``top_frac`` of the
    remaining pairs are returned (1 = all).
    """
    pg = results.get("predictive_graph") or {}
    if not pg.get("rho"):
        return []
    rho = np.asarray(pg["rho"], dtype=float)
    theta = float(pg["theta"]) if min_rho is None else float(min_rho)
    sym = np.maximum(rho, rho.T)
    n = sym.shape[0]
    keep = np.zeros_like(sym, dtype=bool)
    for i in range(n):
        order = np.argsort(-sym[i])
        picked = 0
        for j in order:
            if j == i or abs(i - j) < min_seq_sep or sym[i, j] <= theta:
                continue
            keep[i, j] = keep[j, i] = True
            picked += 1
            if max_partners and picked >= max_partners:
                break
    pairs = [
        (i, j, float(sym[i, j]))
        for i in range(n)
        for j in range(i + 1, n)
        if keep[i, j]
    ]
    if pairs and top_frac < 1:
        pairs.sort(key=lambda p: -p[2])
        pairs = pairs[: max(1, int(round(top_frac * len(pairs))))]
    return pairs


def build_pml(
    results,
    pdb,
    first_residue=None,
    edges="significant",
    max_partners=3,
    min_seq_sep=4,
    top_frac=0.3,
    surface_alpha=0.65,
    name=None,
):
    """Return the text of the ``.pml`` script."""
    n_res = int((results.get("data") or {}).get("n_residues") or 0)
    rmap = residue_map(pdb, n_res, first_residue)
    units = unit_table(results)
    roles = role_table(results)
    pdb = Path(pdb)
    title = name or pdb.stem

    def color_name(k):
        return f"pdu_u{k}" if k < len(UNIT_COLORS) else "pdu_overflow"

    lines = []
    w = lines.append
    w(f"# PDU units for {title}: {len(units)} units")
    w("# scenes: F1 = units_cartoon, F2 = units_detail, F3 = units_confidence")
    w("# residue numbers are the PDB's own; colours are set below")
    w("reinitialize")
    # two copies of the structure so each scene keeps its own settings
    w(f"load {pdb.name}, pdu_cartoon")
    w(f"load {pdb.name}, pdu_detail")
    w(f"load {pdb.name}, pdu_conf")
    w("bg_color white")
    w("set ray_shadow, 0")
    w("set antialias, 2")
    w("set sphere_scale, 0.5")
    w("set stick_radius, 0.22")
    w("set cartoon_fancy_helices, 1")
    for k in range(len(units)):
        if k < len(UNIT_COLORS):
            w(f"set_color pdu_u{k}, {_rgb(UNIT_COLORS[k])}")
    w(f"set_color pdu_overflow, {_rgb(OVERFLOW_COLOR)}")
    for role, hexcolor in ROLE_COLORS.items():
        w(f"set_color pdu_{role}, {_rgb(hexcolor)}")
    w(f"set_color pdu_link, {_rgb(LINK_COLOR)}")
    for name_, hexcolor in BAND_COLORS.items():
        w(f"set_color pdu_{name_.replace(' ', '_')}, {_rgb(hexcolor)}")

    # selections (residues only; applied to one object at a time below)
    all_core, all_port = [], []
    for u in units:
        k = u["unit"]
        w(f"select u{k}, {_sel(u['core'] + u['ports'], rmap)}")
        w(f"select u{k}_core, {_sel(u['core'], rmap)}")
        w(f"select u{k}_port, {_sel(u['ports'], rmap)}")
        all_core += u["core"]
        all_port += u["ports"]
    bridge = [i for i, c in roles.items() if c == "bridge"]
    loose = [i for i, c in roles.items() if c == "loose"]
    w(f"select pdu_core, {_sel(all_core, rmap)}")
    w(f"select pdu_port, {_sel(all_port, rmap)}")
    w(f"select pdu_bridge, {_sel(bridge, rmap)}")
    w(f"select pdu_loose, {_sel(loose, rmap)}")

    # couplings: cylinders between CA atoms, one CGO object
    pairs = []
    if edges != "none":
        pairs = significant_edges(
            results,
            max_partners=0 if edges == "all" else max_partners,
            min_seq_sep=min_seq_sep,
            top_frac=1.0 if edges == "all" else top_frac,
        )
    unit_of = {}
    for u in units:
        for i in u["core"] + u["ports"]:
            unit_of.setdefault(i, u["unit"])
    if pairs:
        smax = max(p[2] for p in pairs)
        w("python")
        w("from pymol import cmd")
        w("from pymol.cgo import CYLINDER")
        w("net = []")
        for i, j, s in pairs:
            same = i in unit_of and unit_of.get(i) == unit_of.get(j)
            col = (
                _rgb(UNIT_COLORS[unit_of[i] % len(UNIT_COLORS)])
                if same
                else _rgb(LINK_COLOR)
            )
            r = 0.04 + 0.11 * s / smax
            p, q = rmap[i][2], rmap[j][2]
            w(
                f"net += [CYLINDER, {p[0]}, {p[1]}, {p[2]}, {q[0]}, {q[1]}, "
                f"{q[2]}, {r:.3f}, {col[0]}, {col[1]}, {col[2]}, "
                f"{col[0]}, {col[1]}, {col[2]}]"
            )
        w("cmd.load_cgo(net, 'pdu_network')")
        w("python end")

    # scene 1: cartoon, one colour per unit
    w("")
    w("# ---- scene 1: units as coloured cartoon")
    w("hide everything, pdu_cartoon")
    w("show cartoon, pdu_cartoon")
    w("set cartoon_color, grey85, pdu_cartoon")
    for u in units:
        w(
            f"set cartoon_color, {color_name(u['unit'])}, "
            f"pdu_cartoon and u{u['unit']}"
        )
    w("orient pdu_cartoon")
    w("disable pdu_detail")
    w("disable pdu_conf")
    if pairs:
        w("disable pdu_network")
    w("scene units_cartoon, store")

    # scene 2: cores, ports, bridges, unit surfaces, couplings
    w("")
    w("# ---- scene 2: cores, ports, bridges, unit surfaces, couplings")
    w("enable pdu_detail")
    w("disable pdu_conf")
    w("hide everything, pdu_detail")
    w("show cartoon, pdu_detail")
    w("set cartoon_color, grey85, pdu_detail")
    for u in units:
        k = u["unit"]
        w(f"create surf_u{k}, pdu_detail and u{k}")
        w(f"hide everything, surf_u{k}")
        w(f"show surface, surf_u{k}")
        w(f"color {color_name(k)}, surf_u{k}")
        w(f"set transparency, {surface_alpha}, surf_u{k}")
    w("show sticks, pdu_detail and pdu_core")
    w("color pdu_core, pdu_detail and pdu_core and elem C")
    w("hide everything, pdu_detail and elem H")
    w("show spheres, pdu_detail and (pdu_port or pdu_bridge) and name CA")
    w("color pdu_port, pdu_detail and pdu_port and name CA")
    w("color pdu_bridge, pdu_detail and pdu_bridge and name CA")
    if loose:
        w("show spheres, pdu_detail and pdu_loose and name CA")
        w("color pdu_loose, pdu_detail and pdu_loose and name CA")
        w("set sphere_scale, 0.3, pdu_detail and pdu_loose")
    if pairs:
        w("enable pdu_network")
    w("orient pdu_detail")
    w("scene units_detail, store")

    # scene 3: residue confidence score (RCS) as pLDDT-style colours
    scores = confidence.residue_scores(results)
    if scores:
        w("")
        w(
            "# ---- scene 3: RCS (residue confidence score) in the B-factor column"
        )
        w("disable pdu_detail")
        w("disable pdu_cartoon")
        if pairs:
            w("disable pdu_network")
        for k in range(len(units)):
            w(f"disable surf_u{k}")
        w("enable pdu_conf")
        w("alter pdu_conf, b=0")
        by_value = {}
        for i, d in scores.items():
            by_value.setdefault(round(d["rcs"], 1), []).append(i)
        for value, idx in sorted(by_value.items()):
            w(f"alter pdu_conf and ({_sel(idx, rmap)}), b={value}")
        w("hide everything, pdu_conf")
        w("show cartoon, pdu_conf")
        w("set cartoon_color, grey85, pdu_conf")
        groups = {}
        for i, d in scores.items():
            groups.setdefault(confidence.band(d["rcs"]), []).append(i)
        for name_, idx in groups.items():
            w(
                f"set cartoon_color, pdu_{name_.replace(' ', '_')}, "
                f"pdu_conf and ({_sel(idx, rmap)})"
            )
        w("orient pdu_conf")
        w("scene units_confidence, store")
    w("")
    w("python")
    w("from pymol import cmd")
    w("cmd.set_key('F1', cmd.scene, ('units_cartoon', 'recall'))")
    w("cmd.set_key('F2', cmd.scene, ('units_detail', 'recall'))")
    if scores:
        w("cmd.set_key('F3', cmd.scene, ('units_confidence', 'recall'))")
    w("python end")
    w("scene units_cartoon, recall")
    return "\n".join(lines) + "\n"


def main(argv=None):
    """Command line: ``pdu plot pml RUN_DIR --pdb protein.pdb``."""
    ap = argparse.ArgumentParser(
        prog="pdu plot pml",
        description="write a PyMOL script showing units and couplings",
    )
    ap.add_argument("run", help="run directory (or results.json)")
    ap.add_argument("--pdb", required=True, help="topology PDB of the run")
    ap.add_argument("-o", "--out", help="output .pml (default RUN/units.pml)")
    ap.add_argument(
        "--edges",
        choices=["significant", "all", "none"],
        default="significant",
        help=(
            "couplings drawn: significant = rho above the null threshold, "
            "each residue's strongest partners, strongest fraction only; all = every "
            "significant pair; none = no network"
        ),
    )
    ap.add_argument(
        "--max-partners",
        type=int,
        default=3,
        help="strongest significant partners kept per residue",
    )
    ap.add_argument(
        "--min-seq-sep",
        type=int,
        default=4,
        help="drop couplings between residues closer than this in sequence",
    )
    ap.add_argument(
        "--top-frac",
        type=float,
        default=0.3,
        help="draw only this fraction of the strongest remaining couplings",
    )
    ap.add_argument(
        "--alpha",
        type=float,
        default=0.65,
        help="surface transparency (0 = opaque, 1 = invisible)",
    )
    ap.add_argument(
        "--first-residue",
        type=int,
        help="number the PDB's first chain from this value if the numbers "
        "differ from the run",
    )
    args = ap.parse_args(argv)
    results = ResultsReader.load(args.run)
    rundir = results.rundir
    text = build_pml(
        results,
        args.pdb,
        first_residue=args.first_residue,
        edges=args.edges,
        max_partners=args.max_partners,
        min_seq_sep=args.min_seq_sep,
        top_frac=args.top_frac,
        surface_alpha=args.alpha,
    )
    out = Path(args.out) if args.out else rundir / "units.pml"
    out.write_text(text)
    print(
        f"wrote {out}  (open in PyMOL from the PDB's folder; F1/F2/F3 = scenes)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
