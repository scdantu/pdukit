"""Residue networks: ``pdu plot network`` and ``pdu plot compare``.

Nodes sit on a circle in sequence order.  An arrow ``j -> i`` means residue
j's present predicts residue i's future beyond what i knows about itself
(``rho[i, j]`` above the null threshold ``theta``); thickness follows
``rho``.  ``network`` draws one protein; ``compare`` draws several variants
on the same circle and the difference to the first one.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from pdukit.io.results.reader import ResultsReader
from pdukit.viz.pymol import residue_map, significant_edges
from pdukit.viz.style import (
    DARK,
    GREY,
    labels_from_pdb,
    panel_label,
    pyplot,
    save,
    unit_color,
)

IN_COLOR = "#1F77B4"
OUT_COLOR = "#C0392B"
GAIN_COLOR = "#C0392B"
LOSS_COLOR = "#1F77B4"


def _rho(results):
    pg = results.get("predictive_graph") or {}
    if not pg.get("rho"):
        raise SystemExit("this run has no predictive-graph rho to draw")
    return np.asarray(pg["rho"], dtype=float), float(pg["theta"])


def directed_links(
    results,
    max_partners=3,
    min_seq_sep=4,
    top_frac=0.3,
    oneway=False,
    residues=None,
):
    """Return ``{(source, target): rho}`` of significant directed links.

    Pairs are picked as in :func:`significant_edges` (strongest partners,
    not sequence neighbours, strongest fraction); each picked pair gives an
    arrow for every direction above ``theta``.  ``oneway`` keeps only pairs
    significant in a single direction.  ``residues`` restricts both ends.
    """
    rho, theta = _rho(results)
    pairs = significant_edges(results, max_partners, min_seq_sep, top_frac)
    keep = set(residues) if residues is not None else None
    links = {}
    for i, j, _ in pairs:
        if keep is not None and not (i in keep and j in keep):
            continue
        fwd, bwd = rho[i, j] > theta, rho[j, i] > theta  # j->i, i->j
        if oneway and fwd and bwd:
            continue
        if fwd:
            links[(j, i)] = float(rho[i, j])
        if bwd:
            links[(i, j)] = float(rho[j, i])
    return links


def parse_residues(text, n_res, pdb=None):
    """Parse ``--residues``: PDB residue numbers (with PDB) or indices."""
    if not text:
        return None
    wanted = [t.strip() for t in text.split(",") if t.strip()]
    if pdb:
        rmap = residue_map(pdb, n_res)
        by_resi = {rmap[i][1]: i for i in range(n_res)}
        missing = [w for w in wanted if w not in by_resi]
        if missing:
            raise SystemExit(f"residues not in the PDB: {missing}")
        return [by_resi[w] for w in wanted]
    return [int(w) - 1 for w in wanted]


def _circle(nodes):
    ang = np.pi / 2 - 2 * np.pi * np.arange(len(nodes)) / max(len(nodes), 1)
    return {n: (np.cos(a), np.sin(a)) for n, a in zip(nodes, ang)}


def draw_network(
    ax,
    nodes,
    links,
    labels,
    unit_of=None,
    focus=None,
    highlight=(),
    colors=None,
    title=None,
    label_all=True,
):
    """Draw nodes on a circle and the directed ``links`` between them."""
    from matplotlib.patches import FancyArrowPatch

    pos = _circle(nodes)
    smax = max(links.values()) if links else 1.0
    for (src, tgt), val in sorted(links.items(), key=lambda kv: kv[1]):
        if src not in pos or tgt not in pos:
            continue
        color = GREY
        if colors and (src, tgt) in colors:
            color = colors[(src, tgt)]
        elif focus is not None and tgt == focus:
            color = IN_COLOR
        elif focus is not None and src == focus:
            color = OUT_COLOR
        ax.add_patch(
            FancyArrowPatch(
                pos[src], pos[tgt], connectionstyle="arc3,rad=0.18",
                arrowstyle="-|>", mutation_scale=5, shrinkA=4, shrinkB=4,
                lw=0.4 + 1.6 * val / smax, color=color,
                alpha=0.9 if color != GREY else 0.55,
            )
        )  # fmt: skip
    for n in nodes:
        x, y = pos[n]
        edge = unit_color(unit_of[n]) if unit_of and n in unit_of else DARK
        ring = n in highlight
        ax.scatter(
            [x], [y], s=34 if ring else 20, facecolor="white",
            edgecolor="black" if ring else edge,
            linewidths=1.6 if ring else 1.0, zorder=3,
        )  # fmt: skip
        if label_all or ring or n == focus:
            ax.text(
                x * 1.17, y * 1.17, labels[n], ha="center", va="center",
                fontsize=5.5, fontweight="bold" if ring or n == focus else None,
                color=DARK,
            )  # fmt: skip
    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-1.45, 1.45)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=7)


def _unit_of(results):
    from pdukit.numerics.confidence import residue_scores

    return {i: d["unit"] for i, d in residue_scores(results).items()}


def _node_set(results_list, links_list, residues):
    if residues is not None:
        return sorted(set(residues))
    nodes = set()
    for links in links_list:
        for s, t in links:
            nodes.update((s, t))
    return sorted(nodes)


def _labels(pdb, n_res):
    return labels_from_pdb(pdb, n_res) or [str(i + 1) for i in range(n_res)]


def _legend(ax, focus_label=None, extra=()):
    from matplotlib.lines import Line2D

    items = []
    if focus_label:
        items += [
            Line2D(
                [0], [0], color=IN_COLOR, lw=2, label=f"into {focus_label}"
            ),
            Line2D(
                [0], [0], color=OUT_COLOR, lw=2, label=f"out of {focus_label}"
            ),
        ]
    items += list(extra)
    items.append(Line2D([0], [0], color=GREY, lw=2, label="other link"))
    ax.legend(handles=items, loc="lower center", ncol=len(items), frameon=False,
              bbox_to_anchor=(0.5, -0.12), fontsize=6)  # fmt: skip


def network_figure(
    results, out, pdb=None, focus=None, highlight=(), residues=None, **kw
):
    """One protein: significant directed links among its residues."""
    plt = pyplot()
    n_res = int(results["data"]["n_residues"])
    labels = _labels(pdb, n_res)
    links = directed_links(results, residues=residues, **kw)
    nodes = _node_set([results], [links], residues)
    fig, ax = plt.subplots(figsize=(3.6, 3.8))
    n_links = len(links)
    draw_network(
        ax, nodes, links, labels, unit_of=_unit_of(results), focus=focus,
        highlight=set(highlight),
        title=f"{n_links} significant links among {len(nodes)} residues",
        label_all=len(nodes) <= 40,
    )  # fmt: skip
    _legend(ax, labels[focus] if focus is not None else None)
    save(fig, out)
    plt.close(fig)


def compare_figure(
    variants, out, pdb=None, focus=None, highlight=(), residues=None, **kw
):
    """Several variants on one circle, plus differences to the first.

    ``variants`` maps a name to its ``results`` dict (same topology).  Row 1
    is each variant; row 2 is ``variant - reference``: links gained (red),
    lost (blue) and shared (grey).
    """
    from matplotlib.lines import Line2D

    plt = pyplot()
    names = list(variants)
    sizes = {int(v["data"]["n_residues"]) for v in variants.values()}
    if len(sizes) != 1:
        raise SystemExit(
            "variants must have the same number of residues "
            f"(got {sorted(sizes)})"
        )
    n_res = sizes.pop()
    labels = _labels(pdb, n_res)
    links = {
        k: directed_links(v, residues=residues, **kw)
        for k, v in variants.items()
    }
    nodes = _node_set(list(variants.values()), list(links.values()), residues)
    ref = names[0]
    rows = 1 if len(names) == 1 else 2
    fig, axes = plt.subplots(
        rows, len(names), figsize=(3.4 * len(names), 3.6 * rows),
        squeeze=False,
    )  # fmt: skip
    for c, name in enumerate(names):
        draw_network(
            axes[0, c], nodes, links[name], labels,
            unit_of=_unit_of(variants[name]), focus=focus,
            highlight=set(highlight),
            title=f"{name}: {len(links[name])} links",
            label_all=len(nodes) <= 40,
        )  # fmt: skip
        panel_label(axes[0, c], "abcdefgh"[c], x=0.02, y=0.98)
    if rows == 2:
        axes[1, 0].axis("off")
        for c, name in enumerate(names[1:], start=1):
            gained = set(links[name]) - set(links[ref])
            lost = set(links[ref]) - set(links[name])
            shared = set(links[name]) & set(links[ref])
            merged = {**links[ref], **links[name]}
            colors = {k: GAIN_COLOR for k in gained}
            colors.update({k: LOSS_COLOR for k in lost})
            draw_network(
                axes[1, c], nodes, {k: merged[k] for k in gained | lost | shared},
                labels, colors=colors, highlight=set(highlight), focus=None,
                title=f"{name} vs {ref}: +{len(gained)} / −{len(lost)} "
                f"/ ={len(shared)}", label_all=len(nodes) <= 40,
            )  # fmt: skip
        axes[1, 0].legend(
            handles=[
                Line2D([0], [0], color=GAIN_COLOR, lw=2, label=f"gained vs {ref}"),
                Line2D([0], [0], color=LOSS_COLOR, lw=2, label=f"lost vs {ref}"),
                Line2D([0], [0], color=GREY, lw=2, label="shared"),
            ],
            loc="center", frameon=False, fontsize=6.5,
        )  # fmt: skip
    if focus is not None:
        axes[0, 0].legend(
            handles=[
                Line2D([0], [0], color=IN_COLOR, lw=2, label=f"into {labels[focus]}"),
                Line2D([0], [0], color=OUT_COLOR, lw=2, label=f"out of {labels[focus]}"),
            ],
            loc="lower center", ncol=2, frameon=False, fontsize=6,
            bbox_to_anchor=(0.5, -0.1),
        )  # fmt: skip
    fig.tight_layout()
    save(fig, out)
    plt.close(fig)


def _common(ap):
    ap.add_argument("--pdb", help="PDB (residue labels like F114)")
    ap.add_argument(
        "--residues",
        help="comma-separated residues to draw (PDB numbers with --pdb, "
        "else 1-based indices); default: every residue with a drawn link",
    )
    ap.add_argument(
        "--focus",
        help="residue whose in/out links are coloured (as --residues)",
    )
    ap.add_argument(
        "--highlight", help="residues drawn with a bold ring (as --residues)"
    )
    ap.add_argument(
        "--unit",
        type=int,
        help="draw only the residues of this final unit (core + ports)",
    )
    ap.add_argument("--max-partners", type=int, default=3)
    ap.add_argument("--min-seq-sep", type=int, default=4)
    ap.add_argument("--top-frac", type=float, default=0.3)
    ap.add_argument(
        "--oneway",
        action="store_true",
        help="draw only pairs that are significant in one direction",
    )
    ap.add_argument("-o", "--out")


def _unit_residues(results, unit):
    from pdukit.viz.pymol import unit_table

    for u in unit_table(results):
        if u["unit"] == unit:
            return sorted(set(u["core"]) | set(u["ports"]))
    raise SystemExit(f"no final unit {unit}")


def _opts(args, n_res, results):
    residues = parse_residues(args.residues, n_res, args.pdb)
    if args.unit is not None:
        residues = _unit_residues(results, args.unit)
    focus = parse_residues(args.focus, n_res, args.pdb)
    high = parse_residues(args.highlight, n_res, args.pdb) or []
    return {
        "pdb": args.pdb,
        "focus": focus[0] if focus else None,
        "highlight": high,
        "residues": residues,
        "max_partners": args.max_partners,
        "min_seq_sep": args.min_seq_sep,
        "top_frac": args.top_frac,
        "oneway": args.oneway,
    }


def main_network(argv=None):
    """Command line: ``pdu plot network RUN_DIR``."""
    ap = argparse.ArgumentParser(
        prog="pdu plot network",
        description="residue network of one protein (circle layout)",
    )
    ap.add_argument("run")
    _common(ap)
    args = ap.parse_args(argv)
    results = ResultsReader.load(args.run)
    rundir = results.rundir
    opts = _opts(args, int(results["data"]["n_residues"]), results)
    out = Path(args.out) if args.out else rundir / "network.png"
    network_figure(results, out, **opts)
    return 0


def main_compare(argv=None):
    """Command line: ``pdu plot compare NAME=RUN NAME=RUN ...``."""
    ap = argparse.ArgumentParser(
        prog="pdu plot compare",
        description="residue networks of several variants and their "
        "differences to the first (the reference)",
    )
    ap.add_argument("runs", nargs="+", help="NAME=RUN_DIR, reference first")
    _common(ap)
    args = ap.parse_args(argv)
    variants = {}
    for spec in args.runs:
        name, _, path = spec.partition("=")
        if not path:
            name, path = Path(spec).resolve().name, spec
        variants[name] = ResultsReader.load(path)
    first = next(iter(variants.values()))
    opts = _opts(args, int(first["data"]["n_residues"]), first)
    compare_figure(variants, args.out or "network_compare.png", **opts)
    return 0


if __name__ == "__main__":
    sys.exit(main_network())
