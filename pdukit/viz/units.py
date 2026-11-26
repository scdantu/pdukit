"""Figures about units: ``pdu plot units`` and ``pdu plot lag``."""

import argparse
import sys
from pathlib import Path

import numpy as np

from pdukit.io.results.reader import ResultsReader
from pdukit.numerics import confidence
from pdukit.viz.pymol import residue_map
from pdukit.viz.style import (
    BRIDGE,
    DARK,
    GREY,
    PORT_ALPHA,
    band_color,
    panel_label,
    pyplot,
    save,
    unit_color,
)


def units_figure(results, out, pdb=None, title=None):
    """Units with their sizes and confidence, plus the RCS along the chain.

    Panel a: one bar per unit (core solid, ports pale) with its UCS and
    band; hatched when fewer than ``confidence.LOW_EVIDENCE`` seeds support
    it.  Panel b: RCS of every residue along the sequence, coloured by
    band, a ribbon with the unit of each residue and ticks for bridge and
    loose residues.
    """
    plt = pyplot()
    n_res = int(results["data"]["n_residues"])
    units = confidence.unit_scores(results)
    scores = confidence.residue_scores(results)
    classes = {
        int(r["residue"]): r["cls"]
        for r in (results.get("incoming_ports") or {}).get("residues", [])
    }
    numbers = None
    if pdb:
        rmap = residue_map(pdb, n_res)
        numbers = [rmap[i][1] for i in range(n_res)]

    fig = plt.figure(figsize=(7.2, 4.6))
    gs = fig.add_gridspec(
        2, 1, height_ratios=[max(len(units), 2) * 0.55 + 0.6, 2.2], hspace=0.5
    )
    ax = fig.add_subplot(gs[0])
    for row, u in enumerate(units):
        y = len(units) - 1 - row
        color = unit_color(u["unit"])
        hatch = "////" if u["low_evidence"] else None
        ax.barh(
            y, u["size"], color=color, edgecolor="white", hatch=hatch,
            linewidth=0.4, height=0.7,
        )  # fmt: skip
        ax.barh(
            y, u["n_ports"], left=u["size"], color=color, alpha=PORT_ALPHA,
            edgecolor="white", linewidth=0.4, height=0.7,
        )  # fmt: skip
        total = u["size"] + u["n_ports"]
        flag = "\u2020" if u["low_evidence"] else ""
        plural = "seed" if u["n_seeds"] == 1 else "seeds"
        ax.plot(
            [total + 1.0], [y], marker="o", ms=5, color=band_color(u["band"]),
            clip_on=False,
        )  # fmt: skip
        ax.text(
            total + 2.2, y, f"UCS {u['ucs']:.0f}{flag}  ({u['band']}, "
            f"{u['n_seeds']} {plural})", va="center", fontsize=6.5,
            color=DARK,
        )  # fmt: skip
    ax.set_yticks(range(len(units)))
    ax.set_yticklabels([f"u{u['unit']}" for u in units[::-1]])
    ax.set_xlabel("residues (core solid, ports pale)")
    ax.set_xlim(0, max(u["size"] + u["n_ports"] for u in units) * 1.7)
    ax.tick_params(axis="y", length=0, pad=4)
    panel_label(ax, "a")
    if any(u["low_evidence"] for u in units):
        fig.text(
            0.01, 0.01, "\u2020 hatched: supported by fewer than "
            f"{confidence.LOW_EVIDENCE} seeds, score is indicative only",
            ha="left", va="bottom", fontsize=6, color=DARK,
        )  # fmt: skip

    bx = fig.add_subplot(gs[1])
    x = np.arange(n_res)
    rcs = confidence.rcs_vector(results, n_res)
    colors = [
        band_color(confidence.band(rcs[i])) if i in scores else GREY
        for i in range(n_res)
    ]
    bx.bar(x, rcs, color=colors, width=0.9, linewidth=0)
    for i, d in scores.items():
        bx.add_patch(
            plt.Rectangle(
                (i - 0.45, 104), 0.9, 7, color=unit_color(d["unit"]),
                alpha=1.0 if d["role"] == "core" else PORT_ALPHA, lw=0,
                clip_on=False,
            )
        )  # fmt: skip
    for i, cls in classes.items():
        if cls == "bridge":
            bx.plot([i], [-6], marker="^", ms=3, color=BRIDGE, clip_on=False)
        elif cls == "loose":
            bx.plot([i], [-6], marker="|", ms=3, color=GREY, clip_on=False)
    for level in (50, 70, 90):
        bx.axhline(level, color=GREY, lw=0.4, ls=":")
    bx.set_ylim(0, 100)
    bx.set_xlim(-1, n_res)
    bx.set_ylabel("RCS")
    if numbers:
        step = max(1, n_res // 8)
        bx.set_xticks(range(0, n_res, step))
        bx.set_xticklabels([numbers[i] for i in range(0, n_res, step)])
    bx.set_xlabel("residue")
    panel_label(bx, "b", y=1.12)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=band_color(n)) for n in
        ("very high", "high", "low", "very low")
    ]  # fmt: skip
    bx.legend(
        handles, ["≥90", "70–90", "50–70", "<50"],
        title="RCS", ncol=4, loc="upper right", frameon=False,
        bbox_to_anchor=(1.0, -0.3), fontsize=6, title_fontsize=6,
    )  # fmt: skip
    if title:
        fig.suptitle(title, fontsize=8, y=0.98)
    save(fig, out)
    plt.close(fig)


def _lag_row(path):
    results = ResultsReader.load(path)
    args = results.get("args") or {}
    counts = (results.get("incoming_ports") or {}).get("counts") or {}
    n_res = int(results["data"]["n_residues"])
    lag_ns = float(args["lag"]) * float(args["frame_ps"]) / 1000.0
    unassigned = counts.get("bridge", 0) + counts.get("loose", 0)
    return {
        "lag_ns": lag_ns,
        "pct_unassigned": 100.0 * unassigned / n_res,
        "n_units": len(results.get("units_final") or []),
        "n_residues": n_res,
    }


def lag_figure(groups, out):
    """Lag versus % unassigned residues (and unit count), one line a group.

    ``groups`` maps a label to a list of run directories made at different
    lags.  Unassigned = bridge + loose residues.
    """
    plt = pyplot()
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.6))
    for k, (label, runs) in enumerate(groups.items()):
        rows = sorted((_lag_row(r) for r in runs), key=lambda r: r["lag_ns"])
        lag = [r["lag_ns"] for r in rows]
        color = unit_color(k)
        a.plot(
            lag, [r["pct_unassigned"] for r in rows], marker="o", ms=3.5,
            color=color, label=label, lw=1.0,
        )  # fmt: skip
        b.plot(
            lag, [r["n_units"] for r in rows], marker="s", ms=3.5,
            color=color, label=label, lw=1.0,
        )  # fmt: skip
    for ax, ylabel in ((a, "% residues unassigned"), (b, "units detected")):
        ax.set_xlabel("lag τ (ns)")
        ax.set_ylabel(ylabel)
        if len({r for g in groups.values() for r in g}) > 3:
            ax.set_xscale("log")
        ax.set_ylim(bottom=0)
    a.legend(frameon=False)
    panel_label(a, "a")
    panel_label(b, "b")
    fig.tight_layout()
    save(fig, out)
    plt.close(fig)


def _groups(specs):
    """Parse ``NAME=DIR`` (or bare ``DIR``) arguments into groups."""
    groups = {}
    for spec in specs:
        name, _, path = spec.partition("=")
        if not path:
            name, path = Path(spec).resolve().parent.name, spec
        groups.setdefault(name, []).append(path)
    return groups


def main_units(argv=None):
    """Command line: ``pdu plot units RUN_DIR``."""
    ap = argparse.ArgumentParser(
        prog="pdu plot units",
        description="units, sizes and confidence (UCS / RCS) figure",
    )
    ap.add_argument("run", help="run directory (or results.json)")
    ap.add_argument("--pdb", help="PDB for real residue numbers on the axis")
    ap.add_argument("--title")
    ap.add_argument(
        "-o", "--out", help="output .png/.pdf/.svg (default RUN/units.png)"
    )
    args = ap.parse_args(argv)
    results = ResultsReader.load(args.run)
    rundir = results.rundir
    out = Path(args.out) if args.out else rundir / "units.png"
    units_figure(results, out, pdb=args.pdb, title=args.title)
    return 0


def main_lag(argv=None):
    """Command line: ``pdu plot lag NAME=RUN ...``."""
    ap = argparse.ArgumentParser(
        prog="pdu plot lag",
        description="lag versus % unassigned residues across runs",
    )
    ap.add_argument(
        "runs",
        nargs="+",
        help="run directories, optionally NAME=DIR to group runs of one "
        "protein into one line",
    )
    ap.add_argument("-o", "--out", default="lag_unassigned.png")
    args = ap.parse_args(argv)
    lag_figure(_groups(args.runs), args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main_units())
