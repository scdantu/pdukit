"""``pdu plot movie``: growth, merge and finalisation of the units.

All growth seeds are drawn as small maps of the protein (one per seed) in
a grid, each in its own colour, growing step by step (phase GRO).  Then the
seeds that found a unit change to that unit's colour (MRG), the large map on
the right settles on the final cores (SPL), and finally ports and bridge
residues appear (POR).  So the many seed colours converge to one colour per
unit, and the residues that belong to no unit are shown at the end.

The frame plan (:func:`build_plan`) is plain numpy and testable without
matplotlib; :func:`render` needs matplotlib and ffmpeg for an mp4.
"""

import argparse
import colorsys
import math
import shutil
import sys
from pathlib import Path

import numpy as np

from pdukit.io.results.reader import ResultsReader
from pdukit.numerics import confidence
from pdukit.viz.pymol import UNIT_COLORS, unit_table
from pdukit.viz.style import BRIDGE, coords_2d, pyplot

GREY = np.array([0.80, 0.80, 0.80])


def _hex(h):
    h = h.lstrip("#")
    return np.array([int(h[i : i + 2], 16) / 255 for i in (0, 2, 4)])


def seed_colors(n):
    """Return ``n`` well separated RGB colours (golden-ratio hues)."""
    out = []
    for i in range(n):
        h = (i * 0.61803398875) % 1.0
        out.append(np.array(colorsys.hsv_to_rgb(h, 0.6 + 0.25 * (i % 2), 0.9)))
    return out


def build_plan(results, max_growth_frames=60, hold=14):
    """Return the frame plan: static data plus a list of frame states.

    Each frame is a dict with ``phase`` (GRO / MRG / SPL / POR), ``text``,
    ``members`` (seeds x residues bool), ``seed_rgb`` (seeds x 3),
    ``seed_alpha`` (seeds), ``big_rgb`` (residues x 3), ``big_alpha`` and
    ``big_size`` (residues) and ``bridge`` / ``loose`` masks.
    """
    n_res = int(results["data"]["n_residues"])
    runs = (results.get("incoming_growth") or {}).get("runs") or []
    if not runs:
        raise SystemExit("this run has no growth history to animate")
    n_seed = len(runs)
    assign = confidence.seed_assignment(results)
    units = {u["unit"]: u for u in unit_table(results)}
    roles = {
        int(r["residue"]): r["cls"]
        for r in (results.get("incoming_ports") or {}).get("residues", [])
    }
    own = seed_colors(n_seed)
    ucol = {k: _hex(UNIT_COLORS[k % len(UNIT_COLORS)]) for k in units}
    core_unit = {i: k for k, u in units.items() for i in u["core"]}
    port_unit = {i: k for k, u in units.items() for i in u["ports"]}

    def members_at(step):
        m = np.zeros((n_seed, n_res), bool)
        for s, run in enumerate(runs):
            hist = run["history"]
            m[s, hist[min(step, len(hist) - 1)]["residues"]] = True
        return m

    def blend(members, rgb, alpha):
        big_rgb = np.tile(GREY, (n_res, 1))
        big_a = np.full(n_res, 0.25)
        weight = members * alpha[:, None]
        tot = weight.sum(axis=0)
        for i in np.flatnonzero(tot > 0):
            big_rgb[i] = (weight[:, i, None] * rgb).sum(axis=0) / tot[i]
            big_a[i] = min(1.0, 0.35 + 0.2 * tot[i])
        return big_rgb, big_a

    n_steps = max(len(r["history"]) for r in runs)
    stride = max(1, math.ceil(n_steps / max_growth_frames))
    steps = list(range(0, n_steps, stride))
    if steps[-1] != n_steps - 1:
        steps.append(n_steps - 1)
    frames = []
    size0 = np.full(n_res, 14.0)
    none = np.zeros(n_res, bool)
    alpha1 = np.ones(n_seed)
    rgb0 = np.array(own)

    for t in steps:
        m = members_at(t)
        br, ba = blend(m, rgb0, alpha1)
        frames.append(
            dict(phase="GRO", text=f"growth step {t}/{n_steps - 1}",
                 members=m, seed_rgb=rgb0, seed_alpha=alpha1, big_rgb=br,
                 big_alpha=ba, big_size=size0, bridge=none, loose=none)
        )  # fmt: skip
    m_end = members_at(n_steps)
    for _ in range(hold // 2):
        br, ba = blend(m_end, rgb0, alpha1)
        n_ok = sum(1 for v in assign.values() if v is not None)
        frames.append(
            dict(phase="GRO", text=f"{n_ok} of {n_seed} seeds found a unit",
                 members=m_end, seed_rgb=rgb0, seed_alpha=alpha1, big_rgb=br,
                 big_alpha=ba, big_size=size0, bridge=none, loose=none)
        )  # fmt: skip

    target = np.array(
        [
            ucol[assign[s]] if assign[s] is not None else GREY
            for s in range(n_seed)
        ]
    )
    end_alpha = np.array(
        [1.0 if assign[s] is not None else 0.3 for s in range(n_seed)]
    )
    n_merge = 16
    for f in range(1, n_merge + 1):
        w = f / n_merge
        rgb = (1 - w) * rgb0 + w * target
        al = (1 - w) * alpha1 + w * end_alpha
        br, ba = blend(m_end, rgb, al)
        frames.append(
            dict(phase="MRG", text="seed colours converge to their unit",
                 members=m_end, seed_rgb=rgb, seed_alpha=al, big_rgb=br,
                 big_alpha=ba, big_size=size0, bridge=none, loose=none)
        )  # fmt: skip
    br_m, ba_m = blend(m_end, target, end_alpha)
    final_core = np.tile(GREY, (n_res, 1))
    final_alpha = np.full(n_res, 0.25)
    for i, k in core_unit.items():
        final_core[i], final_alpha[i] = ucol[k], 1.0
    n_split = 12
    for f in range(1, n_split + 1):
        w = f / n_split
        frames.append(
            dict(phase="SPL", text="unit cores after the cohesion split",
                 members=m_end, seed_rgb=target, seed_alpha=end_alpha,
                 big_rgb=(1 - w) * br_m + w * final_core,
                 big_alpha=(1 - w) * ba_m + w * final_alpha,
                 big_size=size0, bridge=none, loose=none)
        )  # fmt: skip
    bridge = np.array([roles.get(i) == "bridge" for i in range(n_res)])
    loose = np.array([roles.get(i) == "loose" for i in range(n_res)])
    n_port = 12
    for f in range(1, n_port + 1):
        w = f / n_port
        rgb, al, size = final_core.copy(), final_alpha.copy(), size0.copy()
        for i, k in port_unit.items():
            pale = 0.55 * np.ones(3) + 0.45 * ucol[k]
            rgb[i] = (1 - w) * GREY + w * pale
            al[i] = (1 - w) * 0.25 + w * 1.0
        size[bridge] = 14 + 22 * w
        frames.append(
            dict(phase="POR", text="ports (pale) and bridges (magenta)",
                 members=m_end, seed_rgb=target, seed_alpha=end_alpha,
                 big_rgb=rgb, big_alpha=al, big_size=size,
                 bridge=bridge if w > 0.3 else none,
                 loose=loose if w > 0.3 else none)
        )  # fmt: skip
    last = dict(frames[-1], text="final units; bridges between them")
    frames.extend([last] * hold)
    return {
        "n_res": n_res,
        "n_seed": n_seed,
        "frames": frames,
        "unit_colors": ucol,
        "units": units,
    }


def _ffmpeg():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return None


def render(results, out, pdb=None, fps=12, max_growth_frames=60, size=(1280, 720),
           frames_dir=None, title=None):  # fmt: skip
    """Render the movie to ``out`` (mp4), or PNG frames if no ffmpeg."""
    plt = pyplot()
    plan = build_plan(results, max_growth_frames)
    n_res, n_seed = plan["n_res"], plan["n_seed"]
    xy = coords_2d(pdb, n_res)
    xy = (xy - xy.mean(axis=0)) / (np.abs(xy - xy.mean(axis=0)).max() + 1e-9)
    cols = min(n_seed, max(1, math.ceil(math.sqrt(n_seed * 1.3))))
    rows = math.ceil(n_seed / cols)
    w, h = size
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100, facecolor="white")
    gax = fig.add_axes([0.01, 0.04, 0.62, 0.84])
    bax = fig.add_axes([0.64, 0.04, 0.35, 0.84])
    cell = 2.2
    off = np.array(
        [[(s % cols) * cell, -(s // cols) * cell] for s in range(n_seed)]
    )
    pts = (xy[None, :, :] * 0.95 + off[:, None, :]).reshape(-1, 2)
    gax.scatter(pts[:, 0], pts[:, 1], s=3.0, c="#d9d9d9", linewidths=0)
    live = gax.scatter(pts[:, 0], pts[:, 1], s=5.0, c="k", linewidths=0)
    for s in range(n_seed):
        gax.text(off[s, 0] - 1.0, off[s, 1] + 1.0, str(s + 1), fontsize=5,
                 color="#777777", va="top")  # fmt: skip
    gax.set_xlim(-1.2, cols * cell - 1.0)
    gax.set_ylim(-(rows - 1) * cell - 1.2, 1.2)
    gax.set_aspect("equal")
    gax.axis("off")
    big = bax.scatter(xy[:, 0], xy[:, 1], s=14, c="k", linewidths=0, zorder=2)
    ring = bax.scatter([], [], s=60, facecolors="none", edgecolors=BRIDGE,
                       linewidths=1.2, zorder=3)  # fmt: skip
    loose_sc = bax.scatter(
        [], [], s=8, c="#888888", marker="x", linewidths=0.6
    )
    bax.set_xlim(-1.15, 1.15)
    bax.set_ylim(-1.15, 1.15)
    bax.set_aspect("equal")
    bax.axis("off")
    head = fig.text(
        0.01, 0.95, "", fontsize=13, fontweight="bold", va="center"
    )
    sub = fig.text(0.01, 0.915, "", fontsize=9, color="#444444", va="center")
    fig.text(0.01, 0.012, "each small map = one growth seed; its colour is the "
             "seed's own until it joins a unit", fontsize=7, color="#666666")  # fmt: skip
    if title:
        fig.text(0.99, 0.95, title, fontsize=10, ha="right", va="center")
    names = {"GRO": "growth", "MRG": "merge", "SPL": "split", "POR": "ports"}

    def draw(frame):
        mem = frame["members"]
        rgba = np.zeros((n_seed, n_res, 4))
        rgba[:] = (0.0, 0.0, 0.0, 0.0)
        for s in range(n_seed):
            rgba[s, mem[s], :3] = frame["seed_rgb"][s]
            rgba[s, mem[s], 3] = frame["seed_alpha"][s]
        live.set_facecolor(rgba.reshape(-1, 4))
        live.set_sizes(np.where(mem.reshape(-1), 15.0, 0.0))
        brgba = np.c_[frame["big_rgb"], frame["big_alpha"]]
        big.set_facecolor(brgba)
        big.set_sizes(frame["big_size"])
        ring.set_offsets(
            xy[frame["bridge"]] if frame["bridge"].any() else np.empty((0, 2))
        )
        loose_sc.set_offsets(
            xy[frame["loose"]] if frame["loose"].any() else np.empty((0, 2))
        )
        head.set_text(f"{frame['phase']}  {names[frame['phase']]}")
        sub.set_text(frame["text"])

    outp = Path(out)
    exe = _ffmpeg()
    if frames_dir or not exe:
        d = Path(frames_dir or outp.with_suffix("")).resolve()
        d.mkdir(parents=True, exist_ok=True)
        for k, fr in enumerate(plan["frames"]):
            draw(fr)
            fig.savefig(d / f"frame_{k:04d}.png", dpi=100, facecolor="white")
        print(f"wrote {len(plan['frames'])} PNG frames to {d}"
              + ("" if exe else " (ffmpeg not found: install it to get an mp4)"))  # fmt: skip
    else:
        from matplotlib import animation

        plt.rcParams["animation.ffmpeg_path"] = exe
        writer = animation.FFMpegWriter(
            fps=fps, codec="libx264",
            extra_args=["-pix_fmt", "yuv420p", "-crf", "20"],
        )  # fmt: skip
        with writer.saving(fig, str(outp), dpi=100):
            for fr in plan["frames"]:
                draw(fr)
                writer.grab_frame(facecolor="white")
        print(f"wrote {outp} ({len(plan['frames'])} frames, {fps} fps)")
    plt.close(fig)


def main(argv=None):
    """Command line: ``pdu plot movie RUN_DIR``."""
    ap = argparse.ArgumentParser(
        prog="pdu plot movie",
        description="movie of all growth seeds growing, merging into units "
        "and finalising",
    )
    ap.add_argument("run", help="run directory (or results.json)")
    ap.add_argument("--pdb", help="PDB: lay the maps out from the CA atoms")
    ap.add_argument("-o", "--out", help="output .mp4 (default RUN/growth.mp4)")
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--max-growth-frames", type=int, default=60)
    ap.add_argument("--frames-dir", help="also/only write PNG frames here")
    ap.add_argument("--title")
    args = ap.parse_args(argv)
    results = ResultsReader.load(args.run)
    rundir = results.rundir
    out = Path(args.out) if args.out else rundir / "growth.mp4"
    render(results, out, pdb=args.pdb, fps=args.fps,
           max_growth_frames=args.max_growth_frames,
           frames_dir=args.frames_dir, title=args.title)  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
