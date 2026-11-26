"""``pdu plot``: figures and PyMOL scripts from a finished run."""

import sys

TOOLS = {
    "pml": "PyMOL script: units as cartoon, detail, and confidence scenes",
    "units": "units, sizes and confidence (UCS) + residue confidence (RCS)",
    "lag": "lag versus % unassigned residues across runs",
    "network": "residue network of one protein (circle layout)",
    "compare": "residue networks of several variants + differences",
    "movie": "movie: all growth seeds grow, merge into units, finalise",
}


def main(argv=None):
    """Dispatch ``pdu plot <tool> ...``."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print("usage: pdu plot <tool> RUN_DIR [options]\n")
        for name, text in TOOLS.items():
            print(f"  {name:<8} {text}")
        print("\npdu plot <tool> -h   shows the options of a tool.")
        return 0
    tool, rest = argv[0], argv[1:]
    if tool == "pml":
        from pdukit.viz.pymol import main as run_tool
    elif tool == "units":
        from pdukit.viz.units import main_units as run_tool
    elif tool == "lag":
        from pdukit.viz.units import main_lag as run_tool
    elif tool == "network":
        from pdukit.viz.network import main_network as run_tool
    elif tool == "compare":
        from pdukit.viz.network import main_compare as run_tool
    elif tool == "movie":
        from pdukit.viz.movie import main as run_tool
    else:
        print(f"pdu plot: unknown tool {tool!r}", file=sys.stderr)
        return 2
    return run_tool(rest) or 0


if __name__ == "__main__":
    sys.exit(main())
