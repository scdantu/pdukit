"""The command line of ``pdu run``: eight arguments, everything else in a ppf.

    pdu run -p my.ppf -f rep1.xtc rep2.xtc -s top.pdb -o out
    pdu run -p my.ppf --nt 8              # use 8 CPU threads
    pdu run -p my.ppf --set lag=20 dec-seeds=64
    pdu run -p my.ppf --check            # print resolved settings, run nothing

Old command lines that spell a setting as its own flag (``--dec-seeds 64``,
``--faces``) still work: they are turned into ``--set`` with a warning.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pdukit.config.config import _ROWS, Config, ConfigError, normalise_key

ARGUMENTS = (
    "-f", "-s", "-p", "-o", "--nt", "--set", "--log-level", "--check"
)


def build_parser(description, prog="pdu run"):
    """Return the parser of the eight run arguments."""
    ap = argparse.ArgumentParser(prog=prog, description=description)
    ap.add_argument("-f", "--traj", "--xtc", nargs="+", metavar="FILE",
                    help="trajectory files, one per replica (or xtc = ... "
                    "in the ppf)")
    ap.add_argument("-s", "--top", "--xtc-topology", metavar="FILE",
                    help="topology file (PDB, GRO, TPR)")
    ap.add_argument("-p", "--ppf", "--params", metavar="FILE",
                    help="parameter file (key = value); the place for "
                    "nearly every setting.  A shipped preset can be named "
                    "without its path: -p workstation")
    ap.add_argument("-o", "--out", "--outdir", metavar="DIR",
                    help="output folder")
    ap.add_argument("--nt", "-nt", type=int, metavar="N",
                    help="CPU threads for numpy/scipy and torch "
                    "(default -1 = all cores)")
    ap.add_argument("--set", nargs="+", metavar="KEY=VALUE", default=[],
                    action="extend",
                    help="override any setting of the ppf, e.g. --set lag=20 "
                    "dec-seeds=64")
    ap.add_argument("--log-level", choices=["quiet", "info", "debug"],
                    help="progress log detail")
    ap.add_argument("--check", action="store_true",
                    help="print the resolved settings and where each came "
                    "from; run nothing")
    return ap


def shipped_preset(name):
    """Return the path of ``name`` if it is not a file but a shipped preset.

    ``-p workstation`` (or ``workstation.ppf``) finds
    ``pdukit/params/workstation.ppf`` when there is no such file here, so
    the presets can be used without knowing where pip installed them.
    """
    if not name or Path(name).exists():
        return name
    stem = Path(name).name
    stem = stem if stem.endswith(".ppf") else stem + ".ppf"
    params = Path(__file__).resolve().parents[1] / "params"
    hits = [params / stem] + sorted(params.glob(f"*/{stem}"))
    for hit in hits:
        if hit.is_file():
            sys.stderr.write(f"pdu: using the shipped preset {hit}\n")
            return str(hit)
    return name


def _legacy(tokens):
    """Turn old ``--key value`` tokens into ``(key, text)`` pairs."""
    pairs = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if not tok.startswith("--"):
            raise ConfigError(f"unexpected argument {tok!r} (see pdu run -h)")
        key = normalise_key(tok[2:].split("=", 1)[0])
        if "=" in tok:
            values = [tok.split("=", 1)[1]]
            i += 1
        else:
            values = []
            i += 1
            while i < len(tokens) and not (
                tokens[i].startswith("--") and len(tokens[i]) > 2
                and not tokens[i][2].isdigit()
            ):
                values.append(tokens[i])
                i += 1
        row = _ROWS.get(key)
        if row is None:
            raise ConfigError(
                f"unknown option --{key} (settings go in the ppf or --set)"
            )
        if row[3] in ("flag", "noflag") and not values:
            values = ["1"]
        sys.stderr.write(
            f"pdu: warning: --{key} is deprecated; use --set {key}="
            f"{' '.join(values)}\n"
        )
        pairs.append((key, " ".join(values)))
    return pairs


def parse_run(argv, description="pdu run: detect protein dynamical units",
              prog="pdu run", planted=False):
    """Return ``(options, args)``; exit with a message on bad settings."""
    ap = build_parser(description, prog)
    args, rest = ap.parse_known_args(argv)
    args.ppf = shipped_preset(args.ppf)
    try:
        sets = _legacy(rest)
        if planted:
            sets.append(("planted", "1"))
        for key, value in (("xtc", args.traj and " ".join(args.traj)),
                           ("xtc-topology", args.top),
                           ("outdir", args.out),
                           ("params", args.ppf),
                           ("log-level", args.log_level),
                           ("nt", None if args.nt is None else str(args.nt))):
            if value:
                sets.append((key, value))
        sets += list(args.set)
        cfg = Config.build(ppf=args.ppf, sets=sets,
                           require_outdir=not args.check,
                           require_input=not args.check)
    except ConfigError as exc:
        ap.error(str(exc))
    return cfg, args
