"""The ``pdu`` command: one entry point with a module per task.

    pdu run       -f R1.xtc R2.xtc -s top.tpr -p t4l.ppf -o out
    pdu bench     -p bench.ppf -o out          (planted system, known answer)
    pdu check     -f ... -s ... -p t4l.ppf     (resolve and print; run nothing)
    pdu summarise RESULTS_ROOT
    pdu compare   RUN_DIR RUN_DIR ...
    pdu fasttest  --out DIR ...

Settings live in a parameter file (``-p``, ``key = value``, see
`pdukit.lib.pdu.ppf`); a command-line option overrides the file.
"""

from __future__ import annotations

import sys

MODULES = {
    "run": "run the pipeline on MD trajectories",
    "bench": "run the pipeline on a planted system with a known answer",
    "check": "resolve all settings and print them; run nothing",
    "summarise": "one table from many runs",
    "compare": "do several runs find the same units?",
    "fasttest": "the fast test grid of planted designs",
}


def _version():
    try:
        from importlib.metadata import version

        return version("pdukit")
    except Exception:  # not installed (running from a checkout)
        return "unknown"


def _usage():
    lines = [f"usage: pdu <module> [options]    (pdukit {_version()})", ""]
    lines += [f"  {name:<10} {text}" for name, text in MODULES.items()]
    lines += ["", "pdu <module> -h   shows the options of a module."]
    return "\n".join(lines)


def _run(argv, planted=False):
    from .lib.pdu.cog2_cli import parse_args
    from .lib.pdu.cog2_pipeline import Cog2Run

    args = parse_args(
        argv + (["--planted"] if planted else []),
        description="pdu run: detect protein dynamical units",
    )
    return Cog2Run(args).run()


def _check(argv):
    from .lib.pdu.cog2_cli import parse_args

    args = parse_args(
        argv,
        description="pdu check: resolve all settings and print them",
        require_outdir=False,
    )
    sys.stdout.write(args._ppf)
    return 0


def main(argv=None):
    """Dispatch ``pdu <module> ...``; return the exit code."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(_usage())
        return 0
    if argv[0] in ("-V", "--version"):
        print(f"pdukit {_version()}")
        return 0
    module, rest = argv[0], argv[1:]
    if module == "run":
        return _run(rest)
    if module == "bench":
        return _run(rest, planted=True)
    if module == "check":
        return _check(rest)
    if module == "summarise":
        from .lib.pdu.cog2_summarise import main as run_module
    elif module == "compare":
        from .lib.pdu.cog2_compare import main as run_module
    elif module == "fasttest":
        from .lib.pdu.cog2_fasttest import main as run_module
    else:
        print(f"pdu: unknown module {module!r}\n\n{_usage()}", file=sys.stderr)
        return 2
    return run_module(rest) or 0


if __name__ == "__main__":
    sys.exit(main())
