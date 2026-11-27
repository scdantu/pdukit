"""The ``pdu`` command: one entry point with a module per task.

    pdu run       -p t4l.ppf -f R1.xtc R2.xtc -s top.tpr -o out
    pdu run       -p t4l.ppf --set lag=20 dec-seeds=64 ...   (any setting)
    pdu run       -p t4l.ppf --check           (resolve and print; run nothing)
    pdu bench     -p bench.ppf -o out          (planted system, known answer)
    pdu summarise RESULTS_ROOT
    pdu compare   RUN_DIR RUN_DIR ...
    pdu score-bench RUN_DIR ... -o TABLES    (synthetic suite tables)

Settings live in a parameter file (``-p``, ``key = value``, see
`pdukit.config`); ``--set KEY=VALUE`` overrides the file.
"""

from __future__ import annotations

import sys

MODULES = {
    "run": "run the pipeline on MD trajectories",
    "bench": "run the pipeline on a planted system with a known answer",
    "check": "resolve all settings and print them; run nothing",
    "export": "write a run's results.json as CSV and ASCII tables",
    "plot": "figures and PyMOL scripts from a run (pdu plot -h)",
    "summarise": "one table from many runs",
    "compare": "do several runs find the same units?",
    "score-bench": "score pdu bench runs against the planted answer",
    "converge": "how much trajectory is needed? compare with a full run",
    "track": "follow residues and units across lag times",
    "refmodes": "units vs fixed reference global modes (C-alpha, dihedral)",
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
    from pdukit.cli.arguments import parse_run
    from pdukit.cog.pipeline import Pipeline
    from pdukit.cog.state import RunState

    cfg, args = parse_run(
        argv,
        prog="pdu bench" if planted else "pdu run",
        planted=planted,
    )
    if args.check:
        sys.stdout.write(cfg.to_ppf(sources=True))
        return 0
    from pdukit.errors import ResourceError

    try:
        return Pipeline(RunState(cfg)).run()
    except ResourceError as exc:
        print(f"pdu: stopped: {exc}", file=sys.stderr)
        return 3


def _check(argv):
    return _run(list(argv) + ["--check"])


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
        from pdukit.analysis.summary import main as run_module
    elif module == "plot":
        from pdukit.cli.plot import main as run_module
    elif module == "export":
        from pdukit.analysis.tables import main as run_module
    elif module == "compare":
        from pdukit.analysis.compare import main as run_module
    elif module == "score-bench":
        from pdukit.analysis.synthetic import main as run_module
    elif module == "converge":
        from pdukit.analysis.convergence import main as run_module
    elif module == "track":
        from pdukit.analysis.track import main as run_module
    elif module == "refmodes":
        from pdukit.cog.reference_modes import main as run_module
    else:
        print(f"pdu: unknown module {module!r}\n\n{_usage()}", file=sys.stderr)
        return 2
    return run_module(rest) or 0


def cog2_main(argv=None):
    """Run the old ``pdu-cog2`` command (``pdu run``, same arguments)."""
    return main(["run", *(sys.argv[1:] if argv is None else argv)])


if __name__ == "__main__":
    sys.exit(main())
