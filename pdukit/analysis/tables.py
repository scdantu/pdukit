"""Turn a run's ``results.json`` into CSV and plain-text (ASCII) tables.

``pdu export RUN_DIR`` writes one file per table into ``RUN_DIR/export/``:
``<name>.csv`` for spreadsheets and R/pandas, ``<name>.txt`` for reading in
a terminal or pasting into an email, plus ``report.txt`` that stacks the
small tables into one summary.  ``--list`` shows the tables, ``--tables``
picks some, and ``--key a.b.c`` flattens any part of ``results.json`` that
is not a named table.

A table is ``(columns, rows)``: rows are lists of plain values; lists of
numbers are written space-separated and missing values as empty cells.
"""

import argparse
import csv
import io
import json
import sys
from pathlib import Path

from pdukit.io.results.reader import ResultsReader
from pdukit.numerics import confidence

REPORT_TABLES = (
    "overview",
    "units",
    "calibration",
    "timings",
    "residue_classes",
)


# ----------------------------------------------------------------------
# cell formatting and generic flattening
# ----------------------------------------------------------------------


def _cell(value):
    """Return a plain CSV/text cell for ``value``."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (list, tuple)):
        if all(not isinstance(v, (list, dict)) for v in value):
            return " ".join(_cell(v) for v in value)
        return json.dumps(value)
    if isinstance(value, dict):
        return json.dumps(value)
    return str(value)


def _records(rows, first=None):
    """Turn a list of dicts into ``(columns, rows)`` keeping key order."""
    cols = list(first or [])
    for row in rows:
        for key in row:
            if key not in cols:
                cols.append(key)
    return cols, [[row.get(c) for c in cols] for row in rows]


def _pairs(mapping, prefix=""):
    """Flatten a nested dict into ``(dotted key, value)`` pairs."""
    out = []
    for key, value in mapping.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict) and value:
            out.extend(_pairs(value, name + "."))
        else:
            out.append((name, value))
    return out


def _per_size(results, section, fields):
    """Rows ``size, field...`` from dicts keyed by unit size."""
    sec = results.get(section) or {}
    sizes = sorted({int(k) for f in fields for k in sec.get(f, {})})
    rows = [[k] + [sec.get(f, {}).get(str(k)) for f in fields] for k in sizes]
    return ["size"] + list(fields), rows


def flatten(value):
    """Flatten any JSON value to ``(columns, rows)``.

    A list of dicts becomes one row per item; a dict becomes ``key,value``
    rows with dotted keys; a list of numbers becomes ``index,value``.
    """
    if (
        isinstance(value, list)
        and value
        and all(isinstance(v, dict) for v in value)
    ):
        flat = [dict(_pairs(v)) for v in value]
        return _records(flat)
    if isinstance(value, dict):
        return ["key", "value"], [[k, v] for k, v in _pairs(value)]
    if isinstance(value, list):
        return ["index", "value"], [[i, v] for i, v in enumerate(value)]
    return ["value"], [[value]]


# ----------------------------------------------------------------------
# named tables
# ----------------------------------------------------------------------


def _overview(r):
    data = r.get("data") or {}
    final = r.get("units_final") or []
    items = [
        ("version", r.get("version")),
        ("mode", r.get("mode")),
        ("frames", data.get("n_frames")),
        ("residues", data.get("n_residues")),
        ("features", data.get("n_feat")),
        ("lag_frames", (r.get("args") or {}).get("lag")),
        ("discovery_replicas", (r.get("split") or {}).get("discovery")),
        ("holdout_replicas", (r.get("split") or {}).get("holdout")),
        ("n_final_units", len(final)),
        ("final_unit_sizes", [u.get("size") for u in final]),
        ("residue_classes", (r.get("incoming_ports") or {}).get("counts")),
        ("git_commit", (r.get("environment") or {}).get("git_commit")),
    ]
    return ["item", "value"], [[k, v] for k, v in items if v is not None]


def _units(r):
    """Join final units with their cohesion and port information."""
    final = r.get("units_final") or r.get("units_split") or []
    ports = {
        u["unit"]: u for u in (r.get("incoming_ports") or {}).get("units", [])
    }
    split = {u["unit"]: u for u in r.get("units_split") or []}
    rows = []
    for u in final:
        p, s = ports.get(u["unit"], {}), split.get(u["unit"], {})
        rows.append(
            {
                "unit": u["unit"],
                "size": u.get("size", len(u["residues"])),
                "residues": u["residues"],
                "ports": p.get("ports"),
                "extended": p.get("extended"),
                "cohesion": s.get("cohesion", p.get("cohesion_core")),
                "cohesion_extended": p.get("cohesion_extended"),
                "nu_in": s.get("nu_in", p.get("nu_in_core")),
                "nu_out": s.get("nu_out", p.get("nu_out_core")),
                "auc_listener": s.get("auc_listener"),
                "auc_driver": s.get("auc_driver"),
            }
        )
    score = {r["unit"]: r for r in confidence.unit_scores(r)}
    for row in rows:
        sc = score.get(row["unit"], {})
        row["ucs"] = sc.get("ucs")
        row["confidence"] = sc.get("band")
        row["n_seeds"] = sc.get("n_seeds")
    return _records(rows, ["unit", "size", "residues", "ucs", "n_seeds"])


def _rcs(r):
    """Per-residue confidence score for every residue in a unit."""
    rows = [
        {
            "residue": i,
            "unit": d["unit"],
            "role": d["role"],
            "rcs": d["rcs"],
            "confidence": confidence.band(d["rcs"]),
            "n_seeds": d["n_seeds"],
        }
        for i, d in sorted(confidence.residue_scores(r).items())
    ]
    return _records(rows)


def _candidates(r):
    rows = [
        {k: v for k, v in u.items() if k not in ("rho_in", "rho_out")}
        for u in r.get("units_discovery") or []
    ]
    return _records(rows, ["unit", "size", "residues"])


def _families(r):
    fams = (r.get("incoming_merge") or {}).get("families") or []
    keep = (
        "family size decision members member_sizes member_nu_in "
        "best_member union_size core majority residues"
    ).split()
    return _records([{k: f.get(k) for k in keep} for f in fams], keep)


def _nodes(r):
    rows = []
    for n in (r.get("incoming_split") or {}).get("nodes") or []:
        whole = n.get("whole_protein") or {}
        rows.append(
            {
                "node": n.get("node"),
                "parent": n.get("parent"),
                "root": n.get("root"),
                "depth": n.get("depth"),
                "size": n.get("size"),
                "split": n.get("split"),
                "cohesion": whole.get("cohesion"),
                "graph_null_p": n.get("graph_null_p"),
                "bootstrap_cut_lower_bound": n.get(
                    "bootstrap_cut_lower_bound"
                ),
                "cut_min_directional_auc": n.get("cut_min_directional_auc"),
                "stop_reason": n.get("stop_reason"),
                "residues": n.get("residues"),
            }
        )
    return _records(rows)


def _ports(r):
    rows = [
        {k: v for k, v in u.items() if k not in ("core", "extended")}
        for u in (r.get("incoming_ports") or {}).get("units") or []
    ]
    return _records(rows)


def _classes(r):
    rows = (r.get("incoming_ports") or {}).get("residues") or []
    return _records(rows, ["residue", "cls", "unit"])


def _edges(r):
    edges = (r.get("predictive_graph") or {}).get("edges") or []
    return ["residue_i", "residue_j"], [list(e) for e in edges]


def _rho(r):
    rho = (r.get("predictive_graph") or {}).get("rho") or []
    cols = ["residue"] + [str(j) for j in range(len(rho))]
    return cols, [[i] + list(row) for i, row in enumerate(rho)]


def _modes(r):
    return _records((r.get("global_mode_recovery") or {}).get("modes") or [])


def _combinations(r):
    return _records(
        (r.get("global_mode_recovery") or {}).get("combinations") or []
    )


def _timings(r):
    return ["step", "seconds"], [
        [k.replace("_seconds", ""), v]
        for k, v in (r.get("timings") or {}).items()
    ]


def _settings(r):
    return ["option", "value"], [
        [k, v] for k, v in (r.get("args") or {}).items()
    ]


def _environment(r):
    return ["item", "value"], [
        [k, v] for k, v in (r.get("environment") or {}).items()
    ]


def _calibration(r):
    cal = r.get("calibration") or {}
    cols, rows = _per_size(
        r,
        "calibration",
        [
            "theta_in",
            "theta_out",
            "baseline_median",
            "n_supports",
            "stopped_by",
        ],
    )
    iqr = cal.get("baseline_iqr") or {}
    cols = cols + ["baseline_q25", "baseline_q75"]
    rows = [
        row + [(iqr.get(str(row[0])) or [None, None])[i] for i in (0, 1)]
        for row in rows
    ]
    return cols, rows


TABLES = {
    "overview": (_overview, "run summary: sizes, final unit count, versions"),
    "units": (_units, "final units with cohesion, nu_in/out and ports"),
    "residue_scores": (_rcs, "RCS: per-residue confidence score (0-100)"),
    "candidates": (_candidates, "grown candidates before merge and split"),
    "merge_families": (_families, "candidates merged into families"),
    "split_nodes": (_nodes, "recursive split tree, one row per node"),
    "ports_units": (_ports, "core vs core+ports statistics per unit"),
    "residue_classes": (_classes, "core / port / bridge / loose per residue"),
    "calibration": (_calibration, "null thresholds and baseline by size"),
    "timings": (_timings, "seconds spent in each stage"),
    "settings": (_settings, "every setting the run used"),
    "environment": (_environment, "host, library versions, git commit"),
    "predictive_edges": (_edges, "edges of the predictive graph"),
    "predictive_rho": (_rho, "residue-by-residue coupling matrix"),
    "global_modes": (_modes, "reference global modes and timescales"),
    "global_combinations": (_combinations, "unit combinations tested"),
}


# ----------------------------------------------------------------------
# writers
# ----------------------------------------------------------------------


def to_csv(columns, rows):
    """Return the table as CSV text."""
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(columns)
    writer.writerows([[_cell(c) for c in row] for row in rows])
    return buf.getvalue()


def to_text(columns, rows, title=None, max_width=60):
    """Return the table as aligned ASCII text (long cells are clipped)."""
    body = [[_cell(c) for c in row] for row in rows]
    body = [
        [c if len(c) <= max_width else c[: max_width - 3] + "..." for c in row]
        for row in body
    ]
    widths = [
        max([len(str(h))] + [len(row[i]) for row in body])
        for i, h in enumerate(columns)
    ]
    line = "  ".join("-" * w for w in widths)
    out = []
    if title:
        out += [title, "=" * len(title)]
    out.append("  ".join(str(h).ljust(w) for h, w in zip(columns, widths)))
    out.append(line)
    out += ["  ".join(c.ljust(w) for c, w in zip(row, widths)) for row in body]
    if not body:
        out.append("(no rows)")
    return "\n".join(out) + "\n"


def build(results, names=None, key=None):
    """Return ``{name: (columns, rows)}`` for the requested tables."""
    if key:
        node = results
        for part in key.split("."):
            if isinstance(node, list):
                node = node[int(part)]
            elif isinstance(node, dict) and part in node:
                node = node[part]
            else:
                raise SystemExit(f"--key {key!r}: no {part!r} in results.json")
        return {key.replace(".", "_"): flatten(node)}
    names = list(names or TABLES)
    unknown = [n for n in names if n not in TABLES]
    if unknown:
        raise SystemExit(
            f"unknown table(s) {unknown}; see 'pdu export --list'"
        )
    out = {}
    for name in names:
        columns, rows = TABLES[name][0](results)
        if rows:
            out[name] = (columns, rows)
    return out


def write_all(tables, outdir, fmt="both", report=True):
    """Write the tables (and ``report.txt``); return the file names."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, (cols, rows) in tables.items():
        if fmt in ("csv", "both"):
            (outdir / f"{name}.csv").write_text(to_csv(cols, rows))
            written.append(f"{name}.csv")
        if fmt in ("txt", "both"):
            (outdir / f"{name}.txt").write_text(to_text(cols, rows, name))
            written.append(f"{name}.txt")
    parts = [n for n in REPORT_TABLES if n in tables]
    if report and parts and fmt in ("txt", "both"):
        # the report stays narrow: residue classes keep three columns
        short = {"residue_classes": 3}
        text = "\n".join(
            to_text(
                tables[n][0][: short.get(n)],
                [row[: short.get(n)] for row in tables[n][1]],
                title=n.replace("_", " "),
            )
            for n in parts
        )
        (outdir / "report.txt").write_text(text)
        written.append("report.txt")
    return written


def main(argv=None):
    """Command line: ``pdu export RUN_DIR [options]``."""
    ap = argparse.ArgumentParser(
        prog="pdu export",
        description="write results.json as CSV and ASCII text tables",
    )
    ap.add_argument("run", nargs="?", help="run directory (or results.json)")
    ap.add_argument(
        "-f",
        "--format",
        choices=["csv", "txt", "both"],
        default="both",
        help="file types to write",
    )
    ap.add_argument(
        "-o", "--outdir", help="output directory (default RUN/export)"
    )
    ap.add_argument(
        "--tables", help="comma-separated table names (default: all)"
    )
    ap.add_argument(
        "--key",
        help="flatten one part of results.json, e.g. incoming_split.nodes",
    )
    ap.add_argument(
        "--print",
        dest="show",
        metavar="TABLE",
        help="print one table to the terminal",
    )
    ap.add_argument(
        "--list", action="store_true", help="list the available tables"
    )
    args = ap.parse_args(argv)
    if args.list:
        for name, (_, text) in TABLES.items():
            print(f"{name:<20} {text}")
        return 0
    if not args.run:
        ap.error("RUN_DIR is required")
    results = ResultsReader.load(args.run)
    rundir = results.rundir
    if args.show:
        tables = build(results, [args.show])
        if args.show not in tables:
            print(f"{args.show}: nothing to show in this run", file=sys.stderr)
            return 1
        sys.stdout.write(to_text(*tables[args.show], title=args.show))
        return 0
    names = args.tables.split(",") if args.tables else None
    tables = build(results, names, args.key)
    outdir = Path(args.outdir) if args.outdir else rundir / "export"
    written = write_all(tables, outdir, args.format, report=not args.key)
    skipped = [
        n for n in (names or TABLES) if n not in tables and not args.key
    ]
    print(f"wrote {len(written)} files to {outdir}")
    if skipped:
        print("empty in this run (skipped): " + ", ".join(skipped))
    return 0


if __name__ == "__main__":
    sys.exit(main())
