#!/usr/bin/env python3
"""Find code in pdukit that nothing reaches.  Standard library only.

1. Modules: import graph (relative imports, imports inside functions too) from the entry points
   (pdukit.cli.main, the pyproject console scripts, hpc/ and analysis/ scripts).  A module nobody
   imports is dead.
2. Definitions: name-based fixpoint.  Roots are the names used at module level of reachable
   modules, plus the entry-point mains.  A def becomes live when a live def mentions its name
   (identifier, attribute or a string equal to the name, for getattr and dispatch tables).
   Name-based means a name shared by two defs keeps both alive: the list errs towards keeping code.

Usage: python3 tools/dead_code_report.py [--json out.json]
"""
from __future__ import annotations

import ast
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "pdukit"
ENTRY_MODULES = {"pdukit.cli.main", "pdukit.__main__",
                 "pdukit.analysis.summary"}
SCRIPT_DIRS = [ROOT / "analysis", ROOT / "tools"]


def modname(p: Path) -> str:
    rel = p.relative_to(ROOT).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def py_files(base: Path):
    return [p for p in base.rglob("*.py") if "__pycache__" not in p.parts and "pypdu" not in p.parts]


def imports_of(tree, mod: str, is_pkg: bool):
    out = set()
    pkg = mod if is_pkg else mod.rpartition(".")[0]
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                out.add(a.name)
        elif isinstance(n, ast.ImportFrom):
            base = n.module or ""
            if n.level:
                parts = pkg.split(".")
                parts = parts[: len(parts) - (n.level - 1)]
                base = ".".join(parts + ([n.module] if n.module else []))
            out.add(base)
            for a in n.names:
                out.add(f"{base}.{a.name}")
    return out


def refs_in(node) -> set[str]:
    s = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name):
            s.add(n.id)
        elif isinstance(n, ast.Attribute):
            s.add(n.attr)
        elif isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value.isidentifier():
            s.add(n.value)
    return s     # an import alone is not a use


def split_module(tree):
    """-> (module-level refs excluding def bodies, [(qualname, node, lineno, end)])"""
    defs, top = [], []

    def visit(body, prefix):
        for st in body:
            if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef)):
                defs.append((prefix + st.name, st, st.lineno, st.end_lineno))
            elif isinstance(st, ast.ClassDef):
                defs.append((prefix + st.name, None, st.lineno, st.end_lineno))   # class itself
                top.extend(ast.walk(ast.Module(body=[ast.Expr(d) for d in st.decorator_list], type_ignores=[])))
                for b in st.bases:
                    top.extend(ast.walk(b))
                visit(st.body, prefix + st.name + ".")
            else:
                top.append(st)
    visit(tree.body, "")
    return top, defs


def main():
    files = {modname(p): p for p in py_files(PKG) if "hpc" not in p.parts}   # hpc/ = standalone scripts
    trees = {m: ast.parse(p.read_text(), str(p)) for m, p in files.items()}
    is_test = lambda m: ".tests." in m or m.endswith(".tests")
    scripts = {}
    for d in SCRIPT_DIRS:
        for p in py_files(d):
            if p == Path(__file__).resolve():
                continue
            m = modname(p)
            if m not in files:
                scripts[m] = (p, ast.parse(p.read_text(), str(p)))

    # ---- module reachability
    graph = {}
    for m, t in trees.items():
        graph[m] = imports_of(t, m, files[m].name == "__init__.py")
    seeds = set(ENTRY_MODULES)
    for m, (p, t) in scripts.items():
        seeds |= {i for i in imports_of(t, m, False) if i in files}
    # a package import pulls its __init__
    seen, stack = set(), [s for s in seeds if s in files]
    while stack:
        m = stack.pop()
        if m in seen:
            continue
        seen.add(m)
        parts = m.split(".")
        for k in range(1, len(parts)):
            par = ".".join(parts[:k])
            if par in files and par not in seen:
                stack.append(par)
        for i in graph[m]:
            for cand in (i, i.rpartition(".")[0]):
                if cand in files and cand not in seen and not is_test(cand):
                    stack.append(cand)
    mods = sorted(m for m in files if not is_test(m))
    dead_mods = [m for m in mods if m not in seen]
    test_only = []
    tests_import = set()
    for m in files:
        if is_test(m):
            for i in graph[m]:
                for cand in (i, i.rpartition(".")[0]):
                    if cand in files:
                        tests_import.add(cand)
    test_only = [m for m in dead_mods if m in tests_import]

    # ---- definitions in reachable modules
    live_names = set()
    defs_by_name = defaultdict(list)
    per_def_refs = {}
    for m in sorted(seen):
        top, defs = split_module(trees[m])
        for st in top:
            live_names |= refs_in(st) if isinstance(st, ast.AST) else set()
        for q, node, a, b in defs:
            if node is not None:
                per_def_refs[(m, q)] = refs_in(node)
            else:
                per_def_refs[(m, q)] = set()
            defs_by_name[q.rpartition(".")[2]].append((m, q, a, b))
    for m in ENTRY_MODULES:
        live_names.add("main")
    # scripts and hpc call into lib
    for m, (p, t) in scripts.items():
        live_names |= refs_in(t)
    # class bodies: methods of a live class that are dunder/properties stay live
    def keep_by_form(name):
        return name.startswith("__") and name.endswith("__")
    # class -> its methods' refs only count once the method itself is live
    changed = True
    live = set()
    while changed:
        changed = False
        for name, items in defs_by_name.items():
            if name in live:
                continue
            if name in live_names or keep_by_form(name):
                live.add(name)
                for (m, q, a, b) in items:
                    new = per_def_refs[(m, q)] - live_names
                    if new:
                        live_names |= per_def_refs[(m, q)]
                changed = True
        # a live class makes all its methods that are dunder live (done above)
    test_refs = set()
    for m in files:
        if is_test(m):
            test_refs |= refs_in(trees[m])
    dead_defs = []
    for name, items in defs_by_name.items():
        if name in live:
            continue
        for (m, q, a, b) in items:
            dead_defs.append({"module": m, "def": q, "lines": [a, b], "n": b - a + 1,
                              "tested": name in test_refs})
    dead_defs.sort(key=lambda d: (d["module"], d["lines"][0]))

    total = {m: len(files[m].read_text().splitlines()) for m in mods}
    print(f"modules: {len(mods)} (non-test), {len(dead_mods)} unreachable from any entry point or script")
    for m in dead_mods:
        print(f"  DEAD MODULE {m}: {total[m]} lines{'  (imported by tests)' if m in test_only else ''}")
    print(f"\ndefinitions never reached from live code: {len(dead_defs)}")
    byfile = defaultdict(list)
    for d in dead_defs:
        byfile[d["module"]].append(d)
    for m, ds in sorted(byfile.items(), key=lambda kv: -sum(x["n"] for x in kv[1])):
        outer = [d for d in ds if "." not in d["def"] or d["def"].rpartition(".")[0] not in {e["def"] for e in ds}]
        print(f"  {m}: {sum(x['n'] for x in outer)} lines in {len(outer)} defs")
        for d in outer:
            print(f"      {d['def']:55s} {d['lines'][0]:5d}-{d['lines'][1]:5d} {d['n']:4d}{'  tested' if d['tested'] else ''}")
    if "--json" in sys.argv:
        Path(sys.argv[sys.argv.index("--json") + 1]).write_text(json.dumps(
            {"dead_modules": dead_mods, "dead_defs": dead_defs}, indent=1))


if __name__ == "__main__":
    main()
