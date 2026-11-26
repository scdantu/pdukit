"""Apply ``moves.MOVES``: git mv the files, rewrite every import and reference.

    python tools/restructure/move.py [--dry-run]

Imports become absolute (``from pdukit.cog.units.grow import ...``).  Dotted
names and ``pdukit/...`` paths in strings, docs and config are rewritten with
the same table.
"""

import ast
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from moves import MOVES, NEW_INITS, PACKAGES  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "pdukit"


def dotted(rel):
    """Return the module name of a path relative to pdukit/."""
    parts = rel[:-3].split("/")
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(["pdukit", *parts])


MODMAP = {dotted(o): dotted(n) for o, n in MOVES.items()}
OLDMODS = {dotted(str(p.relative_to(PKG))) for p in PKG.rglob("*.py")}
# package prefixes for modules that have no file entry of their own
PKGMAP = dict(PACKAGES)
PKGMAP.update({"pdukit.io.trajectory": "pdukit.io.trajectory"})


def map_module(name):
    """Return the new name of module ``name`` (unchanged if it stays)."""
    if name in MODMAP:
        return MODMAP[name]
    best = None
    for old in PKGMAP:
        if name == old or name.startswith(old + "."):
            if best is None or len(old) > len(best):
                best = old
    if best:
        return PKGMAP[best] + name[len(best):]
    return name


def resolve(module, level, file_mod, is_init):
    """Absolute name of a (possibly relative) import in file ``file_mod``."""
    if not level:
        return module or ""
    base = file_mod.split(".")
    if not is_init:
        base = base[:-1]
    base = base[: len(base) - (level - 1)]
    return ".".join(base + ([module] if module else []))


def render(module, names):
    """Return an import statement for ``module`` and ``names``."""
    items = [n if not a else f"{n} as {a}" for n, a in names]
    one = f"from {module} import {', '.join(items)}"
    if len(one) <= 79:
        return one
    body = "".join(f"    {i},\n" for i in items)
    return f"from {module} import (\n{body})"


def rewrite_py(path, text, file_mod, is_init):
    """Rewrite the imports of one source file; return the new text."""
    tree = ast.parse(text)
    lines = text.split("\n")
    edits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            absmod = resolve(node.module, node.level, file_mod, is_init)
            if not absmod.startswith("pdukit"):
                continue
            groups = {}
            for a in node.names:
                full = f"{absmod}.{a.name}"
                if full in MODMAP or (
                    a.name != "*" and full in OLDMODS and full in PKGMAP
                ):
                    new = map_module(full)
                    parent, _, leaf = new.rpartition(".")
                    groups.setdefault(parent, []).append((leaf, a.asname))
                else:
                    groups.setdefault(map_module(absmod), []).append(
                        (a.name, a.asname)
                    )
            indent = " " * node.col_offset
            new_text = "\n".join(
                indent + render(m, ns).replace("\n", "\n" + indent)
                for m, ns in groups.items()
            )
            edits.append((node.lineno, node.end_lineno, new_text))
        elif isinstance(node, ast.Import):
            changed = False
            names = []
            for a in node.names:
                n = map_module(a.name) if a.name.startswith("pdukit") else a.name
                changed |= n != a.name
                names.append(f"{n} as {a.asname}" if a.asname else n)
            if changed:
                edits.append((
                    node.lineno, node.end_lineno,
                    " " * node.col_offset + "import " + ", ".join(names),
                ))
    for start, end, new in sorted(edits, reverse=True):
        lines[start - 1:end] = new.split("\n")
    return "\n".join(lines)


DOT = re.compile(r"\bpdukit(?:\.[A-Za-z_][A-Za-z0-9_]*)+")
SLASH = re.compile(r"\bpdukit(?:/[A-Za-z_][A-Za-z0-9_.]*)+")


def rewrite_strings(text):
    """Rewrite dotted names and pdukit/ paths that are not import lines."""
    def dot(m):
        return map_module(m.group(0))

    def slash(m):
        s = m.group(0)
        rel = s[len("pdukit/"):]
        if rel in MOVES:
            return "pdukit/" + MOVES[rel]
        as_mod = map_module("pdukit." + rel.replace("/", "."))
        if as_mod != "pdukit." + rel.replace("/", "."):
            return as_mod.replace(".", "/")
        return s

    return SLASH.sub(slash, DOT.sub(dot, text))


def main():
    """Move the files and rewrite the references."""
    dry = "--dry-run" in sys.argv
    files = [p for p in ROOT.rglob("*") if p.is_file()
             and ".git" not in p.parts and "archive" not in p.parts
             and "__pycache__" not in p.parts and ".egg" not in str(p)
             and p.suffix in (".py", ".md", ".toml", ".yml", ".yaml", ".slurm",
                              ".ppf", ".txt", ".cfg")]
    new_text = {}
    for p in files:
        rel = p.relative_to(ROOT)
        text = p.read_text()
        if p.suffix == ".py" and rel.parts[0] in ("pdukit", "tools", "analysis"):
            if rel.parts[0] == "pdukit":
                inner = str(p.relative_to(PKG))
            else:
                inner = None
            file_mod = dotted(inner) if inner else ".".join(rel.with_suffix("").parts)
            text = rewrite_py(p, text, file_mod, p.name == "__init__.py")
        new_text[p] = rewrite_strings(text)
    if dry:
        n = sum(1 for p in files if new_text[p] != p.read_text())
        print(f"dry run: {n} files would change, {len(MOVES)} would move")
        return
    for p, text in new_text.items():
        if text != p.read_text():
            p.write_text(text)
    for old, new in MOVES.items():
        dest = PKG / new
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "mv", str(PKG / old), str(dest)], check=True,
                       cwd=ROOT)
    for pkg, doc in NEW_INITS.items():
        init = PKG / pkg / "__init__.py"
        init.parent.mkdir(parents=True, exist_ok=True)
        if not init.exists():
            init.write_text(f'"""{doc}"""\n')
    for left in ("graph", "modes", "data", "data/trajectory"):
        init = PKG / left / "__init__.py"
        if init.exists():
            subprocess.run(["git", "rm", "-q", "-f", str(init)], check=True,
                           cwd=ROOT)
    print(f"moved {len(MOVES)} files; rewrote references")


if __name__ == "__main__":
    main()
