"""Rename classes everywhere (identifiers only, not words in prose).

    python tools/restructure/rename.py

Uses the syntax tree, so ``Calibrate`` in a docstring is left alone while the
class, its imports and its uses change.  Dotted names inside strings (for
``patch(...)``) are handled separately.
"""

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

RENAMES = {
    "Cog2Scorer": "FluxScorer",
    "BatchedEngine": "TorchEngine",
    "_Direction": "FluxDirection",
    "PpfIO": "PpfReader",
    "RunResults": "ResultsReader",
    "RunWriter": "ResultsWriter",
    "DomainTrajectory": "MdcathDomain",
    "AtomTable": "MdcathAtoms",
    "LoadFeatures": "LoadFeaturesStage",
    "SplitReplicas": "SplitReplicasStage",
    "ScanWindow": "ScanWindowStage",
    "Preprocess": "PreprocessStage",
    "BuildWorkspace": "BuildWorkspaceStage",
    "BuildGraph": "BuildGraphStage",
    "Calibrate": "CalibrateStage",
    "DiscoverUnits": "DiscoverUnitsStage",
    "ReportModes": "ReportModesStage",
}


def positions(tree, lines):
    """Yield (line, col, old) for every identifier that is a renamed class."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in RENAMES:
            yield node.lineno, node.col_offset, node.id
        elif isinstance(node, ast.Attribute) and node.attr in RENAMES:
            yield (node.end_lineno,
                   node.end_col_offset - len(node.attr), node.attr)
        elif isinstance(node, ast.ClassDef) and node.name in RENAMES:
            m = re.search(r"class\s+(\w+)", lines[node.lineno - 1])
            yield node.lineno, m.start(1), node.name
        elif isinstance(node, (ast.ImportFrom, ast.Import)):
            for a in node.names:
                if a.name in RENAMES:
                    yield a.lineno, a.col_offset, a.name
        elif isinstance(node, ast.arg) and node.arg in RENAMES:
            continue


def main():
    """Apply every rename to every python file under the repo."""
    changed = 0
    for path in ROOT.rglob("*.py"):
        if any(x in path.parts for x in (".git", "archive", "__pycache__")):
            continue
        text = path.read_text()
        if not any(k in text for k in RENAMES):
            continue
        lines = text.split("\n")
        for line, col, old in sorted(set(positions(ast.parse(text), lines)),
                                     reverse=True):
            s = lines[line - 1]
            assert s[col:col + len(old)] == old, (path, line, old)
            lines[line - 1] = s[:col] + RENAMES[old] + s[col + len(old):]
        new = "\n".join(lines)
        if new != text:
            path.write_text(new)
            changed += 1
    print(f"renamed in {changed} files")


if __name__ == "__main__":
    sys.exit(main())
