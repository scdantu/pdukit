"""Plain tables: write a list of row dictionaries as CSV."""

from __future__ import annotations

import csv
from pathlib import Path


class CsvWriter:
    """Write rows (dicts with the same keys) to one CSV file."""

    def __init__(self, path):
        """Remember the target file (parent folders are created on write)."""
        self.path = Path(path)

    def write(self, rows, note=None):
        """Write the rows; ``note`` becomes a first ``# ...`` line.

        Columns are the keys of the rows in first-seen order.  Returns the
        path, or None when there are no rows (no file is written).
        """
        rows = list(rows)
        if not rows:
            return None
        cols = []
        for r in rows:
            cols += [k for k in r if k not in cols]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", newline="") as fh:
            if note:
                fh.write(f"# {note}\n")
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        return self.path
