#!/usr/bin/env python3
"""Compare two PDU run folders file by file.

    python compare_runs.py OLD_RUN NEW_RUN [--rtol 0] [--atol 0] [-q]

Exit status 0 when the runs match, 1 when they differ, 2 on a usage error.

What is compared
    results.json and the other .json files   every value, by its path in the file
    .csv files                               every cell
    .npz files                               every array
    pdu_out.ppf                              every setting (values only)
    everything else (.md, ...)               byte for byte

What is skipped, and only this
    * timings (any key ending in "seconds", anything under /timings/);
    * where the run was made: outdir, feature_cache, /environment/*;
    * "schema_version", which newer results.json files carry;
    * the 37 settings that were retired (they only exist in runs made by
      the older code, and nothing read them).

With --rtol 0 --atol 0 (the default) two numbers must be exactly equal
(NaN equals NaN).  With a tolerance a pair of numbers passes when
|a - b| <= atol + rtol * max(|a|, |b|).  In both modes the report ends with
the largest difference seen, so run it on two runs of the SAME code first
to learn how much a GPU wanders by itself.
"""
import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

RETIRED_SETTINGS = {
    'block-check', 'boundary-tol', 'channel-shadow', 'cog1-null',
    'cohesion-baseline-n', 'cohesion-margin', 'compare-cog1', 'dec-iters',
    'face-balance', 'face-delta', 'incoming-mc-steps', 'merge-jaccard',
    'merge-mode', 'no-cohesion-split', 'no-keep-best', 'no-ledger-loo',
    'pdudef', 'pdudef-frames', 'pdudef-random', 'pdudef-shifts',
    'per-replica-check', 'resolve-margin', 'resolve-seeds', 'rho-scan',
    'rho-scan-null', 'rho-scan-us', 'size-penalty', 'stab-cycles',
    'stab-kmax', 'stab-margin', 'stab-null', 'stab-objective',
    'stab-patience', 'stab-protect', 'stab-protect-alpha', 'stab-steps',
    'units-from',
}
# the same settings under the names results.json uses (underscores; the
# "no-ledger-loo" switch is stored as ledger_loo)
RETIRED_ARGS = {k.replace('-', '_') for k in RETIRED_SETTINGS} - {'no_ledger_loo'}
RETIRED_ARGS.add('ledger_loo')

SKIP = re.compile(
    r'seconds$|^/timings/|/outdir$|/feature_cache$|^/environment/'
    r'|^/schema_version$|^/args/(%s)$' % '|'.join(sorted(RETIRED_ARGS))
)


class Tally:
    """Counts the numbers compared and remembers the largest difference."""

    def __init__(self, rtol, atol):
        self.rtol, self.atol = rtol, atol
        self.n = 0
        self.max_abs = 0.0
        self.max_rel = 0.0
        self.where = ''

    def same(self, a, b, where):
        if isinstance(a, bool) or isinstance(b, bool):
            return a == b
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            if math.isnan(a) and math.isnan(b):
                return True
            if math.isnan(a) or math.isnan(b):
                return False
            if a == b:
                self.n += 1
                return True
            diff = abs(a - b)
            scale = max(abs(a), abs(b))
            rel = diff / scale if scale else 0.0
            self.n += 1
            if diff > self.max_abs:
                self.max_abs, self.max_rel, self.where = diff, rel, where
            return diff <= self.atol + self.rtol * scale
        return a == b


def flat(obj, path=''):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from flat(v, f'{path}/{k}')
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from flat(v, f'{path}[{i}]')
    else:
        yield path, obj


def compare_pairs(pairs_a, pairs_b, tally, skip=None):
    """pairs: dict path -> value.  Returns a list of problems."""
    bad = []
    for key in sorted(set(pairs_a) | set(pairs_b)):
        if skip and skip.search(key):
            continue
        if key not in pairs_a:
            bad.append(f'{key}: only in the new run')
        elif key not in pairs_b:
            bad.append(f'{key}: only in the old run')
        elif not tally.same(pairs_a[key], pairs_b[key], key):
            bad.append(f'{key}: {pairs_a[key]!r} vs {pairs_b[key]!r}')
    return bad


def compare_json(a, b, tally):
    ja = dict(flat(json.loads(a.read_text())))
    jb = dict(flat(json.loads(b.read_text())))
    return compare_pairs(ja, jb, tally, SKIP)


def parse_ppf(path):
    out = {}
    for line in path.read_text().splitlines():
        line = line.split(';')[0].strip()
        if '=' in line:
            k, v = line.split('=', 1)
            key = k.strip().replace('_', '-').lower()
            if key not in RETIRED_SETTINGS and key not in ('outdir', 'feature-cache'):
                out[key] = v.strip()
    return out


def compare_ppf(a, b, tally):
    pa, pb = parse_ppf(a), parse_ppf(b)
    bad = []
    for key in sorted(set(pa) | set(pb)):
        if pa.get(key) != pb.get(key):
            bad.append(f'{key}: {pa.get(key)!r} vs {pb.get(key)!r}')
    return bad


def cell(text):
    try:
        return float(text)
    except ValueError:
        return text


def compare_csv(a, b, tally):
    with a.open(newline='') as fa, b.open(newline='') as fb:
        ra, rb = list(csv.reader(fa)), list(csv.reader(fb))
    bad = []
    if len(ra) != len(rb):
        bad.append(f'{len(ra)} rows vs {len(rb)} rows')
    for i, (x, y) in enumerate(zip(ra, rb)):
        if len(x) != len(y):
            bad.append(f'row {i}: {len(x)} cells vs {len(y)}')
            continue
        for j, (u, v) in enumerate(zip(x, y)):
            if u == v:
                continue
            if not tally.same(cell(u), cell(v), f'row {i} col {j}'):
                bad.append(f'row {i} col {j}: {u!r} vs {v!r}')
    return bad


def compare_npz(a, b, tally):
    import numpy as np
    bad = []
    with np.load(a) as za, np.load(b) as zb:
        if set(za.files) != set(zb.files):
            return [f'arrays differ: {sorted(za.files)} vs {sorted(zb.files)}']
        for name in sorted(za.files):
            x, y = za[name], zb[name]
            if x.shape != y.shape:
                bad.append(f'{name}: shape {x.shape} vs {y.shape}')
                continue
            if x.size == 0:
                continue
            fx = x.astype(float).ravel()
            fy = y.astype(float).ravel()
            both_nan = np.isnan(fx) & np.isnan(fy)
            fx, fy = np.where(both_nan, 0.0, fx), np.where(both_nan, 0.0, fy)
            diff = np.abs(fx - fy)
            scale = np.maximum(np.abs(fx), np.abs(fy))
            tally.n += fx.size
            k = int(diff.argmax())
            if diff[k] > tally.max_abs:
                tally.max_abs = float(diff[k])
                tally.max_rel = float(diff[k] / scale[k]) if scale[k] else 0.0
                tally.where = f'{a.name}:{name}'
            if not (diff <= tally.atol + tally.rtol * scale).all():
                bad.append(f'{name}: max difference {diff[k]:.3g}')
    return bad


def files(root):
    return {p.relative_to(root) for p in root.rglob('*') if p.is_file()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('old')
    ap.add_argument('new')
    ap.add_argument('--rtol', type=float, default=0.0)
    ap.add_argument('--atol', type=float, default=0.0)
    ap.add_argument('-q', '--quiet', action='store_true',
                    help='print only the verdict')
    args = ap.parse_args(argv)
    old, new = Path(args.old), Path(args.new)
    for p in (old, new):
        if not (p / 'results.json').is_file():
            print(f'no results.json in {p}', file=sys.stderr)
            return 2
    tally = Tally(args.rtol, args.atol)
    fa, fb = files(old), files(new)
    failures = []
    for p in sorted(fa ^ fb):
        failures.append((p, ['only in the ' + ('old' if p in fa else 'new') + ' run']))
    for p in sorted(fa & fb):
        x, y = old / p, new / p
        if p.suffix == '.json':
            bad = compare_json(x, y, tally)
        elif p.suffix == '.csv':
            bad = compare_csv(x, y, tally)
        elif p.suffix == '.npz':
            bad = compare_npz(x, y, tally)
        elif p.suffix == '.ppf':
            bad = compare_ppf(x, y, tally)
        else:
            bad = [] if x.read_bytes() == y.read_bytes() else ['bytes differ']
        if bad:
            failures.append((p, bad))
    mode = ('exact' if args.rtol == 0 and args.atol == 0
            else f'rtol={args.rtol:g} atol={args.atol:g}')
    if not args.quiet:
        for p, bad in failures:
            print(f'  DIFFERENT {p}: {len(bad)} difference(s)')
            for line in bad[:4]:
                print(f'      {line[:160]}')
    print(f'compared {len(fa & fb)} files and {tally.n} numbers ({mode})')
    if tally.max_abs:
        print(f'largest numeric difference: {tally.max_abs:.3g} absolute, '
              f'{tally.max_rel:.3g} relative, at {tally.where}')
    else:
        print('largest numeric difference: none (every number equal)')
    print('RESULT: ' + ('IDENTICAL' if not failures else f'{len(failures)} FILE(S) DIFFER'))
    return 0 if not failures else 1


if __name__ == '__main__':
    sys.exit(main())
