"""Plain-language text of the settings reference (docs/OPTIONS.md).

The tables in OPTIONS.md come from the settings table (`schema.SETTINGS`);
the explanations around them live here, so the generated page reads as a
guide and not only as a list.  A test checks that every key named here is a
real setting and that every group has an introduction.
"""

INTRO = """\
Every setting is a `key = value` line in a parameter file (a `.ppf`).  Start
from a shipped preset (listed at the end of this page; `pdu check -p
workstation > my.ppf` writes a copy) or from `pdu check > my.ppf`, which
writes every setting at its default with a comment.  Then run

    pdu run -p my.ppf -f rep1.xtc rep2.xtc -s top.pdb -o out

`pdu run` has eight arguments: `-f` trajectories (one file per replica), `-s`
topology, `-p` parameter file, `-o` output folder, `--nt` CPU threads,
`--set` (change any setting for one run: `--set lag=20 n-null=32`),
`--log-level` and `--check` (print what will be used, and where each value
came from, then stop).  Precedence: built-in default, then the parameter
file, then the command line.  Case, `-` and `_` do not matter in keys; an
unknown key is an error with a suggestion.  Flags are written `1` or `0`.

Times are in **frames**, not ps: `lag = 10` with 20 ps frames is 200 ps.

Every run writes `pdu.log` (the screen log) and `pdu_out.ppf` (every setting
it really used) into the output folder; `pdu run -p out/pdu_out.ppf ...`
repeats it.
"""

START_HERE = [
    ("lag", "The time step at which coupling is measured, in frames.  A "
     "unit is a group of residues whose future depends on the rest of the "
     "protein as little as possible, AT THIS TIME SCALE.  Choose it from the "
     "motion you care about: 100 ps - 10 ns are typical.  Units change with "
     "the lag; a scan over several lags is the honest way to report them."),
    ("frame-ps", "Time between saved frames in ps.  Only used to print "
     "times in the log; set it so the log speaks in ps and ns."),
    ("highpass-window", "Removes motions slower than this many frames "
     "inside every replica before anything is measured.  Slow drift that a "
     "trajectory does not sample makes everything look correlated with "
     "everything and hides real units.  20 x lag is our rule; 0 = off; "
     "'auto' scans a few windows to choose one (slow: about one workspace "
     "per window)."),
    ("highpass-before-features", "1 = filter the raw distances before each "
     "residue's slow modes are chosen (recommended with a high-pass window)."),
    ("rse-mode", "How K, the number of slow modes (tICA) kept per "
     "residue, is set: fixed (you give K in rse-components) or auto.  K sets "
     "the size of the problem: memory grows with (residues x K)^2.  At short "
     "lags each residue has many similar fast modes (no gap in its "
     "spectrum), so the automatic K grows quickly as the lag shortens."),
    ("rse-coverage", "Automatic K, condition 1: keep the fewest modes that "
     "hold this share of a residue's slow motion (sum of lambda^2), e.g. "
     "0.9.  K is the median over residues."),
    ("rse-min-share", "Automatic K, condition 2: keep only modes that each "
     "add at least this share (e.g. 0.03 = 3%).  Each residue stops at "
     "whichever condition comes first, so K gets as close to rse-coverage "
     "as modes worth keeping allow.  0 = off."),
    ("rse-max-components", "Upper limit on the automatic K: a safety cap "
     "for memory; 8-12 fits a workstation."),
    ("n-null", "Time shifts used to build the null (the 'what would chance "
     "give' reference).  64 is precise; 32 halves the memory and time of "
     "the null with slightly noisier thresholds."),
    ("dec-seeds", "Number of growth starts.  Keep seeds at or below a "
     "quarter of the residues (see seed-fraction) so units are found by "
     "growth, not by cutting overlapping candidates apart."),
    ("seed-fraction", "Caps the seeds at this fraction of the residues "
     "(0.25 recommended)."),
    ("seed-group", "Residues in each growth start: 2 = the seed plus its "
     "strongest partner.  A single residue has too few slow modes to "
     "predict anything, so the first addition often looks worse; starting "
     "from pairs avoids that barrier."),
    ("min-size", "Smallest unit kept.  Use at least twice seed-group "
     "(4 with pairs), or pairs that never grew are reported as units."),
    ("device", "Where the heavy algebra runs: auto (a CUDA GPU when torch "
     "can use it), cpu or cuda.  pdu works in double precision: data-centre "
     "GPUs (A100, H100, GH200) are fast at it, consumer cards (GeForce, RTX "
     "A-series) are not, so use cpu there."),
    ("nt", "CPU threads (`--nt N` on the command line).  -1 = all the cores "
     "the job may use; on a shared machine give the number you were "
     "allocated."),
    ("log-level", "quiet, info (one line per step) or debug (every growth "
     "step).  Lines starting with ! are warnings."),
]

RUN_SIZE = """\
**Memory.**  The estimator stacks every residue's K modes into one vector of
length D = residues x K and keeps D x D matrices in double precision: three
for the data and two for every null shift.  So

    bytes ~ (6 + 2 x n-null) x (residues x K)^2 x 8  +  12 x frames x D

| residues | K | n-null | matrices (without the frames term) |
|---|---|---|---|
| 160 | 8 | 64 | ~2 GB + frames term |
| 400 | 8 | 32 | ~6 GB |
| 400 | 12 | 32 | ~13 GB |
| 400 | 20 | 64 | ~69 GB |

The frames term matters for long trajectories: 2.6 million frames at
D = 4400 add about 140 GB.  Before each large step the log prints the
estimate and the free memory (`memory for the workspace: ~13.0 GB ...`);
if it does not fit, the run stops at once and names the settings that do
(`rse-max-components`, `n-null`).  `no-memory-check = 1` skips this.

**Time.**  Building the null costs about frames x D^2 x n-null operations,
so the same three numbers set the run time.  Fewer frames (`stride = 2`
keeps every second frame; then halve `lag` and `highpass-window`, which
are in frames) is the other lever.

**On a workstation** (32-64 GB, consumer GPU or none): start from the
`workstation` preset.  It runs on the CPU, keeps at most 12 modes per
residue and 32 shifts, and fits a 400-residue protein with 60,000 frames in
about 16 GB.  Write your own copy, set the time settings (`frame-ps`, `lag`,
`highpass-window`) in it, and run:

    pdu check -p workstation > my.ppf
    pdu run -p my.ppf -f rep1.xtc rep2.xtc -s top.pdb -o out --nt 8

(A shipped preset can be named without its path: `-p workstation`.)

**On a GPU**: install torch for your CUDA first (CUDA 12.1:
`pip install "torch<2.6" --index-url https://download.pytorch.org/whl/cu121`).
The log line `WSP ... backend torch on cuda` confirms the GPU is used; a
warning names cards that are slow in double precision.  The GPU must also
hold the matrices above, so on an 8 GB card with 32 shifts D = residues x K
must stay below about 3,800 (e.g. 300 residues at K = 12).
"""

GROUP_INTRO = {
    "files and general options": "Run bookkeeping: random seed, log "
    "detail, threads, and where the files are.",
    "planted benchmark": "`pdu bench` runs the pipeline on a synthetic "
    "system whose units are known, to check an install or a setting.  "
    "These settings describe that system; they are ignored for MD data.",
    "MD data": "How trajectories become per-residue features: which frames "
    "are read, and how each residue's environment (its distances to every "
    "residue within rse-cutoff) is compressed into K slow modes.",
    "estimator": "How coupling is measured: the lag, how many slow "
    "coordinates describe a unit and its environment, the null that sets "
    "the noise threshold, and where the algebra runs.",
    "calibration": "Before the search, random groups of each size are "
    "scored to learn what a group of that size looks like by chance.  "
    "Units must beat that baseline.  These settings trade calibration "
    "precision for time.",
    "discovery": "The search itself: high-pass filtering and its window "
    "scan, seeds, growth (add the residue that best lowers incoming "
    "dependence), merging of repeated candidates, the recursive split test, "
    "ports, and performance switches.  Most runs change only the seed "
    "settings, min-size and the high-pass window.",
    "verification": "After discovery the units are scored again at other "
    "lags, to show which hold across time scales.",
    "global-mode recovery": "At the end the units' own slow motions are "
    "combined to see how much of the protein's global slow motion they "
    "explain.",
}
