# Split-method benchmark

Exploratory comparison for the grow-on-incoming, split-on-cohesion plan.
See `REPORT.md` for results, assumptions, and the weighted-library limitation.
No production growth, scoring or MC code is modified.

## Reproduction

Use the repository Python environment with numpy, scipy, scikit-learn and
matplotlib. The nested-model workers require a separate graph-tool installation.
The original cached `results.json` files must be available in the adjacent
`isambard` directory; `manifest.json` records their absolute paths and hashes.

From the repository root:

```bash
export PDU_REPO="$PWD"
export PDU_SPLIT_OUT="$PWD/analysis/results/split_methods_2026-10-17"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/pdu_split_mpl
python tools/split_benchmark/benchmark.py prepare
python tools/split_benchmark/checks.py
python tools/split_benchmark/benchmark.py run --nulls 199
python tools/split_benchmark/benchmark.py raw
# Use the Python interpreter in which graph-tool is installed:
python tools/split_benchmark/nested_sbm.py --binary
python tools/split_benchmark/weighted_mdl_bank.py
python tools/split_benchmark/summarise.py
```

The normal `run` action resumes completed datasets. Use a fresh output directory
when changing benchmark settings or implementations.

On the test host graph-tool used Python 3.15 with
`PYTHONPATH=/opt/homebrew/opt/graph-tool/libexec/lib/python3.15/site-packages`.
The other scripts used the repository Python 3.13 environment. The unrestricted
weighted worker (without `--binary`) was attempted, but graph-tool 3.9 crashed in
native weighted nested-state updates on a 12-node diagnostic. Turning off parallel
updates produced `RuntimeError: vector`; an alternative initialisation and
multiflip updates also failed. These failures are implementation/runtime evidence,
not evidence that the weighted SBM scientific model is unsuitable.

`weighted_mdl_bank.py` avoids those native moves by constructing fixed candidate
hierarchies and evaluating the actual graph-tool nested weighted-model description
length. It includes unpruned proposal trees as well as cohesion-stopped trees.
Its search is explicitly restricted and shares proposals with the other methods.

## Checks

`checks.py` verifies agreement with production cohesion, recovery of two exact
blocks, and absence of subdivision in a homogeneous graph, for all five
spectral/modularity/stability variants. The full benchmark adds the known-answer
time-series controls. Graph-tool outputs and MDL candidate banks are checked for
complete coverage and valid residue partitions during summarisation.

## Interpretation limits

The graph-null permutations are not a temporal bootstrap. Graph edge dependence,
per-residue strengths and temporal uncertainty are not all preserved by that null.
The cached protein matrices permit cut proposal comparison, not recalculation of
new child interface spectra. No new biological acceptance criterion is imposed.
The procedures have different stopping rules; their performance differences cannot
be attributed solely to their cut-proposal algorithms.
