"""The file moves of the restructure (step 1): old path -> new path.

Paths are relative to ``pdukit/``.  ``tools/restructure/move.py`` applies
them (git mv, then every import in the repo is rewritten to the new
absolute module names); the plan document is generated from this table.
"""

MOVES = {
    # settings
    "config/cli.py": "cli/arguments.py",
    # io: everything that reads or writes a file format
    "config/ppf_reader.py": "io/ppf.py",
    "data/trajectory/xtc.py": "io/trajectory/mdanalysis.py",
    "data/trajectory/hdf5_layout.py": "io/mdcath/layout.py",
    "data/trajectory/domain.py": "io/mdcath/domain.py",
    "results/writer.py": "io/results/writer.py",
    "results/reader.py": "io/results/reader.py",
    "results/stage_files.py": "io/results/stage_files.py",
    # numerics: pure maths
    "maths/geometry.py": "numerics/geometry.py",
    "maths/kernel.py": "numerics/kernel.py",
    "flux/features.py": "numerics/features.py",
    # cog: the phases of a run
    "protocols.py": "cog/protocols.py",
    "runinfo.py": "cog/runinfo.py",
    "pipeline/run.py": "cog/pipeline.py",
    "pipeline/state.py": "cog/state.py",
    "pipeline/prepare.py": "cog/stages/prepare.py",
    "pipeline/search.py": "cog/stages/search.py",
    "pipeline/discover.py": "cog/stages/discover.py",
    "data/features.py": "cog/features.py",
    "graph/predictive.py": "cog/graph.py",
    "modes/recovery.py": "cog/modes.py",
    "calibration/bootstrap.py": "cog/calibration/bootstrap.py",
    "calibration/controls.py": "cog/calibration/controls.py",
    "calibration/null.py": "cog/calibration/null.py",
    "calibration/scans.py": "cog/calibration/scans.py",
    "calibration/scorer.py": "cog/calibration/scorer.py",
    "flux/backend.py": "cog/scoring/backend.py",
    "flux/engine_torch.py": "cog/scoring/engine_torch.py",
    "flux/factory.py": "cog/scoring/factory.py",
    "flux/pair.py": "cog/scoring/pair.py",
    "flux/scorer.py": "cog/scoring/scorer.py",
    "flux/scorer_masked.py": "cog/scoring/scorer_masked.py",
    "flux/workspace.py": "cog/scoring/workspace.py",
    "flux/workspace_schur.py": "cog/scoring/workspace_schur.py",
    "units/cohesion.py": "cog/units/cohesion.py",
    "units/grow.py": "cog/units/grow.py",
    "units/ports.py": "cog/units/ports.py",
    "units/seeds.py": "cog/units/seeds.py",
    "units/split.py": "cog/units/split.py",
    # analysis: what is made from finished results
    "results/tables.py": "analysis/tables.py",
    "results/confidence.py": "analysis/confidence.py",
    "results/compare.py": "analysis/compare.py",
    "results/summary.py": "analysis/summary.py",
    # viz: figures and PyMOL scripts
    "plots/cli.py": "cli/plot.py",
    "plots/movie.py": "viz/movie.py",
    "plots/network.py": "viz/network.py",
    "plots/pymol.py": "viz/pymol.py",
    "plots/style.py": "viz/style.py",
    "plots/units.py": "viz/units.py",
    # bench: synthetic systems
    "data/planted.py": "bench/planted.py",
    # command line
    "cli.py": "cli/main.py",
    # package files that carry content
    "flux/__init__.py": "cog/scoring/__init__.py",
    "results/__init__.py": "io/results/__init__.py",
    "calibration/__init__.py": "cog/calibration/__init__.py",
    "units/__init__.py": "cog/units/__init__.py",
    "plots/__init__.py": "viz/__init__.py",
    "maths/__init__.py": "numerics/__init__.py",
    "pipeline/__init__.py": "cog/stages/__init__.py",
}

# new packages that get an __init__.py with this docstring
NEW_INITS = {
    "cog": "The phases of COG: the pipeline, its stages, scoring, units.",
    "io": "Reading and writing files: trajectories, topology, ppf, results.",
    "io/trajectory": "Trajectory readers (xtc, dcd, trr, ... via MDAnalysis).",
    "io/mdcath": "The mdCATH hdf5 layout and one domain loaded from it.",
    "analysis": "What is made from finished results: tables, scores, comparison.",
    "bench": "Synthetic systems with a known answer, and demo data.",
    "cli": "The command line: one module per command.",
}

# old package -> new package, for the __init__.py files and docstrings
PACKAGES = {
    "pdukit.numerics": "pdukit.numerics",
    "pdukit.cog.scoring": "pdukit.cog.scoring",
    "pdukit.cog.stages": "pdukit.cog",
    "pdukit.cog.units": "pdukit.cog.units",
    "pdukit.cog.calibration": "pdukit.cog.calibration",
    "pdukit.cog.modes": "pdukit.cog.modes",
    "pdukit.cog.graph": "pdukit.cog.graph",
    "pdukit.viz": "pdukit.viz",
    "pdukit.io.results": "pdukit.analysis",
    "pdukit.cog": "pdukit.cog",
}
