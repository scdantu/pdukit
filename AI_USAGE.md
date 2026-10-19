# AI usage statement

This file records how generative AI was used in preparing this software, in
line with the disclosure policies of Springer Nature (*Nature Computational
Science*) and JOSS, and the recommendations of Bridgeford et al., PLOS
Comput Biol 22(7): e1014428 (2026).

## Tool

Claude Code (Anthropic) with Claude models. The release preparation
described below was done with `claude-sonnet-5-5`. The serving model can
change between sessions, so exact versions are not guaranteed to be the same
throughout. Claude Code was used interactively, with the author reviewing
each step, not as an unattended agent.

## What AI assisted with

1. Refactoring the single-file pipeline into the `Cog2Run` class, with one
   method per stage, and splitting the code into modules.
2. Formatting and linting to PEP 8, PEP 257 and PEP 440 (ruff configuration,
   docstrings, import order).
3. Designing and implementing the `pdu` command-line interface and the
   `.ppf` parameter-file reader (a GROMACS-style `.mdp` syntax).
4. Packaging (`pyproject.toml`, PyPI build, package data), Slurm scripts,
   README and documentation, and this file.
5. Repository housekeeping (branch restructuring, tags, helper scripts).

## What AI did not do

The scientific method is the author's own: the definition of protein dynamical
units, the COG criterion, the gating, cohesion and resolution rules, the
choice of test systems and the interpretation of all results. Where AI
suggested implementation details or defaults, the author decided whether to
adopt them.

## How the output was verified

1. A set of seeded planted-system regression configurations (golden tests)
   was run before and after each refactoring step. Outputs were compared after
   removing only timings, environment and output-path fields, and were
   identical.
2. The new command line and `.ppf` presets were checked to resolve to the same
   parameters as the earlier scripts and `.mdp` files.
3. `pytest` and `ruff` are run on the released code (see the README).
4. The author reviewed the code, the commit history and the documentation.

## Responsibility

The author is fully responsible for the content, correctness and licensing of
this software. AI systems are not authors.

## Provenance of the AI workflow

Prompts and sessions are not archived and the process is not exactly
reproducible, because model outputs vary between runs and model versions
change. The version-controlled code, the regression tests and this statement
are the record of the result.
