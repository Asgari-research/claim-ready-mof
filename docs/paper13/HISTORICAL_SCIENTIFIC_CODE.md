# Historical scientific-code release note

The public `src/` layer preserves the two Python files supplied as the Paper 13 scientific-analysis source.

## Preservation choice

The release candidate makes **module-docstring-only** edits to remove stale machine-specific path examples and accurately distinguish the historical analysis code from the final standalone figure renderer. Function bodies, imports, constants, model/split logic, identifier handling, chemistry rules, output logic, and other executable statements are unchanged.

A normalized AST comparison, after removing docstrings from both original and candidate trees, is identical for both files. This is a static behavior-preservation check for the docstring-only edit; it is not a runtime certificate.

## Why the code is not structurally refactored

The large analysis pipeline contains repeated top-level definitions and historical aliases. Cosmetic deduplication/reordering could change active behavior. The public release therefore preserves this structure instead of making the source appear cleaner at the cost of scientific traceability.

## Historical vs final figure generation

The historical scripts contain executable logic for an earlier publication architecture, including an older main Figure 6. This is retained as provenance. The **final manuscript artwork** is governed by `reproduce/figures/paper13/`, which produces Main Figures 1–5 and SI Figures S0–S9. The former main Figure 6 is represented in the final artwork as SI Figure S9.

## Environment boundary

The historical dependency list was unpinned and is not an exact environment lockfile. Raw third-party data and all historical intermediate products are not distributed here. Accordingly, the presence of `src/` documents the supplied scientific workflow but does not, by itself, establish complete end-to-end rerun reproducibility.
