# Historical scientific-analysis source

The `src/` directory contains the two Python files supplied as the scientific-analysis source associated with the completed historical run.

## Preservation

The public copies differ from the supplied run-source files only in module-level documentation text used to remove stale local-path examples and align the description with the final manuscript title. Function bodies, imports, constants, model/split logic, identifier handling, chemistry rules, and output logic are unchanged.

A normalized Python AST comparison after removing docstrings gives identical hashes for each original/public pair; hashes are recorded in `provenance/source_hashes.json`.

A controlled two-table smoke test was run against both the supplied run-source and the public copy. Both completed successfully, produced the same output file set, and produced identical profile statistics; the only CSV difference was the measured `read_seconds` timing field.

The large historical pipeline intentionally retains repeated top-level definitions and aliases. Cosmetic deduplication or reordering was not performed because it could change active behavior.

## Final figure renderer

The historical asset generator contains logic from an earlier manuscript layout. The canonical final figure layer is `reproduce/figures/`, which operates on frozen derived tables and produces Main Figures 1–5 and SI Figures S0–S9.
