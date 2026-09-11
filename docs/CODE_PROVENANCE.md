# Code provenance

This release separates the historical scientific-analysis code from the later standalone figure-rendering layer.

## Scientific-analysis source

The two files under `src/` are behavior-preserving public copies of the supplied run-source files. Only module-level documentation text was normalized for public portability and manuscript naming. `provenance/source_hashes.json` records both file hashes and normalized executable-AST hashes.

The completed historical run is documented by sanitized run metadata under `provenance/` and summarized in `docs/HISTORICAL_RUN.md`.

## Final figure renderer

The final renderer under `reproduce/figures/` operates only on frozen derived figure-source tables. It does not refit ML models, rerun joins, regenerate descriptors, or alter chemistry/evidence classifications. The committed PDFs under `figures/` remain the canonical publication artwork.

Development-tool history is not used as scientific evidence. Repository claims are based on source preservation, recorded run metadata, direct execution tests, and output checks. Any disclosure required by a journal or institution should be handled truthfully according to that policy.
