# Data and reproducibility

## Public repository contents

The repository includes:

- the preserved scientific-analysis source under `src/`;
- canonical final publication PDFs under `figures/`;
- sanitized frozen derived figure-source tables under `reproduce/figures/data/`;
- a standalone cross-platform figure renderer under `reproduce/figures/`.

## Figure-source tables

The figure-source CSVs are derived outputs prepared from the completed analysis. They are sufficient to regenerate the final figure layer without rerunning the expensive upstream profiling, joins, chemistry rules, or ML stages.

Five inventory/profile tables originally contained absolute historical filesystem roots. For public portability those roots are replaced by `<DATA_ROOT>`. Row counts, columns, numerical values, identifiers, and other scientific fields are unchanged.

## Raw third-party resources

Raw third-party MOF databases and CIF archives used in the historical analysis are not redistributed here. Users who wish to reconstruct the full upstream run must obtain the relevant resources from their original providers and respect those providers’ terms.

## Environment boundary

The exact historical run captured Python and core numerical package versions; versions for `pyarrow` and `openpyxl` were not recorded. `environment/historical_analysis.yml` is therefore a reconstructed environment specification, not a cryptographic lockfile.
