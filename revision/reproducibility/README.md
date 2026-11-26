# Reconstructed-input reruns (November 2026)

The historical run (Python 3.11.15, Windows) read 509 input files from a local `data_raw/` folder
that is not distributed. These reruns rebuild that folder from public sources and rerun the frozen
pipeline unchanged on Linux, then compare every archived figure-source table in
`reproduce/figures/data/` with the rerun outputs.

All three are **reconstructed-input reruns**, not exact reproductions: five inputs differ from
the historical copy (see "Remaining input gaps").

## How to repeat

```bash
python tools/fetch_historical_inputs.py --work ~/mof_fetch --data-root ~/crmof_data
python src/chemistry_ready_mof_analysis_pipeline_v2_2_claim_ready_sciml_visual_final.py \
    --data-root ~/crmof_data --out-dir ~/crmof_runs/rerun \
    --save-mode balanced --ram-mode light --comprehensive-level thorough \
    --n-jobs 1 --random-seed 42 --force
python tools/compare_reproducibility.py --rerun ~/crmof_runs/rerun --build-publication-assets
```

The data root must not contain the words checked by the historical path rules
(`ion`, `ncr`, `salt`, `charged`, `asr`, `fsr`, `check`, `valid`, ...); see "Install-folder
effect" below. `--build-publication-assets` runs `src/make_publication_assets_v2_3.py` on a
copy of the run's `tables/` and `source_data/`, which writes the main-figure panel tables.

## Runs

| Run | Inputs matched (of 509) | Data root contains `ion` | Pipeline time |
|---|---|---|---|
| 1 | 503 (no `clean_data.csv`) | yes | 24 h 34 min |
| 2 | 504 | yes | about 24 h |
| 3 | 504 | no | 16 h 36 min |

All 13 pipeline stages completed in each run, and all 31 key outputs recorded in
`provenance/historical_pipeline_state_summary.json` were produced.

## Result of run 3 (30 archived tables)

| Status | Tables |
|---|---:|
| IDENTICAL | 15 |
| CLOSE | 4 |
| DIFFERENT | 8 |
| BLOCKED | 3 |

- **IDENTICAL:** Figure 1; Figure 2b, 2c; Figure 3a, 3d; Figure 4b; Figure 5a/b, 5c, 5d; Figure 6a, 6b, 6c;
  the headline endpoint summary; the style sheet; SI Table S15.
- **CLOSE:** SI Table S1 (508 of 509 files; the private inventory file is absent), S2 (682,446 of
  682,496 values), S6 (59,957 of 60,000 values) and S8 (399 of 400 values).
- **DIFFERENT:** Figure 2a, 2d, 3b, 3c, 4a, 4c (both copies) and SI Table S3. Every differing value
  is in a row derived from the CSD MOF Collection (41,537 rows here versus 39,982 historically), or is a
  resource rank in Figure 4c that moves because the CSD-derived score moved. The "Other/unknown"
  score in Figure 4c also moves slightly, most likely because the private inventory file is absent.
- **BLOCKED:** SI Tables S0, S0b, S0c need `PROJECT_DATA_ARCHIVE_STRUCTURE_AND_CONTENTS.txt`.

The 43 differing S6 values are top-k recovery for HistGradientBoosting, each a change of one or
two structures in about 600 test structures. They are the same in runs 2 and 3, so they do not come
from the inputs or the folder; they are consistent with floating-point differences between the
historical Windows build and Linux.

`SI_headline_endpoint_repeated_summary` is not written by any script in the repository. It equals
the per-endpoint mean and SD of SI Table S6 over repeats (to 1e-15 on the archived tables), and the
comparison rebuilds it that way.

## Install-folder effect (reviewer comment R1.2)

`row_flags()` marks a file as an ionic context if its **absolute path** contains `ion`, `charged`,
`salt` or `ncr`, and `role_of()` assigns `curation_or_validation` on similar substrings. In runs 1
and 2 the data root was under `.../Documents/Professional/...`, and "profess**ion**al" contains
`ion`. As a result:

- every one of 278,492 chemistry-trust rows was treated as ionic;
- non-zero charges became `charged_context` instead of `nonzero_charge_uncertain`, so MOSAEC-DB lost
  its `flagged_or_inconsistent` rows (SI Table S8: 200 -> 159 rows) and the rule cards changed
  (SI Table S15: 32 -> 24 rows);
- about 7,240 of 22,016 SI Table S2 rows changed role to `curation_or_validation`.

Run 3 used the same code and inputs from a neutral folder and restored S8, S15 and Figure 3d.
The behaviour is pinned by `tests/test_historical_known_issues.py::test_R1_2_*`; the corrected rule
(`revision/claimready_fixes/ionic_context.py`) considers only whole tokens inside the data root.

## Remaining input gaps

| File | Status |
|---|---|
| CSD MOF Collection `Framework details.csv`, `Frameworks with hydrogen added.csv`, `Charged frameworks.csv`, `Suspect chemistry frameworks.csv` | Public copies are a newer CSD release than the historical 6.01 copy |
| `PROJECT_DATA_ARCHIVE_STRUCTURE_AND_CONTENTS.txt` | Private inventory of the original archive |

With the historical copies of these five files, the remaining tables are expected to match.
