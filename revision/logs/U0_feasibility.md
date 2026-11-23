# U0 — Feasibility and reproducibility report (draft)

Owner: Uzair · Status: PARTIAL · Reviewer IDs: E1, R1.2–R1.5, R2.2, R2.3, R3.3
Baseline: branch `test-ujaira02`, parent commit `fe29766`.

## 1. What runs today

| Check | Result | What it actually covers |
|---|---|---|
| `tools/validate_release.py` | PASS | File presence, syntax, forbidden strings, figure hashes |
| Historical smoke test | PASS | Profiling stage only, two 2-row CSVs, ML skipped |
| Figure regeneration | PASS (~24 s) | Re-renders 15 figures from frozen CSV tables; content matches canonical PDFs; page size differs slightly without Arial |
| New pytest suite | 35 passed | Python 3.11 + pinned historical versions, and Python 3.13 + latest versions |

## 2. What cannot run yet

A fresh end-to-end rerun is not possible from the repository: the July 2026 raw
inputs (509 table-like files) are not included. Figure regeneration is not
end-to-end reproducibility.

## 3. Reviewer-flagged behaviour confirmed in the live code

Evidence: `tests/test_historical_known_issues.py`.

1. **R1.2** – Ionic context is a substring test on the file *path*:
   `adsorption`, `regression`, `version`, `selection` all count as ionic.
2. **R1.4 / R2.6** – Joins keep the first row per material; two pressure
   states collapse to one, and the kept label depends on row order.
3. **R2.2** – Under `sym_stripped`, `…pcu_sym.10` receives the descriptors of
   `…pcu_sym.11` (a different structure).
4. **R1.5 / R2.4** – Rank-`first` quantile groups split tied descriptor values
   across groups (9 of 51 values in a synthetic test).
5. **R2.3** – Missingness / constant / variance screening uses all rows before splitting.
6. **R1.3** – Each descriptor family is joined to targets separately, so cohorts differ.

## 4. Maintainability

- Main script: 9,256 lines; 61 names defined more than once; 4,167 lines are
  superseded dead code. Only the last copy runs — see `live_definitions.md`.
- CI previously installed unpinned packages; now pinned to the recorded run.
- Input reads are capped at 120,000 rows (`nrows`, file head, not random) and
  model rows at 60,000 (random sample, seed 42). Check whether any historical
  target file exceeded the cap.

## 5. Inputs / access needed

| Need | From | Blocks |
|---|---|---|
| Original data root (509 files) or acquisition manifest with versions/licences | Shayan | G1 rerun, E1, R1.4 |
| ARC-MOF CO2/N2 source rows, column definitions, S(g1) formula | Shayan | R2.6, condition keys |
| Archived outputs: split indices, join tables, per-row chemistry labels | Shayan | baseline comparison |
| Metal / topology / family metadata | Shayan / Natalia | R2.4 holdout |
| CSD licence position for a public 100–200 MOF example | Mehrdad | E1, R3.1 |
| Write access to the upstream repository / revision branch policy | Mehrdad | merging |

## 6. Open scientific questions (not coding decisions)

- Should `NCR` mark an ionic context? The historical rule treats it as one;
  in CoRE MOF naming NCR usually means "not computation-ready". The token-aware
  rule keeps it for now so only the substring defect changes.
- Which identifier-variant levels are acceptable for joins? The corrected join
  uses exact identity only; any fuzzier level needs a measured false-match rate.
- Primary endpoint and condition for the common-cohort benchmark.
