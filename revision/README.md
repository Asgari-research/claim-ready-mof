# Revision workspace — Digital Discovery DD-ART-10-2026-000896

Work for the major revision lives here. The historical baseline is **frozen**:

- `src/` (historical analysis code) and `figures/` (canonical V6 artwork) are not edited.
  `tests/test_release_integrity.py` fails if either changes.
- New or corrected code goes in `revision/`, with tests in `tests/`.

## Layout

| Path | Purpose |
|---|---|
| `claimready_fixes/` | Corrected, tested helpers replacing reviewer-flagged behaviour (no edits to `src/`) |
| `reviewer_matrix.md` | Reviewer comment → owner → evidence → status |
| `logs/U0_feasibility.md` | U0 handoff: environment, reproducibility and missing-input report |
| `logs/live_definitions.md` | Which copy of each redefined function in `src/` actually runs |

Planned (per the master plan, created when work starts): `target_context/`,
`chemistry_validation/`, `joins/`, `benchmarks/`, `example/`, `figures/`, `response/`.

## Run the checks

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
python tools/validate_release.py
python -m pytest
python reproduce/figures/run_figures.py --allow-font-fallback
```

## Test groups

| File | What it shows |
|---|---|
| `tests/test_release_integrity.py` | Frozen source + figures match recorded hashes |
| `tests/test_historical_smoke.py` | Historical pipeline profiling stage runs |
| `tests/test_historical_known_issues.py` | Executable evidence of each reviewer-flagged behaviour in the frozen code (R1.2, R1.3, R1.4, R1.5, R2.2, R2.3) |
| `tests/test_revision_fixes.py` | Corrected helpers behave as specified |

## Handoff rule

Each handoff to Shayan states: reviewer IDs; COMPLETE / PARTIAL / BLOCKED; commit;
input versions; commands; outputs; baseline comparison; tests; limitations; decision requested.
A passing test or merged PR is not scientific sign-off.
