# Reviewer matrix (computational items)

Status values: PENDING EVIDENCE · IN PROGRESS · HANDED OFF · ACCEPTED · NEEDS CHANGES · BLOCKED.
Owners are proposals until confirmed at the first coordination meeting.

| ID | Demand (short) | Proposed lead | Evidence so far | Status |
|---|---|---|---|---|
| E1 / R3.3 | Executed 100–200 MOF open example, raw → figure | Uzair; Shayan accepts | Figure layer regenerates only; full pipeline needs raw inputs | PENDING EVIDENCE |
| R1.2 | Validate heuristic chemistry classification | Natalia design; Shayan chemistry; Uzair code | Substring ionic rule demonstrated (`test_R1_2_*`); install-folder effect observed in reruns (`reproducibility/README.md`); token-aware, root-relative rule without NCR in `claimready_fixes/ionic_context.py` | IN PROGRESS |
| R1.3 | Same-cohort descriptor comparison (Fig 5C) | Uzair; Shayan verifies | Families joined separately confirmed (`test_R1_3_*`) | PENDING EVIDENCE |
| R1.4 | End-to-end traceability | Uzair; Natalia + Shayan review | First-row join loses conditions (`test_R1_4_*`); condition-preserving join in `claimready_fixes/condition_keys.py` | IN PROGRESS |
| R1.5 | Dedup + realistic generalisation | Uzair; Shayan verifies | Tied values split across quantile groups (`test_R1_5_*`); tie-safe groups + leakage checks in `claimready_fixes/splits.py` | IN PROGRESS |
| R2.2 | Identifier normalisation / duplicates | Natalia design; Shayan; Uzair code | `sym_stripped` false match shown (`test_R2_2_*`) | IN PROGRESS |
| R2.3 | Feature selection before splitting | Uzair; Natalia + Shayan interpret | Pre-split screening confirmed (`test_R2_3_*`); train-only pipeline in `claimready_fixes/screening.py` | IN PROGRESS |
| R2.4 | Stronger grouped holdout | Natalia + Shayan define; Uzair runs | Needs metal/topology/family metadata | PENDING EVIDENCE |
| R2.6 | S(g1) and adsorption conditions | Shayan; Uzair trace checks | Needs ARC-MOF source rows | BLOCKED (inputs) |
| R3.1 | Free CSD MOF compatibility + data manifest | Uzair | Not started | PENDING EVIDENCE |
| R3.2 | Folder READMEs | Uzair; Natalia reviews | `revision/README.md` added | IN PROGRESS |
| R3.4 | Directory naming conventions + tests | Uzair; Shayan + Natalia | Path-token rule partly addresses `ion` example | PENDING EVIDENCE |
