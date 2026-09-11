# Historical run record

The supplied completed-run metadata records a successful full analysis on **29–30 July 2026**. All **13** recorded pipeline stages have status `ok`. The sum of recorded stage durations is **24.1 h**.

Key run settings:

- Python: **3.11.15** (conda-forge, Windows)
- NumPy: **2.4.6**
- pandas: **3.0.3**
- Matplotlib: **3.11.0**
- scikit-learn: **1.9.0**
- SciPy: **1.17.1**
- save mode: `balanced`
- RAM mode: `light`
- comprehensive level: `thorough`
- `n_jobs = 1`
- random seed: `42`
- repeated ML splits configured: `10`
- models configured: dummy, ridge, histogram gradient boosting, random forest, extra trees
- split families configured: random and grouped

The historical log records **509 table-like files** discovered during profiling and an archive inventory containing **9,335 example files** across **1,852 folders**.

Equivalent command structure:

```bash
python src/chemistry_ready_mof_analysis_pipeline_v2_2_claim_ready_sciml_visual_final.py \
  --data-root /path/to/project_data \
  --out-dir /path/to/output \
  --inventory-report /path/to/PROJECT_DATA_ARCHIVE_STRUCTURE_AND_CONTENTS.zip \
  --save-mode balanced \
  --ram-mode light \
  --comprehensive-level thorough \
  --n-jobs 1 \
  --random-seed 42 \
  --force
```

Raw third-party inputs are not included in this repository. Consequently, this record documents the run that produced the archived results; it does not claim that a fresh end-to-end rerun can be performed from this repository alone.
