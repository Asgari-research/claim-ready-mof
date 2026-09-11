# Figure regeneration

This directory regenerates the final publication figure set from frozen, derived source tables.

- Main Figures **1–5**
- Supporting Information Figures **S0–S9**
- The former development-stage main Figure 6 is represented in the final manuscript as **SI Figure S9**.

The renderer does **not** refit models, regenerate descriptors, rerun simulations, or reclassify chemistry/evidence states.

## Environment

```bash
conda env create -f environment.yml
conda activate claim_ready_mof_figures
```

Alternatively, install `requirements.txt` in Python 3.11.

## Run

On Windows, Linux, or macOS:

```bash
python run_figures.py
```

Final publication rendering uses Arial from a lawful local installation. On Linux/WSL, if Arial is installed outside the standard font path:

```bash
export CLAIM_READY_MOF_ARIAL_DIR="$HOME/.local/share/fonts/Arial"
python run_figures.py
```

For CI or non-publication QA on a machine without Arial:

```bash
python run_figures.py --allow-font-fallback
```

Expected output: **15 figures × 3 formats = 45 assets** under `outputs/`. The committed PDFs under the repository-level `figures/` directory remain the canonical publication artwork and are never overwritten by this runner.

## Data

`data/` contains sanitized, frozen **derived figure-source tables**. It does not contain raw third-party MOF databases or CIF archives. Historical machine-specific root paths in five inventory/profile tables are replaced by `<DATA_ROOT>`; numerical/scientific values are unchanged.
