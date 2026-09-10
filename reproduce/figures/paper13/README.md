# Paper 13 figure regeneration

Regenerates the final Paper 13 artwork from frozen saved CSV outputs.

## Final set
- Main Figures **1–5**
- SI Figures **S0–S9**
- Former main Figure 6 is now **Figure S9**.

## Environment
```bash
conda env create -f environment.yml
conda activate paper13_figures
```
or install `requirements.txt` in Python 3.11.

## Run under Linux/WSL
```bash
python scripts/redesign_main_figures.py --figures 1 2 3 4 5
python scripts/redesign_si_figures.py --figures S0 S1 S2 S3 S4 S5 S6 S7 S8 S9
python scripts/qa_outputs.py
```

Final rendering requires Arial from a lawful local installation. The repository must not redistribute Arial font files.

Expected output: **15 figures × 3 formats = 45 assets**.

## Regenerate and synchronize into repository figure folders

When this directory lives at `reproduce/figures/paper13/` inside the target repository, run:

```bash
bash RUN_AND_SYNC_WSL.sh
```

This runs all generators, performs file-level QA, and copies the regenerated outputs to `figures/main/` and `figures/supplementary/`.
