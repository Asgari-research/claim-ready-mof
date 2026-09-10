#!/usr/bin/env bash
set -euo pipefail
python scripts/redesign_main_figures.py --figures 1 2 3 4 5
python scripts/redesign_si_figures.py --figures S0 S1 S2 S3 S4 S5 S6 S7 S8 S9
python scripts/qa_outputs.py
echo "Done: main Figures 1-5 + SI Figures S0-S9; QA passed."
