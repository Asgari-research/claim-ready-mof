#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE"
python scripts/redesign_main_figures.py --figures 1 2 3 4 5
python scripts/redesign_si_figures.py --figures S0 S1 S2 S3 S4 S5 S6 S7 S8 S9
python scripts/qa_outputs.py
REPO_ROOT=$(cd "$HERE/../../.." && pwd)
mkdir -p "$REPO_ROOT/figures/main" "$REPO_ROOT/figures/supplementary"
cp -a outputs/main/. "$REPO_ROOT/figures/main/"
cp -a outputs/si/. "$REPO_ROOT/figures/supplementary/"
echo "Regenerated and synchronized final Paper 13 figures into $REPO_ROOT/figures/."
