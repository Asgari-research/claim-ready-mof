#!/usr/bin/env python3
"""Cross-platform entry point for regenerating the final figure set.

Outputs are written under ``reproduce/figures/outputs``. The committed PDFs in
``figures/`` are canonical publication artwork and are not overwritten.
"""
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def run(cmd):
    print("+", " ".join(map(str, cmd)))
    subprocess.run(cmd, cwd=ROOT, check=True)

def main():
    ap=argparse.ArgumentParser(description="Regenerate Main Figures 1-5 and SI Figures S0-S9 from frozen source tables.")
    ap.add_argument("--allow-font-fallback", action="store_true", help="Use DejaVu Sans if Arial is unavailable; QA only.")
    args=ap.parse_args()
    main=[sys.executable,"scripts/redesign_main_figures.py","--figures","1","2","3","4","5"]
    si=[sys.executable,"scripts/redesign_si_figures.py","--figures","S0","S1","S2","S3","S4","S5","S6","S7","S8","S9"]
    if args.allow_font_fallback:
        main.append("--allow-font-fallback"); si.append("--allow-font-fallback")
    run(main); run(si); run([sys.executable,"scripts/qa_outputs.py"])
    print("Done. Regenerated outputs are under reproduce/figures/outputs/.")

if __name__ == "__main__":
    main()
