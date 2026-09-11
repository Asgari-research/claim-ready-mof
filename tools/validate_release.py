#!/usr/bin/env python3
from pathlib import Path
import ast, hashlib, json, re, subprocess, sys
ROOT=Path(__file__).resolve().parents[1]
errors=[]
required=[
 'README.md','LICENSE','AUTHORS.md','CITATION.cff','docs/HISTORICAL_RUN.md','docs/CODE_PROVENANCE.md',
 'src/chemistry_ready_mof_analysis_pipeline_v2_2_claim_ready_sciml_visual_final.py','src/make_publication_assets_v2_3.py',
 'reproduce/figures/run_figures.py','reproduce/figures/scripts/redesign_main_figures.py','reproduce/figures/scripts/redesign_si_figures.py']
for r in required:
    if not (ROOT/r).is_file(): errors.append(f"missing {r}")
main=list((ROOT/'figures/main').glob('*.pdf')); si=list((ROOT/'figures/supplementary').glob('*.pdf'))
if len(main)!=5: errors.append(f"expected 5 main PDFs, got {len(main)}")
if len(si)!=10: errors.append(f"expected 10 SI PDFs, got {len(si)}")
data=list((ROOT/'reproduce/figures/data').rglob('*.csv'))
if len(data)!=30: errors.append(f"expected 30 figure-source CSVs, got {len(data)}")
# Syntax
for p in ROOT.rglob('*.py'):
    if any(x in p.parts for x in ['outputs','qa']): continue
    try: ast.parse(p.read_text(encoding='utf-8'))
    except Exception as e: errors.append(f"syntax: {p.relative_to(ROOT)}: {e}")
# No public development-number naming or private paths.
patterns=[re.compile(r'paper\s*13',re.I),re.compile(r'[A-Za-z]:\\projects\\our_group',re.I),re.compile(r'/home/shayan',re.I),re.compile(r'/mnt/c/Users/abaei',re.I)]
for p in ROOT.rglob('*'):
    if '.git' in p.parts: continue
    if not p.is_file() or p.suffix.lower() in {'.pdf','.png','.svg'}: continue
    if p.resolve() == Path(__file__).resolve(): continue
    try: txt=p.read_text('utf-8',errors='ignore')
    except: continue
    for pat in patterns:
        if pat.search(txt): errors.append(f"forbidden public string in {p.relative_to(ROOT)}: {pat.pattern}")
# Figure hashes.
fh=ROOT/'provenance/final_figure_sha256.csv'
if fh.exists():
    import csv
    for row in csv.DictReader(fh.open()):
        p=ROOT/row['file']
        if hashlib.sha256(p.read_bytes()).hexdigest()!=row['sha256']: errors.append(f"figure hash mismatch: {row['file']}")
if errors:
    print('RELEASE VALIDATION: FAIL'); [print(' -',e) for e in errors]; sys.exit(1)
print('RELEASE VALIDATION: PASS')
print(f'  main PDFs: {len(main)} | SI PDFs: {len(si)} | figure-source CSVs: {len(data)}')
