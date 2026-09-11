#!/usr/bin/env python3
from pathlib import Path
import csv, subprocess, sys, tempfile
ROOT=Path(__file__).resolve().parents[1]
script=ROOT/'src/chemistry_ready_mof_analysis_pipeline_v2_2_claim_ready_sciml_visual_final.py'
with tempfile.TemporaryDirectory() as td:
    td=Path(td); data=td/'data'; out=td/'out'; data.mkdir()
    (data/'a.csv').write_text('mof_id,formula,metal,charge,pld,lcd,target\nMOF1,C8H4O4Zn,Zn,0,4.2,6.8,1.2\nMOF2,C6H3O6Cu,Cu,0,3.5,5.1,2.1\n')
    (data/'b.csv').write_text('id,density,asa,uptake\nMOF1,0.9,1200,2.2\nMOF2,1.1,900,1.8\n')
    cmd=[sys.executable,str(script),'--data-root',str(data),'--out-dir',str(out),'--save-mode','minimal','--ram-mode','ultra-light','--comprehensive-level','screening','--n-jobs','1','--force','--profile-only','--skip-ml','--no-zip']
    subprocess.run(cmd,check=True)
    if not (out/'profiles/file_level_profile.csv').exists(): raise SystemExit('smoke test failed: profile output missing')
print('HISTORICAL PIPELINE SMOKE TEST: PASS')
