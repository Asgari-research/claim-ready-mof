#!/usr/bin/env python
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
MAIN=[
'Figure_1_claim_readiness_framework','Figure_2_data_resource_atlas',
'Figure_3_chemistry_evidence_trust_regimes','Figure_4_trust_readiness_risk_matrix',
'Figure_5_ml_stress_test']
SI=[
'SI_Figure_S0_inventory_aware_input_map','SI_Figure_S1_inventory_profile',
'SI_Figure_S2_missingness_column_types','SI_Figure_S3_join_accounting',
'SI_Figure_S4_coverage_profiles','SI_Figure_S5_trust_score_sensitivity',
'SI_Figure_S6_ml_repeated_splits','SI_Figure_S7_duplicate_leakage',
'SI_Figure_S8_rule_cards','SI_Figure_S9_decision_framework']
rows=[]
for group,stems in [('main',MAIN),('si',SI)]:
    for stem in stems:
        for ext in ['pdf','svg','png']:
            p=ROOT/'outputs'/group/f'{stem}.{ext}'
            rows.append((p.relative_to(ROOT).as_posix(),p.exists(),p.stat().st_size if p.exists() else 0))
missing=[r for r in rows if not r[1] or r[2]<1000]
report=['# Paper 13 figure-output QA','','File-level QA after regeneration. This does not replace scientific manuscript review.','',
        f'- Expected figures: {len(MAIN)+len(SI)} (5 main + 10 SI, S0-S9)',
        f'- Expected exported assets: {len(rows)} (PDF + SVG + PNG)',
        f'- Missing/too-small assets: {len(missing)}','', '## Files']
report += [f"- {'OK' if ok and size>=1000 else 'FAIL'} - `{name}` - {size:,} bytes" for name,ok,size in rows]
if missing:
    report += ['', '## Action required']+[f'- Regenerate `{m[0]}`' for m in missing]
(ROOT/'qa').mkdir(exist_ok=True)
(ROOT/'qa'/'QA_REPORT.md').write_text('\n'.join(report),encoding='utf-8')
print('\n'.join(report[:9]))
raise SystemExit(1 if missing else 0)
