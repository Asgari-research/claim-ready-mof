#!/usr/bin/env python
from __future__ import annotations
import argparse, hashlib, json, math, os, platform, sys, textwrap
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, BoundaryNorm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

from config import *

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "main"
OUT = ROOT / "outputs" / "main"
QA = ROOT / "qa"


def _sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def setup_font(allow_fallback=False):
    candidates=[]
    env=os.environ.get("PAPER13_ARIAL_DIR")
    if env:
        candidates += [Path(env)/"arial.ttf", Path(env)/"arialbd.ttf"]
    if platform.system().lower().startswith("win"):
        w=Path(os.environ.get("WINDIR", r"C:\Windows"))/"Fonts"
        candidates += [w/"arial.ttf", w/"arialbd.ttf", w/"ariali.ttf"]
    for p in candidates:
        if p.exists():
            try: font_manager.fontManager.addfont(str(p))
            except Exception: pass
    try:
        resolved=font_manager.findfont(FONT_FAMILY, fallback_to_default=False)
        family=FONT_FAMILY
    except Exception:
        if not allow_fallback:
            raise SystemExit(
                "Arial was not found. On Windows it should normally be in C:\\Windows\\Fonts. "
                "Set PAPER13_ARIAL_DIR to a lawful installed Arial folder, or use --allow-font-fallback only for QA previews."
            )
        resolved=font_manager.findfont("DejaVu Sans")
        family="DejaVu Sans"
    plt.rcParams.update({
        "font.family": family, "font.size": FONT_SIZES["base"],
        "axes.titlesize": FONT_SIZES["title"], "axes.labelsize": FONT_SIZES["axis"],
        "xtick.labelsize": FONT_SIZES["tick"], "ytick.labelsize": FONT_SIZES["tick"],
        "legend.fontsize": FONT_SIZES["legend"], "axes.spines.top": False,
        "axes.spines.right": False, "axes.linewidth": .75,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    })
    return family, resolved


def df(name): return pd.read_csv(DATA/name, low_memory=False)
def wrap(s,w=34): return "\n".join(textwrap.wrap(str(s), width=w, break_long_words=False, break_on_hyphens=False))

def panel(ax, letter, x=-0.11, y=1.19):
    ax.text(x, y, letter.upper(), transform=ax.transAxes, ha="left", va="top",
            fontsize=15.5, fontweight="bold", clip_on=False)

def clean(ax, grid=True, axis="both"):
    if grid: ax.grid(True, axis=axis, color=GRID, lw=.55, alpha=.55, zorder=0)
    ax.tick_params(length=3, width=.7)

def save(fig, stem):
    OUT.mkdir(parents=True,exist_ok=True)
    for ext in ["pdf","svg"]:
        fig.savefig(OUT/f"{stem}.{ext}", bbox_inches="tight")
    fig.savefig(OUT/f"{stem}.png", dpi=PNG_DPI, bbox_inches="tight")
    plt.close(fig)


def fig1():
    stages=df("Figure_1_framework.csv")
    fig=plt.figure(figsize=(FULL_WIDTH_IN,5.25))
    gs=fig.add_gridspec(2,3,height_ratios=[0.70,1.30],width_ratios=[0.95,1.13,1.12],hspace=.16,wspace=.27)
    a=fig.add_subplot(gs[0,:]); b=fig.add_subplot(gs[1,0]); c=fig.add_subplot(gs[1,1]); d=fig.add_subplot(gs[1,2])

    # A: compact, fully in-bounds evidence chain with a richer publication palette.
    a.axis('off'); panel(a,'A',-.005,1.16)
    a.set_title("Evidence chain from data files to defensible claims",fontweight='bold',pad=5)
    xs=np.linspace(.095,.905,len(stages))
    stage_cols=['#777C86','#3D8FC2','#297C9D','#C7962B','#33866E']
    for i,(x,(_,r)) in enumerate(zip(xs,stages.iterrows())):
        col=stage_cols[i % len(stage_cols)]
        a.add_patch(FancyBboxPatch((x-.080,.290),.160,.325,boxstyle="round,pad=.010,rounding_size=.018",
                                   fc=col,ec=col,lw=.9,alpha=.16,clip_on=False))
        a.text(x,.525,str(r.stage),ha='center',va='center',fontweight='bold',fontsize=8.8)
        a.text(x,.385,wrap(r.operational_meaning,18),ha='center',va='center',fontsize=7.65,color='#383838',linespacing=1.10)
        if i<len(xs)-1:
            a.add_patch(FancyArrowPatch((x+.081,.455),(xs[i+1]-.083,.455),arrowstyle='-|>',mutation_scale=11,lw=1.0,color='#666666'))
    a.text(.50,.145,"Model-ready does not automatically mean evidence-ready for a chemistry-sensitive claim.",ha='center',fontsize=8.5,color='#3D3D3D')

    for ax,L,title in [(b,'B','Claim determines evidence'),(c,'C','Keep evidence states distinct'),(d,'D','Complementary resource roles')]:
        ax.axis('off'); panel(ax,L,-.02,1.16); ax.set_title(title,fontweight='bold',pad=5)

    rows=[("Prediction","features · targets · units","#D6E8F5"),("Screening","rank stability · traceability","#D5EFE6"),("Chemistry claim","formula · charge · curation","#FCE9B7"),("Mechanism","interpretable domain evidence","#E9E1F2")]
    y=.895
    for t,s,col in rows:
        b.add_patch(FancyBboxPatch((.025,y-.090),.95,.145,boxstyle='round,pad=.012,rounding_size=.018',fc=col,ec='none'))
        b.text(.070,y+.008,t,fontweight='bold',fontsize=8.8,va='center')
        b.text(.070,y-.054,s,fontsize=8.0,va='center',color='#404040')
        y-=.205

    regimes=[("Validated + observable","curated and directly visible",TRUST_COLORS['validated_observable']),
             ("Validated, not observable","trusted source; field absent",TRUST_COLORS['validated_not_observable']),
             ("Uncertain + observable","visible but not independently validated",TRUST_COLORS['uncertain_observable']),
             ("Uncertain, unobservable","missing from parsed representation",TRUST_COLORS['uncertain_unobservable']),
             ("Flagged / inconsistent","warning or detected conflict",TRUST_COLORS['flagged_or_inconsistent'])]
    y=.905
    for t,s,col in regimes:
        c.add_patch(FancyBboxPatch((.018,y-.073),.964,.132,boxstyle='round,pad=.010,rounding_size=.015',fc=col,ec='none',alpha=.20))
        c.text(.055,y+.012,t,fontweight='bold',fontsize=8.05,va='center')
        c.text(.055,y-.024,wrap(s,33),fontsize=7.20,va='top',color='#404040',linespacing=1.10)
        y-=.165
    c.text(.025,.052,"Not observable ≠ chemically invalid.",fontsize=8.3,fontweight='bold',color='#444')

    roles=[("MOSAEC / CoRE","chemistry and provenance anchors","#D5EFE6"),("ARC-MOF","adsorption/process ML stress test","#EFDCEA"),("QMOF","DFT / quantum-property contrast","#FCE9B7"),("Reusable output","claim maps + split-aware benchmarks","#EEEEF0")]
    y=.895
    for t,s,col in roles:
        d.add_patch(FancyBboxPatch((.020,y-.095),.96,.160,boxstyle='round,pad=.012,rounding_size=.018',fc=col,ec='#C8C8C8',lw=.55))
        d.text(.068,y+.010,t,fontweight='bold',fontsize=8.8,va='center')
        d.text(.068,y-.060,wrap(s,29),fontsize=7.85,va='center',color='#404040',linespacing=1.16)
        y-=.205
    fig.subplots_adjust(left=.040,right=.990,top=.965,bottom=.045)
    save(fig,"Figure_1_claim_readiness_framework")

def fig2():
    A=df('Figure_2a_resource_summary.csv'); B=df('Figure_2b_modality_presence.csv'); C=df('Figure_2c_missingness_composition.csv'); D=df('Figure_2d_readiness_scores.csv'); regimes=df('Figure_3b_validation_observability_regimes.csv')
    order=list(A.sort_values('rows',ascending=True).resource)
    fig,axs=plt.subplots(2,2,figsize=(FULL_WIDTH_IN,5.78)); a,b,c,d=axs.ravel()
    panel(a,'A'); clean(a,True,'x')
    aa=A.set_index('resource').loc[order].reset_index(); y=np.arange(len(aa))
    a.barh(y,aa.rows,color=[RESOURCE_COLORS[x] for x in aa.resource],height=.62,zorder=3)
    a.set_xscale('log'); a.set_yticks(y); a.set_yticklabels([SHORT[x] for x in aa.resource]); a.set_xlabel('Profiled / counted rows (log scale)'); a.set_title('Local data-product scale',fontweight='bold')
    for yi,r in enumerate(aa.itertuples()): a.text(r.rows*1.08,yi,f"{int(r.n_files)} files",va='center',fontsize=8.0,color='#444')

    panel(b,'B'); b.set_title('Observed data modalities',fontweight='bold')
    bm=B.copy().set_index('resource').loc[[x for x in A.resource if x in set(B.resource)]]; bm.index=[SHORT[x] for x in bm.index]
    arr=bm.values.astype(float); cmap=ListedColormap(['#F2F3F4','#173A5E']); b.imshow(arr,aspect='auto',cmap=cmap,vmin=0,vmax=1)
    b.set_xticks(range(bm.shape[1])); b.set_xticklabels(bm.columns,rotation=28,ha='right'); b.set_yticks(range(bm.shape[0])); b.set_yticklabels(bm.index); b.tick_params(length=0)
    for sp in b.spines.values(): sp.set_visible(False)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]): b.text(j,i,'●' if arr[i,j] else '–',ha='center',va='center',fontsize=10,color='white' if arr[i,j] else '#777',fontweight='bold')

    panel(c,'C'); c.set_title('Column-missingness composition',fontweight='bold'); clean(c,False)
    miss_order=['0%','0–5%','5–25%','25–75%','>75%']; miss_cols=['#E8F3F8','#B9DFEA','#73C2C6','#F1C27D','#D87555']
    cc=C.pivot(index='resource',columns='missingness_bin',values='fraction').fillna(0)
    cc=cc.loc[[x for x in A.resource if x in cc.index]]
    left=np.zeros(len(cc))
    for lab,col in zip(miss_order,miss_cols):
        vals=cc[lab].values if lab in cc else np.zeros(len(cc)); c.barh(np.arange(len(cc)),vals,left=left,height=.62,color=col,label=lab); left+=vals
    c.set_yticks(range(len(cc))); c.set_yticklabels([SHORT[x] for x in cc.index]); c.set_xlim(0,1); c.set_xlabel('Fraction of profiled columns'); c.invert_yaxis(); c.legend(frameon=False,ncol=5,loc='upper center',bbox_to_anchor=(.5,-.18),columnspacing=.7,handlelength=1)

    panel(d,'D',x=-.11,y=1.20); d.set_title('Row-level chemistry-evidence observation',fontweight='bold',pad=9); clean(d,True,'x')
    dd=D.set_index('resource').loc[[x for x in A.resource if x in set(D.resource)]].reset_index()
    not_scanned=set(regimes.loc[regimes.trust_regime.eq('not_scanned'),'resource'])
    y=np.arange(len(dd)); vals=dd.row_level_trust_observed_fraction.astype(float).values
    for i,r in enumerate(dd.itertuples()):
        if r.resource in not_scanned:
            d.barh(i,1.0,height=.58,color='#F2F2F2',edgecolor='#9A9A9A',hatch='///'); d.text(.50,i,'not scanned',ha='center',va='center',fontsize=8.1,color='#555')
        else:
            d.barh(i,vals[i],height=.58,color=RESOURCE_COLORS[r.resource]); d.text(min(vals[i]+.025,.93),i,f"{100*vals[i]:.0f}%",va='center',fontsize=8.2,color='#444')
    d.set_yticks(y); d.set_yticklabels([SHORT[x] for x in dd.resource]); d.set_xlim(0,1); d.invert_yaxis(); d.set_xlabel('Rows with observed row-level trust/evidence fields')

    # Wider inter-column gutter; explanatory caveats are intentionally left for the caption.
    fig.subplots_adjust(left=.115,right=.990,top=.94,bottom=.115,wspace=.52,hspace=.52)
    save(fig,'Figure_2_data_resource_atlas')

def fig3():
    A=df('Figure_3a_domain_evidence_observability.csv'); B=df('Figure_3b_validation_observability_regimes.csv'); C=df('Figure_3c_charge_status.csv'); cards=df('Figure_3d_representative_rule_cards.csv')
    fig,axs=plt.subplots(2,2,figsize=(FULL_WIDTH_IN,6.48)); a,b,c,d=axs.ravel()
    panel(a,'A',x=-.16,y=1.20); a.set_title('Direct domain-evidence observability',fontweight='bold',pad=9)
    labels=['Formula','Metal','Oxidation state','Charge','Curation']; cols=['formula_observable','metal_observable','oxidation_state_observable','charge_observable','curation_observable']
    aa=A.set_index('resource')[cols]; aa.index=[SHORT[x] for x in aa.index]
    arr=aa.values; cmap=LinearSegmentedColormap.from_list('obs',['#F5F5F5','#D6EEF3','#0072B2']); a.imshow(arr,aspect='auto',cmap=cmap,vmin=0,vmax=1)
    a.set_xticks(range(5)); a.set_xticklabels(labels,rotation=25,ha='right'); a.set_yticks(range(len(aa))); a.set_yticklabels(aa.index); a.tick_params(length=0)
    for sp in a.spines.values(): sp.set_visible(False)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]): a.text(j,i,f"{100*arr[i,j]:.0f}%",ha='center',va='center',fontsize=8.1,color='white' if arr[i,j]>.55 else '#333')

    panel(b,'B'); b.set_title('Validation–observability regimes',fontweight='bold')
    resources=['MOSAEC-DB','CoRE MOF 2024','CoRE MOF 2025 metadata','ARC-MOF','QMOF','CSD-derived context']
    pivot=B.pivot(index='resource',columns='trust_regime',values='fraction').fillna(0).reindex(resources)
    keys=['validated_observable','validated_not_observable','uncertain_observable','uncertain_unobservable','flagged_or_inconsistent','not_scanned']
    left=np.zeros(len(pivot))
    for k in keys:
        v=pivot[k].values if k in pivot else np.zeros(len(pivot)); b.barh(np.arange(len(pivot)),v,left=left,height=.62,color=TRUST_COLORS[k],label=k.replace('_',' ')); left+=v
    b.set_yticks(range(len(pivot))); b.set_yticklabels([SHORT[x] for x in pivot.index]); b.set_xlim(0,1); b.invert_yaxis(); b.set_xlabel('Fraction of scanned rows / explicit no-scan state',labelpad=7)
    b.legend(frameon=False,fontsize=7.25,ncol=2,loc='upper center',bbox_to_anchor=(.50,-.245),columnspacing=.8,borderaxespad=0)

    panel(c,'C'); c.set_title('Charge-status decomposition',fontweight='bold')
    cp=C.pivot(index='resource',columns='charge_status',values='fraction').fillna(0).reindex(resources)
    status=['neutral_or_balanced','nonzero_charge_uncertain','not_observable','not_scanned']; scol=['#56B4E9','#D55E00','#AFAFAF','#ECECEC']
    left=np.zeros(len(cp))
    for k,col in zip(status,scol):
        v=cp[k].values if k in cp else np.zeros(len(cp)); c.barh(np.arange(len(cp)),v,left=left,height=.62,color=col,label=k.replace('_',' ')); left+=v
    c.set_yticks(range(len(cp))); c.set_yticklabels([SHORT[x] for x in cp.index]); c.set_xlim(0,1); c.invert_yaxis(); c.set_xlabel('Fraction of scanned rows / explicit no-scan state',labelpad=7)
    c.legend(frameon=False,fontsize=7.3,ncol=2,loc='upper center',bbox_to_anchor=(.50,-.245),borderaxespad=0)

    panel(d,'D',x=-.15,y=1.30); d.axis('off'); d.set_title('Representative evidence-state examples',fontweight='bold',pad=11)
    ytops=[.952,.635,.318]
    for y,(_,r) in zip(ytops,cards.iterrows()):
        cls=str(r.rule_card_class); col=TRUST_COLORS.get(cls,'#DDDDDD')
        d.add_patch(FancyBboxPatch((.016,y-.258),.968,.264,boxstyle='round,pad=.012,rounding_size=.018',fc=col,ec=col,lw=.7,alpha=.16))
        d.text(.050,y-.030,cls.replace('_',' '),fontweight='bold',fontsize=8.20,va='top')
        meta=f"{SHORT.get(r.resource,r.resource)} · metal {r.metals_detected} · trust {r.chemistry_trust_score_row:.2f} · observability {r.observability_score:.2f}"
        d.text(.050,y-.092,meta,fontsize=6.85,va='top',color='#333')
        txt=str(r.display_interpretation) if pd.notna(r.display_interpretation) else str(r.interpretation)
        d.text(.050,y-.148,wrap(txt,42),fontsize=6.88,va='top',color='#444',linespacing=1.18)
    fig.subplots_adjust(left=.12,right=.990,top=.94,bottom=.165,wspace=.35,hspace=.84)
    save(fig,'Figure_3_chemistry_evidence_trust_regimes')

def fig4():
    A=df('Figure_4a_claim_readiness_map.csv'); B=df('Figure_4b_claim_risk_matrix.csv'); C=df('Figure_4c_score_weight_sensitivity.csv')
    fig=plt.figure(figsize=(FULL_WIDTH_IN,6.05)); gs=fig.add_gridspec(2,2,height_ratios=[.88,1.12],hspace=.43,wspace=.50)
    a=fig.add_subplot(gs[0,0]); c=fig.add_subplot(gs[0,1]); b=fig.add_subplot(gs[1,:])
    panel(a,'A'); clean(a); a.set_title('Claim-readiness coordinates',fontweight='bold')
    for _,r in A.iterrows():
        x=float(r.ml_readiness_score); y=float(r.chemistry_trust_score); lo=float(r.chemistry_trust_ci95_low); hi=float(r.chemistry_trust_ci95_high)
        a.errorbar(x,y,yerr=[[max(0,y-lo)],[max(0,hi-y)]],fmt='none',ecolor=RESOURCE_COLORS[r.resource],elinewidth=1.2,capsize=2,zorder=2)
        a.scatter(x,y,s=60,color=RESOURCE_COLORS[r.resource],edgecolor='white',linewidth=.7,zorder=3)
        a.annotate(SHORT[r.resource],(x,y),xytext=(5,4),textcoords='offset points',fontsize=7.8,color='#333')
    a.set_xlim(.35,.90); a.set_ylim(.15,.82); a.set_xlabel('Generic ML-readiness index'); a.set_ylabel('Chemistry-evidence reporting index')

    panel(c,'C'); clean(c,True,'x'); c.set_title('Sensitivity to score weighting',fontweight='bold')
    cc=C.sort_values('mean_rank',ascending=False); yy=np.arange(len(cc))
    for i,r in enumerate(cc.itertuples()):
        c.plot([r.min_rank,r.max_rank],[i,i],lw=2.2,color='#B0B0B0',solid_capstyle='round'); c.scatter(r.mean_rank,i,s=46,color=RESOURCE_COLORS.get(r.resource,'#888'),edgecolor='white',linewidth=.6,zorder=3)
    c.set_yticks(yy); c.set_yticklabels([SHORT.get(x,x) for x in cc.resource]); c.set_xlabel('Rank across weights (1 = highest)'); c.set_xlim(.7,7.3); c.invert_xaxis()

    panel(b,'B',x=-.060,y=1.20); b.set_title('Author-defined claim-risk rubric (severity category, not failure probability)',fontweight='bold',pad=10)
    use=['adsorption ranking','process screening','quantum-property modeling','generative training','mechanistic interpretation']
    issues=list(dict.fromkeys(B.risk_issue)); piv=B.pivot(index='risk_issue',columns='use_case',values='risk_score_0_to_3').reindex(index=issues,columns=use)
    arr=piv.values.astype(float)
    cmap=ListedColormap(['#F6F6F7','#BBDCF0','#F4D27A','#E36D4F'])
    norm=BoundaryNorm([-.5,.5,1.5,2.5,3.5],cmap.N)
    im=b.imshow(arr,aspect='auto',cmap=cmap,norm=norm)
    b.set_xticks(range(len(use))); b.set_xticklabels(['Adsorption\nranking','Process\nscreening','Quantum-property\nmodeling','Generative\ntraining','Mechanistic\ninterpretation'])
    b.set_yticks(range(len(issues))); b.set_yticklabels([wrap(x,31) for x in issues]); b.tick_params(length=0)
    for sp in b.spines.values(): sp.set_visible(False)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]): b.text(j,i,str(int(arr[i,j])),ha='center',va='center',fontsize=8.7,fontweight='bold',color='white' if arr[i,j]>=2.5 else '#24303A')
    cbar=fig.colorbar(im, ax=b, orientation='vertical', fraction=.028, pad=.018, ticks=[0,1,2,3])
    cbar.ax.set_ylabel('Risk severity score', rotation=90, labelpad=8, fontsize=8.0)
    cbar.ax.tick_params(labelsize=7.6, length=2)
    fig.subplots_adjust(left=.22,right=.965,top=.94,bottom=.08)
    save(fig,'Figure_4_trust_readiness_risk_matrix')

def _r2_sd_table():
    s=df('SI_headline_endpoint_repeated_summary.csv')
    return s[['target_column','descriptor_family','split_type','mean_r2','sd_r2','n_repeats']].copy()

def fig5():
    A=df('Figure_5ab_selected_endpoint_cases.csv'); C=df('Figure_5c_descriptor_family_comparison.csv'); D=df('Figure_5d_ranking_percentile.csv'); sd=_r2_sd_table()
    fig,axs=plt.subplots(2,2,figsize=(FULL_WIDTH_IN,6.38)); a,b,c,d=axs.ravel()
    labels=A.case_label.tolist(); y=np.arange(len(A))
    panel(a,'A'); clean(a,True,'x'); a.set_title('Random vs grouped R²',fontweight='bold',pad=8)
    for i,r in A.iterrows():
        sr=sd[(sd.target_column==r.target_column)&(sd.descriptor_family==r.descriptor_family)]
        rv=sr[sr.split_type=='random']; gv=sr[sr.split_type=='descriptor_grouped']
        rsd=float(rv.sd_r2.iloc[0]) if len(rv) else np.nan; gsd=float(gv.sd_r2.iloc[0]) if len(gv) else np.nan
        a.plot([r.mean_r2_descriptor_grouped,r.mean_r2_random],[i,i],color='#B5B5B5',lw=1.4,zorder=1)
        a.errorbar(r.mean_r2_random,i,xerr=rsd,fmt='o',ms=6.2,color=SPLIT_COLORS['random'],ecolor=SPLIT_COLORS['random'],capsize=2,label='random' if i==0 else None,zorder=3)
        a.errorbar(r.mean_r2_descriptor_grouped,i,xerr=gsd,fmt='s',ms=5.8,color=SPLIT_COLORS['descriptor_grouped'],ecolor=SPLIT_COLORS['descriptor_grouped'],capsize=2,label='descriptor grouped' if i==0 else None,zorder=3)
    a.set_yticks(y); a.set_yticklabels(labels); a.invert_yaxis(); a.set_xlabel('Mean R² ± SD across 5 repeats')
    a.legend(frameon=False,loc='upper center',bbox_to_anchor=(.50,-.20),ncol=2,columnspacing=1.2,handletextpad=.5,borderaxespad=0)

    panel(b,'B',x=-.12,y=1.28); clean(b,True,'x'); b.set_title('Random vs grouped top-10% recovery',fontweight='bold',pad=8)
    for i,r in A.iterrows():
        b.plot([r.mean_top_10pct_recovery_descriptor_grouped,r.mean_top_10pct_recovery_random],[i,i],color='#B5B5B5',lw=1.4,zorder=1)
        b.scatter(r.mean_top_10pct_recovery_random,i,s=46,color=SPLIT_COLORS['random'],zorder=3)
        b.scatter(r.mean_top_10pct_recovery_descriptor_grouped,i,s=46,marker='s',color=SPLIT_COLORS['descriptor_grouped'],zorder=3)
    b.set_yticks(y); b.set_yticklabels(labels); b.invert_yaxis(); b.set_xlabel('Mean top-10% recovery'); b.set_xlim(.45,.78)

    panel(c,'C'); c.set_title('Descriptor-family comparison',fontweight='bold',pad=8)
    cc=C.set_index('index'); cc.index=[x.replace('_',' ') for x in cc.index]
    cols=['R² random','R² grouped','Top-10 random','Top-10 grouped']; arr=cc[cols].values.astype(float)
    cmap=LinearSegmentedColormap.from_list('cmp',['#F5F5F5','#B9DDEE','#0072B2']); c.imshow(arr,aspect='auto',cmap=cmap,vmin=np.nanmin(arr),vmax=np.nanmax(arr))
    c.set_xticks(range(4)); c.set_xticklabels(['R²\nrandom','R²\ngrouped','Top-10\nrandom','Top-10\ngrouped']); c.set_yticks(range(len(cc))); c.set_yticklabels(cc.index); c.tick_params(length=0)
    for sp in c.spines.values(): sp.set_visible(False)
    threshold=np.nanmedian(arr)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]): c.text(j,i,f"{arr[i,j]:.2f}",ha='center',va='center',fontsize=8.5,color='white' if arr[i,j]>=threshold else '#333')

    panel(d,'D',x=-.15,y=1.22); clean(d); d.set_title('Rank-percentile agreement',fontweight='bold',pad=10)
    d.hexbin(D.observed_percentile,D.predicted_percentile,gridsize=34,mincnt=1,cmap='Blues',linewidths=0)
    d.plot([0,1],[0,1],'--',lw=1,color='#555'); d.axvline(.9,ls=':',lw=.8,color='#888'); d.axhline(.9,ls=':',lw=.8,color='#888')
    d.set_xlim(0,1); d.set_ylim(0,1); d.set_xlabel('Observed target percentile'); d.set_ylabel('Predicted target percentile')
    rho=pd.Series(D.observed_percentile).corr(pd.Series(D.predicted_percentile),method='spearman')
    r0=D.iloc[0]; d.text(.035,.965,f"RAC · v/v · ExtraTrees\ndescriptor-grouped · repeat {int(r0['repeat'])}\nSpearman = {rho:.2f} · n = {len(D):,}",transform=d.transAxes,va='top',fontsize=8.0,bbox=dict(boxstyle='round,pad=.25',fc='white',ec='#CCCCCC',lw=.6,alpha=.94))
    fig.subplots_adjust(left=.15,right=.990,top=.94,bottom=.11,wspace=.38,hspace=.70)
    save(fig,'Figure_5_ml_stress_test')

FIGFUN={1:fig1,2:fig2,3:fig3,4:fig4,5:fig5}

def main():
    ap=argparse.ArgumentParser(description='Regenerate redesigned Paper 13 main figures 1-5 from frozen saved CSVs only; the decision framework is SI Figure S9.')
    ap.add_argument('--figures',nargs='*',type=int,default=[1,2,3,4,5],choices=[1,2,3,4,5])
    ap.add_argument('--allow-font-fallback',action='store_true',help='QA preview only; final Windows run should use Arial.')
    args=ap.parse_args()
    family,fontpath=setup_font(args.allow_font_fallback)
    print("[Paper13 PATCH v5] Main generator loaded")
    OUT.mkdir(parents=True,exist_ok=True); QA.mkdir(parents=True,exist_ok=True)
    for ext in ['pdf','svg','png']:
        stale=OUT/f'Figure_6_decision_framework.{ext}'
        if stale.exists():
            stale.unlink()
            print(f'[Paper13] Removed stale main-figure file: {stale.name}')
    for n in args.figures: print(f'[Paper13] Rendering Figure {n} ...'); FIGFUN[n]()
    inputs={p.name:_sha256(p) for p in sorted(DATA.glob('*.csv'))}
    outputs={p.name:_sha256(p) for p in sorted(OUT.glob('*')) if p.is_file()}
    manifest={'generated_utc':datetime.now(timezone.utc).isoformat(),'font_family':family,'font_file':str(fontpath),'figures':args.figures,'input_sha256':inputs,'output_sha256':outputs,'note':'Saved results only; no model fitting, descriptor generation, or chemistry reclassification.'}
    (QA/'main_generation_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(f'[Paper13] Done. Outputs: {OUT}')
    print(f'[Paper13] Font resolved: {family} -> {fontpath}')
if __name__=='__main__': main()
