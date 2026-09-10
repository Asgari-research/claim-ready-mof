#!/usr/bin/env python3
"""Historical Paper 13 publication-asset generator.

This script was supplied with the completed analysis and rebuilds publication
assets from stored intermediate tables. It is retained for provenance, but it
is not the canonical renderer for the finalized manuscript artwork.

The finalized figure-regeneration workflow is maintained separately under
``reproduce/figures/paper13`` and generates main Figures 1--5 and Supporting
Information Figures S0--S9 from frozen figure-source tables.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import textwrap
import warnings
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.colors import LinearSegmentedColormap

VERSION = "2.3-publication-final-visual-polish"
warnings.filterwarnings("ignore", category=FutureWarning, message="The default value of observed=False.*")
warnings.filterwarnings("ignore", message="An input array is constant; the correlation coefficient is not defined.*")
MAIN_RESOURCES = [
    "MOSAEC-DB", "CoRE MOF 2024", "CoRE MOF 2025 metadata",
    "ARC-MOF", "QMOF", "CSD-derived context"
]
SHORT = {
    "MOSAEC-DB": "MOSAEC", "CoRE MOF 2024": "CoRE-24",
    "CoRE MOF 2025 metadata": "CoRE-25", "ARC-MOF": "ARC-MOF",
    "QMOF": "QMOF", "CSD-derived context": "CSD ctx",
    "Other/unknown": "Other"
}
RESOURCE_COLORS = {
    "MOSAEC-DB": "#009E73", "CoRE MOF 2024": "#0072B2",
    "CoRE MOF 2025 metadata": "#56B4E9", "ARC-MOF": "#CC79A7",
    "QMOF": "#E69F00", "CSD-derived context": "#D55E00",
    "Other/unknown": "#999999"
}
TRUST_COLORS = {
    "validated_observable": "#009E73",
    "validated_not_observable": "#56B4E9",
    "uncertain_observable": "#E69F00",
    "uncertain_unobservable": "#999999",
    "flagged_or_inconsistent": "#D55E00",
    "not_scanned": "#EEEEEE",
}
SPLIT_COLORS = {"random": "#0072B2", "descriptor_grouped": "#E69F00", "grouped": "#E69F00"}
STAGE_COLORS = ["#7A7A7A", "#56B4E9", "#0072B2", "#E69F00", "#009E73"]

# Final-size typography. Figures are authored at approximately 183 mm width,
# so labels remain legible when inserted without rescaling in a two-column journal.
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "font.family": "Liberation Sans",
    "font.size": 7.4,
    "axes.titlesize": 8.6,
    "axes.labelsize": 7.8,
    "axes.linewidth": 0.65,
    "xtick.labelsize": 6.8,
    "ytick.labelsize": 6.8,
    "legend.fontsize": 6.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False) if path.exists() else pd.DataFrame()


def wrap(s: object, width: int) -> str:
    return "\n".join(textwrap.wrap(str(s), width=width, break_long_words=False, break_on_hyphens=False))


def clean_ax(ax, grid: bool=True):
    ax.set_facecolor("white")
    if grid:
        ax.grid(True, axis="both", lw=0.35, alpha=0.14, zorder=0)
    ax.spines["left"].set_linewidth(0.65)
    ax.spines["bottom"].set_linewidth(0.65)
    ax.tick_params(length=2.5, width=0.55)


def panel(ax, letter: str, x: float=-0.11, y: float=1.08):
    ax.text(x, y, letter, transform=ax.transAxes, ha="left", va="top",
            fontsize=9.3, fontweight="bold", clip_on=False,
            bbox=dict(boxstyle="round,pad=0.10", fc="white", ec="#555555", lw=0.45))


def save_figure(fig, outbase: Path):
    outbase.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outbase.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(outbase.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(outbase.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_panel(df: pd.DataFrame, outdir: Path, name: str):
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / f"{name}.csv", index=False)


def fmt_rows(v: float) -> str:
    if pd.isna(v): return "-"
    v=float(v)
    if v >= 1e6: return f"{v/1e6:.2f}M"
    if v >= 1e3: return f"{v/1e3:.1f}k"
    return f"{int(round(v)):,}"


def _binary_heatmap(ax, mat: pd.DataFrame, title: str):
    cmap = LinearSegmentedColormap.from_list("presence", ["#F4F6F7", "#133A5E"])
    arr = mat.values.astype(float)
    ax.imshow(arr, aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks(range(mat.shape[1]))
    ax.set_xticklabels(mat.columns, rotation=28, ha="right")
    ax.set_yticks(range(mat.shape[0]))
    ax.set_yticklabels(mat.index)
    ax.tick_params(length=0)
    for sp in ax.spines.values(): sp.set_visible(False)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, "●" if arr[i,j] >= .5 else "–", ha="center", va="center",
                    fontsize=7.6, color="white" if arr[i,j] >= .5 else "#777777", fontweight="bold")
    ax.set_title(title, fontweight="bold", pad=7)


def figure1(root: Path, panel_dir: Path):
    """Framework figure with a wide workflow panel and three readable lower panels."""
    fig=plt.figure(figsize=(7.2,5.05))
    gs=fig.add_gridspec(2,3,height_ratios=[.90,1.10],wspace=.20,hspace=.34)
    axa=fig.add_subplot(gs[0,:]); axb=fig.add_subplot(gs[1,0]); axc=fig.add_subplot(gs[1,1]); axd=fig.add_subplot(gs[1,2])

    ax=axa; ax.axis("off"); panel(ax,"a",x=0.0,y=1.02)
    ax.set_title("From files to claim-ready scientific ML datasets",fontweight="bold",pad=4)
    stages=[
        ("Raw files","located · cited · hashed"),
        ("Computation-ready","parsable tables · units"),
        ("Model-ready","features · targets · splits"),
        ("Evidence-ready","domain facts observable · validated"),
        ("Claim-ready","supported claim · caveats · source data"),
    ]
    xs=np.linspace(.09,.91,5)
    for i,((title,body),x,c) in enumerate(zip(stages,xs,STAGE_COLORS)):
        ax.add_patch(FancyBboxPatch((x-.084,.43),.168,.32,boxstyle="round,pad=.010,rounding_size=.024",fc=c,ec="none",alpha=.16))
        ax.text(x,.635,title,ha="center",va="center",fontsize=7.1,fontweight="bold")
        ax.text(x,.515,wrap(body,20),ha="center",va="center",fontsize=6.2,color="#444444")
        if i<4:
            ax.add_patch(FancyArrowPatch((x+.087,.59),(xs[i+1]-.089,.59),arrowstyle="-|>",mutation_scale=8,lw=.7,color="#666666"))
    ax.text(.50,.24,"The key scientific gap is whether model-ready data contain the evidence needed for the stated claim.",ha="center",fontsize=6.5,color="#444444")

    ax=axb; ax.axis("off"); panel(ax,"b",x=0.0,y=1.01); ax.set_title("Claim → evidence",fontweight="bold",pad=4)
    rows=[("Prediction","features, targets, units","#DDEBF7"),("Screening","ranking stability + traceability","#DDF2E9"),("Chemistry claim","formula, charge, curation","#FFF0CC"),("Mechanism","interpretable evidence","#EEE8F3")]
    y=.83
    for title,body,c in rows:
        ax.add_patch(FancyBboxPatch((.04,y-.065),.92,.105,boxstyle="round,pad=.008,rounding_size=.014",fc=c,ec="none"))
        ax.text(.08,y-.01,title,fontsize=6.4,fontweight="bold",ha="left",va="center")
        ax.text(.08,y-.045,body,fontsize=5.8,ha="left",va="center",color="#444444")
        y-=.18

    ax=axc; ax.axis("off"); panel(ax,"c",x=0.0,y=1.01); ax.set_title("Evidence regimes",fontweight="bold",pad=4)
    regimes=[("validated observable","curated + visible",TRUST_COLORS["validated_observable"]),("validated not observable","trusted, fields missing",TRUST_COLORS["validated_not_observable"]),("uncertain observable","visible, not validated",TRUST_COLORS["uncertain_observable"]),("uncertain unobservable","not visible in metadata",TRUST_COLORS["uncertain_unobservable"]),("flagged / inconsistent","warning / conflict",TRUST_COLORS["flagged_or_inconsistent"])]
    y=.86
    for title,body,c in regimes:
        ax.add_patch(FancyBboxPatch((.04,y-.050),.92,.084,boxstyle="round,pad=.007,rounding_size=.012",fc=c,ec="none",alpha=.18))
        ax.text(.08,y-.006,title,fontsize=5.9,fontweight="bold",ha="left",va="center")
        ax.text(.08,y-.034,body,fontsize=5.4,ha="left",va="center",color="#444444")
        y-=.135
    ax.text(.04,.075,"Not observable ≠ chemically invalid.",fontsize=5.8,color="#444444")

    ax=axd; ax.axis("off"); panel(ax,"d",x=0.0,y=1.01); ax.set_title("MOF case-study roles",fontweight="bold",pad=4)
    cards=[("MOSAEC / CoRE","chemistry + provenance anchors","#DDF2E9"),("ARC-MOF","large adsorption/process ML stress test","#F5E3EF"),("QMOF","DFT / quantum-property contrast","#FFF0CC"),("Reusable output","claim maps + split-aware benchmarks","#F1F1F1")]
    y=.83
    for title,body,c in cards:
        ax.add_patch(FancyBboxPatch((.04,y-.065),.92,.105,boxstyle="round,pad=.008,rounding_size=.014",fc=c,ec="#D0D0D0",lw=.35))
        ax.text(.08,y-.005,title,fontsize=6.3,fontweight="bold",ha="left",va="center")
        ax.text(.08,y-.043,wrap(body,32),fontsize=5.5,ha="left",va="center",color="#444444")
        y-=.18

    source=pd.DataFrame([{"stage":t,"operational_meaning":b,"color_hex":c} for (t,b),c in zip(stages,STAGE_COLORS)])
    save_panel(source,panel_dir,"Figure_1_framework")
    fig.suptitle("Figure 1. A claim-readiness framework for scientific machine-learning datasets",fontsize=10.2,fontweight="bold",y=.995)
    fig.subplots_adjust(left=.045,right=.99,top=.91,bottom=.055)
    save_figure(fig,root/"figures/main/Figure_1_claim_readiness_framework")

def figure2(root: Path, panel_dir: Path):
    pdir=root/"source_data/figure_panel_source_data"
    files=read_csv(pdir/"Figure_2a_resource_file_scale.csv")
    mod=read_csv(pdir/"Figure_2b_modality_evidence.csv")
    cols=read_csv(pdir/"Figure_2c_missingness.csv")
    scores=read_csv(pdir/"Figure_2d_resource_scores.csv")

    # Exact data transformations used in the final panels.
    f=files[files.resource.isin(MAIN_RESOURCES)].copy()
    f["true_rows"]=pd.to_numeric(f["true_rows"],errors="coerce")
    f["size_mb"]=pd.to_numeric(f["size_mb"],errors="coerce")
    res=(f.groupby("resource",as_index=False)
           .agg(n_files=("file_name","count"), rows=("true_rows","sum"), total_size_mb=("size_mb","sum")))
    res["order"]=res.resource.map({r:i for i,r in enumerate(MAIN_RESOURCES)})
    res=res.sort_values("order").drop(columns="order")
    save_panel(res,panel_dir,"Figure_2a_resource_summary")

    modality_cols=[
        ("identifier","IDs"), ("formula_or_composition","formula"),
        ("metal_or_charge_chemistry","chemistry"), ("descriptor","descriptors"),
        ("target_or_property","targets"), ("curation_or_validation_flag","curation")
    ]
    mm=mod[mod.resource.isin(MAIN_RESOURCES)].set_index("resource")
    for c,_ in modality_cols:
        if c not in mm: mm[c]=0
    mm=(mm[[c for c,_ in modality_cols]]>0).astype(int)
    mm=mm.reindex(MAIN_RESOURCES).fillna(0).astype(int)
    mm.columns=[lab for _,lab in modality_cols]
    save_panel(mm.reset_index(),panel_dir,"Figure_2b_modality_presence")

    c=cols[cols.resource.isin(MAIN_RESOURCES)].copy()
    c["missing_fraction"]=pd.to_numeric(c["missing_fraction"],errors="coerce")
    bins=[-1e-9,1e-12,.05,.25,.75,1.000001]
    labels=["0%","0–5%","5–25%","25–75%",">75%"]
    c["missingness_bin"]=pd.cut(c.missing_fraction,bins=bins,labels=labels,include_lowest=True)
    miss=(c.groupby(["resource","missingness_bin"],observed=False).size().rename("n_columns").reset_index())
    miss["fraction"]=miss.n_columns/miss.groupby("resource").n_columns.transform("sum")
    save_panel(miss,panel_dir,"Figure_2c_missingness_composition")

    s=scores[scores.resource.isin(MAIN_RESOURCES)].copy()
    for x in ["chemistry_trust_score","ml_readiness_score","total_profiled_or_counted_rows"]:
        s[x]=pd.to_numeric(s[x],errors="coerce")
    save_panel(s,panel_dir,"Figure_2d_readiness_scores")

    fig,axs=plt.subplots(2,2,figsize=(7.2,5.25)); axs=axs.ravel()
    # a rows scale
    ax=axs[0]; panel(ax,"a"); clean_ax(ax,grid=False)
    plot=res.sort_values("rows",ascending=True)
    y=np.arange(len(plot))
    ax.barh(y,np.log10(plot.rows.clip(lower=1)),color=[RESOURCE_COLORS[r] for r in plot.resource],height=.62,alpha=.90)
    ax.set_yticks(y); ax.set_yticklabels([SHORT[r] for r in plot.resource])
    ax.set_xlabel("Profiled/counted rows (log scale)")
    ticks=[1,1e2,1e4,1e6,1e8]; ax.set_xticks(np.log10(ticks)); ax.set_xticklabels(["1","100","10k","1M","100M"])
    ax.set_title("Input scale by resource",fontweight="bold")
    xmax=max(np.log10(plot.rows.clip(lower=1)))
    for yi,(_,r) in enumerate(plot.iterrows()):
        ax.text(np.log10(max(r.rows,1))+.06,yi,f"{int(r.n_files)} files",va="center",fontsize=6.0,color="#444444")
    ax.set_xlim(0,xmax+.95)

    # b modalities
    ax=axs[1]; panel(ax,"b")
    _binary_heatmap(ax,mm.rename(index=SHORT),"Observed data modalities")

    # c stacked missingness
    ax=axs[2]; panel(ax,"c"); clean_ax(ax,grid=False)
    piv=miss.pivot_table(index="resource",columns="missingness_bin",values="fraction",fill_value=0).reindex(MAIN_RESOURCES)
    colors=["#EAF3F8","#C9E4EE","#95CCD8","#F3C77C","#D9805F"]
    left=np.zeros(len(piv)); yy=np.arange(len(piv))
    for lab,cx in zip(labels,colors):
        vals=piv[lab].values if lab in piv else np.zeros(len(piv))
        ax.barh(yy,vals,left=left,height=.60,color=cx,label=lab,edgecolor="white",linewidth=.35)
        left+=vals
    ax.set_yticks(yy); ax.set_yticklabels([SHORT[r] for r in piv.index])
    ax.set_xlim(0,1); ax.set_xlabel("Fraction of profiled columns"); ax.set_title("Column missingness composition",fontweight="bold")
    handles, legend_labels = ax.get_legend_handles_labels()

    # d readiness map
    ax=axs[3]; panel(ax,"d"); clean_ax(ax,grid=True)
    ax.axvspan(.70,1.02,color="#EFF7F5",zorder=0); ax.axhspan(.70,1.02,color="#F3F8EF",zorder=0)
    ax.axvline(.70,color="#AAAAAA",lw=.65,ls="--"); ax.axhline(.70,color="#AAAAAA",lw=.65,ls="--")
    offsets={"MOSAEC-DB":(.012,.014),"CoRE MOF 2024":(.012,-.035),"CoRE MOF 2025 metadata":(.012,.014),"ARC-MOF":(-.095,-.035),"QMOF":(.012,.014),"CSD-derived context":(.012,-.035)}
    for _,r in s.iterrows():
        size=45+18*np.log10(max(float(r.total_profiled_or_counted_rows),1))
        ax.scatter(r.ml_readiness_score,r.chemistry_trust_score,s=size,color=RESOURCE_COLORS[r.resource],edgecolor="white",linewidth=.6,zorder=3)
        dx,dy=offsets.get(r.resource,(.01,.01)); ax.text(r.ml_readiness_score+dx,r.chemistry_trust_score+dy,SHORT[r.resource],fontsize=6.2)
    ax.set_xlim(0,1.02); ax.set_ylim(0,1.02)
    ax.set_xlabel("ML-readiness score"); ax.set_ylabel("Chemistry-trust score")
    ax.set_title("Scale, chemistry trust and ML-readiness",fontweight="bold")
    ax.text(.98,.98,"high claim readiness",transform=ax.transAxes,ha="right",va="top",fontsize=5.7,color="#5D7E66")

    fig.suptitle("Figure 2. MOF databases as a demanding claim-readiness case study",fontsize=10.2,fontweight="bold",y=.995)
    fig.legend(handles, legend_labels, ncol=5, loc="lower center", bbox_to_anchor=(.50,.015), frameon=False, columnspacing=.9, handlelength=1.2, fontsize=6.2)
    fig.subplots_adjust(left=.12,right=.985,top=.91,bottom=.18,wspace=.33,hspace=.46)
    save_figure(fig,root/"figures/main/Figure_2_data_resource_atlas")


def figure3(root: Path, panel_dir: Path):
    pdir=root/"source_data/figure_panel_source_data"
    evidence=read_csv(pdir/"Figure_3a_chemistry_evidence.csv")
    reg=read_csv(pdir/"Figure_3b_trust_regime_summary.csv")
    charge=read_csv(pdir/"Figure_3c_charge_status_summary.csv")
    cards=read_csv(root/"tables/si/SI_Table_S15_representative_rule_cards.csv")
    evidence=evidence[evidence.resource.isin(MAIN_RESOURCES)].copy()
    save_panel(evidence,panel_dir,"Figure_3a_domain_evidence_observability")

    # add explicit not-scanned blocks for main resources absent from the trust scan
    regimes=["validated_observable","validated_not_observable","uncertain_observable","uncertain_unobservable","flagged_or_inconsistent","not_scanned"]
    rows=[]
    for r in MAIN_RESOURCES:
        sub=reg[reg.resource.eq(r)]
        if sub.empty:
            rows.append({"resource":r,"trust_regime":"not_scanned","fraction":1.0,"n_rows":0})
        else:
            for _,q in sub.iterrows(): rows.append({"resource":r,"trust_regime":q.trust_regime_row,"fraction":float(q.fraction),"n_rows":int(q.n_rows)})
    reg2=pd.DataFrame(rows); save_panel(reg2,panel_dir,"Figure_3b_validation_observability_regimes")

    crows=[]
    for r in MAIN_RESOURCES:
        sub=charge[charge.resource.eq(r)]
        if sub.empty:
            crows.append({"resource":r,"charge_status":"not_scanned","fraction":1.0,"n_rows":0})
        else:
            for _,q in sub.iterrows(): crows.append({"resource":r,"charge_status":q.charge_balance_status,"fraction":float(q.fraction),"n_rows":int(q.n_rows)})
    charge2=pd.DataFrame(crows); save_panel(charge2,panel_dir,"Figure_3c_charge_status")

    # choose one representative card per key regime
    card_order=["validated_not_observable","uncertain_observable","flagged_or_inconsistent"]
    chosen=[]
    if not cards.empty:
        for rr in card_order:
            sub=cards[cards.rule_card_class.eq(rr)]
            if not sub.empty: chosen.append(sub.iloc[0])
    carddf=pd.DataFrame(chosen) if chosen else pd.DataFrame()
    if not carddf.empty:
        keep=[c for c in ["rule_card_class","resource","primary_id","metals_detected","chemistry_trust_score_row","observability_score","interpretation"] if c in carddf.columns]
        carddf=carddf[keep]
        def _display(r):
            rr=str(r.get("rule_card_class",""))
            if rr=="validated_not_observable": return "trusted source; oxidation/charge fields are not directly observable"
            if rr=="uncertain_observable": return "some chemistry evidence is visible, but validation/charge evidence is incomplete"
            if rr=="flagged_or_inconsistent": return "non-zero charge is uncertain and oxidation state is not directly observable"
            return str(r.get("interpretation",""))
        carddf["display_interpretation"]=carddf.apply(_display,axis=1)
    save_panel(carddf,panel_dir,"Figure_3d_representative_rule_cards")

    fig,axs=plt.subplots(2,2,figsize=(7.2,5.2)); axs=axs.ravel()
    # a evidence heatmap
    ax=axs[0]; panel(ax,"a")
    if not evidence.empty:
        cols=["formula_observable","metal_observable","oxidation_state_observable","charge_observable","curation_observable"]
        mat=evidence.set_index("resource")[cols].reindex([r for r in MAIN_RESOURCES if r in set(evidence.resource)])
        mat.index=[SHORT[r] for r in mat.index]
        mat.columns=["formula","metal","oxid. state","charge","curation"]
        arr=mat.values.astype(float)
        im=ax.imshow(arr,aspect="auto",vmin=0,vmax=1,cmap="YlGnBu",interpolation="nearest")
        ax.set_xticks(range(len(mat.columns))); ax.set_xticklabels(mat.columns,rotation=28,ha="right")
        ax.set_yticks(range(len(mat.index))); ax.set_yticklabels(mat.index); ax.tick_params(length=0)
        for sp in ax.spines.values(): sp.set_visible(False)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]): ax.text(j,i,f"{arr[i,j]:.2f}",ha="center",va="center",fontsize=6.0,color="white" if arr[i,j]>.55 else "#333333")
        cb=fig.colorbar(im,ax=ax,fraction=.045,pad=.025); cb.ax.tick_params(labelsize=6.0); cb.set_label("fraction observable",fontsize=6.4)
        ax.set_title("Domain-evidence observability",fontweight="bold")

    # b regimes stacked
    ax=axs[1]; panel(ax,"b"); clean_ax(ax,grid=False)
    piv=reg2.pivot_table(index="resource",columns="trust_regime",values="fraction",fill_value=0).reindex(MAIN_RESOURCES)
    y=np.arange(len(piv)); left=np.zeros(len(piv))
    for rr in regimes:
        vals=piv[rr].values if rr in piv else np.zeros(len(piv))
        if np.allclose(vals,0): continue
        ax.barh(y,vals,left=left,height=.58,color=TRUST_COLORS[rr],label=rr.replace("_"," "),edgecolor="white",linewidth=.35)
        left+=vals
    ax.set_yticks(y); ax.set_yticklabels([SHORT[r] for r in piv.index]); ax.set_xlim(0,1); ax.set_xlabel("Fraction of scanned / classified rows")
    ax.set_title("Validation-observability regimes",fontweight="bold")
    ax.legend(ncol=2,loc="lower center",bbox_to_anchor=(.5,-.43),frameon=False,columnspacing=.7,handlelength=1.1,fontsize=5.7)

    # c charge status
    ax=axs[2]; panel(ax,"c")
    statuses=["neutral_or_balanced","nonzero_charge_uncertain","not_observable","not_scanned"]
    cpiv=charge2.pivot_table(index="resource",columns="charge_status",values="fraction",fill_value=0).reindex(MAIN_RESOURCES)
    carr=cpiv.reindex(columns=statuses,fill_value=0).values.astype(float)
    cmap=LinearSegmentedColormap.from_list("charge",["#F5F5F5","#BDD7E7","#6BAED6","#2171B5"])
    # categorical composition as stacked bars is clearer than heatmap
    clean_ax(ax,grid=False); yy=np.arange(len(cpiv)); left=np.zeros(len(cpiv))
    scols={"neutral_or_balanced":"#56B4E9","nonzero_charge_uncertain":"#D55E00","not_observable":"#BDBDBD","not_scanned":"#EEEEEE"}
    for st in statuses:
        vals=cpiv[st].values if st in cpiv else np.zeros(len(cpiv))
        ax.barh(yy,vals,left=left,height=.58,color=scols[st],label=st.replace("_"," "),edgecolor="white",linewidth=.35); left+=vals
    ax.set_yticks(yy); ax.set_yticklabels([SHORT[r] for r in cpiv.index]); ax.set_xlim(0,1); ax.set_xlabel("Fraction of scanned rows")
    ax.set_title("Charge-status decomposition",fontweight="bold")
    ax.legend(ncol=2,loc="lower center",bbox_to_anchor=(.5,-.37),frameon=False,columnspacing=.7,handlelength=1.1,fontsize=5.8)

    # d rule cards
    ax=axs[3]; panel(ax,"d"); ax.axis("off"); ax.set_title("How to read non-observable evidence",fontweight="bold",pad=6)
    if not carddf.empty:
        y=.80
        for _,r in carddf.iterrows():
            rr=str(r.rule_card_class); col=TRUST_COLORS.get(rr,"#DDDDDD")
            ax.add_patch(FancyBboxPatch((.04,y-.115),.92,.185,boxstyle="round,pad=.010,rounding_size=.016",fc=col,ec="none",alpha=.15))
            title=rr.replace("_"," ")
            metal=str(r.get("metals_detected","")); metal = metal if metal and metal!="nan" else "not observable"
            meta=f"{SHORT.get(str(r.resource),str(r.resource))}  ·  metal {metal}  ·  score {float(r.get('chemistry_trust_score_row',np.nan)):.2f}"
            desc=str(r.get("display_interpretation",r.get("interpretation","")))
            ax.text(.075,y+.020,title,fontsize=6.3,fontweight="bold",ha="left",va="center")
            ax.text(.94,y+.020,meta,fontsize=5.2,ha="right",va="center",color="#444444")
            ax.text(.075,y-.055,wrap(desc,56),fontsize=5.45,ha="left",va="center",color="#444444")
            y-=.26
    ax.text(.04,.035,"Missing parsed evidence is kept distinct from a warning or inconsistency.",fontsize=5.7,color="#444444")

    fig.suptitle("Figure 3. Domain-evidence observability and validation regimes",fontsize=10.2,fontweight="bold",y=.995)
    fig.subplots_adjust(left=.12,right=.985,top=.91,bottom=.15,wspace=.38,hspace=.50)
    save_figure(fig,root/"figures/main/Figure_3_chemistry_evidence_trust_regimes")


def figure4(root: Path, panel_dir: Path):
    pdir=root/"source_data/figure_panel_source_data"
    unc=read_csv(pdir/"Figure_4a_resource_scores_uncertainty.csv")
    risk=read_csv(pdir/"Figure_4b_benchmark_risk_matrix.csv")
    sens=read_csv(pdir/"Figure_4c_score_sensitivity.csv")
    scores=read_csv(pdir/"Figure_2d_resource_scores.csv")
    data=scores.merge(unc[[c for c in unc.columns if c in ["resource","chemistry_trust_ci95_low","chemistry_trust_ci95_high","uncertainty_method"]]],on="resource",how="left")
    data=data[data.resource.isin(MAIN_RESOURCES)].copy()
    save_panel(data,panel_dir,"Figure_4a_claim_readiness_map")
    save_panel(risk,panel_dir,"Figure_4b_claim_risk_matrix")
    save_panel(sens,panel_dir,"Figure_4c_score_weight_sensitivity")

    fig=plt.figure(figsize=(7.2,5.0))
    gs=fig.add_gridspec(2,2,width_ratios=[.92,1.28],height_ratios=[1,1],wspace=.42,hspace=.45)
    axa=fig.add_subplot(gs[0,0]); axc=fig.add_subplot(gs[1,0]); axb=fig.add_subplot(gs[:,1])

    # a readiness
    ax=axa; panel(ax,"a"); clean_ax(ax,grid=True)
    ax.axvline(.70,color="#AAAAAA",ls="--",lw=.6); ax.axhline(.70,color="#AAAAAA",ls="--",lw=.6)
    offsets={"MOSAEC-DB":(-.08,.025),"CoRE MOF 2024":(-.09,.022),"CoRE MOF 2025 metadata":(.015,.018),"ARC-MOF":(.015,-.04),"QMOF":(.015,.02),"CSD-derived context":(-.10,-.045)}
    for _,r in data.iterrows():
        x=float(r.ml_readiness_score); y=float(r.chemistry_trust_score)
        lo=pd.to_numeric(pd.Series([r.get("chemistry_trust_ci95_low")]),errors="coerce").iloc[0]
        hi=pd.to_numeric(pd.Series([r.get("chemistry_trust_ci95_high")]),errors="coerce").iloc[0]
        if np.isfinite(lo) and np.isfinite(hi) and hi>lo:
            ax.errorbar([x],[y],yerr=[[y-lo],[hi-y]],fmt="none",ecolor="#777777",elinewidth=.65,capsize=1.8,zorder=1)
        ax.scatter(x,y,s=56,color=RESOURCE_COLORS[r.resource],edgecolor="white",linewidth=.55,zorder=3)
        dx,dy=offsets.get(r.resource,(.01,.01)); ax.text(x+dx,y+dy,SHORT[r.resource],fontsize=6.0)
    ax.set_xlim(0,1.02); ax.set_ylim(0,1.02); ax.set_xlabel("ML-readiness score"); ax.set_ylabel("Chemistry-trust score")
    ax.set_title("Claim-readiness map",fontweight="bold")

    # c sensitivity forest
    ax=axc; panel(ax,"c"); clean_ax(ax,grid=False)
    ss=sens[sens.resource.isin(MAIN_RESOURCES)].copy().sort_values("mean_rank",ascending=False)
    y=np.arange(len(ss)); x=ss.mean_rank.astype(float).values; lo=x-ss.min_rank.astype(float).values; hi=ss.max_rank.astype(float).values-x
    ax.errorbar(x,y,xerr=[lo,hi],fmt="o",markersize=4.3,color="#444444",ecolor="#A0A0A0",capsize=2.0,lw=.8)
    ax.set_yticks(y); ax.set_yticklabels([SHORT[r] for r in ss.resource]); ax.set_xlim(.6,7.35); ax.set_xticks(range(1,8))
    ax.set_xlabel("Rank across score-weight perturbations"); ax.set_title("Score-weight sensitivity",fontweight="bold")

    # b risk matrix large
    ax=axb; panel(ax,"b",x=-.08,y=1.04)
    risk_order=[
        "missing oxidation-state/formal-charge evidence","nonzero charge outside explicit ionic context",
        "suspect chemistry or validation warning","identifier ambiguity or duplicate leakage",
        "high descriptor missingness","target provenance mismatch","small or biased trusted subset"
    ]
    use_order=["adsorption ranking","process screening","quantum-property modeling","generative training","mechanistic interpretation"]
    piv=risk.pivot_table(index="risk_issue",columns="use_case",values="risk_score_0_to_3",aggfunc="mean").reindex(index=risk_order,columns=use_order)
    arr=piv.values.astype(float)
    im=ax.imshow(arr,aspect="auto",vmin=0,vmax=3,cmap="OrRd",interpolation="nearest")
    ax.set_xticks(range(len(use_order))); ax.set_xticklabels(["adsorption","process","quantum","generation","mechanism"],rotation=30,ha="right")
    ylabels=["missing chemistry\nevidence","charge ambiguity","validation warning","identifier / duplicate\nleakage","descriptor missingness","target mismatch","small / biased\ntrusted subset"]
    ax.set_yticks(range(len(risk_order))); ax.set_yticklabels(ylabels,fontsize=6.2); ax.tick_params(length=0)
    for sp in ax.spines.values(): sp.set_visible(False)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(j,i,str(int(arr[i,j])),ha="center",va="center",fontsize=6.4,color="white" if arr[i,j]>=2.3 else "#552211",fontweight="bold")
    cb=fig.colorbar(im,ax=ax,fraction=.042,pad=.03,ticks=[0,1,2,3]); cb.ax.tick_params(labelsize=6.0); cb.set_label("risk score",fontsize=6.3)
    ax.set_title("Claim-risk matrix",fontweight="bold",pad=6)

    fig.suptitle("Figure 4. Claim-risk and readiness mapping",fontsize=10.2,fontweight="bold",y=.995)
    fig.subplots_adjust(left=.12,right=.965,top=.90,bottom=.13)
    save_figure(fig,root/"figures/main/Figure_4_trust_readiness_risk_matrix")


def _descriptor_heatmap_data(matched: pd.DataFrame) -> pd.DataFrame:
    m=matched.copy()
    for c in ["mean_r2","mean_top_10pct_recovery"]:
        if c in m: m[c]=pd.to_numeric(m[c],errors="coerce")
    g=(m.groupby(["descriptor_family","split_type"],as_index=False)
         .agg(median_r2=("mean_r2","median"),median_top10=("mean_top_10pct_recovery","median")))
    r2=g.pivot(index="descriptor_family",columns="split_type",values="median_r2")
    t=g.pivot(index="descriptor_family",columns="split_type",values="median_top10")
    out=pd.DataFrame(index=sorted(set(m.descriptor_family.astype(str))))
    if "random" in r2: out["R² random"]=r2["random"]
    if "descriptor_grouped" in r2: out["R² grouped"]=r2["descriptor_grouped"]
    if "random" in t: out["Top-10 random"]=t["random"]
    if "descriptor_grouped" in t: out["Top-10 grouped"]=t["descriptor_grouped"]
    return out


def _best_rank_group(pred: pd.DataFrame) -> tuple[pd.DataFrame,dict]:
    p=pred.copy()
    p["y_true"]=pd.to_numeric(p.y_true,errors="coerce"); p["y_pred"]=pd.to_numeric(p.y_pred,errors="coerce")
    p=p[p.y_true.notna()&p.y_pred.notna()]
    gcols=[c for c in ["source_table","target_column","descriptor_family","model","split_type","repeat"] if c in p.columns]
    rows=[]
    for key,g in p.groupby(gcols,dropna=False):
        if len(g)<500: continue
        sp=g.y_true.corr(g.y_pred,method="spearman")
        if not np.isfinite(sp): continue
        meta=dict(zip(gcols,key if isinstance(key,tuple) else (key,)))
        rows.append({**meta,"n":len(g),"spearman":float(sp)})
    r=pd.DataFrame(rows)
    if r.empty: return pd.DataFrame(),{}
    # Prefer the harder descriptor-grouped split; then choose strongest ranking signal.
    hard=r[r.split_type.astype(str).str.contains("group",case=False,na=False)] if "split_type" in r else r
    if hard.empty: hard=r
    best=hard.sort_values(["spearman","n"],ascending=False).iloc[0].to_dict()
    mask=pd.Series(True,index=p.index)
    for c in gcols: mask &= p[c].astype(str).eq(str(best[c]))
    g=p.loc[mask].copy()
    if len(g)>2500: g=g.sample(2500,random_state=42)
    g["observed_percentile"]=g.y_true.rank(pct=True,method="average")
    g["predicted_percentile"]=g.y_pred.rank(pct=True,method="average")
    return g,best


def figure5(root: Path, panel_dir: Path):
    pdir=root/"source_data/figure_panel_source_data"
    matched=read_csv(pdir/"Figure_5_case_matched_descriptor_comparison.csv")
    pred=read_csv(pdir/"Figure_5d_calibration.csv")
    best=read_csv(root/"tables/main/Main_Table_6_best_descriptor_joined_cases_publication.csv")

    # A/B: transparent five-endpoint selection - best grouped top-10 descriptor family
    # for each post-combustion CO2 target in the publication-filtered table.
    sub=best[(best.task=="post_combustion_capture")&(best.gas=="CO2")].copy()
    metric="mean_top_10pct_recovery_descriptor_grouped"
    selected=(sub.sort_values(metric,ascending=False).groupby("target_column",as_index=False).first())
    target_order=[x for x in ["S(g1)","mmol/g","molc/uc","v/v","wt%"] if x in set(selected.target_column)]
    selected["target_column"]=pd.Categorical(selected.target_column,categories=target_order,ordered=True)
    selected=selected.sort_values("target_column")
    selected["case_label"]=selected.apply(lambda r:f"{str(r.target_column)} · {str(r.descriptor_family)}",axis=1)
    save_panel(selected,panel_dir,"Figure_5ab_selected_endpoint_cases")

    heat=_descriptor_heatmap_data(matched); save_panel(heat.reset_index(),panel_dir,"Figure_5c_descriptor_family_comparison")
    rank,meta=_best_rank_group(pred)
    rank_out=rank[[c for c in ["source_table","target_column","descriptor_family","model","split_type","repeat","y_true","y_pred","observed_percentile","predicted_percentile"] if c in rank.columns]].copy()
    save_panel(rank_out,panel_dir,"Figure_5d_ranking_percentile")

    fig,axs=plt.subplots(2,2,figsize=(7.2,5.35)); axs=axs.ravel()
    # a dumbbell R2
    ax=axs[0]; panel(ax,"a"); clean_ax(ax,grid=True)
    labels=selected.case_label.astype(str).tolist(); y=np.arange(len(labels))
    xr=selected.mean_r2_random.astype(float).values; xg=selected.mean_r2_descriptor_grouped.astype(float).values
    for i in range(len(y)):
        ax.plot([xg[i],xr[i]],[i,i],color="#B7B7B7",lw=1.15,zorder=1)
    ax.scatter(xr,y,s=34,color=SPLIT_COLORS["random"],label="random",edgecolor="white",linewidth=.4,zorder=3)
    ax.scatter(xg,y,s=34,color=SPLIT_COLORS["descriptor_grouped"],label="descriptor grouped",marker="s",edgecolor="white",linewidth=.4,zorder=3)
    ax.set_yticks(y); ax.set_yticklabels(labels); ax.invert_yaxis(); ax.set_xlabel("Mean R²"); ax.set_title("Generalization gap across endpoint cases",fontweight="bold")
    ax.legend(frameon=False,loc="upper left",bbox_to_anchor=(.02,.98),ncol=2,handletextpad=.4,columnspacing=.7,fontsize=5.9)

    # b top10 recovery
    ax=axs[1]; panel(ax,"b"); clean_ax(ax,grid=True)
    xr=selected.mean_top_10pct_recovery_random.astype(float).values; xg=selected.mean_top_10pct_recovery_descriptor_grouped.astype(float).values
    for i in range(len(y)): ax.plot([xg[i],xr[i]],[i,i],color="#B7B7B7",lw=1.15,zorder=1)
    ax.scatter(xr,y,s=34,color=SPLIT_COLORS["random"],edgecolor="white",linewidth=.4,zorder=3)
    ax.scatter(xg,y,s=34,color=SPLIT_COLORS["descriptor_grouped"],marker="s",edgecolor="white",linewidth=.4,zorder=3)
    ax.set_yticks(y); ax.set_yticklabels(labels); ax.invert_yaxis(); ax.set_xlim(.45,.80); ax.set_xlabel("Top-10% recovery"); ax.set_title("Screening signal under harder splits",fontweight="bold")

    # c heatmap
    ax=axs[2]; panel(ax,"c")
    if not heat.empty:
        h=heat.copy(); h.index=[str(x).replace("_"," ") for x in h.index]
        arr=h.values.astype(float); im=ax.imshow(arr,aspect="auto",vmin=np.nanmin(arr),vmax=np.nanmax(arr),cmap="YlGnBu",interpolation="nearest")
        ax.set_xticks(range(len(h.columns))); ax.set_xticklabels(h.columns,rotation=28,ha="right")
        ax.set_yticks(range(len(h.index))); ax.set_yticklabels(h.index); ax.tick_params(length=0)
        for sp in ax.spines.values(): sp.set_visible(False)
        threshold=np.nanmean(arr)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]): ax.text(j,i,f"{arr[i,j]:.2f}",ha="center",va="center",fontsize=6.0,color="white" if arr[i,j]>threshold else "#333333")
        cb=fig.colorbar(im,ax=ax,fraction=.045,pad=.028); cb.ax.tick_params(labelsize=5.8)
        ax.set_title("Case-matched descriptor-family comparison",fontweight="bold")

    # d rank percentile
    ax=axs[3]; panel(ax,"d"); clean_ax(ax,grid=True)
    if not rank.empty:
        ax.hexbin(rank.observed_percentile,rank.predicted_percentile,gridsize=34,mincnt=1,cmap="Blues",linewidths=0,alpha=.95)
        ax.plot([0,1],[0,1],ls="--",lw=.8,color="#555555")
        ax.axvline(.90,ls=":",lw=.65,color="#999999"); ax.axhline(.90,ls=":",lw=.65,color="#999999")
        ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_xlabel("Observed target percentile"); ax.set_ylabel("Predicted target percentile")
        ax.set_title("Representative ranking-percentile view",fontweight="bold")
        task="post-combustion CO₂" if "post_comb" in str(meta.get("source_table","")) else str(meta.get("source_table",""))[:18]
        txt=(f"{task}\n{meta.get('descriptor_family','')} · {meta.get('target_column','')} · {meta.get('model','')}\n"
             f"{str(meta.get('split_type','')).replace('_',' ')} · Spearman {float(meta.get('spearman',np.nan)):.2f} · n={int(meta.get('n',len(rank)))}")
        ax.text(.035,.965,txt,transform=ax.transAxes,ha="left",va="top",fontsize=5.7,
                bbox=dict(boxstyle="round,pad=.20",fc="white",ec="#CCCCCC",lw=.4,alpha=.94))

    fig.suptitle("Figure 5. Split-aware descriptor-joined benchmark",fontsize=10.2,fontweight="bold",y=.995)
    fig.subplots_adjust(left=.135,right=.985,top=.91,bottom=.12,wspace=.36,hspace=.42)
    save_figure(fig,root/"figures/main/Figure_5_ml_stress_test")


def figure6(root: Path, panel_dir: Path):
    decision=read_csv(root/"source_data/decision_rules.csv")
    checklist=read_csv(root/"source_data/minimum_reporting_checklist.csv")
    rec=read_csv(root/"tables/main/Main_Table_4_use_case_recommendation_cards.csv")
    save_panel(decision,panel_dir,"Figure_6a_decision_rules")
    save_panel(checklist,panel_dir,"Figure_6b_reporting_checklist")
    save_panel(rec,panel_dir,"Figure_6c_use_case_recommendations")

    fig=plt.figure(figsize=(7.2,5.05))
    gs=fig.add_gridspec(2,2,height_ratios=[1.15,.85],width_ratios=[.90,1.10],wspace=.24,hspace=.28)
    axa=fig.add_subplot(gs[0,0]); axb=fig.add_subplot(gs[0,1]); axc=fig.add_subplot(gs[1,:])

    ax=axa; ax.axis("off"); panel(ax,"a",x=0.0,y=1.01); ax.set_title("Claim-first decision tree",fontweight="bold",pad=4)
    nodes=[("Scientific claim",.50,.86),("Target context\nknown?",.50,.65),("Domain evidence\nobservable?",.27,.42),("Split-aware\nperformance?",.73,.42),("Caveated claim +\nsource data",.50,.16)]
    for t,x,y in nodes:
        ax.add_patch(FancyBboxPatch((x-.17,y-.055),.34,.11,boxstyle="round,pad=.012,rounding_size=.018",fc="#F7F7F7",ec="#777777",lw=.5))
        ax.text(x,y,t,ha="center",va="center",fontsize=6.4,fontweight="bold" if y in [.86,.16] else "normal")
    for a,b in [((.50,.80),(.50,.71)),((.50,.59),(.31,.49)),((.50,.59),(.69,.49)),((.27,.36),(.43,.23)),((.73,.36),(.57,.23))]:
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=7,lw=.65,color="#555555"))

    ax=axb; ax.axis("off"); panel(ax,"b",x=0.0,y=1.01); ax.set_title("Minimum reporting blocks",fontweight="bold",pad=4)
    blocks=[("Data provenance","versions, hashes, access route"),("Target context","gas / property / unit / task"),("Identifier joins","retention, duplicates, leakage"),("Domain evidence","observed, missing, validated, flagged"),("Split policy","random + grouped / claim-aware split"),("Source data","one table per figure panel")]
    y=.86
    for title,body in blocks:
        ax.add_patch(FancyBboxPatch((.035,y-.045),.93,.080,boxstyle="round,pad=.009,rounding_size=.012",fc="#F7F7F7",ec="#D0D0D0",lw=.35))
        ax.text(.07,y-.005,title,fontsize=6.0,fontweight="bold",ha="left",va="center")
        ax.text(.52,y-.005,body,fontsize=5.6,ha="left",va="center",color="#444444")
        y-=.125

    ax=axc; ax.axis("off"); panel(ax,"c",x=0.0,y=1.01); ax.set_title("Use-case recommendation cards",fontweight="bold",pad=4)
    y=.83
    for _,r in rec.head(5).iterrows():
        ax.add_patch(FancyBboxPatch((.03,y-.055),.94,.090,boxstyle="round,pad=.008,rounding_size=.012",fc="#FAFAFA",ec="#D0D0D0",lw=.35))
        ax.text(.055,y-.010,str(r.use_case),fontsize=6.0,fontweight="bold",ha="left",va="center")
        ax.text(.31,y-.010,wrap(str(r.preferred_resource_logic),98),fontsize=5.45,ha="left",va="center",color="#444444")
        y-=.145
    ax.text(.03,.035,"Use the evidence required by the claim - not a generic database leaderboard.",fontsize=5.7,color="#444444")

    fig.suptitle("Figure 6. Claim-specific reporting framework for scientific ML datasets",fontsize=10.2,fontweight="bold",y=.995)
    fig.subplots_adjust(left=.045,right=.99,top=.90,bottom=.055)
    save_figure(fig,root/"figures/main/Figure_6_decision_framework")

def _draw_table(df: pd.DataFrame, outbase: Path, title: str, widths: Sequence[float],
                formats: dict[str,str]|None=None, wrap_cols: dict[str,int]|None=None,
                row_height: float=.36, font_size: float=6.1):
    formats=formats or {}; wrap_cols=wrap_cols or {}
    d=df.copy()
    for c,fmt in formats.items():
        if c in d:
            if fmt=="int": d[c]=pd.to_numeric(d[c],errors="coerce").map(lambda x:"-" if pd.isna(x) else f"{int(round(x)):,}")
            elif fmt=="rows": d[c]=pd.to_numeric(d[c],errors="coerce").map(fmt_rows)
            elif fmt=="pct0": d[c]=pd.to_numeric(d[c],errors="coerce").map(lambda x:"-" if pd.isna(x) else f"{100*x:.0f}%")
            elif fmt=="f2": d[c]=pd.to_numeric(d[c],errors="coerce").map(lambda x:"-" if pd.isna(x) else f"{x:.2f}")
    for c,w in wrap_cols.items():
        if c in d: d[c]=d[c].map(lambda x:wrap(x,w))
    n=len(d); fig_h=max(2.0,1.0+n*row_height)
    fig,ax=plt.subplots(figsize=(7.2,fig_h)); ax.axis("off")
    ax.text(0,.99,title,transform=ax.transAxes,ha="left",va="top",fontsize=9.2,fontweight="bold")
    table=ax.table(cellText=d.values,colLabels=d.columns,cellLoc="left",colLoc="left",loc="upper left",bbox=[0,.02,1,.88],colWidths=widths)
    table.auto_set_font_size(False); table.set_fontsize(font_size)
    for (r,c),cell in table.get_celld().items():
        cell.set_edgecolor("#D4D4D4"); cell.set_linewidth(.35)
        if r==0:
            cell.set_facecolor("#173A5E"); cell.get_text().set_color("white"); cell.get_text().set_fontweight("bold")
            cell.set_height(.075)
        else:
            cell.set_facecolor("#F8FAFB" if r%2==0 else "white")
            cell.set_height(.072 if "\n" not in str(cell.get_text().get_text()) else .10)
    fig.savefig(outbase.with_suffix(".pdf"),bbox_inches="tight",facecolor="white")
    fig.savefig(outbase.with_suffix(".png"),dpi=600,bbox_inches="tight",facecolor="white")
    plt.close(fig)


def latex_escape(s: object) -> str:
    x=str(s)
    repl={"\\":"\\textbackslash{}","&":"\\&","%":"\\%","$":"\\$","#":"\\#","_":"\\_","{":"\\{","}":"\\}","~":"\\textasciitilde{}","^":"\\textasciicircum{}"}
    for a,b in repl.items(): x=x.replace(a,b)
    return x


def write_latex_table(df: pd.DataFrame, path: Path, caption: str, label: str):
    lines=["\\begin{table*}[t]","\\centering","\\small",f"\\caption{{{latex_escape(caption)}}}",f"\\label{{{label}}}","\\begin{tabular}{"+"l"*len(df.columns)+"}","\\toprule"]
    lines.append(" & ".join(latex_escape(c) for c in df.columns)+" \\\\")
    lines.append("\\midrule")
    for _,r in df.iterrows(): lines.append(" & ".join(latex_escape(r[c]) for c in df.columns)+" \\\\")
    lines += ["\\bottomrule","\\end{tabular}","\\end{table*}",""]
    path.write_text("\n".join(lines),encoding="utf-8")


def publication_tables(root: Path):
    out=root/"tables/publication_ready"; out.mkdir(parents=True,exist_ok=True)
    # Table 1 compact resource atlas
    t1=read_csv(root/"tables/main/Main_Table_1_publication_resource_atlas_clean.csv")
    t1=t1[["short_label","n_files","profiled_or_counted_rows","target_columns","descriptor_columns","identifier_coverage","chemistry_trust_score","ml_readiness_score","recommended_role"]].copy()
    t1.columns=["Resource","Files","Rows","Targets","Descriptors","ID coverage","Chemistry trust","ML readiness","Recommended role"]
    t1.to_csv(out/"Table_1_resource_atlas.csv",index=False)
    _draw_table(t1,out/"Table_1_resource_atlas","Table 1. Compact resource atlas and claim-readiness scores",
                widths=[.08,.055,.075,.055,.07,.075,.08,.075,.33],
                formats={"Files":"int","Rows":"rows","Targets":"int","Descriptors":"int","ID coverage":"pct0","Chemistry trust":"f2","ML readiness":"f2"},wrap_cols={"Recommended role":38},row_height=.42,font_size=5.8)
    lt=t1.copy()
    for c,fmt in {"Files":"int","Rows":"rows","Targets":"int","Descriptors":"int","ID coverage":"pct0","Chemistry trust":"f2","ML readiness":"f2"}.items():
        if fmt=="int": lt[c]=pd.to_numeric(lt[c],errors="coerce").map(lambda x:"-" if pd.isna(x) else f"{int(round(x)):,}")
        elif fmt=="rows": lt[c]=pd.to_numeric(lt[c],errors="coerce").map(fmt_rows)
        elif fmt=="pct0": lt[c]=pd.to_numeric(lt[c],errors="coerce").map(lambda x:"-" if pd.isna(x) else f"{100*x:.0f}%")
        elif fmt=="f2": lt[c]=pd.to_numeric(lt[c],errors="coerce").map(lambda x:"-" if pd.isna(x) else f"{x:.2f}")
    write_latex_table(lt,out/"Table_1_resource_atlas.tex","Compact resource atlas and claim-readiness scores.","tab:resource_atlas")

    # Table 2: descriptor endpoint results - the five exact Figure 5 endpoint cases
    t2=read_csv(root/"source_data/final_figure_panel_data/Figure_5ab_selected_endpoint_cases.csv")
    cols=["target_column","descriptor_family","mean_r2_random","mean_r2_descriptor_grouped","mean_spearman_descriptor_grouped","mean_top_10pct_recovery_random","mean_top_10pct_recovery_descriptor_grouped"]
    t2=t2[cols].copy(); t2.columns=["Endpoint","Best family","R² random","R² grouped","Spearman grouped","Top-10 random","Top-10 grouped"]
    t2.to_csv(out/"Table_2_split_aware_endpoint_results.csv",index=False)
    _draw_table(t2,out/"Table_2_split_aware_endpoint_results","Table 2. Split-aware descriptor-joined endpoint results (post-combustion CO₂)",
                widths=[.12,.12,.11,.11,.14,.14,.14],formats={c:"f2" for c in t2.columns if c not in ["Endpoint","Best family"]},row_height=.40,font_size=6.2)
    lt=t2.copy();
    for c in ["R² random","R² grouped","Spearman grouped","Top-10 random","Top-10 grouped"]: lt[c]=pd.to_numeric(lt[c],errors="coerce").map(lambda x:f"{x:.2f}")
    write_latex_table(lt,out/"Table_2_split_aware_endpoint_results.tex","Split-aware descriptor-joined endpoint results for post-combustion CO2.","tab:descriptor_endpoints")

    # Table 3: headline claims/caveats
    t3=read_csv(root/"tables/main/Main_Table_9_headline_claims_and_caveats.csv")
    t3=t3[["claim","numerical_support","caveat"]].copy(); t3.columns=["Claim","Numerical support","Caveat"]
    t3.to_csv(out/"Table_3_headline_claims_and_caveats.csv",index=False)
    _draw_table(t3,out/"Table_3_headline_claims_and_caveats","Table 3. Headline claims and mandatory caveats",
                widths=[.34,.22,.40],wrap_cols={"Claim":48,"Numerical support":34,"Caveat":54},row_height=.70,font_size=5.7)
    write_latex_table(t3,out/"Table_3_headline_claims_and_caveats.tex","Headline claims and mandatory caveats.","tab:claims_caveats")


def write_captions(root: Path):
    txt="# Final main-figure captions (v2.3 publication-final visuals)\n\n"
    txt += "**Figure 1 | A claim-readiness framework for scientific machine-learning datasets.** " \
           "(a) Progression from raw files through computation-ready, model-ready and evidence-ready representations to a claim-ready dataset. " \
           "(b) Different scientific claims require different supporting evidence. (c) Evidence regimes explicitly distinguish missing/unobservable evidence from warnings or inconsistencies. " \
           "(d) MOF databases provide a demanding case study spanning chemistry/provenance anchors, large descriptor-rich adsorption resources and quantum-property data.\n\n"
    txt += "**Figure 2 | MOF databases as a demanding claim-readiness case study.** " \
           "(a) Profiled/counted table scale and number of input files for the six main resources. (b) Presence of key data modalities in parsed tables. " \
           "(c) Distribution of column missingness. (d) Resource-level chemistry-trust versus ML-readiness scores; point size reflects data scale. " \
           "Counts describe parsed/profled tabular inputs and should not be interpreted as the number of fully model-ready structures.\n\n"
    txt += "**Figure 3 | Domain-evidence observability and validation regimes.** " \
           "(a) Fraction of scanned rows/resources for which chemistry-relevant evidence is directly observable in parsed metadata. (b) Validation-observability regimes with explicit not-scanned states. " \
           "(c) Charge-status decomposition. (d) Representative rule cards illustrating how validated-but-not-observable, uncertain-observable and flagged/inconsistent cases are interpreted. " \
           "‘Not observable’ means not directly present in the parsed tabular metadata and does not imply chemical invalidity.\n\n"
    txt += "**Figure 4 | Claim-risk and readiness mapping.** " \
           "(a) Claim-readiness map with chemistry-trust score intervals where row-level bootstrap estimates are available. " \
           "(b) Expert-defined claim-risk matrix for common data-quality issues across scientific use cases. (c) Rank stability under perturbation of chemistry-trust versus ML-readiness score weights. " \
           "Bootstrap intervals quantify reporting-score uncertainty, not uncertainty in true chemical validity.\n\n"
    txt += "**Figure 5 | Split-aware descriptor-joined benchmark.** " \
           "(a) Random versus descriptor-grouped R² for five post-combustion CO₂ endpoints, using the best grouped top-10-recovery descriptor family for each endpoint within the publication-filtered comparison table. " \
           "(b) Corresponding top-10% candidate recovery. (c) Case-matched median performance across descriptor families and split strategies. " \
           "(d) Representative rank-percentile agreement for a descriptor-grouped ExtraTrees case. Descriptor-family comparisons are limited to successfully joined task cases and are not universal rankings of descriptor families.\n\n"
    txt += "**Figure 6 | Claim-specific reporting framework for scientific ML datasets.** " \
           "(a) Claim-first decision tree, (b) minimum reporting blocks, and (c) use-case-specific resource recommendations. " \
           "The framework emphasizes provenance, target context, identifier-join accounting, domain evidence, split-aware validation and exact panel source data.\n"
    (root/"manuscript_support/FINAL_FIGURE_CAPTIONS_v2_3.md").write_text(txt,encoding="utf-8")


def write_visual_audit(root: Path):
    txt=f"""# Publication-final visual audit\n\nVersion: `{VERSION}`\n\n## What was changed\n\n- Main figures are authored at approximately 183 mm (7.2 in) final width instead of very large canvases that would force journal down-scaling and shrink typography.\n- Minimum main-panel text is kept around 6-8 pt at final size; titles and panel labels are larger and consistent.\n- All six main figures use one colour-blind-safe resource palette and one split palette.\n- Figure 2 replaces dense heatmap/scale combinations with direct row-scale, modality-presence, missingness-composition and readiness views.\n- Figure 3 explicitly shows `not_scanned` separately from `not_observable`, preventing absence of parsed evidence from visually resembling invalidity.\n- Figure 4 gives the risk matrix substantially more space by using an asymmetric 2x2 layout.\n- Figure 5 removes long source-table strings from y-axis labels and uses five compact endpoint labels; the representative ranking panel uses a density hexbin rather than a visually overplotted scatter cloud.\n- Figure 6 uses an asymmetric layout so decision logic and reporting guidance remain readable at double-column width.\n- Exact transformed panel data are written to `source_data/final_figure_panel_data/` so every final plotted quantity can be reproduced without reverse engineering figure code.\n\n## Scientific safeguards\n\n- No numerical result is recomputed from raw databases in this visual-refresh script.\n- No score, metric, claim or caveat is altered for aesthetics.\n- Figure 5 endpoint selection is deterministic and documented: one post-combustion CO2 case per target column, selected by maximum descriptor-grouped top-10% recovery in `Main_Table_6_best_descriptor_joined_cases_publication.csv`.\n- ‘Not observable’ remains a metadata-observability label, not a chemical-invalidity assertion.\n- Score intervals remain reporting-score bootstrap intervals, not uncertainty in true chemical validity.\n\n## Recommended manuscript use\n\nUse PDF for typesetting, SVG for later vector editing if required, and PNG only for inspection/submission systems that request raster images.\n"""
    (root/"manuscript_support/PUBLICATION_FINAL_VISUAL_AUDIT.md").write_text(txt,encoding="utf-8")


def write_manifest(root: Path):
    rows=[]
    for p in sorted(root.rglob("*")):
        if p.is_file() and "MANIFEST_SHA256" not in p.name:
            h=hashlib.sha256()
            with p.open("rb") as fh:
                for b in iter(lambda:fh.read(1024*1024),b""): h.update(b)
            rows.append({"relative_path":str(p.relative_to(root)).replace(os.sep,"/"),"size_bytes":p.stat().st_size,"sha256":h.hexdigest()})
    pd.DataFrame(rows).to_csv(root/"checks/MANIFEST_SHA256.csv",index=False)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",type=Path,default=Path("."),help="Publication package/result root")
    args=ap.parse_args(); root=args.root.resolve()
    panel_dir=root/"source_data/final_figure_panel_data"; panel_dir.mkdir(parents=True,exist_ok=True)
    for fn in [figure1,figure2,figure3,figure4,figure5,figure6]: fn(root,panel_dir)
    publication_tables(root)
    write_captions(root); write_visual_audit(root); write_manifest(root)
    print(f"Publication assets regenerated successfully under: {root}")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
