#!/usr/bin/env python
"""Paper 13 SI figure redesign/regeneration: S0 through S9.

All scientific values are read from frozen CSV outputs copied from All_Results.zip.
This script performs plotting/aggregation only; it does not rerun profiling, joins,
chemistry rules, scoring, or machine-learning models.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import math
import textwrap

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

from redesign_main_figures import setup_font, panel, clean
from config import *

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "si"
MAIN_DATA = ROOT / "data" / "main"
OUT = ROOT / "outputs" / "si"


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / name, low_memory=False)

def read_main(name: str) -> pd.DataFrame:
    return pd.read_csv(MAIN_DATA / name, low_memory=False)


def short(resource: str) -> str:
    return SHORT.get(resource, resource)


def wrap(s, width=44):
    return "\n".join(textwrap.wrap(str(s), width=width, break_long_words=False, break_on_hyphens=False))


def save(fig, stem: str):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=PNG_DPI, bbox_inches="tight")
    plt.close(fig)


def _resource_order(values):
    preferred = [
        "MOSAEC-DB", "CoRE MOF 2024", "CoRE MOF 2025 metadata",
        "ARC-MOF", "QMOF", "CSD-derived context", "Other/unknown"
    ]
    present = set(values)
    return [x for x in preferred if x in present] + sorted(present - set(preferred))


def s0():
    inv = read("SI_Table_S0_inventory_resource_role_format_summary.csv")
    local = read("SI_Table_S1_complete_file_level_profile.csv")
    res = (inv.groupby("resource", as_index=False).agg(inventory_files=("n_inventory_files", "sum"),inventory_mb=("inventory_size_mb", "sum")))
    res["inventory_gb"] = res["inventory_mb"] / 1024.0
    order = list(res.sort_values("inventory_gb", ascending=True)["resource"])
    roles = (inv.groupby("role", as_index=False).agg(files=("n_inventory_files", "sum"), mb=("inventory_size_mb", "sum")).sort_values("files", ascending=True))
    role_keep = roles.tail(9)
    role_matrix = inv.groupby(["resource", "role"])["n_inventory_files"].sum().unstack(fill_value=0)
    top_roles = list(roles.sort_values("files", ascending=False).head(8)["role"])
    role_matrix = role_matrix.reindex(_resource_order(role_matrix.index))[top_roles]
    local_counts = local.groupby("resource").size().rename("local_profiled_files")
    compare = res.set_index("resource")[["inventory_files"]].join(local_counts, how="left").fillna(0).reindex(order)

    fig, axs = plt.subplots(2, 2, figsize=(FULL_WIDTH_IN, 6.88)); a, b, c, d = axs.ravel()
    panel(a, "A"); clean(a, True, "x")
    rr = res.set_index("resource").loc[order].reset_index(); y = np.arange(len(rr))
    a.barh(y, rr.inventory_gb, color=[RESOURCE_COLORS.get(x, "#888888") for x in rr.resource], height=.62)
    a.set_yticks(y); a.set_yticklabels([short(x) for x in rr.resource]); a.set_xlabel("Inventory footprint (GB)"); a.set_title("Archive inventory footprint", fontweight="bold")
    for i, r in enumerate(rr.itertuples()): a.text(r.inventory_gb + max(rr.inventory_gb) * .012, i, f"{int(r.inventory_files):,} files", va="center", fontsize=7.6)

    panel(b, "B", x=-.16); b.set_title("Inventory file roles by resource", fontweight="bold")
    arr = np.log10(role_matrix.values.astype(float) + 1); cmap = LinearSegmentedColormap.from_list("role", ["#F5F7F8", "#B9DDE9", "#0072B2"])
    b.imshow(arr, aspect="auto", cmap=cmap); b.set_yticks(range(len(role_matrix))); b.set_yticklabels([short(x) for x in role_matrix.index]); b.set_xticks(range(len(top_roles))); b.set_xticklabels([x.replace("_", " ") for x in top_roles], rotation=37, ha="right", fontsize=7.4); b.tick_params(length=0)
    for sp in b.spines.values(): sp.set_visible(False)
    for i in range(role_matrix.shape[0]):
        for j in range(role_matrix.shape[1]):
            v = int(role_matrix.iloc[i, j])
            if v: b.text(j, i, f"{v:,}", ha="center", va="center", fontsize=6.3, color="white" if arr[i, j] > arr.max() * .55 else "#333333")

    panel(c, "C"); clean(c, True, "x"); yy = np.arange(len(role_keep))
    c.barh(yy, role_keep.files, color="#5F8FAF", height=.62); c.set_yticks(yy); c.set_yticklabels([x.replace("_", " ") for x in role_keep.role]); c.set_xscale("log"); c.set_xlabel("Inventory-listed files (log scale)"); c.set_title("Scientific roles in the archive", fontweight="bold")
    for i, r in enumerate(role_keep.itertuples()): c.text(r.files * 1.08, i, f"{int(r.files):,}", va="center", fontsize=7.5)

    panel(d, "D", x=-.17, y=1.20); clean(d, True, "x"); yy = np.arange(len(compare)); h = .30
    d.barh(yy + h/2, compare.inventory_files, height=h, color="#D0D0D0", label="inventory-listed files")
    d.barh(yy - h/2, compare.local_profiled_files, height=h,color=[RESOURCE_COLORS.get(x, "#888888") for x in compare.index], label="locally profiled data files")
    d.set_yticks(yy); d.set_yticklabels([short(x) for x in compare.index]); d.set_xscale("log"); d.set_xlabel("File count (log scale)"); d.set_title("Archive inventory vs local profiling inputs", fontweight="bold"); d.legend(frameon=False, fontsize=7.6, loc="lower right")

    fig.subplots_adjust(left=.15, right=.990, top=.94, bottom=.10, wspace=.50, hspace=.60)
    save(fig, "SI_Figure_S0_inventory_aware_input_map")

def s1():
    x = read("SI_Table_S1_complete_file_level_profile.csv")
    order = _resource_order(x.resource.unique())
    status = x.groupby(["resource", "read_status"]).size().unstack(fill_value=0).reindex(order)
    depth = x[x.read_status.eq("ok")].copy(); depth["profile_mode"] = np.where(depth["profile_is_full_file"].astype(str).str.lower().eq("true"), "full-file", "sampled")
    prof = depth.groupby(["resource", "profile_mode"]).size().unstack(fill_value=0).reindex(order).fillna(0)

    fig, (a, b) = plt.subplots(1, 2, figsize=(FULL_WIDTH_IN, 3.85))
    panel(a, "A",x=-.11,y=1.20); a.set_title("File loading status", fontweight="bold",pad=8)
    y = np.arange(len(order)); left = np.zeros(len(order)); status_colors = {"ok":"#009E73", "failed":"#D55E00", "skipped":"#999999"}
    for k in [c for c in ["ok", "failed", "skipped"] if c in status.columns]:
        vals = status[k].values.astype(float); a.barh(y, vals, left=left, height=.62, color=status_colors[k], label=k); left += vals
    a.set_yticks(y); a.set_yticklabels([short(r) for r in order]); a.invert_yaxis(); a.set_xlabel("Number of profiled data files"); a.legend(frameon=False, ncol=3, loc="lower right"); clean(a, True, "x")
    for i, total in enumerate(status.sum(axis=1).values): a.text(total + max(status.sum(axis=1))*0.015, i, f"n={int(total)}", va="center", fontsize=7.6)

    panel(b, "B",x=-.15,y=1.22); b.set_title("Profiling depth among successfully read files", fontweight="bold",pad=10)
    left = np.zeros(len(order))
    for k, col in [("full-file", "#0072B2"), ("sampled", "#E69F00")]:
        vals = prof[k].values if k in prof else np.zeros(len(order)); b.barh(y, vals, left=left, height=.62, color=col, label=k); left += vals
    b.set_yticks(y); b.set_yticklabels([short(r) for r in order]); b.invert_yaxis(); b.set_xlabel("Number of successfully read files"); b.legend(frameon=False, loc="lower right"); clean(b, True, "x")
    for i, r in enumerate(order):
        total = prof.loc[r].sum() if r in prof.index else 0; sampled = prof.loc[r, "sampled"] if (r in prof.index and "sampled" in prof.columns) else 0
        if total: b.text(total + max(prof.sum(axis=1))*0.015, i, f"{100*sampled/total:.0f}% sampled", va="center", fontsize=7.4)
    fig.subplots_adjust(left=.15, right=.985, top=.88, bottom=.17, wspace=.42)
    save(fig, "SI_Figure_S1_inventory_profile")

def s2():
    x = read("SI_Table_S2_complete_column_level_profile.csv")
    order = _resource_order(x.resource.unique())
    mod = (x.groupby(["resource", "inferred_modality"]).size().unstack(fill_value=0).reindex(order).fillna(0))
    # retain all categories, sorted by global prevalence
    cols = list(mod.sum(axis=0).sort_values(ascending=False).index)
    mod = mod[cols]
    miss = x.groupby("resource")["missing_fraction"].agg(["mean", "median"]).reindex(order)

    fig, (a, b) = plt.subplots(1, 2, figsize=(FULL_WIDTH_IN, 4.20), gridspec_kw={"width_ratios":[1.55, .95]})
    panel(a, "A", x=-.12)
    a.set_title("Column-type composition", fontweight="bold")
    arr = np.log10(mod.values.astype(float) + 1)
    cmap = LinearSegmentedColormap.from_list("mod", ["#F5F3F8", "#B5A4D6", "#4B2E83"])
    a.imshow(arr, aspect="auto", cmap=cmap)
    a.set_yticks(range(len(order))); a.set_yticklabels([short(r) for r in order])
    a.set_xticks(range(len(cols))); a.set_xticklabels([c.replace("_", " ") for c in cols], rotation=50, ha="right")
    a.tick_params(length=0)
    for sp in a.spines.values(): sp.set_visible(False)
    for i in range(mod.shape[0]):
        for j in range(mod.shape[1]):
            v = int(mod.iloc[i, j])
            if v >= 100:
                a.text(j, i, f"{v:,}", ha="center", va="center", fontsize=6.2,
                       color="white" if arr[i,j] > arr.max()*.55 else "#333")

    panel(b, "B", x=-.18)
    b.set_title("Column missingness by resource", fontweight="bold")
    y = np.arange(len(order))
    b.barh(y, miss["mean"], color=[RESOURCE_COLORS.get(r, "#888") for r in order], height=.58, label="mean")
    b.scatter(miss["median"], y, marker="D", s=28, color="#222222", zorder=3, label="median")
    b.set_yticks(y); b.set_yticklabels([short(r) for r in order]); b.invert_yaxis()
    b.set_xlim(0, max(.30, float(miss["mean"].max())*1.12))
    b.set_xlabel("Missing fraction across profiled columns")
    b.legend(frameon=False, loc="lower right")
    clean(b, True, "x")

    fig.subplots_adjust(left=.13, right=.985, top=.90, bottom=.25, wspace=.43)
    save(fig, "SI_Figure_S2_missingness_column_types")


def s3():
    x = read("SI_Table_S3_join_accounting.csv")
    for c in ["intersection_ids_profiled", "left_retention_fraction", "right_retention_fraction"]: x[c] = pd.to_numeric(x[c], errors="coerce")
    x["pair"] = x["left_file"].astype(str) + " ↔ " + x["right_file"].astype(str)
    idx = x.groupby("pair")["intersection_ids_profiled"].idxmax(); u = x.loc[idx].sort_values("intersection_ids_profiled", ascending=False).head(18).copy()
    u["min_retention"] = u[["left_retention_fraction", "right_retention_fraction"]].min(axis=1); u = u.sort_values("intersection_ids_profiled", ascending=True)

    fig, ax = plt.subplots(figsize=(FULL_WIDTH_IN, 5.20)); clean(ax, True, "x")
    y = np.arange(len(u)); ret=u.min_retention.fillna(0).values
    cmap=LinearSegmentedColormap.from_list('join_retention',['#9FD8D2','#3E9FA6','#145A72'])
    denom=max(1e-9,float(np.nanmax(ret)-np.nanmin(ret))); cols=[cmap((v-float(np.nanmin(ret)))/denom) for v in ret]
    ax.barh(y, u.intersection_ids_profiled, color=cols, height=.62)
    ax.set_yticks(y); ax.set_yticklabels([wrap(p, 31) for p in u.pair], fontsize=7.6); ax.set_xlabel("Profiled shared identifiers")
    mx = float(u.intersection_ids_profiled.max())
    for i, r in enumerate(u.itertuples()):
        retv = r.min_retention; txt = f"{int(r.intersection_ids_profiled):,} · min retention {100*retv:.0f}%" if pd.notna(retv) else f"{int(r.intersection_ids_profiled):,}"
        ax.text(r.intersection_ids_profiled + mx*.012, i, txt, va="center", fontsize=7.2, color="#404040")
    fig.subplots_adjust(left=.37, right=.970, top=.985, bottom=.10)
    save(fig, "SI_Figure_S3_join_accounting")

def s4():
    x = read("SI_Table_S8_trust_flags_by_metal.csv")
    x["n"] = pd.to_numeric(x["n"], errors="coerce").fillna(0)
    totals = x.groupby("metal_family_primary")["n"].sum().sort_values(ascending=False)
    top = totals.head(18).sort_values(ascending=True)
    top_metals = [m for m in totals.index if m != "not_observable"][:10]
    comp = (x[x.metal_family_primary.isin(top_metals)]
            .groupby(["metal_family_primary", "trust_regime_row"])["n"].sum().unstack(fill_value=0))
    comp = comp.reindex(top_metals)
    comp_frac = comp.div(comp.sum(axis=1), axis=0).fillna(0)

    fig, (a, b) = plt.subplots(1, 2, figsize=(FULL_WIDTH_IN, 5.05), gridspec_kw={"width_ratios":[.95,1.25]})
    panel(a, "A", x=-.16)
    clean(a, True, "x")
    y = np.arange(len(top))
    colors = ["#B5B5B5" if m == "not_observable" else "#2878B5" for m in top.index]
    a.barh(y, top.values, color=colors, height=.62)
    a.set_yticks(y); a.set_yticklabels(top.index)
    a.set_xlabel("Profiled rows")
    a.set_title("Metal / chemistry coverage", fontweight="bold")
    mx = float(top.max())
    for i, v in enumerate(top.values): a.text(v+mx*.012, i, f"{int(v):,}", va="center", fontsize=7.2)

    panel(b, "B", x=-.14)
    b.set_title("Trust-regime composition for common observable metals", fontweight="bold")
    keys = ["validated_observable", "validated_not_observable", "uncertain_observable", "uncertain_unobservable", "flagged_or_inconsistent"]
    y = np.arange(len(comp_frac)); left=np.zeros(len(comp_frac))
    for k in keys:
        if k not in comp_frac.columns: continue
        vals=comp_frac[k].values
        b.barh(y, vals, left=left, height=.62, color=TRUST_COLORS.get(k,"#999"), label=k.replace("_"," "))
        left += vals
    b.set_yticks(y); b.set_yticklabels(comp_frac.index); b.invert_yaxis(); b.set_xlim(0,1)
    b.set_xlabel("Fraction of profiled rows")
    b.legend(frameon=False, fontsize=7.0, ncol=2, loc="upper center", bbox_to_anchor=(.52,-.12), columnspacing=.8)

    fig.subplots_adjust(left=.10, right=.985, top=.91, bottom=.18, wspace=.37)
    save(fig, "SI_Figure_S4_coverage_profiles")


def s5():
    x = read("Figure_4c_score_sensitivity.csv").sort_values("mean_rank", ascending=True)
    fig, ax = plt.subplots(figsize=(FULL_WIDTH_IN, 4.05)); clean(ax, True, "both")
    for _, r in x.iterrows():
        col = RESOURCE_COLORS.get(r.resource, "#777777")
        xerr = np.array([[r.mean_rank-r.min_rank], [r.max_rank-r.mean_rank]])
        yerr = np.array([[r.sd_combined_score], [r.sd_combined_score]])
        ax.errorbar(r.mean_rank, r.mean_combined_score, xerr=xerr, yerr=yerr,fmt='o', ms=7, color=col, ecolor=col, elinewidth=1.3, capsize=3,markeredgecolor='white', markeredgewidth=.7, zorder=3)
    ax.set_xlabel("Mean sensitivity rank across score-weight perturbations (1 = highest)"); ax.set_ylabel("Mean combined score"); ax.set_xlim(.65, 7.2)
    handles=[Line2D([0],[0],marker='o',linestyle='None',markersize=6.5,markerfacecolor=RESOURCE_COLORS.get(r.resource,'#777'),markeredgecolor='white',label=short(r.resource)) for r in x.itertuples()]
    ax.legend(handles=handles,frameon=False,ncol=4,loc='lower center',bbox_to_anchor=(.5,1.015),columnspacing=1.0,handletextpad=.35,borderaxespad=0,fontsize=7.6)
    fig.subplots_adjust(left=.12, right=.985, top=.86, bottom=.15)
    save(fig, "SI_Figure_S5_trust_score_sensitivity")

def s6():
    x = read("SI_Table_S6_complete_model_results.csv")
    x["r2"] = pd.to_numeric(x["r2"], errors="coerce")
    combos = []
    # preserve stable order by split then model
    model_order = ["dummy", "ridge", "hgb", "randomforest", "extratrees"]
    split_order = ["descriptor_grouped", "random"]
    for sp in split_order:
        for m in model_order:
            if ((x.split_type == sp) & (x.model == m)).any(): combos.append((sp,m))
    labels = [f"{'Grouped' if sp=='descriptor_grouped' else 'Random'}\n{'ExtraTrees' if m=='extratrees' else ('RF' if m=='randomforest' else m.upper() if m=='hgb' else m.title())}" for sp,m in combos]
    groups = [x.loc[(x.split_type==sp)&(x.model==m), "r2"].dropna().values for sp,m in combos]

    fig, (a, b) = plt.subplots(2, 1, figsize=(FULL_WIDTH_IN, 6.05), gridspec_kw={"height_ratios":[1.35,.75]})
    panel(a, "A", x=-.07, y=1.05)
    try:
        _mv = tuple(int(part) for part in matplotlib.__version__.split('.')[:2])
    except Exception:
        _mv = (99, 0)
    _label_kw = {"tick_labels": labels} if _mv >= (3, 9) else {"labels": labels}
    bp = a.boxplot(groups, patch_artist=True, showfliers=False, widths=.62,
                   medianprops={"color":"#333333","linewidth":1.2},
                   whiskerprops={"color":"#777777"}, capprops={"color":"#777777"},
                   **_label_kw)
    for patch,(sp,m) in zip(bp['boxes'],combos):
        patch.set_facecolor(SPLIT_COLORS.get(sp, "#999999")); patch.set_alpha(.55); patch.set_edgecolor("#555555")
    a.axhline(0, color="#777777", lw=.8, ls='--')
    a.set_ylim(-1.0, 1.02)
    a.set_ylabel("R²")
    a.set_title("Repeated-split performance distributions", fontweight="bold")
    a.tick_params(axis='x', labelrotation=32)
    for lab in a.get_xticklabels(): lab.set_ha('right')
    clean(a, True, "y")
    xtrans = a.get_xaxis_transform()
    for i,g in enumerate(groups, start=1):
        a.text(i, .985, f"n={len(g)}", transform=xtrans, ha='center', va='top', fontsize=6.9, color='#555')

    panel(b, "B", x=-.07, y=1.08)
    neg = [np.mean(g < 0) if len(g) else np.nan for g in groups]
    severe = [np.mean(g < -1) if len(g) else np.nan for g in groups]
    pos=np.arange(len(groups)); w=.36
    b.bar(pos-w/2, neg, width=w, color="#56B4E9", label="R² < 0")
    b.bar(pos+w/2, severe, width=w, color="#D55E00", label="R² < -1")
    b.set_xticks(pos); b.set_xticklabels(labels, rotation=32, ha='right')
    b.set_ylabel("Fraction of runs")
    b.set_ylim(0, max(.10, max(neg+severe)*1.20))
    b.set_title("Lower-tail failure frequency", fontweight="bold")
    b.legend(frameon=False, ncol=2, loc='upper right')
    clean(b, True, "y")
    fig.subplots_adjust(left=.10, right=.985, top=.94, bottom=.10, hspace=.48)
    save(fig, "SI_Figure_S6_ml_repeated_splits")


def s7():
    x = read("SI_Table_S1_complete_file_level_profile.csv")
    g = (x.groupby("resource")[["constant_columns", "all_missing_columns"]].sum()
         .reindex(_resource_order(x.resource.unique())).fillna(0))
    fig, (a,b) = plt.subplots(1,2,figsize=(FULL_WIDTH_IN,3.85))
    y=np.arange(len(g))
    panel(a,"A")
    a.barh(y,g.constant_columns,color=[RESOURCE_COLORS.get(r,"#777") for r in g.index],height=.62)
    a.set_yticks(y); a.set_yticklabels([short(r) for r in g.index]); a.invert_yaxis(); a.set_xlabel("Summed constant-column count")
    a.set_title("Constant columns in profiled files",fontweight='bold'); clean(a,True,'x')
    panel(b,"B")
    b.barh(y,g.all_missing_columns,color="#D55E00",height=.62)
    b.set_yticks(y); b.set_yticklabels([short(r) for r in g.index]); b.invert_yaxis(); b.set_xlabel("Summed all-missing-column count")
    b.set_title("All-missing columns in profiled files",fontweight='bold'); clean(b,True,'x')
    fig.subplots_adjust(left=.15,right=.985,top=.90,bottom=.14,wspace=.40)
    save(fig,"SI_Figure_S7_duplicate_leakage")


def _pick_card(group: pd.DataFrame) -> pd.Series:
    # Deterministic display-oriented representative: prefer a non-empty, compact ID,
    # then closeness to the class median trust score, then source file / ID.
    g=group.copy()
    g["chemistry_trust_score_row"] = pd.to_numeric(g["chemistry_trust_score_row"], errors="coerce")
    g["id_text"] = g["primary_id"].fillna("").astype(str).str.strip()
    g=g[g["id_text"].ne("")].copy() if g["id_text"].ne("").any() else g.copy()
    med=g["chemistry_trust_score_row"].median(); g["dist"]=(g["chemistry_trust_score_row"]-med).abs(); g["id_len"]=g["id_text"].str.len()
    return g.sort_values(["id_len","dist","source_file","id_text"], na_position='last').iloc[0]

def s8():
    x=read("SI_Table_S15_representative_rule_cards.csv")
    class_order=["validated_not_observable","uncertain_observable","uncertain_unobservable","flagged_or_inconsistent"]
    cards=[]
    for cls in class_order:
        g=x[x.rule_card_class.eq(cls)]
        if not g.empty: cards.append(_pick_card(g))

    fig,axs=plt.subplots(2,2,figsize=(FULL_WIDTH_IN,5.55)); axs=axs.ravel()
    header_cols={'validated_not_observable':'#4E9CC9','uncertain_observable':'#D8A137','uncertain_unobservable':'#7B7F94','flagged_or_inconsistent':'#C46F4B'}
    for i,(ax,r) in enumerate(zip(axs,cards)):
        panel(ax, chr(ord('A')+i), x=-.045, y=1.08); ax.axis('off')
        cls=str(r.rule_card_class); col=header_cols.get(cls,"#777777")
        ax.add_patch(FancyBboxPatch((.025,.025),.95,.93,boxstyle="round,pad=.012,rounding_size=.024",fc="#FCFCFC",ec=col,lw=1.0))
        ax.add_patch(FancyBboxPatch((.025,.825),.95,.13,boxstyle="round,pad=.012,rounding_size=.024",fc=col,ec=col,lw=1.0,alpha=.22))
        ax.text(.07,.888,cls.replace('_',' '),fontweight='bold',fontsize=9.0,va='center',color='#222')

        resource=short(str(r.resource)); source=str(r.source_file); rid=str(r.primary_id) if pd.notna(r.primary_id) else 'not available'
        metal=str(r.metals_detected) if pd.notna(r.metals_detected) and str(r.metals_detected).strip() else 'not directly observed'
        trust=float(r.chemistry_trust_score_row) if pd.notna(r.chemistry_trust_score_row) else np.nan; obs=float(r.observability_score) if pd.notna(r.observability_score) else np.nan
        ax.text(.07,.765,"Resource",fontweight='bold',fontsize=7.7,va='center'); ax.text(.31,.765,resource,fontsize=7.7,va='center',color='#333')
        ax.text(.56,.765,"Metal",fontweight='bold',fontsize=7.7,va='center'); ax.text(.72,.765,metal,fontsize=7.7,va='center',color='#333')
        ax.text(.07,.690,"Source",fontweight='bold',fontsize=7.7,va='center'); ax.text(.31,.690,wrap(source,30),fontsize=7.5,va='center',color='#333')
        ax.text(.07,.600,"Identifier",fontweight='bold',fontsize=7.7,va='top')
        ax.add_patch(FancyBboxPatch((.285,.515),.64,.115,boxstyle="round,pad=.008,rounding_size=.010",fc='#F4F6F7',ec='#DFE3E5',lw=.5))
        ax.text(.305,.608,wrap(rid,44),fontsize=7.0,va='top',color='#2F2F2F',linespacing=1.10)
        ax.text(.07,.445,"Chemistry trust",fontweight='bold',fontsize=7.25,va='center'); ax.text(.39,.445,f"{trust:.2f}" if np.isfinite(trust) else "n/a",fontsize=7.6,va='center',ha='left')
        ax.text(.59,.445,"Observability",fontweight='bold',fontsize=7.25,va='center'); ax.text(.87,.445,f"{obs:.2f}" if np.isfinite(obs) else "n/a",fontsize=7.6,va='center',ha='left')
        ax.text(.07,.355,"Interpretation",fontweight='bold',fontsize=7.7,va='top')
        ax.text(.07,.303,wrap(r.interpretation,51),fontsize=7.15,va='top',color='#404040',linespacing=1.17)
    for ax in axs[len(cards):]: ax.axis('off')
    fig.subplots_adjust(left=.035,right=.990,top=.975,bottom=.035,wspace=.14,hspace=.12)
    save(fig,"SI_Figure_S8_rule_cards")

def s9():
    """Decision/reporting framework moved from former main Figure 6 to SI Figure S9."""
    rec=read_main('Figure_6c_use_case_recommendations.csv')
    fig=plt.figure(figsize=(FULL_WIDTH_IN,6.00))
    gs=fig.add_gridspec(2,2,height_ratios=[.92,1.08],width_ratios=[1.02,.98],hspace=.28,wspace=.30)
    a=fig.add_subplot(gs[0,0]); b=fig.add_subplot(gs[0,1]); c=fig.add_subplot(gs[1,:])

    # A: complete the evidence path first, then branch cleanly to three possible outcomes.
    a.axis('off'); panel(a,'A',-.02,1.18); a.set_title('Claim-first evidence path',fontweight='bold',pad=7)
    step_cols=['#E8EEF5','#DCEBF3','#E2F0E9','#F2E7D8']
    steps=[('1  Define claim','ranking · process · quantum ·\ngeneration · mechanism'),
           ('2  Verify target context','property · gas · unit ·\nconditions'),
           ('3  Check evidence state','observable · validated ·\nmissing · flagged'),
           ('4  Stress-test evaluation','joins · duplicates ·\ngrouped split')]
    ys=[.835,.605,.375,.145]
    for i,((title,sub),yy) in enumerate(zip(steps,ys)):
        a.add_patch(FancyBboxPatch((.025,yy-.088),.555,.182,boxstyle='round,pad=.010,rounding_size=.016',fc=step_cols[i],ec='#AEB8C0',lw=.6))
        a.text(.066,yy+.043,title,fontweight='bold',fontsize=8.05,va='center')
        a.text(.066,yy-.020,sub,fontsize=7.00,color='#3F4448',va='center',linespacing=1.12)
        if i<len(ys)-1:
            a.add_patch(FancyArrowPatch((.302,yy-.095),(.302,ys[i+1]+.102),arrowstyle='-|>',mutation_scale=9,lw=.8,color='#7A8085'))

    outcomes=[('SUPPORTED\nSCOPE','#CDE9DF','#286F5F'),('NARROW /\nCAVEAT','#F8E5B6','#8A6420'),('EVIDENCE\nTO OBTAIN','#D5E7F4','#2A6288')]
    oys=[.67,.40,.13]
    for (txt,fc,ec),yy in zip(outcomes,oys):
        a.add_patch(FancyBboxPatch((.730,yy-.078),.240,.156,boxstyle='round,pad=.010,rounding_size=.018',fc=fc,ec=ec,lw=.8))
        a.text(.850,yy,txt,ha='center',va='center',fontweight='bold',fontsize=7.50,color='#25313A',linespacing=1.02)

    # One completed assessment leads to a clear branch spine and three explicit outcomes.
    trunk_x=.665
    start_x=.592
    start_y=ys[-1]
    a.text(.850,.805,'ASSESSMENT OUTCOMES',ha='center',va='center',fontsize=6.45,fontweight='bold',color='#6A7075')
    a.add_patch(FancyArrowPatch((start_x,start_y),(trunk_x,start_y),arrowstyle='-|>',mutation_scale=9,lw=.8,color='#7A8085'))
    a.plot([trunk_x,trunk_x],[oys[-1],oys[0]],color='#7A8085',lw=.8,solid_capstyle='round')
    a.plot([trunk_x],[start_y],marker='o',ms=2.8,color='#7A8085')
    for oy in oys:
        a.add_patch(FancyArrowPatch((trunk_x,oy),(.718,oy),arrowstyle='-|>',mutation_scale=8.5,lw=.8,color='#7A8085'))
    # B: slightly taller cards and shorter sublabels.
    b.axis('off'); panel(b,'B',-.02,1.18); b.set_title('Minimum reporting blocks',fontweight='bold',pad=7)
    items=[('File provenance','version · size · access','#DCEBF3'),('Rows + columns','missingness','#E3F0EA'),
           ('Target roles','label · coordinate\nmetadata','#F7E6BA'),('Target context','gas · property · unit','#EADFF2'),
           ('Identifiers + joins','normalize · retain','#DCEBF3'),('Evidence regimes','validation · observability','#E3F0EA'),
           ('Duplicate / leakage','policy · grouped splits','#F7E6BA'),('Figure source data','exact panel inputs','#EADFF2')]
    pos=[(.02,.735),(.515,.735),(.02,.505),(.515,.505),(.02,.275),(.515,.275),(.02,.045),(.515,.045)]
    for i,((title,sub,fc),(x0,y0)) in enumerate(zip(items,pos),1):
        b.add_patch(FancyBboxPatch((x0,y0),.455,.185,boxstyle='round,pad=.010,rounding_size=.013',fc=fc,ec='#C5C9CC',lw=.55))
        b.text(x0+.040,y0+.126,str(i),fontweight='bold',fontsize=8.0,va='center')
        b.text(x0+.095,y0+.126,title,fontweight='bold',fontsize=7.45,va='center')
        b.text(x0+.095,y0+.061,sub,fontsize=6.80,color='#444',va='center',linespacing=1.02)

    c.axis('off'); panel(c,'C',-.005,1.16); c.set_title('Use-case recommendations and minimum evidence',fontweight='bold',pad=7)
    c.text(.255,.928,'Resource logic',fontweight='bold',fontsize=8.0,color='#333'); c.text(.690,.928,'Minimum evidence',fontweight='bold',fontsize=8.0,color='#333')
    short_rec={
        'Adsorption ranking': ('ARC-MOF adsorption/process data; explicit target context; grouped splits; chemistry-observability caveats.', 'gas/pressure/T/property/unit · joins · top-k stability · trust regime'),
        'Process screening': ('Use process tables only after target definitions and leakage-prone cycle descriptors are documented.', 'cycle target definition · row retention · duplicate policy · grouped split'),
        'Quantum-property prediction': ('Use QMOF/DFT-consistent resources when electronic-structure consistency is central.', 'DFT settings · units · failed calculations · chemistry flags'),
        'Generative-model training': ('Use chemistry-validated, duplicate-controlled subsets; keep not-observable cases explicit.', 'validity filters · duplicate groups · charge/oxidation warnings'),
        'Mechanistic interpretation': ('Prioritize chemically interpretable subsets over maximum resource size.', 'rule cards · manual-check examples'),
    }
    row_cols=['#E9F2F7','#E8F3EE','#F8EAC5','#EEE5F4','#F8E5DC']
    y=.845
    for idx,(_,r) in enumerate(rec.iterrows()):
        logic,minimum=short_rec.get(str(r.use_case),(str(r.preferred_resource_logic),str(r.minimum_reporting)))
        c.add_patch(FancyBboxPatch((.015,y-.128),.97,.125,boxstyle='round,pad=.008,rounding_size=.012',fc=row_cols[idx%len(row_cols)],ec='#D0D0D0',lw=.45))
        c.text(.045,y-.066,wrap(str(r.use_case),19),fontweight='bold',fontsize=7.7,va='center',linespacing=1.05)
        c.text(.255,y-.061,wrap(logic,58),fontsize=7.05,va='center',color='#373D40',linespacing=1.08)
        c.text(.690,y-.061,wrap(minimum,43),fontsize=7.05,va='center',color='#373D40',linespacing=1.08)
        y-=.145
    fig.subplots_adjust(left=.045,right=.990,top=.945,bottom=.050)
    save(fig,'SI_Figure_S9_decision_framework')

FUN={"S0":s0,"S1":s1,"S2":s2,"S3":s3,"S4":s4,"S5":s5,"S6":s6,"S7":s7,"S8":s8,"S9":s9}


def main():
    ap=argparse.ArgumentParser(description="Regenerate Paper 13 SI figures S0-S9 from frozen CSV outputs; S9 is the former main Figure 6 decision framework.")
    ap.add_argument("--figures",nargs="*",default=list(FUN),choices=list(FUN))
    ap.add_argument("--allow-font-fallback",action="store_true",help="QA only: permit DejaVu Sans when Arial is unavailable.")
    args=ap.parse_args()
    family,resolved=setup_font(args.allow_font_fallback)
    print("[Paper13 PATCH v6] SI generator loaded")
    print(f"[Paper13] font={family} resolved={resolved}")
    for f in args.figures:
        print(f"[Paper13] Rendering SI {f}")
        FUN[f]()
    print(f"[Paper13] SI outputs written to: {OUT}")

if __name__=="__main__":
    main()
