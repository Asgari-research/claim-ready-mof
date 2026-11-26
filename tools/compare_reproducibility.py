#!/usr/bin/env python3
"""Compare a pipeline rerun against the archived figure-source tables.

For every archived table in ``reproduce/figures/data/{main,si}`` this finds the rerun output with the
same name (CSV preferred, then Parquet, then Excel), aligns rows on their identifying columns, compares
every shared numeric value with a tolerance, and writes:

- ``<report-dir>/reproducibility_summary.csv``  one row per archived table
- ``<report-dir>/diffs/<table>.csv``             the largest differences per table
- ``<report-dir>/REPRODUCIBILITY_REPORT.md``     a readable summary

Machine-specific paths are normalised (``...\\data_raw\\x`` and ``/home/.../data_raw/x`` both become
``<DATA_ROOT>/data_raw/x``) so that path differences alone are not counted. Differences in rows that
mention a known input gap are attributed to it.

Usage:
    python tools/compare_reproducibility.py --rerun RUN_DIR --report-dir RUN_DIR/reproducibility
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "reproduce" / "figures" / "data"
SUFFIX_ORDER = {".csv": 0, ".parquet": 1, ".xlsx": 2}
KNOWN_GAPS = {
    "CSD MOF Collection: newer public release": re.compile(r"csd_mof_collection|CSD_MOF_Collection|framework details|charged frameworks|suspect chemistry|hydrogen added|CSD-derived", re.I),
    "private inventory file not public": re.compile(r"PROJECT_DATA_ARCHIVE_STRUCTURE", re.I),
}
# Tables that only exist when the private archive inventory file is present.
BLOCKED = {
    "SI_Table_S0_inventory_resource_role_format_summary": "needs PROJECT_DATA_ARCHIVE_STRUCTURE_AND_CONTENTS.txt (private)",
    "SI_Table_S0b_inventory_folder_summary": "needs PROJECT_DATA_ARCHIVE_STRUCTURE_AND_CONTENTS.txt (private)",
    "SI_Table_S0c_inventory_example_files": "needs PROJECT_DATA_ARCHIVE_STRUCTURE_AND_CONTENTS.txt (private)",
}
# Row keys known to identify rows in each archived table (used when unique in both tables).
KEY_HINTS = {
    "Figure_2a_resource_summary": ["resource"],
    "Figure_2b_modality_presence": ["resource"],
    "Figure_2c_missingness_composition": ["resource", "missingness_bin"],
    "Figure_2d_readiness_scores": ["resource"],
    "Figure_3a_domain_evidence_observability": ["resource"],
    "Figure_3b_validation_observability_regimes": ["resource", "trust_regime"],
    "Figure_3c_charge_status": ["resource", "charge_status"],
    "Figure_3d_representative_rule_cards": ["rule_card_class"],
    "Figure_4a_claim_readiness_map": ["resource"],
    "Figure_4b_claim_risk_matrix": ["risk_issue", "use_case"],
    "Figure_4c_score_weight_sensitivity": ["resource"],
    "Figure_4c_score_sensitivity": ["resource"],
    "Figure_5ab_selected_endpoint_cases": ["target_column", "task", "gas", "descriptor_family", "source_table"],
    "Figure_5c_descriptor_family_comparison": ["index"],
    "Figure_5d_ranking_percentile": ["source_table", "target_column", "descriptor_family", "model", "split_type", "repeat", "y_true"],
    "Figure_6a_decision_rules": ["step"],
    "Figure_6b_reporting_checklist": ["item"],
    "Figure_6c_use_case_recommendations": ["use_case"],
    "SI_headline_endpoint_repeated_summary": ["source_table", "target_column", "feature_source", "descriptor_family", "model", "split_type"],
    "SI_Table_S1_complete_file_level_profile": ["relative_path"],
    "SI_Table_S2_complete_column_level_profile": ["relative_path", "column_name"],
    "SI_Table_S3_join_accounting": ["left_path", "left_id_column", "right_path", "right_id_column"],
    "SI_Table_S6_complete_model_results": ["source_table", "target_column", "model", "split_type", "feature_source", "descriptor_family", "repeat"],
    "SI_Table_S8_trust_flags_by_metal": ["resource", "metal_family_primary", "trust_regime_row"],
    "SI_Table_S15_representative_rule_cards": ["rule_card_class", "resource", "source_file", "primary_id"],
}
PUB_SCRIPT = ROOT / "src" / "make_publication_assets_v2_3.py"
PREFERRED_DIR = "final_figure_panel_data"   # where make_publication_assets_v2_3.py writes panel tables
PATH_RE = re.compile(r"^.*?[\\/](data_raw[\\/].*)$")
RTOL, ATOL = 1e-6, 1e-9          # "identical" tolerance for floats written by different library builds
CLOSE_RTOL = 1e-3                # "close" tolerance


def norm_text(v):
    if pd.isna(v):
        return v
    s = str(v).replace("\\", "/")
    m = PATH_RE.match(s)
    if m:
        s = "<DATA_ROOT>/" + m.group(1)
    return s.replace("<DATA_ROOT>/<DATA_ROOT>/", "<DATA_ROOT>/")


def load(path: Path) -> pd.DataFrame:
    if path.suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_excel(path)


def index_outputs(rerun: Path):
    idx = {}
    for p in rerun.rglob("*"):
        if p.suffix.lower() in SUFFIX_ORDER and p.is_file() and "reproducibility" not in p.parts:
            idx.setdefault(p.stem.lower(), []).append(p)
    for k in idx:
        idx[k].sort(key=lambda p: (0 if PREFERRED_DIR in p.parts else 1, SUFFIX_ORDER[p.suffix.lower()], len(p.parts), str(p)))
    return idx


def numeric_cols(df: pd.DataFrame, cols):
    out = []
    for c in cols:
        s = df[c]
        if pd.api.types.is_bool_dtype(s):
            continue
        conv = pd.to_numeric(s, errors="coerce")
        nonnull = s.notna().sum()
        if nonnull and conv.notna().sum() >= 0.9 * nonnull:
            out.append(c)
    return out


VOLATILE_RE = re.compile(r"(^|_)(seconds|secs|time|times|timestamp|date|created|modified|elapsed|duration|runtime)($|_)", re.I)
IDLIKE_RE = re.compile(r"path|file|name|^id$|_id$|key|table|target|descriptor|model|split|repeat|resource|family|metal|"
                       r"column|panel|rule|endpoint|case|feature|regime|status|claim|use|step|block|label|category|role", re.I)


def volatile(cols):
    """Columns that legitimately differ between machines and runs (timings, file dates)."""
    return [c for c in cols if VOLATILE_RE.search(str(c))]


def choose_keys(a: pd.DataFrame, b: pd.DataFrame, common, num, name=None):
    hint = [c for c in KEY_HINTS.get(name, []) if c in common]
    if hint and not a.duplicated(hint).any() and not b.duplicated(hint).any():
        return hint
    vol = set(volatile(common))
    text = [c for c in common if c not in num and c not in vol]
    ranked = sorted(text, key=lambda c: (0 if IDLIKE_RE.search(str(c)) else 1,
                                         a[c].astype(str).str.len().mean()))
    small_int = []
    for c in num:
        if c in vol:
            continue
        v = pd.to_numeric(a[c], errors="coerce").dropna().to_numpy(float)
        if len(v) and np.all(v == np.round(v)) and a[c].nunique() <= 50:
            small_int.append(c)

    def ok(keys):
        if not keys or a.duplicated(keys).any() or b.duplicated(keys).any():
            return False
        def joined(df):
            return df[keys].astype(object).where(df[keys].notna(), "").astype(str).agg("|".join, axis=1)
        shared = joined(a).isin(set(joined(b)))
        return shared.mean() >= 0.5

    for c in ranked:
        if ok([c]):
            return [c]
    head = ranked[:6] + small_int
    for i in range(len(head)):
        for j in range(i + 1, len(head)):
            if ok([head[i], head[j]]):
                return [head[i], head[j]]
    for keys in (text, text + small_int):
        if ok(keys):
            return keys
    return None


HEADLINE = "SI_headline_endpoint_repeated_summary"
HEADLINE_KEYS = ["source_table", "target_column", "feature_source", "descriptor_family", "model", "split_type"]


def derive_headline(archived: pd.DataFrame, s6: pd.DataFrame) -> pd.DataFrame:
    """The archived headline table is the per-endpoint mean/SD of S6 over repeats (no script in the
    repository writes it). Rebuild it from the rerun's S6 for the endpoints listed in the archive."""
    keys = [k for k in HEADLINE_KEYS if k in s6.columns]
    metrics = ["rmse", "mae", "r2", "spearman", "top_1pct_recovery", "top_5pct_recovery",
               "top_10pct_recovery", "ndcg_10pct"]
    agg = {"n_repeats": ("repeat", "nunique")}
    for m in metrics:
        if m in s6.columns:
            agg[f"mean_{m}"] = (m, "mean")
    for m in ("rmse", "r2"):
        if m in s6.columns:
            agg[f"sd_{m}"] = (m, "std")
    for c in ["task", "gas", "target_property", "target_family", "n_train", "n_test",
              "n_numeric_features", "n_categorical_features"]:
        if c in s6.columns:
            agg[c] = (c, "first")
    g = s6.groupby(keys, dropna=False).agg(**agg).reset_index()
    sel = archived[keys].astype(str)
    g = g[g[keys].astype(str).apply(tuple, axis=1).isin(set(sel.apply(tuple, axis=1)))]
    return g[[c for c in archived.columns if c in g.columns]]


def attribute(texts) -> list[str]:
    blob = " ".join(map(str, texts))
    return [name for name, rx in KNOWN_GAPS.items() if rx.search(blob)]


def compare_table(name: str, old: pd.DataFrame, new: pd.DataFrame):
    old = old.copy(); new = new.copy()
    for df in (old, new):
        for c in df.columns:
            if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c]):
                df[c] = df[c].map(norm_text)
    common = [c for c in old.columns if c in new.columns]
    vol = volatile(common)
    num = [c for c in numeric_cols(old, common) if c in numeric_cols(new, common) and c not in vol]
    res = {
        "table": name, "rows_archived": len(old), "rows_rerun": len(new),
        "cols_archived": old.shape[1], "cols_rerun": new.shape[1],
        "cols_only_archived": ";".join(c for c in old.columns if c not in new.columns),
        "cols_only_rerun": ";".join(c for c in new.columns if c not in old.columns),
        "ignored_volatile_cols": ";".join(vol),
    }
    diffs = []
    keys = choose_keys(old, new, common, num, name)
    if keys:
        m = old[keys + [c for c in common if c not in keys]].merge(
            new[keys + [c for c in common if c not in keys]], on=keys, how="outer",
            suffixes=("__a", "__r"), indicator=True)
        only_a, only_r = m[m["_merge"] == "left_only"], m[m["_merge"] == "right_only"]
        both = m[m["_merge"] == "both"]
        res.update(alignment="keys:" + ",".join(keys), rows_matched=len(both),
                   rows_only_archived=len(only_a), rows_only_rerun=len(only_r))
        for _, r in only_a.head(100).iterrows():
            diffs.append({"kind": "row only in archive", **{k: r[k] for k in keys}})
        for _, r in only_r.head(100).iterrows():
            diffs.append({"kind": "row only in rerun", **{k: r[k] for k in keys}})
        cells = close = identical = 0
        max_rel = 0.0
        attrib_rows = []
        for c in [c for c in num if c not in keys]:
            x = pd.to_numeric(both[c + "__a"], errors="coerce").to_numpy(float)
            y = pd.to_numeric(both[c + "__r"], errors="coerce").to_numpy(float)
            both_nan = np.isnan(x) & np.isnan(y)
            same = both_nan | np.isclose(x, y, rtol=RTOL, atol=ATOL)
            near = both_nan | np.isclose(x, y, rtol=CLOSE_RTOL, atol=1e-6)
            cells += len(x); identical += int(same.sum()); close += int(near.sum())
            with np.errstate(divide="ignore", invalid="ignore"):
                rel = np.abs(x - y) / np.maximum(np.abs(x), 1e-12)
            rel = np.where(both_nan, 0, rel)
            rel = np.where(np.isnan(rel), np.inf, rel)
            if len(rel):
                max_rel = max(max_rel, float(np.nanmax(rel)))
            for i in np.where(~same)[0][:50]:
                row = both.iloc[i]
                diffs.append({"kind": f"value differs: {c}", "archived": x[i], "rerun": y[i],
                              "rel_diff": rel[i], **{k: row[k] for k in keys}})
                attrib_rows.append(" ".join(str(row[k]) for k in keys))
        for c in [c for c in common if c not in keys and c not in num and c not in vol]:
            x = both[c + "__a"].astype(str).where(both[c + "__a"].notna(), "")
            y = both[c + "__r"].astype(str).where(both[c + "__r"].notna(), "")
            same = (x.to_numpy() == y.to_numpy())
            cells += len(same); identical += int(same.sum()); close += int(same.sum())
            for i in np.where(~same)[0][:50]:
                row = both.iloc[i]
                diffs.append({"kind": f"text differs: {c}", "archived": x.iloc[i], "rerun": y.iloc[i],
                              **{k: row[k] for k in keys}})
                attrib_rows.append(" ".join(str(row[k]) for k in keys))
        res.update(cells_compared=cells, cells_identical=identical, cells_close=close,
                   max_rel_diff=(None if cells == 0 else max_rel))
        unmatched_text = [" ".join(map(str, r)) for r in pd.concat([only_a[keys], only_r[keys]]).itertuples(index=False)]
        res["attributed_to"] = "; ".join(attribute(unmatched_text + attrib_rows))
    else:
        res.update(alignment="unaligned (no unique key); compared column summaries", rows_matched=None,
                   rows_only_archived=None, rows_only_rerun=None)
        cells = close = identical = 0
        for c in num:
            sa = pd.to_numeric(old[c], errors="coerce").describe()[["count", "mean", "std", "min", "max"]]
            sb = pd.to_numeric(new[c], errors="coerce").describe()[["count", "mean", "std", "min", "max"]]
            for stat in sa.index:
                x, y = float(sa[stat]), float(sb[stat])
                cells += 1
                ok = (np.isnan(x) and np.isnan(y)) or np.isclose(x, y, rtol=RTOL, atol=ATOL)
                identical += int(ok)
                close += int(ok or np.isclose(x, y, rtol=CLOSE_RTOL, atol=1e-6))
                if not ok:
                    diffs.append({"kind": f"summary differs: {c}.{stat}", "archived": x, "rerun": y})
        res.update(cells_compared=cells, cells_identical=identical, cells_close=close, max_rel_diff=None,
                   attributed_to="")
    same_shape = res["rows_archived"] == res["rows_rerun"] and not res["cols_only_archived"] and not res["cols_only_rerun"]
    unmatched = (res.get("rows_only_archived") or 0) + (res.get("rows_only_rerun") or 0)
    if cells and identical == cells and same_shape and unmatched == 0:
        status = "IDENTICAL"
    elif cells and close >= 0.99 * cells and unmatched <= 0.01 * max(len(old), 1):
        status = "CLOSE"
    elif cells == 0 and same_shape and unmatched == 0:
        status = "IDENTICAL (no numeric columns)"
    else:
        status = "DIFFERENT"
    res["status"] = status
    return res, pd.DataFrame(diffs)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rerun", required=True, help="pipeline output folder (the run's --out-dir)")
    ap.add_argument("--report-dir", help="where to write the report (default: RERUN/reproducibility)")
    ap.add_argument("--archive", default=str(ARCHIVE), help="archived figure-source tables")
    ap.add_argument("--build-publication-assets", action="store_true",
                    help="first run src/make_publication_assets_v2_3.py on a copy of the rerun's tables/ and "
                         "source_data/ (in RERUN/publication_assets), which produces the main-figure panel tables")
    a = ap.parse_args(argv)
    rerun = Path(a.rerun).expanduser().resolve()
    report = Path(a.report_dir).expanduser().resolve() if a.report_dir else rerun / "reproducibility"
    (report / "diffs").mkdir(parents=True, exist_ok=True)
    if a.build_publication_assets:
        pub = rerun / "publication_assets"
        if pub.exists():
            shutil.rmtree(pub)
        for sub in ("tables", "source_data"):
            if (rerun / sub).exists():
                shutil.copytree(rerun / sub, pub / sub)
        for sub in ("manuscript_support", "checks", "figures/main"):
            (pub / sub).mkdir(parents=True, exist_ok=True)
        print(f"Building publication assets in {pub}", flush=True)
        r = subprocess.run([sys.executable, str(PUB_SCRIPT), "--root", str(pub)], capture_output=True, text=True)
        (report / "publication_assets.log").write_text(r.stdout + "\n" + r.stderr)
        print(("ok" if r.returncode == 0 else f"FAILED (exit {r.returncode}); see {report / 'publication_assets.log'}"), flush=True)
    idx = index_outputs(rerun)
    rows = []
    for arch in sorted(Path(a.archive).rglob("*.csv")):
        name = arch.stem
        cands = idx.get(name.lower(), [])
        if not cands and name == HEADLINE and idx.get("si_table_s6_complete_model_results"):
            s6_path = idx["si_table_s6_complete_model_results"][0]
            try:
                arch_df = load(arch)
                res, diffs = compare_table(name, arch_df, derive_headline(arch_df, load(s6_path)))
                res.update(archived_file=str(arch.relative_to(Path(a.archive))),
                           rerun_file=f"derived from {s6_path.relative_to(rerun)}")
                rows.append(res)
                if len(diffs):
                    diffs.head(500).to_csv(report / "diffs" / f"{name}.csv", index=False)
                print(f"{res['status']:<30} {name} (derived from S6)", flush=True)
            except Exception as e:
                rows.append({"table": name, "archived_file": str(arch.relative_to(Path(a.archive))),
                             "status": f"ERROR: {e}"})
            continue
        if not cands:
            status = f"BLOCKED: {BLOCKED[name]}" if name in BLOCKED else "NOT_FOUND_IN_RERUN"
            rows.append({"table": name, "archived_file": str(arch.relative_to(Path(a.archive))), "status": status})
            continue
        try:
            res, diffs = compare_table(name, load(arch), load(cands[0]))
        except Exception as e:  # keep going; record the problem
            rows.append({"table": name, "archived_file": str(arch.relative_to(Path(a.archive))),
                         "rerun_file": str(cands[0].relative_to(rerun)), "status": f"ERROR: {e}"})
            continue
        res["archived_file"] = str(arch.relative_to(Path(a.archive)))
        res["rerun_file"] = str(cands[0].relative_to(rerun))
        rows.append(res)
        if len(diffs):
            diffs.head(500).to_csv(report / "diffs" / f"{name}.csv", index=False)
        print(f"{res['status']:<30} {name}", flush=True)
    summary = pd.DataFrame(rows)
    first = ["table", "status", "attributed_to", "rows_archived", "rows_rerun", "rows_matched",
             "rows_only_archived", "rows_only_rerun", "cells_compared", "cells_identical", "cells_close",
             "max_rel_diff", "cols_only_archived", "cols_only_rerun", "ignored_volatile_cols", "alignment", "archived_file", "rerun_file"]
    summary = summary[[c for c in first if c in summary.columns] + [c for c in summary.columns if c not in first]]
    summary.to_csv(report / "reproducibility_summary.csv", index=False)

    counts = summary["status"].str.split(r"[ :]").str[0].value_counts()
    lines = [
        "# Reproducibility comparison",
        "",
        f"Rerun: `{rerun}`  ",
        f"Archive: `reproduce/figures/data` ({len(summary)} tables)",
        "",
        "Tolerances: IDENTICAL = every shared value equal (numbers within rel 1e-6) and same rows/columns; "
        "CLOSE = at least 99% of values within rel 1e-3 and at most 1% unmatched rows. "
        "Timing and file-date columns (e.g. read_seconds, modified_time) are ignored and listed per table "
        "in reproducibility_summary.csv.",
        "",
        "| Status | Tables |", "|---|---:|",
        *[f"| {k} | {v} |" for k, v in counts.items()],
        "",
        "| Table | Status | Rows (archived → rerun) | Values identical / compared | Attributed to |",
        "|---|---|---|---|---|",
    ]
    for _, r in summary.iterrows():
        rows_txt = f"{int(r['rows_archived'])} → {int(r['rows_rerun'])}" if pd.notna(r.get("rows_archived")) else "—"
        cells_txt = (f"{int(r['cells_identical'])} / {int(r['cells_compared'])}"
                     if pd.notna(r.get("cells_compared")) else "—")
        lines.append(f"| {r['table']} | {r['status']} | {rows_txt} | {cells_txt} | {r['attributed_to'] if isinstance(r.get('attributed_to'), str) else ''} |")
    lines += ["", "Per-table details: `reproducibility_summary.csv`; largest differences: `diffs/<table>.csv`.", ""]
    (report / "REPRODUCIBILITY_REPORT.md").write_text("\n".join(lines))
    print(f"\nWrote {report / 'REPRODUCIBILITY_REPORT.md'}")
    print(counts.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
