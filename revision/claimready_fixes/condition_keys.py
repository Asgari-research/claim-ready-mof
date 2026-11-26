"""Condition-preserving identity keys and joins (R1.4, R2.2, R2.6).

Historical behaviour (``src/`` ``prepare_variant_joined_table``)::

    td = td[td['__join_id'].notna()].drop_duplicates('__join_id')

keeps the first row per material, so a MOF measured at two pressures keeps
whichever row happens to come first, and the label depends on file order.
Fuzzy identifier variants (e.g. ``serial_stripped``) can also map two
different structures onto one key.

This module:

1. builds an explicit canonical key = resource | identifier | condition columns
   | (optional) source row, so distinct physical observations stay distinct;
2. audits collisions (how many rows share a key, and whether their targets
   disagree) instead of silently deduplicating;
3. joins targets to descriptors on an *exact* normalised identifier, keeping
   every target row and refusing (or reporting) ambiguous descriptor rows.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

DEFAULT_CONDITION_COLUMNS = ("gas", "temperature_K", "pressure_bar")
KEY_SEP = "|"


def normalize_exact_id(s: pd.Series) -> pd.Series:
    """Exact-identity normalisation only: trim, casefold, drop a ``.cif`` suffix.

    Deliberately does NOT strip serial numbers, symmetry suffixes, repeats or
    punctuation; those transformations can merge different structures.
    """
    out = s.astype("string").str.strip().str.casefold()
    out = out.str.replace(r"\.cif$", "", regex=True)
    return out.mask(out.eq("") | out.isna())


def _fmt_condition(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NA:
        return "NA"
    if isinstance(v, (int, float, np.integer, np.floating)):
        # Stable text for floats: 0.15 and 0.150000001 should not differ by noise.
        return format(float(v), ".6g")
    return str(v).strip().casefold()


def build_canonical_key(
    df: pd.DataFrame,
    id_col: str,
    condition_cols: Sequence[str] = DEFAULT_CONDITION_COLUMNS,
    resource: Optional[str] = None,
    include_source_row: bool = False,
    key_col: str = "canonical_key",
) -> pd.DataFrame:
    """Return a copy of ``df`` with ``norm_id``, ``source_row`` and ``key_col`` added.

    Missing condition columns raise, because a key without its condition is the
    defect this module exists to prevent. Pass ``condition_cols=()`` explicitly
    if a table truly has no conditions.
    """
    missing = [c for c in condition_cols if c not in df.columns]
    if missing:
        raise KeyError(f"condition columns missing from table: {missing}")
    out = df.copy()
    out["source_row"] = np.arange(len(out))
    out["norm_id"] = normalize_exact_id(out[id_col])
    parts: List[pd.Series] = []
    if resource is not None:
        parts.append(pd.Series(str(resource), index=out.index))
    parts.append(out["norm_id"].fillna("NA").astype(str))
    for c in condition_cols:
        parts.append(out[c].map(_fmt_condition))
    if include_source_row:
        parts.append(out["source_row"].astype(str))
    key = parts[0]
    for p in parts[1:]:
        key = key + KEY_SEP + p
    out[key_col] = key
    return out


def audit_key_collisions(
    df: pd.DataFrame, key_col: str, target_col: Optional[str] = None, atol: float = 1e-9
) -> pd.DataFrame:
    """Per-key collision table: rows per key and whether targets disagree.

    Only keys with more than one row are returned.
    """
    g = df.groupby(key_col, dropna=False)
    res = g.size().rename("n_rows").to_frame()
    if target_col is not None:
        t = pd.to_numeric(df[target_col], errors="coerce")
        agg = t.groupby(df[key_col], dropna=False).agg(["nunique", "min", "max"])
        res["n_distinct_targets"] = agg["nunique"]
        res["target_min"] = agg["min"]
        res["target_max"] = agg["max"]
        res["targets_conflict"] = (agg["max"] - agg["min"]).abs() > atol
    res = res[res["n_rows"] > 1].reset_index()
    return res.sort_values("n_rows", ascending=False, kind="stable").reset_index(drop=True)


@dataclass
class JoinReport:
    n_target_rows: int
    n_descriptor_rows: int
    n_joined_rows: int
    n_targets_unmatched: int
    n_ambiguous_descriptor_ids: int
    ambiguous_descriptor_ids: List[str] = field(default_factory=list)
    unmatched_target_ids: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, object]:
        return {
            "n_target_rows": self.n_target_rows,
            "n_descriptor_rows": self.n_descriptor_rows,
            "n_joined_rows": self.n_joined_rows,
            "n_targets_unmatched": self.n_targets_unmatched,
            "n_ambiguous_descriptor_ids": self.n_ambiguous_descriptor_ids,
        }


class AmbiguousDescriptorError(ValueError):
    """Raised when one identifier maps to several descriptor rows."""


def condition_preserving_join(
    targets: pd.DataFrame,
    descriptors: pd.DataFrame,
    target_id: str,
    descriptor_id: str,
    on_ambiguous: str = "error",
    how: str = "inner",
):
    """Join descriptors onto every target row by exact normalised identifier.

    - Every target row (i.e. every material x condition observation) is kept;
      there is no ``drop_duplicates`` on the target side.
    - Descriptors must be unique per identifier. If not, ``on_ambiguous``
      decides: ``"error"`` raises, ``"drop"`` excludes those identifiers and
      lists them in the report. Neither option silently picks a first row.

    Returns ``(joined_df, JoinReport)``.
    """
    if on_ambiguous not in {"error", "drop"}:
        raise ValueError("on_ambiguous must be 'error' or 'drop'")
    t = targets.copy()
    d = descriptors.copy()
    t["_jid"] = normalize_exact_id(t[target_id])
    d["_jid"] = normalize_exact_id(d[descriptor_id])
    d = d[d["_jid"].notna()]
    counts = d["_jid"].value_counts()
    ambiguous = sorted(counts[counts > 1].index.astype(str).tolist())
    if ambiguous and on_ambiguous == "error":
        raise AmbiguousDescriptorError(
            f"{len(ambiguous)} identifier(s) map to more than one descriptor row, e.g. {ambiguous[:5]}"
        )
    if ambiguous:
        d = d[~d["_jid"].isin(ambiguous)]
    joined = t.merge(
        d.drop(columns=[descriptor_id]),
        on="_jid",
        how=how,
        validate="many_to_one",
        suffixes=("", "_desc"),
    )
    matched = set(d["_jid"].dropna())
    unmatched = sorted(set(t["_jid"].dropna()) - matched)
    report = JoinReport(
        n_target_rows=len(targets),
        n_descriptor_rows=len(descriptors),
        n_joined_rows=len(joined),
        n_targets_unmatched=int(t["_jid"].isin(unmatched).sum() + t["_jid"].isna().sum()),
        n_ambiguous_descriptor_ids=len(ambiguous),
        ambiguous_descriptor_ids=ambiguous,
        unmatched_target_ids=[str(u) for u in unmatched],
    )
    return joined.drop(columns=["_jid"]), report
