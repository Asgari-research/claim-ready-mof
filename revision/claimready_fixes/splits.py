"""Tie-safe grouping and split-leakage diagnostics (R1.5, R2.4).

Historical behaviour (``src/`` ``choose_group_series_for_joined``)::

    pd.qcut(vals.rank(method='first'), q=..., duplicates='drop')

``rank(method='first')`` breaks ties by row order, so materials with an
identical descriptor value can be placed in different groups and therefore on
both sides of a "grouped" split. ``quantile_groups_tie_safe`` assigns groups
per distinct value, so equal values always share a group.

Note: quantile groups of one geometric descriptor are a weak proxy for
out-of-distribution testing. The revision plan treats them as a retrospective
sensitivity analysis only; chemically defined holdouts (metal / topology /
family) are preferred where metadata support them.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def quantile_groups_tie_safe(values: pd.Series, n_groups: int = 10) -> pd.Series:
    """Assign approximately equal-population groups, never splitting tied values.

    Groups are integers 0..k-1 (k <= n_groups). Missing values get <NA>.
    """
    if n_groups < 2:
        raise ValueError("n_groups must be >= 2")
    v = pd.to_numeric(values, errors="coerce")
    ok = v.notna()
    out = pd.Series(pd.NA, index=values.index, dtype="Int64")
    if not ok.any():
        return out
    counts = v[ok].value_counts().sort_index()
    # Fraction of the population strictly below each distinct value.
    before = (counts.cumsum() - counts) / counts.sum()
    gid = np.floor(before.to_numpy() * n_groups).astype(int)
    # Re-index to consecutive integers in case some bins are empty.
    _, gid = np.unique(gid, return_inverse=True)
    mapping = dict(zip(counts.index, gid))
    out[ok] = v[ok].map(mapping).astype("Int64")
    return out


def tied_values_split_across_groups(values: pd.Series, groups: pd.Series) -> int:
    """Number of distinct values that appear in more than one group (should be 0)."""
    df = pd.DataFrame({"v": values, "g": groups}).dropna()
    return int((df.groupby("v")["g"].nunique() > 1).sum())


def group_overlap(train_groups: pd.Series, test_groups: pd.Series) -> set:
    """Groups present in both train and test (should be empty for a grouped split)."""
    return set(pd.Series(train_groups).dropna()) & set(pd.Series(test_groups).dropna())


def exact_duplicate_rows_across_split(
    X_train: pd.DataFrame, X_test: pd.DataFrame, decimals: Optional[int] = 8
) -> int:
    """Count test rows whose full feature vector also occurs in the training set.

    Rounding (``decimals``) guards against float noise; pass ``None`` for
    bit-exact comparison. Non-zero counts indicate duplicate leakage.
    """
    a = X_train.copy()
    b = X_test.copy()
    if decimals is not None:
        num_a = a.select_dtypes("number").columns
        num_b = b.select_dtypes("number").columns
        a[num_a] = a[num_a].round(decimals)
        b[num_b] = b[num_b].round(decimals)
    ha = set(pd.util.hash_pandas_object(a, index=False))
    hb = pd.util.hash_pandas_object(b, index=False)
    return int(hb.isin(ha).sum())
