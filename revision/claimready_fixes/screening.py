"""Train-only feature screening (R2.3).

Historical behaviour (``src/`` ``xy`` and descriptor-joined ML): columns are
dropped for missingness > 0.65 or a single unique value, and ranked by
variance, using ALL rows before any train/test split. Test-set information
therefore influences which features the model sees (transductive screening).

``TrainOnlyFeatureScreen`` learns the kept columns in ``fit`` (training fold
only) and applies the same selection in ``transform``. Put it first in a
Pipeline so cross-validation refits it inside every fold.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class TrainOnlyFeatureScreen(BaseEstimator, TransformerMixin):
    """Drop high-missingness / constant numeric columns, learned from training rows only.

    Parameters mirror the historical thresholds: ``max_missing=0.65`` and a
    column must have at least ``min_unique=2`` distinct non-missing values.
    ``max_features`` optionally keeps the highest-variance columns (computed on
    training rows only).
    """

    def __init__(self, max_missing: float = 0.65, min_unique: int = 2, max_features: Optional[int] = None):
        self.max_missing = max_missing
        self.min_unique = min_unique
        self.max_features = max_features

    def fit(self, X, y=None):
        X = pd.DataFrame(X).apply(pd.to_numeric, errors="coerce")
        miss = X.isna().mean()
        nun = X.nunique(dropna=True)
        keep = X.columns[(miss <= self.max_missing) & (nun >= self.min_unique)]
        if self.max_features is not None and len(keep) > self.max_features:
            var = X[keep].var(skipna=True).fillna(0.0)
            keep = var.sort_values(ascending=False, kind="stable").index[: self.max_features]
        self.columns_in_ = list(X.columns)
        self.kept_columns_: List = list(keep)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=getattr(self, "columns_in_", None))
        X = X.apply(pd.to_numeric, errors="coerce")
        return X[self.kept_columns_]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.kept_columns_, dtype=object)


def make_train_only_pipeline(estimator, max_missing: float = 0.65, max_features: Optional[int] = None) -> Pipeline:
    """Screen -> median impute -> drop zero-variance -> scale -> estimator, all fold-internal."""
    return Pipeline(
        [
            ("screen", TrainOnlyFeatureScreen(max_missing=max_missing, max_features=max_features)),
            ("impute", SimpleImputer(strategy="median")),
            ("variance", VarianceThreshold(0.0)),
            ("scale", StandardScaler()),
            ("model", estimator),
        ]
    )
