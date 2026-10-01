"""Preprocessing and the model zoo for fraud detection.

`Amount` is the trap. It is the only column with a real business unit, and its
distribution is violently right-skewed: most transactions are a few euros, a few
thousands are tens of thousands, and a handful are the 200,000+ outliers that are
themselves a fraud signal. A standard scaler on the raw column lets those
outliers set the mean and variance, which then flattens the entire normal range
into a few standard deviations -- exactly where the bulk of legitimate traffic
lives. A log transform first compresses that tail without touching the ordering.

The V1..V28 columns are PCA components. Their signs and magnitudes are already
whitened, so they need no scaling. Only `Time` and `Amount` are engineered.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from src.fraud.data import FEATURES, PCA_FEATURES, TARGET

ENGINEERED = ["Time", "Amount"]
SCALED = ENGINEERED  # PCA components are already unit-variance by construction.


def log_amount(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with log1p(Amount). Applied inside the pipeline, before scaling.

    Shifted by 1 because Amount can legitimately be 0, and log(0) is -inf.
    """
    out = df.copy()
    out["Amount"] = np.log1p(out["Amount"].clip(lower=0))
    return out


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            (
                "amount",
                Pipeline(
                    [
                        ("log", FunctionTransformer(log_amount, feature_names_out="one-to-one")),
                        ("scale", StandardScaler()),
                    ]
                ),
                ["Amount"],
            ),
            ("time", StandardScaler(), ["Time"]),
            # Dropped from scaling deliberately: the PCA components are already
            # standardised, and rescaling them is a no-op that costs 28 columns.
            ("pca", "passthrough", PCA_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_pipeline(model_name: str, scale_pos_weight: float) -> Pipeline:
    pre = build_preprocessor()
    steps = [("preprocess", pre)]

    if model_name == "logreg":
        steps.append(
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=42,
                ),
            )
        )
    elif model_name == "rf":
        steps.append(
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=300,
                    class_weight="balanced_subsample",
                    min_samples_leaf=5,
                    n_jobs=-1,
                    random_state=42,
                ),
            )
        )
    elif model_name == "xgb":
        from xgboost import XGBClassifier

        steps.append(
            (
                "clf",
                XGBClassifier(
                    # Computed from the training split's own class balance rather
                    # than the dataset's headline number, so a shifted split does
                    # not silently misweight the loss.
                    scale_pos_weight=scale_pos_weight,
                    n_estimators=500,
                    learning_rate=0.05,
                    max_depth=4,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    eval_metric="aucpr",
                    n_jobs=-1,
                    random_state=42,
                ),
            )
        )
    else:
        raise ValueError(f"unknown model {model_name!r}; expected logreg, rf or xgb")

    return Pipeline(steps)


def model_input_columns() -> list[str]:
    """Column order the API must supply. Explicit because a dict is not ordered."""
    return list(FEATURES)


__all__ = [
    "FEATURES",
    "TARGET",
    "build_pipeline",
    "build_preprocessor",
    "log_amount",
    "model_input_columns",
]
