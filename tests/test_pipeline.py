"""Preprocessing behaviour: the log-transform, the passthrough, the model zoo."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.fraud.data import FEATURES, TARGET
from src.fraud.pipeline import build_pipeline, build_preprocessor, log_amount, model_input_columns

from tests.test_data import make_frame


def test_log_amount_is_monotonic_and_non_negative():
    df = pd.DataFrame({"Amount": [0.0, 1.0, 10.0, 100.0, 1000.0]})
    out = log_amount(df)["Amount"].to_numpy()
    assert (out >= 0).all()
    assert (np.diff(out) > 0).all()


def test_log_amount_handles_zero_without_producing_infinity():
    # log(0) is -inf, which would poison the scaler. log1p is why the shift exists.
    out = log_amount(pd.DataFrame({"Amount": [0.0]}))["Amount"].to_numpy()
    assert np.isfinite(out).all()


def test_log_amount_clamps_negatives_rather_than_going_nan():
    out = log_amount(pd.DataFrame({"Amount": [-50.0]}))["Amount"].to_numpy()
    assert np.isfinite(out).all()


def test_log_amount_does_not_mutate_its_input():
    df = pd.DataFrame({"Amount": [10.0, 20.0]})
    before = df.copy()
    log_amount(df)
    pd.testing.assert_frame_equal(df, before)


def test_log_transform_compresses_the_tail():
    # This is the argument for logging: a standard scaler on raw Amount lets a
    # 200,000 outlier set the mean and variance, collapsing the normal range.
    raw = np.array([5.0] * 999 + [200_000.0])
    logged = np.log1p(raw)
    assert logged.max() / np.median(logged) < raw.max() / np.median(raw)


def test_pca_columns_pass_through_unscaled():
    pre = build_preprocessor()
    df = make_frame(20, fraud_rate=0.2)
    pre.fit(df[FEATURES])  # get_feature_names_out requires a fitted transformer
    out = list(pre.get_feature_names_out())
    # 30 outputs: Amount (log-transformed), Time (scaled), 28 PCA passthrough.
    # verbose_feature_names_out=False keeps the passthrough names bare, which is
    # also what stops Amount/Time from colliding with a same-named PCA column.
    assert len(out) == len(FEATURES)
    assert out == ["Amount", "Time"] + [f"V{i}" for i in range(1, 29)]


def test_pca_components_are_actually_untouched_by_the_transformer():
    # A passthrough must be numerically identical, not merely present in the
    # output: if a future edit routes V1..V28 through the scaler by accident the
    # already-whitened components get distorted and scores quietly shift.
    df = make_frame(50, fraud_rate=0.2)
    pre = build_preprocessor().fit(df[FEATURES])
    out = pre.transform(df[FEATURES])
    assert np.allclose(out[:, -28:], df[[f"V{i}" for i in range(1, 29)]].to_numpy())


def test_unknown_model_name_is_rejected():
    with pytest.raises(ValueError, match="unknown model"):
        build_pipeline("deepseek", scale_pos_weight=578.0)


def test_model_input_columns_match_the_feature_list():
    assert model_input_columns() == list(FEATURES)


@pytest.mark.parametrize("name", ["logreg", "rf", "xgb"])
def test_each_pipeline_fits_and_returns_probabilities(name):
    df = make_frame(400, fraud_rate=0.05)
    pipe = build_pipeline(name, scale_pos_weight=19.0).fit(df[FEATURES], df[TARGET])
    proba = pipe.predict_proba(df[FEATURES].head(10))[:, 1]
    assert proba.shape == (10,)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_pipeline_is_order_insensitive_on_input_columns():
    # The reindex inside the service is what guarantees this; assert the pipeline
    # itself is name-based rather than position-based.
    df = make_frame(300, fraud_rate=0.05)
    pipe = build_pipeline("logreg", scale_pos_weight=19.0).fit(df[FEATURES], df[TARGET])
    row = df[FEATURES].iloc[[0]]
    shuffled = row[list(reversed(FEATURES))]
    assert pipe.predict_proba(shuffled) == pytest.approx(pipe.predict_proba(row))


def test_xgboost_receives_the_supplied_pos_weight():
    pipe = build_pipeline("xgb", scale_pos_weight=578.0)
    assert pipe.named_steps["clf"].scale_pos_weight == 578.0
