"""Chronological splitting, which is the correctness foundation of this project."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.fraud.data import (
    FEATURES,
    TARGET,
    describe,
    imbalance_ratio,
    load_raw,
    time_split,
)


def make_frame(n: int = 1000, fraud_rate: float = 0.02, seed: int = 0) -> pd.DataFrame:
    """A chronological frame: Time ascends with row position, like the real CSV."""
    rng = np.random.default_rng(seed)
    data = {"Time": np.sort(rng.integers(0, 172_000, n))}
    for v in FEATURES:
        if v.startswith("V"):
            data[v] = rng.normal(0, 1, n)
    data["Amount"] = rng.lognormal(3, 1, n)
    data[TARGET] = (rng.random(n) < fraud_rate).astype(int)
    return pd.DataFrame(data)


def test_feature_list_is_time_plus_28_pca_plus_amount():
    assert len(FEATURES) == 30
    assert FEATURES[0] == "Time"
    assert FEATURES[-1] == "Amount"
    assert [f for f in FEATURES if f.startswith("V")] == [f"V{i}" for i in range(1, 29)]


def test_missing_raw_file_names_the_kaggle_command(tmp_path):
    with pytest.raises(FileNotFoundError, match="kaggle datasets download"):
        load_raw(tmp_path)


def test_time_split_never_shuffles():
    df = make_frame(1000)
    train, test = time_split(df, train_fraction=0.8)
    assert len(train) == 800 and len(test) == 200
    # The whole contract: every test row is strictly later than every train row.
    assert train["Time"].max() <= test["Time"].min()


def test_time_split_is_order_preserving_even_for_unsorted_input():
    df = make_frame(500).sample(frac=1.0, random_state=7)  # shuffle the rows
    train, test = time_split(df, train_fraction=0.8)
    assert list(train["Time"]) == sorted(train["Time"])
    assert train["Time"].max() <= test["Time"].min()


def test_time_split_keeps_every_row_exactly_once():
    df = make_frame(640)
    train, test = time_split(df, train_fraction=0.8)
    assert len(pd.concat([train, test])) == len(df)
    assert set(train.index) & set(test.index) == set()


def test_time_split_does_not_mutate_the_input():
    df = make_frame(200)
    before = df.copy()
    time_split(df)
    pd.testing.assert_frame_equal(df, before)


def test_describe_reports_the_fraud_rate():
    df = make_frame(1000, fraud_rate=0.05)
    d = describe(df)
    assert d["rows"] == 1000
    assert d["fraud_rate"] == pytest.approx(d["fraud_count"] / 1000)
    assert d["time_min"] <= d["time_max"]


def test_imbalance_ratio_is_negatives_per_positive():
    df = make_frame(1000, fraud_rate=0.01)
    positives = int(df[TARGET].sum())
    assert positives > 0
    assert imbalance_ratio(df) == pytest.approx((len(df) - positives) / positives)


def test_imbalance_ratio_on_a_fraud_free_split_raises():
    df = make_frame(100, fraud_rate=0.0)
    with pytest.raises(ValueError, match="no fraud rows"):
        imbalance_ratio(df)


def test_a_shuffled_split_would_leak_the_future():
    # Demonstrates why time_split exists, so the README can cite a test rather
    # than an assertion. A random split puts later transactions into training and
    # reports a better score than a chronological one -- the leak, made visible.
    from sklearn.model_selection import train_test_split

    df = make_frame(5000, fraud_rate=0.02, seed=1)
    shuffled_tr, shuffled_te, _, _ = train_test_split(
        df[FEATURES], df[TARGET], test_size=0.2
    )
    leak = shuffled_tr["Time"].max() > shuffled_te["Time"].min()
    assert leak, "expected a shuffled split to contain future rows in training"
