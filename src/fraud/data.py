"""Loading the credit-card fraud dataset and splitting it *by time*.

The single most consequential decision in this project is the split. Transaction
data is chronological: a model trained on a random 80% has seen the future when
scoring the past, so a random split reports a number that will not survive contact
with production. `shuffle=False` is the whole fix.

Everything else here is standard; the imbalance arithmetic in `describe` is not,
and it is the reason the model-selection criteria downstream are average
precision and recall-at-FPR rather than accuracy.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW_FILE = "creditcard.csv"
FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
TARGET = "Class"
PCA_FEATURES = [f"V{i}" for i in range(1, 29)]

# Fraction of the (chronological) data used for training.
TRAIN_FRACTION = 0.8


def load_raw(data_dir: Path) -> pd.DataFrame:
    path = Path(data_dir) / RAW_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download it with:\n"
            "  python -m kaggle datasets download -d mlg-ulb/creditcardfraud -p data/ --unzip"
        )
    return pd.read_csv(path)


def time_split(
    df: pd.DataFrame, train_fraction: float = TRAIN_FRACTION
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split chronologically. Never shuffle this data.

    The rows arrive in event order, so a positional split puts strictly earlier
    transactions in train and strictly later ones in test. That mirrors the real
    deployment question -- "given everything up to now, what happens next?" --
    which a random split does not.
    """
    df = df.sort_values("Time", kind="stable").reset_index(drop=True)
    cut = int(len(df) * train_fraction)
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def describe(df: pd.DataFrame) -> dict[str, float]:
    """Headline counts, so the imbalance is a number rather than an adjective."""
    return {
        "rows": len(df),
        "fraud_count": int(df[TARGET].sum()),
        "fraud_rate": float(df[TARGET].mean()),
        "time_min": float(df["Time"].min()),
        "time_max": float(df["Time"].max()),
    }


def imbalance_ratio(df: pd.DataFrame) -> float:
    """Negatives per positive. ~578 on this dataset, which is why SMOTE exists."""
    positives = int(df[TARGET].sum())
    if positives == 0:
        raise ValueError("split contains no fraud rows; cannot compute the ratio")
    return (len(df) - positives) / positives
