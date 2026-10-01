"""Train the fraud detector, pick thresholds by cost, persist the artifact.

    python train.py

Writes artifacts/model.joblib, consumed by api.py.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from src.fraud.data import (
    FEATURES,
    TARGET,
    describe,
    imbalance_ratio,
    load_raw,
    time_split,
)
from src.fraud.pipeline import build_pipeline, model_input_columns

DATA = Path("data")
ARTIFACTS = Path("artifacts")
MODELS = ["logreg", "rf", "xgb"]

# The operating point, in business terms. A blocked card costs a reissue and some
# goodwill; a missed fraud costs the full transaction and the customer's trust.
# Neither number is measurable from this dataset, so both are stated here and in
# the README rather than buried in a notebook.
#
# The first threshold is "block": the FPR we are willing to inflict on honest
# customers. The second is "review": looser, because a human looks at it.
BLOCK_AT_FPR = 0.005
REVIEW_AT_FPR = 0.02


def threshold_at_fpr(y_true, proba, target_fpr: float) -> float:
    """Smallest threshold whose false-positive rate stays at or below target.

    The FPR has to come from the ROC curve. It cannot be derived from the
    precision-recall curve as `1 - precision`: that quantity is the false
    *discovery* rate, FP / (TP + FP), which is a completely different number from
    the false positive rate, FP / (FP + TN). Substituting one for the other makes
    the threshold collapse to near 1.0 on a rare-event problem and cuts recall to
    single digits while still printing a confident-looking FPR column.

    Returns the smallest threshold that still fits the budget, which is the most
    aggressive cutoff available -- it catches the most fraud for the same number of
    false alarms. The returned value always flags strictly fewer than the whole
    population, so it is never `inf`.
    """
    fpr, _tpr, thresholds = roc_curve(y_true, proba)

    # sklearn returns thresholds DECREASING from inf, paired with fpr INCREASING
    # from 0. The first entry is the degenerate "flag nothing" point
    # (threshold=inf, fpr=0), so it always satisfies any budget -- taking a max
    # over the qualifying set therefore returns inf and blocks nothing at all.
    # The aggressive-but-legal cutoff is the SMALLEST qualifying finite threshold.
    ok = (fpr <= target_fpr) & np.isfinite(thresholds)
    if not ok.any():
        # Budget is tighter than the finest distinction this data allows, e.g. a
        # 0.0 budget on a 1-in-20,000 problem. Flag only the single most suspicious
        # transaction, which is the minimum FPR any threshold can achieve.
        return float(proba.max())
    return float(thresholds[ok].min())


def recall_at_threshold(y_true, proba, threshold: float) -> float:
    positives = int(y_true.sum())
    if positives == 0:
        return 0.0
    return float(y_true[proba >= threshold].sum() / positives)


def main() -> None:
    ARTIFACTS.mkdir(exist_ok=True)
    raw = load_raw(DATA)
    print("full dataset:", json.dumps(describe(raw), indent=2))
    print(f"negatives per positive: {imbalance_ratio(raw):.0f}")

    train, test = time_split(raw)
    print("\ntrain:", json.dumps(describe(train), indent=2))
    print("test: ", json.dumps(describe(test), indent=2))

    X_train, y_train = train[FEATURES], train[TARGET]
    X_test, y_test = test[FEATURES], test[TARGET]

    # The number that justifies everything downstream: a model that never predicts
    # fraud still scores 99.83% accuracy. Accuracy is not a metric here.
    dummy = DummyClassifier(strategy="prior").fit(X_train, y_train)
    dummy_proba = dummy.predict_proba(X_test)[:, 1]
    print(
        f"\nDummyClassifier: accuracy={dummy.score(X_test, y_test):.5f} "
        f"average_precision={average_precision_score(y_test, dummy_proba):.5f}"
    )
    print("(that accuracy is meaningless -- it comes entirely from the majority class)\n")

    pos_weight = imbalance_ratio(train)
    rows, fitted = [], {}
    for name in MODELS:
        pipe = build_pipeline(name, scale_pos_weight=pos_weight).fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        thr = threshold_at_fpr(y_test, proba, BLOCK_AT_FPR)
        rows.append(
            {
                "model": name,
                "roc_auc": roc_auc_score(y_test, proba),
                "average_precision": average_precision_score(y_test, proba),
                "block_threshold": thr,
                "recall_at_block": recall_at_threshold(y_test, proba, thr),
                "fpr_at_block": float(
                    ((proba >= thr) & (y_test == 0)).sum() / (y_test == 0).sum()
                ),
            }
        )
        fitted[name] = (pipe, proba)

    results = pd.DataFrame(rows).sort_values("average_precision", ascending=False)
    print(results.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    results.to_csv(ARTIFACTS / "model_comparison.csv", index=False)

    best_name = str(results.iloc[0]["model"])
    best_pipe, best_proba = fitted[best_name]
    block_thr = threshold_at_fpr(y_test, best_proba, BLOCK_AT_FPR)
    review_thr = threshold_at_fpr(y_test, best_proba, REVIEW_AT_FPR)
    print(f"\nbest model: {best_name}")
    print(f"  block  at FPR <= {BLOCK_AT_FPR}: threshold={block_thr:.5f}")
    print(f"  review at FPR <= {REVIEW_AT_FPR}: threshold={review_thr:.5f}")

    artifact = {
        "pipeline": best_pipe,
        "threshold": block_thr,
        "review_threshold": review_thr,
        "features": model_input_columns(),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "metrics": {
            "model": best_name,
            "roc_auc": float(roc_auc_score(y_test, best_proba)),
            "average_precision": float(average_precision_score(y_test, best_proba)),
            "train_rows": len(train),
            "test_rows": len(test),
            "test_fraud_rate": float(y_test.mean()),
            "pos_weight": pos_weight,
            "block_at_fpr": BLOCK_AT_FPR,
            "review_at_fpr": REVIEW_AT_FPR,
        },
    }
    joblib.dump(artifact, ARTIFACTS / "model.joblib")
    json.dump(artifact["metrics"], (ARTIFACTS / "metrics.json").open("w"), indent=2)

    _plot(best_proba, y_test, results)
    print(f"\nwrote {ARTIFACTS}/model.joblib  ->  serve with:  uvicorn api:app --reload")


def _plot(proba, y_test, results) -> None:
    precision, recall, _ = precision_recall_curve(y_test, proba)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.plot(recall, precision, color="#c0392b")
    ax1.set_xlabel("recall")
    ax1.set_ylabel("precision")
    ax1.set_title("Precision-recall (test set)")

    ax2.bar(results["model"], results["average_precision"], color="#2c3e50")
    ax2.axhline(float(y_test.mean()), ls="--", color="#c0392b", label="random baseline")
    ax2.set_ylabel("average precision")
    ax2.set_title("Average precision by model")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(ARTIFACTS / "curves.png", dpi=150)


if __name__ == "__main__":
    main()
