"""The serving layer: one loaded model, shared by every request.

The point of this module is a single line of discipline -- the joblib artifact is
unpickled once, at application start, not inside the request handler. Loading a
300-tree forest on every call is the difference between 3ms and 300ms, and it is
the single most common way a demo-grade model API becomes unusable in production.

The artifact bundles the fitted pipeline, the decision thresholds, the feature
order, and the metrics from training, so the API cannot drift out of sync with the
model it serves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

DEFAULT_ARTIFACT = Path("artifacts/model.joblib")

DECISIONS = ("approve", "review", "block")


@dataclass(frozen=True)
class Decision:
    fraud_probability: float
    decision: str
    threshold: float
    review_threshold: float


class ModelNotLoaded(RuntimeError):
    """Raised when predict is called before startup loaded the artifact."""


class FraudService:
    """Holds the loaded pipeline and turns a feature dict into a decision."""

    def __init__(self, artifact: dict[str, Any]) -> None:
        self._pipeline = artifact["pipeline"]
        self.threshold: float = float(artifact["threshold"])
        self.review_threshold: float = float(artifact["review_threshold"])
        self.features: list[str] = list(artifact["features"])
        self.metrics: dict[str, Any] = dict(artifact.get("metrics", {}))
        self.trained_at: str = str(artifact.get("trained_at", "unknown"))

    @classmethod
    def from_file(cls, path: Path = DEFAULT_ARTIFACT) -> "FraudService":
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Train first:  python train.py"
            )
        return cls(joblib.load(path))

    def decide(self, fraud_probability: float) -> str:
        if fraud_probability >= self.threshold:
            return "block"
        if fraud_probability >= self.review_threshold:
            return "review"
        return "approve"

    def predict_one(self, record: dict[str, float]) -> Decision:
        missing = [f for f in self.features if f not in record]
        if missing:
            raise KeyError(f"missing features: {missing}")

        # Single-row frame built in the artifact's recorded column order. A dict
        # preserves insertion order, not schema order, and sklearn selects by name,
        # so the reindex is what keeps this correct.
        row = pd.DataFrame([{f: float(record[f]) for f in self.features}])
        proba = float(self._pipeline.predict_proba(row)[0, 1])
        return Decision(
            fraud_probability=proba,
            decision=self.decide(proba),
            threshold=self.threshold,
            review_threshold=self.review_threshold,
        )

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "trained_at": self.trained_at,
            "n_features": len(self.features),
            "thresholds": {"block": self.threshold, "review": self.review_threshold},
            "metrics": self.metrics,
        }


_REGISTRY: dict[str, FraudService] = {}


def register(service: FraudService | None, name: str = "default") -> None:
    """Store the active service, or drop it on shutdown.

    Storing None would leave a key behind and make get() hand back None instead of
    raising, so a post-shutdown call would fail with an AttributeError deep in the
    request path rather than a clear error at the boundary.
    """
    if service is None:
        _REGISTRY.pop(name, None)
    else:
        _REGISTRY[name] = service


def get(name: str = "default") -> FraudService:
    if name not in _REGISTRY:
        raise ModelNotLoaded(
            "no model is loaded; the application lifespan must load the artifact first"
        )
    return _REGISTRY[name]
