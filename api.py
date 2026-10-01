"""FastAPI service for real-time fraud scoring.

    uvicorn api:app --reload

Two decisions worth defending:

1. The request schema is *generated* from the same feature list the artifact was
   trained with, rather than hand-written as 30 explicit fields. Thirty duplicated
   field declarations is thirty chances for the schema to drift from the model,
   and the failure is silent: a request that validates, then predicts on shifted
   columns. One source of truth, one Pydantic model, correct OpenAPI.

2. The model is loaded in the lifespan handler, once per process. Not per request.
   See src/fraud/service.py.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, create_model

from src.fraud.data import FEATURES
from src.fraud.service import DECISIONS, DEFAULT_ARTIFACT, FraudService, get, register


def artifact_path() -> Path:
    """Where to load the model from. Overridable so the container can mount it."""
    return Path(os.environ.get("MODEL_ARTIFACT", DEFAULT_ARTIFACT))


@asynccontextmanager
async def lifespan(app: FastAPI):
    service = FraudService.from_file(artifact_path())
    register(service)
    yield
    register(None)  # type: ignore[arg-type]


# --- request/response schemas ------------------------------------------------

# Time is measured in seconds from the dataset's first transaction, so it is only
# meaningful relative to that origin. Documented rather than hidden, because a
# caller sending a raw epoch timestamp would be silently mispredicted.
_FIELDS = {
    f: (float, Field(..., description="PCA component, already whitened by the source dataset"))
    for f in FEATURES
    if f.startswith("V")
}
_FIELDS["Time"] = (float, Field(..., ge=0, description="seconds since the dataset origin"))
_FIELDS["Amount"] = (float, Field(..., ge=0, description="transaction amount in the dataset's currency"))

TransactionRequest = create_model(
    "TransactionRequest", __config__=ConfigDict(extra="forbid"), **_FIELDS
)


class PredictionResponse(BaseModel):
    fraud_probability: float = Field(..., ge=0.0, le=1.0)
    decision: str = Field(..., description=" | ".join(DECISIONS))
    threshold: float
    review_threshold: float


class HealthResponse(BaseModel):
    status: str
    trained_at: str
    n_features: int
    thresholds: dict[str, float]
    metrics: dict[str, Any]


# --- app ---------------------------------------------------------------------

app = FastAPI(
    title="Fraud Detection API",
    version="1.0.0",
    summary="Scores a card transaction for fraud likelihood.",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    """Liveness plus the model version actually being served.

    Reports the training timestamp deliberately: if this is older than the newest
    data the pipeline has seen, the service is up but serving a stale model, and
    that should be visible to a monitor rather than inferred.
    """
    return HealthResponse(**get().health())


@app.post("/predict", response_model=PredictionResponse, tags=["scoring"])
def predict(tx: TransactionRequest) -> PredictionResponse:
    """Score one transaction.

    A malformed body is rejected by Pydantic with 422 before this handler runs,
    which is why there is no manual validation here to drift out of sync.
    """
    service = get()
    try:
        result = service.predict_one(tx.model_dump())
    except KeyError as exc:
        # Unreachable while the schema is generated from the artifact's own feature
        # list. Kept so that a hand-edited schema fails loudly instead of quietly
        # predicting on a misaligned frame.
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return PredictionResponse(
        fraud_probability=result.fraud_probability,
        decision=result.decision,
        threshold=result.threshold,
        review_threshold=result.review_threshold,
    )
