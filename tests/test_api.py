"""The API contract, including the failure modes.

Trains a small real model into a temporary artifact rather than mocking the
service, so these tests exercise the same path production does: joblib load,
sklearn predict, Pydantic validation.
"""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.fraud.data import FEATURES, TARGET
from src.fraud.pipeline import build_pipeline, model_input_columns

from tests.test_data import make_frame


@pytest.fixture(scope="module")
def artifact(tmp_path_factory) -> Path:
    """A genuine fitted pipeline, saved where the app's lifespan will find it."""
    df = make_frame(600, fraud_rate=0.08, seed=3)
    pipe = build_pipeline("logreg", scale_pos_weight=11.5).fit(df[FEATURES], df[TARGET])
    path = tmp_path_factory.mktemp("artifacts") / "model.joblib"
    joblib.dump(
        {
            "pipeline": pipe,
            "threshold": 0.5,
            "review_threshold": 0.2,
            "features": model_input_columns(),
            "trained_at": "2026-01-01T00:00:00+00:00",
            "metrics": {"model": "logreg", "average_precision": 0.99},
        },
        path,
    )
    return path


@pytest.fixture(scope="module")
def client(artifact, monkeypatch_module) -> TestClient:
    monkeypatch_module.setenv("MODEL_ARTIFACT", str(artifact))
    from api import app  # imported after the env var is set

    # The context manager is what runs the lifespan, i.e. the model load.
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def monkeypatch_module():
    from _pytest.monkeypatch import MonkeyPatch

    mp = MonkeyPatch()
    yield mp
    mp.undo()


def valid_payload(**overrides) -> dict:
    payload = {"Time": 100_000.0, "Amount": 42.0}
    payload.update({f: 0.0 for f in FEATURES if f.startswith("V")})
    payload.update(overrides)
    return payload


# --- happy path --------------------------------------------------------------

def test_health_reports_the_served_model(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["n_features"] == 30
    assert body["trained_at"] == "2026-01-01T00:00:00+00:00"
    # A monitor needs the model identity, not just liveness.
    assert body["metrics"]["model"] == "logreg"


def test_predict_returns_a_probability_and_a_valid_decision(client):
    r = client.post("/predict", json=valid_payload())
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert body["decision"] in {"approve", "review", "block"}
    assert body["threshold"] == 0.5
    assert body["review_threshold"] == 0.2


def test_a_large_amount_is_at_least_as_suspicious_as_a_small_one(client):
    small = client.post("/predict", json=valid_payload(Amount=1.0)).json()
    large = client.post("/predict", json=valid_payload(Amount=90_000.0)).json()
    assert large["fraud_probability"] >= small["fraud_probability"]


def test_predictions_are_reproducible(client):
    a = client.post("/predict", json=valid_payload()).json()
    b = client.post("/predict", json=valid_payload()).json()
    assert a == b


# --- the failures that matter -----------------------------------------------

def test_missing_fields_are_rejected_with_422_not_500(client):
    r = client.post("/predict", json={"Time": 1.0, "Amount": 2.0})
    assert r.status_code == 422


def test_a_single_bad_field_still_rejects_the_whole_request(client):
    r = client.post("/predict", json=valid_payload(Amount="not-a-number"))
    assert r.status_code == 422


def test_negative_amount_is_rejected(client):
    r = client.post("/predict", json=valid_payload(Amount=-5.0))
    assert r.status_code == 422


def test_unknown_extra_fields_are_rejected(client):
    # extra="forbid" stops a typo'd field name from being silently ignored, which
    # would otherwise look like a working request that skipped the intended feature.
    r = client.post("/predict", json=valid_payload(Amountt=1.0))
    assert r.status_code == 422


def test_malformed_json_does_not_crash_the_service(client):
    r = client.post(
        "/predict", content=b"{not json", headers={"Content-Type": "application/json"}
    )
    assert r.status_code == 422
    # And the service is still alive afterwards.
    assert client.get("/health").status_code == 200


def test_wrong_method_returns_405(client):
    assert client.get("/predict").status_code == 405


def test_unknown_route_returns_404(client):
    assert client.get("/score").status_code == 404


# --- openapi -----------------------------------------------------------------

def test_openapi_documents_both_endpoints(client):
    paths = client.get("/openapi.json").json()["paths"]
    assert "/predict" in paths and "post" in paths["/predict"]
    assert "/health" in paths
