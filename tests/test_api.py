"""In-process tests for the serving API (TestClient — no Docker needed)."""
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)

_SAMPLE_PATH = Path(__file__).resolve().parents[1] / "api" / "sample_request.json"
SAMPLE = json.loads(_SAMPLE_PATH.read_text(encoding="utf-8"))


def test_predict_valid_record_returns_prediction():
    resp = client.post("/predict", json=SAMPLE)
    assert resp.status_code == 200
    body = resp.json()
    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["probability"] <= 1.0
    assert body["risk_label"] == ("disease" if body["prediction"] == 1 else "no disease")


def test_predict_directionality_sick_vs_healthy():
    # High-risk profile: older male, asymptomatic chest pain, exercise-induced
    # angina, low peak heart rate, marked ST depression, 3 vessels, reversible
    # defect. Low-risk: young female, strong peak HR, no exercise angina.
    sick = {
        "age": 63, "sex": 1, "cp": 4, "trestbps": 150, "chol": 300, "fbs": 0,
        "restecg": 2, "thalach": 100, "exang": 1, "oldpeak": 3.0, "slope": 2,
        "ca": 3, "thal": 7,
    }
    healthy = {
        "age": 35, "sex": 0, "cp": 3, "trestbps": 115, "chol": 180, "fbs": 0,
        "restecg": 0, "thalach": 185, "exang": 0, "oldpeak": 0.0, "slope": 1,
        "ca": 0, "thal": 3,
    }
    p_sick = client.post("/predict", json=sick).json()["probability"]
    p_healthy = client.post("/predict", json=healthy).json()["probability"]
    assert p_sick > p_healthy


@pytest.mark.parametrize(
    "field,value",
    [
        ("thal", 9),        # unknown categorical code
        ("cp", 5),          # outside enum
        ("ca", 4),          # outside enum
        ("chol", 50),       # below clinical floor
        ("thalach", 300),   # above physiological ceiling
        ("oldpeak", -1.0),  # negative ST depression
    ],
)
def test_predict_rejects_invalid_values(field, value):
    payload = dict(SAMPLE)
    payload[field] = value
    assert client.post("/predict", json=payload).status_code == 422


def test_predict_rejects_missing_field():
    payload = dict(SAMPLE)
    del payload["thal"]
    assert client.post("/predict", json=payload).status_code == 422


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "model_loaded": True}


def test_model_info_exposes_provenance():
    body = client.get("/model-info").json()
    assert "run_id" in body
    assert "metrics" in body


def test_metrics_endpoint_serves_prometheus_format():
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "# HELP" in resp.text
    assert "http_request" in resp.text


def test_predict_increments_prediction_counter():
    before = client.get("/metrics").text
    client.post("/predict", json=SAMPLE)
    after = client.get("/metrics").text
    assert 'heart_disease_predictions_total{risk_label="no disease"}' in after

    def counter_value(text):
        for line in text.splitlines():
            if line.startswith('heart_disease_predictions_total{risk_label="no disease"}'):
                return float(line.rsplit(" ", 1)[1])
        return 0.0

    assert counter_value(after) == counter_value(before) + 1.0


def test_metrics_not_in_openapi_schema():
    paths = client.get("/openapi.json").json()["paths"]
    assert "/metrics" not in paths
