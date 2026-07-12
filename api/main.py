"""FastAPI service for the exported heart-disease pipeline.

The pipeline is loaded once at startup from MODEL_PATH (default: the
committed artifact under models/). The per-request structured log line is
deliberately minimal  -  Phase 7's monitoring builds on it.
"""
import json
import logging
import os
import time
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from prometheus_client import Counter
from prometheus_fastapi_instrumentator import Instrumentator

from api.schemas import PatientFeatures, PredictionResponse

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
)
logger = logging.getLogger("heart_disease_api")

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(
    os.environ.get("MODEL_PATH", REPO_ROOT / "models" / "heart_disease_pipeline.joblib")
)
METADATA_PATH = Path(
    os.environ.get("METADATA_PATH", REPO_ROOT / "models" / "model_metadata.json")
)

app = FastAPI(
    title="Heart Disease Risk API",
    description=(
        "Predicts heart-disease risk from 13 clinical features (UCI Cleveland "
        "schema). Preprocessing and derived clinical features are embedded in "
        "the model pipeline."
    ),
    version="1.0.0",
)

pipeline = joblib.load(MODEL_PATH)
metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))

PREDICTIONS_BY_CLASS = Counter(
    "heart_disease_predictions",
    "Predictions served, by predicted risk label",
    ["risk_label"],
)

Instrumentator().instrument(app).expose(app, include_in_schema=False)


@app.post("/predict", response_model=PredictionResponse)
def predict(features: PatientFeatures) -> PredictionResponse:
    start = time.perf_counter()
    row = pd.DataFrame([features.model_dump()])
    prediction = int(pipeline.predict(row)[0])
    probability = float(pipeline.predict_proba(row)[0, 1])
    latency_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "predict prediction=%d probability=%.4f latency_ms=%.1f",
        prediction,
        probability,
        latency_ms,
    )
    risk_label = "disease" if prediction == 1 else "no disease"
    PREDICTIONS_BY_CLASS.labels(risk_label=risk_label).inc()
    return PredictionResponse(
        prediction=prediction,
        probability=probability,
        risk_label=risk_label,
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": pipeline is not None}


@app.get("/model-info")
def model_info() -> dict:
    return metadata
