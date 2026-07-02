"""Tests for best-run selection and export. Selection logic is tested on
hand-built frames (no store); store-level failure modes use an empty temp
sqlite store — no test here requires the developer's real MLflow store."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from ml.data.preprocess import (
    CATEGORICAL_COLS,
    CONTINUOUS_COLS,
    DEFAULT_CLEAN_PATH,
    TARGET_COL,
)
from ml.models.export import export, pick_best


def runs_frame(rows):
    return pd.DataFrame(rows)


def full_run(run_id, auc):
    return {
        "run_id": run_id,
        "params.quick_mode": "False",
        "metrics.test_roc_auc": auc,
        "tags.model_name": "logistic_regression",
        "tags.feature_set": "clinical",
    }


def test_pick_best_takes_highest_test_roc_auc():
    runs = runs_frame([full_run("a", 0.85), full_run("b", 0.88), full_run("c", 0.83)])
    assert pick_best(runs)["run_id"] == "b"


def test_pick_best_excludes_quick_mode_runs():
    quick = dict(full_run("q", 0.99), **{"params.quick_mode": "True"})
    runs = runs_frame([quick, full_run("a", 0.85)])
    assert pick_best(runs)["run_id"] == "a"


def test_pick_best_raises_on_empty_frame():
    with pytest.raises(RuntimeError, match="No runs"):
        pick_best(pd.DataFrame())


def test_pick_best_raises_when_only_quick_runs():
    quick = dict(full_run("q", 0.99), **{"params.quick_mode": "True"})
    with pytest.raises(RuntimeError, match="full-grid"):
        pick_best(runs_frame([quick]))


def test_export_fails_clearly_when_experiment_missing(tmp_path):
    uri = f"sqlite:///{(tmp_path / 'empty.db').as_posix()}"
    with pytest.raises(RuntimeError, match="not found"):
        export(tracking_uri=uri, out_dir=tmp_path / "models")


ARTIFACT = Path("models/heart_disease_pipeline.joblib")
METADATA = Path("models/model_metadata.json")

needs_artifact = pytest.mark.skipif(
    not ARTIFACT.exists(), reason="exported model artifact not present"
)


def load_features():
    return pd.read_csv(DEFAULT_CLEAN_PATH).drop(columns=[TARGET_COL])


@needs_artifact
def test_committed_artifact_predicts_on_raw_schema():
    pipe = joblib.load(ARTIFACT)
    X = load_features()
    preds = pipe.predict(X)
    proba = pipe.predict_proba(X)[:, 1]
    assert set(np.unique(preds)) <= {0, 1}
    assert proba.min() >= 0.0 and proba.max() <= 1.0
    assert preds.shape == (len(X),)


@needs_artifact
def test_unseen_category_code_still_predicts():
    # Serving-time guard: OneHotEncoder(handle_unknown="ignore") must absorb
    # clinical codes never seen in training rather than raising.
    pipe = joblib.load(ARTIFACT)
    row = load_features().head(1).copy()
    row["thal"] = 9
    assert pipe.predict(row).shape == (1,)


@needs_artifact
def test_metadata_complete_and_consistent():
    meta = json.loads(METADATA.read_text(encoding="utf-8"))
    required = {
        "run_id", "model_name", "feature_set", "metrics", "seed",
        "exported_at_utc", "package_versions", "input_schema",
    }
    assert required <= set(meta)
    assert set(meta["metrics"]) == {
        "cv_roc_auc_mean", "cv_roc_auc_std", "test_accuracy",
        "test_precision", "test_recall", "test_f1", "test_roc_auc",
    }
    assert set(meta["input_schema"]["columns"]) == set(
        CONTINUOUS_COLS + CATEGORICAL_COLS
    )
    assert meta["seed"] == 785
    assert 0.5 < meta["metrics"]["test_roc_auc"] <= 1.0


@pytest.mark.skipif(
    not Path("mlflow.db").exists(), reason="local MLflow store absent (fresh clone)"
)
@needs_artifact
def test_exported_matches_logged_model():
    import mlflow.sklearn

    meta = json.loads(METADATA.read_text(encoding="utf-8"))
    logged = mlflow.sklearn.load_model(f"runs:/{meta['run_id']}/model")
    exported = joblib.load(ARTIFACT)
    X = load_features()
    assert (logged.predict(X) == exported.predict(X)).all()
