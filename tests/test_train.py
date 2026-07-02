"""End-to-end test of the training orchestration in --quick mode.

Uses a temporary MLflow store so the developer's ./mlruns is untouched.
Slow-ish (~30-60s: 6 quick GridSearchCV fits on 237 training rows) but
network-free and CI-safe.
"""
import mlflow
import mlflow.sklearn
import pandas as pd

from ml.data.preprocess import DEFAULT_CLEAN_PATH, TARGET_COL
from ml.models.train import EXPERIMENT_NAME, run_experiments

REQUIRED_METRICS = [
    "cv_roc_auc_mean",
    "cv_roc_auc_std",
    "test_accuracy",
    "test_precision",
    "test_recall",
    "test_f1",
    "test_roc_auc",
]


def test_quick_train_creates_six_complete_runs(tmp_path):
    uri = (tmp_path / "mlruns").as_uri()
    results = run_experiments(quick=True, tracking_uri=uri)

    assert len(results) == 6
    assert set(results["model_name"]) == {"logistic_regression", "random_forest", "xgboost"}
    assert set(results["feature_set"]) == {"raw", "clinical"}

    client = mlflow.tracking.MlflowClient(tracking_uri=uri)
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 6
    for run in runs:
        for metric in REQUIRED_METRICS:
            assert metric in run.data.metrics, f"{metric} missing from run"
        artifact_paths = [a.path for a in client.list_artifacts(run.info.run_id)]
        assert "roc_curve.png" in artifact_paths
        assert "confusion_matrix.png" in artifact_paths
        assert "feature_importance.png" in artifact_paths
        assert "model" in artifact_paths

    # the acceptance contract: the logged model must load from the run and predict
    mlflow.set_tracking_uri(uri)
    some_run = runs[0]
    pipe = mlflow.sklearn.load_model(f"runs:/{some_run.info.run_id}/model")
    X = pd.read_csv(DEFAULT_CLEAN_PATH).drop(columns=[TARGET_COL]).head(3)
    assert pipe.predict(X).shape == (3,)
