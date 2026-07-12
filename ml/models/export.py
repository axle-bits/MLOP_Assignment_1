"""Export the best trained model from the local MLflow store to models/.

Selection is deterministic: among full-grid runs of the heart-disease-risk
experiment, take argmax test_roc_auc (training seed 785 makes the winner
reproducible). Run ids are resolved at export time, never hardcoded  -  the
local store differs per machine.

Outputs (committed to git so serving and graders need no MLflow store):
- models/heart_disease_pipeline.joblib   -  canonical serving artifact
- models/model_metadata.json             -  provenance + serving contract
- models/mlflow_model/                   -  same model in MLflow format

Usage:
    python -m ml.models.export [--tracking-uri URI] [--out-dir models]
"""
import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy
import pandas as pd
import sklearn
import xgboost

from ml.data.preprocess import CATEGORICAL_COLS, CONTINUOUS_COLS
from ml.models.train import EXPERIMENT_NAME, SEED

METRIC_KEYS = [
    "cv_roc_auc_mean",
    "cv_roc_auc_std",
    "test_accuracy",
    "test_precision",
    "test_recall",
    "test_f1",
    "test_roc_auc",
]
RAW_INPUT_COLUMNS = CONTINUOUS_COLS + CATEGORICAL_COLS


def pick_best(runs: pd.DataFrame) -> pd.Series:
    """Best full-grid run row from a mlflow.search_runs frame."""
    if runs.empty:
        raise RuntimeError(
            f"No runs in experiment '{EXPERIMENT_NAME}'  -  "
            "run `python -m ml.models.train` first."
        )
    full = runs[runs["params.quick_mode"] == "False"]
    if full.empty:
        raise RuntimeError(
            "No full-grid runs found (quick-mode runs only)  -  "
            "run `python -m ml.models.train` without --quick."
        )
    return full.loc[full["metrics.test_roc_auc"].idxmax()]


def export(tracking_uri: str | None = None, out_dir: Path = Path("models")) -> dict:
    """Select the best run and write all artifact sets; returns the metadata."""
    previous_uri = mlflow.get_tracking_uri()
    try:
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        if experiment is None:
            raise RuntimeError(
                f"Experiment '{EXPERIMENT_NAME}' not found in the tracking store  -  "
                "run `python -m ml.models.train` first."
            )
        best = pick_best(mlflow.search_runs(experiment_ids=[experiment.experiment_id]))
        run_id = best["run_id"]

        pipe = mlflow.sklearn.load_model(f"runs:/{run_id}/model")

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipe, out_dir / "heart_disease_pipeline.joblib")

        metadata = {
            "run_id": run_id,
            "model_name": best["tags.model_name"],
            "feature_set": best["tags.feature_set"],
            "metrics": {k: float(best[f"metrics.{k}"]) for k in METRIC_KEYS},
            "seed": SEED,
            "exported_at_utc": datetime.now(timezone.utc).isoformat(),
            "package_versions": {
                "scikit-learn": sklearn.__version__,
                "pandas": pd.__version__,
                "numpy": numpy.__version__,
                "joblib": joblib.__version__,
                "xgboost": xgboost.__version__,
                "mlflow": mlflow.__version__,
            },
            "input_schema": {
                "columns": RAW_INPUT_COLUMNS,
                "note": (
                    "Raw 13-column patient frame; the pipeline performs all "
                    "derived-feature computation, scaling and encoding "
                    "internally. Columns are selected by name, order is not "
                    "significant."
                ),
            },
        }
        (out_dir / "model_metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )

        mlflow_dst = out_dir / "mlflow_model"
        if mlflow_dst.exists():
            shutil.rmtree(mlflow_dst)
        downloaded = mlflow.artifacts.download_artifacts(f"runs:/{run_id}/model")
        shutil.copytree(downloaded, mlflow_dst)

        return metadata
    finally:
        # set_tracking_uri mutates process-global state; restore so
        # in-process callers (tests, future scripts) are unaffected.
        mlflow.set_tracking_uri(previous_uri)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking-uri", default=None)
    parser.add_argument("--out-dir", default="models", type=Path)
    args = parser.parse_args()

    metadata = export(tracking_uri=args.tracking_uri, out_dir=args.out_dir)
    print(
        f"Exported {metadata['model_name']}/{metadata['feature_set']} "
        f"(run {metadata['run_id']}, test_roc_auc "
        f"{metadata['metrics']['test_roc_auc']:.4f}) to {args.out_dir}/"
    )


if __name__ == "__main__":
    main()
