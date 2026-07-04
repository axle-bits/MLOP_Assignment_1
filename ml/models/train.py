"""Train 3 classifiers x 2 feature sets with GridSearchCV; track in MLflow.

Usage:
    python -m ml.models.train              # full grids (mlflow store: sqlite:///mlflow.db,
                                            # artifacts in ./mlruns)
    python -m ml.models.train --quick      # single-candidate grids (CI smoke)
"""
import argparse
import os
import tempfile

import matplotlib

matplotlib.use("Agg")  # headless: this module renders plot files only
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from xgboost import XGBClassifier

from ml.data.preprocess import DEFAULT_CLEAN_PATH, TARGET_COL
from ml.models.pipeline import build_pipeline

SEED = 785
EXPERIMENT_NAME = "heart-disease-risk"
C_NO, C_YES = "#2a78d6", "#e34948"  # Phase 1 validated palette

FULL_PARAM_GRIDS = {
    "logistic_regression": {
        "model__C": [0.01, 0.1, 1, 10],
        "model__class_weight": [None, "balanced"],
    },
    "random_forest": {
        "model__n_estimators": [200, 400],
        "model__max_depth": [None, 4, 8],
        "model__min_samples_leaf": [1, 3],
        "model__class_weight": [None, "balanced"],
    },
    "xgboost": {
        "model__n_estimators": [200, 400],
        "model__max_depth": [2, 3, 4],
        "model__learning_rate": [0.05, 0.1],
        "model__subsample": [0.8, 1.0],
    },
}

QUICK_PARAM_GRIDS = {
    "logistic_regression": {"model__C": [1.0]},
    "random_forest": {"model__n_estimators": [100]},
    "xgboost": {"model__n_estimators": [100]},
}


def make_estimator(model_name: str):
    if model_name == "logistic_regression":
        return LogisticRegression(solver="lbfgs", max_iter=2000)
    if model_name == "random_forest":
        return RandomForestClassifier(random_state=SEED)
    if model_name == "xgboost":
        return XGBClassifier(random_state=SEED, eval_metric="logloss")
    raise ValueError(f"unknown model {model_name!r}")


def load_split():
    df = pd.read_csv(DEFAULT_CLEAN_PATH)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)


def evaluate_on_test(pipe, X_test, y_test) -> dict:
    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]
    return {
        "test_accuracy": accuracy_score(y_test, pred),
        "test_precision": precision_score(y_test, pred),
        "test_recall": recall_score(y_test, pred),
        "test_f1": f1_score(y_test, pred),
        "test_roc_auc": roc_auc_score(y_test, proba),
    }


def log_plots(pipe, X_test, y_test, model_name: str):
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_estimator(pipe, X_test, y_test, ax=ax, curve_kwargs={"color": C_YES})
    ax.set_title(f"ROC curve — {model_name} (test)")
    mlflow.log_figure(fig, "roc_curve.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_estimator(pipe, X_test, y_test, ax=ax, cmap="Blues")
    ax.set_title(f"Confusion matrix — {model_name} (test)")
    mlflow.log_figure(fig, "confusion_matrix.png")
    plt.close(fig)

    feature_names = pipe.named_steps["preprocess"].get_feature_names_out()
    model = pipe.named_steps["model"]
    if hasattr(model, "coef_"):
        values = model.coef_[0]
        label = "coefficient"
    else:
        values = model.feature_importances_
        label = "importance"
    order = np.argsort(np.abs(values))[::-1][:15]
    fig, ax = plt.subplots(figsize=(6, 5))
    colors = [C_YES if v > 0 else C_NO for v in values[order]][::-1]
    ax.barh(np.array(feature_names)[order][::-1], values[order][::-1], color=colors)
    ax.set_xlabel(label)
    ax.set_title(f"Top features — {model_name}")
    fig.tight_layout()
    mlflow.log_figure(fig, "feature_importance.png")
    plt.close(fig)


def log_model_as_run_artifact(pipe, X_train):
    """Log the fitted pipeline under the run's own "model" artifact folder.

    mlflow>=3.x's `mlflow.sklearn.log_model(..., name=...)` records the model
    as a separate first-class "LoggedModel" entity (stored under the
    experiment's `models/` tree) rather than as a subfolder of the run's
    artifact root, so it no longer shows up via
    `client.list_artifacts(run_id)`. Saving locally and re-uploading with
    `mlflow.log_artifacts` keeps the familiar run-scoped "model" artifact
    path that downstream tasks (and this test suite) rely on.
    """
    with tempfile.TemporaryDirectory() as local_dir:
        local_model_path = os.path.join(local_dir, "model")
        # Cast to float64 so mlflow's schema inference doesn't emit a
        # UserWarning about integer columns being unable to represent NaNs
        # (verified: without this cast, 6 UserWarnings appear across the run).
        X_example = X_train.head(3).astype("float64")
        signature = infer_signature(X_example, pipe.predict(X_example))
        mlflow.sklearn.save_model(
            pipe,
            path=local_model_path,
            input_example=X_example,
            signature=signature,
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )
        mlflow.log_artifacts(local_model_path, artifact_path="model")


def run_experiments(quick: bool = False, tracking_uri: str | None = None) -> pd.DataFrame:
    # Without this, mlflow 3.14.0 raises on any file:// tracking URI (confirmed
    # empirically): MlflowException("The filesystem tracking backend (e.g.,
    # './mlruns') is in maintenance mode ... set `MLFLOW_ALLOW_FILE_STORE=true`
    # to opt out of this exception."). Scoped here (use-site), not at import time.
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    X_train, X_test, y_train, y_test = load_split()
    grids = QUICK_PARAM_GRIDS if quick else FULL_PARAM_GRIDS
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    rows = []
    for model_name, param_grid in grids.items():
        for feature_set in ("raw", "clinical"):
            pipe = build_pipeline(make_estimator(model_name), feature_set == "clinical")
            search = GridSearchCV(pipe, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1)
            search.fit(X_train, y_train)

            with mlflow.start_run(run_name=f"{model_name}__{feature_set}") as run:
                mlflow.set_tags({"model_name": model_name, "feature_set": feature_set})
                mlflow.log_params(search.best_params_)
                mlflow.log_params(
                    {
                        "seed": SEED,
                        "n_train": len(X_train),
                        "n_test": len(X_test),
                        "cv_folds": 5,
                        "quick_mode": quick,
                    }
                )
                metrics = {
                    "cv_roc_auc_mean": search.cv_results_["mean_test_score"][search.best_index_],
                    "cv_roc_auc_std": search.cv_results_["std_test_score"][search.best_index_],
                    **evaluate_on_test(search.best_estimator_, X_test, y_test),
                }
                mlflow.log_metrics(metrics)
                log_plots(search.best_estimator_, X_test, y_test, model_name)
                log_model_as_run_artifact(search.best_estimator_, X_train)
                rows.append(
                    {
                        "model_name": model_name,
                        "feature_set": feature_set,
                        "run_id": run.info.run_id,
                        **metrics,
                    }
                )

    results = pd.DataFrame(rows).sort_values("test_roc_auc", ascending=False)
    return results.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="single-candidate grids (CI smoke)")
    parser.add_argument(
        "--tracking-uri",
        default=None,
        help="MLflow tracking URI (default: sqlite:///mlflow.db for metadata, "
             "./mlruns for artifacts)",
    )
    args = parser.parse_args()

    results = run_experiments(quick=args.quick, tracking_uri=args.tracking_uri)
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(results.round(4).to_string(index=False))
    best = results.iloc[0]
    print(
        f"\nBest run: {best.model_name} / {best.feature_set} "
        f"(test ROC-AUC {best.test_roc_auc:.4f}, run_id {best.run_id})"
    )


if __name__ == "__main__":
    main()
