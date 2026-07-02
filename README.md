# Heart Disease Risk Prediction — MLOps Pipeline

End-to-end MLOps assignment (BITS Pilani AIMLCZG523): a heart-disease risk
classifier on the UCI Cleveland dataset, with experiment tracking, CI/CD,
a containerized FastAPI service, Kubernetes deployment on AWS EKS, and
monitoring.

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows (source .venv/bin/activate on Linux)
pip install -r requirements.txt
```

## Data pipeline

```bash
python -m ml.data.download     # data/raw/heart_cleveland_raw.csv
python -m ml.data.preprocess   # data/processed/heart_cleveland_clean.csv
pytest                         # unit tests
```

## EDA

`notebooks/01_eda.ipynb` — executed notebook (histograms, correlation
heatmap, class balance, missing-value analysis, and subgroup risk analysis
by sex / age band / chest-pain×angina). Regenerate with:

```bash
jupytext --to notebook --execute notebooks/01_eda.py -o notebooks/01_eda.ipynb
```

Figures are written to `docs/figures/eda/`.

## Model training & experiment tracking

Train all 6 tracked combinations (LR / RF / XGBoost × raw / clinical features):

```bash
python -m ml.models.train           # full grids (~minutes)
python -m ml.models.train --quick   # single-candidate grids (CI smoke)
```

Every run logs params, cross-validation + held-out test metrics, ROC curve,
confusion matrix, feature importances, and the fitted pipeline to the local
MLflow store (run metadata in `mlflow.db`, artifacts under `./mlruns` — both gitignored). Run all commands from the repo root — `mlflow.db` and `./mlruns` are created relative to the working directory. Inspect with:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
```

`notebooks/02_model_comparison.ipynb` reads the store and answers the
project's research question (do the derived clinical features help?).

## Model packaging

Export the best run (highest held-out ROC-AUC) to versioned artifacts:

```bash
python -m ml.models.export
```

This writes — and the repo commits — `models/heart_disease_pipeline.joblib`
(the full preprocessing+model pipeline; loads with scikit-learn + joblib
alone), `models/model_metadata.json` (source run, metrics, package versions,
input schema), and `models/mlflow_model/` (the same model in MLflow format).

Serving contract: send the 13 raw feature columns (see the metadata file);
the pipeline handles derived features, scaling, and encoding internally and
returns a class plus probability.

## Repository structure

```
ml/          data download + preprocessing (importable, tested)
notebooks/   EDA (jupytext-paired .py source + executed .ipynb)
tests/       pytest unit tests
docs/        decision log, specs, report figures
data/        raw + processed CSVs (committed; tiny dataset)
api/ infra/  serving + deployment (later phases)
```
