# Heart Disease Risk Prediction — MLOps Pipeline

[![CI](https://github.com/axle-bits/MLOP_Assignment_1/actions/workflows/ci.yml/badge.svg)](https://github.com/axle-bits/MLOP_Assignment_1/actions/workflows/ci.yml)

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

## CI/CD

Every push and pull request runs the GitHub Actions pipeline
(`.github/workflows/ci.yml`):

1. **Lint** — `ruff check .` (pycodestyle errors, pyflakes, import order)
2. **Test** — full pytest suite on Python 3.13; JUnit results uploaded as a
   workflow artifact on every run, including failures
3. **Train smoke** — `python -m ml.models.train --quick` trains all six
   model combinations end-to-end; the training log and MLflow store are
   uploaded as workflow artifacts

The pipeline fails on any lint finding, test failure, or training error.
Run the same checks locally:

```bash
ruff check .
pytest
```

## Serving the model

Run the API locally:

```bash
uvicorn api.main:app --reload --port 8000
```

Or containerized:

```bash
docker build -t heart-disease-api -f infra/Dockerfile .
docker run -d -p 8000:8000 heart-disease-api
```

Endpoints: `POST /predict` (13 raw features in, prediction + probability
out — see `api/sample_request.json`), `GET /health`, `GET /model-info`,
interactive docs at `/docs`.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @api/sample_request.json
```

## Deploying to Kubernetes

The API deploys to any Kubernetes cluster from the manifests in `infra/k8s/`
(tested on Docker Desktop's built-in cluster):

```bash
docker build -t heart-disease-api:v1 -f infra/Dockerfile .
kubectl apply -f infra/k8s/
kubectl rollout status deployment/heart-disease-api -n heart-disease
```

This creates the `heart-disease` namespace, a 2-replica Deployment with
readiness/liveness probes on `/health`, and a LoadBalancer Service mapping
port 80 to the pods' 8000. On Docker Desktop the service is reachable at
localhost:

```bash
curl http://localhost/health
curl -X POST http://localhost/predict \
  -H "Content-Type: application/json" \
  -d @api/sample_request.json
```

Tear down with `kubectl delete -f infra/k8s/`.

## Monitoring

With the app deployed, add Prometheus and Grafana (both provisioned
declaratively — no manual setup):

```bash
kubectl apply -f infra/k8s/monitoring/
```

- Prometheus scrapes each API pod's `/metrics` every 15s (per-pod targets via a headless Service): http://localhost:9090
- Grafana (anonymous viewer) serves a pre-provisioned "Heart Disease API"
  dashboard — request rate, p50/p95 latency, non-2xx rate, and predictions
  by risk label: http://localhost:3000

The API exposes standard HTTP metrics plus a domain counter,
`heart_disease_predictions_total{risk_label=...}`, incremented on every
prediction. Tear down with `kubectl delete -f infra/k8s/monitoring/`.

## Demo dashboard

A small self-contained dashboard ties the running stack together for
demos: a clinician-style risk assessment form, a live cluster status
board, and a traffic generator that replays real patient records from
the dataset (with occasional invalid requests) so the Grafana panels
show realistic variety.

With the app deployed (see above):

```bash
python demo/server.py
```

Then open http://localhost:8888. The dashboard only calls the API's
public endpoints and kubectl; it changes nothing in the cluster.

## Repository structure

```
ml/          data download + preprocessing (importable, tested)
notebooks/   EDA (jupytext-paired .py source + executed .ipynb)
tests/       pytest unit tests
docs/        decision log, report, figures
data/        raw + processed CSVs (committed; tiny dataset)
api/ infra/  FastAPI serving code, Dockerfile, Kubernetes manifests
demo/        presentation dashboard (server + single-page UI)
screenshots/  report evidence (mlflow, ci/cd, api, deploy)
```
