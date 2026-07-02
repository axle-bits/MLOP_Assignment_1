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
