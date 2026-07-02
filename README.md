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
by sex / age band / chest-pain×angina). One-time setup for notebook execution: `python -m ipykernel install --user` (registers the venv's Jupyter kernel). Regenerate with:

```bash
jupytext --to notebook --execute notebooks/01_eda.py -o notebooks/01_eda.ipynb
```

Figures are written to `docs/figures/eda/`.

## Repository structure

```
ml/          data download + preprocessing (importable, tested)
notebooks/   EDA (jupytext-paired .py source + executed .ipynb)
tests/       pytest unit tests
docs/        decision log, specs, report figures
data/        raw + processed CSVs (committed; tiny dataset)
api/ infra/  serving + deployment (later phases)
```
