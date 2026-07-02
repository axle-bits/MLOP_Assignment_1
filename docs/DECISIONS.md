# Decision Log

Running record of every non-trivial decision, kept current as the project
progresses. Feeds the "steps taken" sections of the final report.

## 2026-07-02 — Dataset variant: Cleveland subset (303 rows)
**Decision:** Use the classic Cleveland subset, fetched via `ucimlrepo` (UCI id=45).
**Rationale:** Verified empirically that id=45 returns exactly Cleveland (303×13). Matches the assignment's dataset link.
**Alternatives considered:** Combined 4-site dataset (~920 rows) — richer but not what id=45 serves; manual CSV download — less reproducible than a pip-installable fetch.

## 2026-07-02 — Target binarization
**Decision:** Map raw `num` (0–4) to binary `target`: 0 → 0 (no disease), 1–4 → 1 (disease present).
**Rationale:** Assignment requires a binary classifier; 0-vs-rest is the established convention for this dataset. Raw distribution 164/55/36/35/13.
**Alternatives considered:** Multiclass severity prediction — out of assignment scope.

## 2026-07-02 — Missing values: drop 6 rows (303 → 297)
**Decision:** Drop rows with NaN in `ca` (4 rows) or `thal` (2 rows).
**Rationale:** 2% of data; both features are categorical clinical measurements (vessel count, thalassemia type) where imputation would invent clinical facts; dropping sidesteps train/test leakage questions entirely.
**Alternatives considered:** Mode imputation in-pipeline (keeps 303 rows but cleaned CSV would still hold NaNs); pre-split imputation (mild leakage a grader could flag).
**Outcome:** post-drop class balance 160/137.

## 2026-07-02 — Repo layout: stage-oriented
**Decision:** Top-level `ml/`, `api/`, `infra/`, `tests/`, `notebooks/`, `docs/`, `screenshots/`.
**Rationale:** Maps 1:1 to CI/CD stages (lint, test, build); deliberately distinct from the heavily-copied cookiecutter-data-science template (academic-integrity requirement).
**Alternatives considered:** cookiecutter-data-science; flat `src/`.

## 2026-07-02 — Derived clinical features (planned for Phase 2)
**Decision:** Rate-pressure product (`trestbps × thalach`), heart-rate reserve (`220 − age − thalach`), `oldpeak × slope` interaction — inside the sklearn pipeline, not in the cleaned CSV.
**Rationale:** Clinically interpretable features supporting the "risk stratification" theme; pipeline placement guarantees identical transformation at training and inference.
**Alternatives considered:** Pulse pressure — REJECTED: dataset has only systolic BP (`trestbps`), no diastolic column.

## 2026-07-02 — Stack decisions
**Decision:** venv+pip with pinned `requirements.txt`; MLflow local file store (`./mlruns`); FastAPI; Docker; GitHub Actions; AWS EKS (ephemeral: create → verify/screenshot → destroy) with ECR as registry; SageMaker notebook available as AWS-side CLI/execution environment.
**Rationale:** Each is the assignment's recommended tool or the lightest option satisfying the rubric; EKS-ephemeral bounds cost.
**Alternatives considered:** Minikube/Docker Desktop (no cloud evidence for report); conda; Flask; Jenkins; k3s-on-EC2.

## 2026-07-02 — EDA figure palette
**Decision:** Fixed two-class palette — no-disease `#2a78d6` (blue), disease `#e34948` (red); diverging blue↔gray↔red for correlation; single-hue blue ramp for prevalence heatmaps. Validated colorblind-safe (worst-pair ΔE 74.6, contrast ≥3:1).
**Rationale:** Consistent, accessible figures across notebook and report; distinct from default-styled tutorial plots.
**Alternatives considered:** seaborn defaults — not validated, visually generic.

## 2026-07-02 — Third model: XGBoost added
**Decision:** Train XGBoost (XGBClassifier, sklearn API) alongside Logistic Regression and Random Forest — 3 models × 2 feature sets = 6 tracked runs.
**Rationale:** Strengthens the model-comparison section; FAQ explicitly lists XGBoost as an accepted choice.
**Alternatives considered:** LR+RF only (meets the ≥2 requirement but a thinner comparison); SVM (no native feature importances, weaker fit for the interpretability angle).

## 2026-07-02 — Phase 2 dependencies pinned
**Decision:** Added scikit-learn, xgboost, mlflow, joblib to requirements.txt, re-frozen with == pins after verifying the full test suite still passes. Installing mlflow forced pandas to be downgraded 3.0.3 → 2.3.3, because mlflow 3.14.0 declares `Requires-Dist: pandas<3` (confirmed via `pip show mlflow` and a `pip install --dry-run pandas==3.0.3 mlflow==3.14.0` check, which raised `ResolutionImpossible: mlflow 3.14.0 depends on pandas<3`); the full test suite was re-verified against the downgraded pin.
**Rationale:** Reproducibility requirement — the pinned set is verified mutually compatible, not assumed.
**Alternatives considered:** Unpinned ranges (irreproducible); conda env (project standardized on venv+pip); keeping pandas 3.0.3 — REJECTED, mlflow's own resolver makes this combination impossible.

## 2026-07-02 — Feature encoding scheme
**Decision:** StandardScaler on the 5 continuous features + `ca` (+ the 3 derived clinical features when enabled); OneHotEncoder(handle_unknown="ignore") on `cp`, `restecg`, `slope`, `thal`; passthrough for the 0/1 features `sex`, `fbs`, `exang`. `ca` is treated numerically because it is an ordinal count of major vessels (0–3), not a nominal code.
**Rationale:** One-hot only where categories are genuinely nominal keeps dimensionality low on 297 rows; handle_unknown="ignore" protects serving-time inputs.
**Alternatives considered:** One-hot everything (wasteful for ordinal/binary); ordinal-encode `thal`/`cp` (imposes a fake ordering on nominal clinical codes).
