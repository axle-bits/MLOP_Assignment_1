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
**Decision:** venv+pip with pinned `requirements.txt`; MLflow local file store (`./mlruns`); FastAPI; Docker; GitHub Actions; AWS EKS (ephemeral: create → verify/screenshot → destroy) with ECR as registry; SageMaker notebook available as AWS-side CLI/execution environment. [Superseded 2026-07-02: mlflow 3.14 deprecates the pure file store; actual setup is sqlite:///mlflow.db metadata + ./mlruns artifacts — see the "mlflow 3.14.0 API adaptations" and tracking-URI entries below.]
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

## 2026-07-02 — Evaluation protocol & tuning
**Decision:** Single stratified 80/20 split with random_state=785 (personalized seed derived from student ID), shared across all 6 runs. GridSearchCV with StratifiedKFold(5, shuffle=True, random_state=785), scoring=roc_auc, tuned on the training split only; the test split is evaluated exactly once per run. Metrics logged: CV mean±std ROC-AUC + test accuracy/precision/recall/F1/ROC-AUC.
**Rationale:** A shared split makes run-to-run differences attributable to model/features alone; touching the test set once avoids selection leakage; ROC-AUC handles the mild 160/137 imbalance.
**Alternatives considered:** RandomizedSearchCV (grids are small enough to enumerate); nested CV (more rigorous but overkill for 297 rows and harder to present); seed 42 (tutorial-universal, weaker originality).

## 2026-07-02 — mlflow 3.14.0 API adaptations in the training script
**Decision:** Three deviations from a literal transcription of the Task 4 brief's `ml/models/train.py`, all forced by the installed mlflow 3.14.0 / scikit-learn 1.9.0 versions: (1) `os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")`, set inside `run_experiments()` immediately before `mlflow.set_tracking_uri`/`set_experiment` (not at module import time), because mlflow 3.x raises on any `file://`/`./mlruns` tracking URI unless this opt-out is set. Empirically verified: removing the line and running `pytest tests/test_train.py` reproduces `mlflow.exceptions.MlflowException: The filesystem tracking backend (e.g., './mlruns') is in maintenance mode and will not receive further updates. Please migrate to a database backend (e.g., 'sqlite:///mlflow.db') to access the latest MLflow features. ... If the filesystem backend is required for your workflow, set \`MLFLOW_ALLOW_FILE_STORE=true\` to opt out of this exception.` raised from `FileStore.__init__`; re-adding the line (scoped to the function) makes the test pass again. (2) `RocCurveDisplay.from_estimator(..., curve_kwargs={"color": C_YES})` instead of the removed direct `color=` kwarg; (3) a `log_model_as_run_artifact()` helper that calls `mlflow.sklearn.save_model()` to a temp dir plus `mlflow.log_artifacts(local_path, artifact_path="model")`, instead of `mlflow.sklearn.log_model(..., name="model", ...)` directly — mlflow 3.x's `log_model` now always records the model as a decoupled "LoggedModel" entity under the experiment's `models/` tree (not the run's own artifact root), so it never appears via `client.list_artifacts(run_id)` regardless of `name=`/`artifact_path=`; the save+log_artifacts route restores the classic run-scoped `model/` artifact folder the test (and downstream tasks) expect, and is now verified end-to-end by a `mlflow.sklearn.load_model("runs:/<id>/model")` + `.predict()` check in the test. Also switched `serialization_format` to `SERIALIZATION_FORMAT_CLOUDPICKLE` because the pipeline's `FunctionTransformer` wraps a first-party function (`ml.features.clinical.add_clinical_features`) that mlflow's new default `skops` format refuses to deserialize as an "untrusted type". An explicit `signature=infer_signature(X_example, pipe.predict(X_example))` was added to `save_model`, with `X_example` cast to `float64` first — empirically, `infer_signature` alone still left 6 `UserWarning`s ("Integer columns in Python cannot represent missing values...") across the 6 runs; adding the `.astype("float64")` cast on top eliminated all 6.
**Rationale:** These are genuine behavioral/API changes between when the brief was authored and the pinned mlflow 3.14.0, not implementation choices — the alternative (downgrading mlflow) would contradict the already-verified, mutually-compatible pinned dependency set. Scoping the env-var set to the function (rather than import time) limits its effect to when tracking actually runs.
**Alternatives considered:** Pin an older mlflow without these changes — REJECTED, contradicts Task 3's verified pin; pass `skops_trusted_types=["ml.features.clinical.add_clinical_features"]` to keep `skops` — works for serialization but does not solve the separate "model not in run artifacts" issue, so cloudpickle + explicit run-artifact upload was simpler and covers both problems at once; `infer_signature` without the float64 cast — REJECTED, verified empirically to still emit the integer-column warning.

## 2026-07-02 — Model-comparison notebook reads the sqlite tracking store, not a pure `./mlruns` file store
**Decision:** In `notebooks/02_model_comparison.py`, `mlflow.set_tracking_uri` points at `sqlite:///{ROOT}/mlflow.db` instead of the brief's literal `(ROOT / "mlruns").as_uri()`. Empirically verified: the six Task 5 full-mode runs have **no `meta.yaml` anywhere under `./mlruns`** (confirmed via a recursive glob for `*.yaml` — only `conda.yaml`/`python_env.yaml` model-env files exist under each run's `artifacts/model/`), so a pure file-store tracking URI finds an empty/malformed experiment (`WARNING:root:Malformed experiment '1' ... Yaml file '...\mlruns\1\meta.yaml' does not exist`) and `search_runs` returns 0 rows, raising `KeyError: 'params.quick_mode'` downstream. The six runs' actual params/metrics/tags live in `./mlflow.db` (sqlite); `SELECT artifact_location FROM experiments` there resolves to `file:.../mlruns/1`, and `SELECT test_roc_auc FROM metrics` for the six runs reproduces the known results exactly (best `logistic_regression`/`clinical` 0.881696..., range 0.831473...-0.881696...), confirming this sqlite db is the genuine, complete Task 5 output — the runs were tracked against sqlite with `./mlruns` retained only as the artifact root, not a pure file store. Loading models via `mlflow.sklearn.load_model(f"runs:/{run_id}/model")` still resolves correctly since each run's recorded `artifact_uri` points into `./mlruns`.
**Rationale:** The notebook's job is to read whatever tracking store Task 5 actually used, not to re-litigate Task 5's setup; retraining or migrating the store is out of scope ("no retraining happens here"). Pointing at the sqlite db is the minimal change that makes the existing, verified-correct run data readable.
**Alternatives considered:** Re-run `python -m ml.models.train` to regenerate a pure-file-store `./mlruns` — REJECTED, would destroy/duplicate the existing verified 6-run experiment and contradicts the notebook's read-only contract; `mlflow migrate-filestore` — inapplicable, migrates file-store → db, not the reverse; keep the literal `(ROOT / "mlruns").as_uri()` — REJECTED, empirically returns zero runs.
