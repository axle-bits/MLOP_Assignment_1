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
