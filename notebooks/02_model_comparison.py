# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Model Comparison  -  did clinical feature engineering help?
#
# Reads the MLflow experiment `heart-disease-risk` (populated by
# `python -m ml.models.train`)  -  no retraining happens here.
# Research question: do the derived clinical features (rate-pressure
# product, heart-rate reserve, oldpeak×slope) improve prediction over the
# 13 raw features?

# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import pandas as pd

ROOT = Path.cwd()
if not (ROOT / "data").exists():  # running from notebooks/
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from ml.features.clinical import CLINICAL_FEATURES
from ml.models.train import EXPERIMENT_NAME

# The Task 5 training run that populated this repo's experiment was tracked
# against the local sqlite backend (./mlflow.db), with artifacts materialized
# under ./mlruns/<experiment_id>/<run_id>/artifacts/  -  there is no file-store
# meta.yaml under ./mlruns (verified: no meta.yaml anywhere in the tree), so
# a pure `(ROOT / "mlruns").as_uri()` tracking URI cannot find the runs. Point
# at the sqlite db, which is where the run/param/metric/tag data actually
# lives; run artifacts (incl. the logged model) still resolve correctly since
# their recorded artifact_uri points into ./mlruns. See docs/DECISIONS.md.
mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")

FIG_DIR = ROOT / "docs" / "figures" / "models"
FIG_DIR.mkdir(parents=True, exist_ok=True)
C_NO, C_YES = "#2a78d6", "#e34948"  # raw / clinical (Phase 1 palette)

METRICS = ["cv_roc_auc_mean", "test_accuracy", "test_precision", "test_recall", "test_f1", "test_roc_auc"]

# %% [markdown]
# ## 1. All runs

# %%
runs = mlflow.search_runs(experiment_names=[EXPERIMENT_NAME])
full = runs[runs["params.quick_mode"] == "False"].copy()
assert len(full) == 6, (
    f"expected exactly one full-grid run per combo, found {len(full)}  -  "
    "rerun `python -m ml.models.train` into a fresh store"
)
table = full[
    ["tags.model_name", "tags.feature_set"] + [f"metrics.{m}" for m in METRICS]
].rename(columns=lambda c: c.split(".", 1)[-1])
table = table.sort_values("test_roc_auc", ascending=False).reset_index(drop=True)
table.round(4)

# %% [markdown]
# ## 2. Test ROC-AUC by model, raw vs clinical

# %%
pivot = table.pivot(index="model_name", columns="feature_set", values="test_roc_auc")
fig, ax = plt.subplots(figsize=(6.5, 3.6))
pivot[["raw", "clinical"]].plot(kind="bar", ax=ax, color=[C_NO, C_YES], width=0.7)
ax.set_ylabel("test ROC-AUC")
ax.set_ylim(0.5, 1.0)
ax.set_title("Test ROC-AUC  -  raw vs clinical feature set")
ax.legend(title="feature set")
ax.tick_params(axis="x", rotation=0)
for container in ax.containers:
    ax.bar_label(container, fmt="%.3f", color="#52514e", fontsize=8)
fig.tight_layout()
fig.savefig(FIG_DIR / "roc_auc_by_model.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 3. Clinical-features delta per model

# %%
delta = (pivot["clinical"] - pivot["raw"]).rename("roc_auc_delta").to_frame()
recall_pivot = table.pivot(index="model_name", columns="feature_set", values="test_recall")
delta["recall_delta"] = recall_pivot["clinical"] - recall_pivot["raw"]
fig, ax = plt.subplots(figsize=(6, 3.2))
x = range(len(delta))
ax.bar([i - 0.18 for i in x], delta["roc_auc_delta"], width=0.36, color=C_YES, label="ROC-AUC Δ")
ax.bar([i + 0.18 for i in x], delta["recall_delta"], width=0.36, color=C_NO, label="recall Δ")
ax.axhline(0, color="#c3c2b7", linewidth=1)
ax.set_xticks(list(x), delta.index)
ax.set_ylabel("clinical − raw")
ax.set_title("Effect of derived clinical features (test split)")
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "clinical_delta.png", dpi=150, bbox_inches="tight")
plt.show()
delta.round(4)

# %% [markdown]
# ## 4. Do the derived features rank highly?
#
# Load the best clinical-feature-set run's pipeline and inspect where the
# three derived features land in its importance ranking.

# %%
best_clinical = table[table["feature_set"] == "clinical"].iloc[0]
best_run = full[
    (full["tags.model_name"] == best_clinical["model_name"])
    & (full["tags.feature_set"] == "clinical")
].iloc[0]
pipe = mlflow.sklearn.load_model(f"runs:/{best_run.run_id}/model")
names = pipe.named_steps["preprocess"].get_feature_names_out()
model = pipe.named_steps["model"]
values = model.coef_[0] if hasattr(model, "coef_") else model.feature_importances_
ranking = (
    pd.Series(abs(pd.Series(values, index=names)), name="abs_weight")
    .sort_values(ascending=False)
    .reset_index()
    .rename(columns={"index": "feature"})
)
ranking["rank"] = ranking.index + 1
ranking[ranking["feature"].str.contains("|".join(CLINICAL_FEATURES))]

# %% [markdown]
# ## 5. Findings
#
# - Best combo overall: **logistic_regression + clinical**, test ROC-AUC
#   **0.8817** (run `514db551...`), narrowly ahead of logistic_regression/raw
#   (0.8761); the full range across all 6 runs is 0.8315 (xgboost/raw) to
#   0.8817.
# - Clinical-vs-raw delta is model-dependent, not uniformly positive:
#   logistic_regression +0.0056 ROC-AUC (recall unchanged, 0.0), xgboost
#   +0.0100 ROC-AUC (its best score, still last place), but random_forest
#   **-0.0089** ROC-AUC  -  clinical features slightly hurt the forest. Recall
#   tells a starker story: it drops -0.1071 (random_forest) and -0.1429
#   (xgboost) with clinical features, while staying flat for logistic
#   regression  -  the derived features cost the tree models real recall even
#   where ROC-AUC ticks up.
# - In the best clinical model (logistic_regression), the three derived
#   features rank in the bottom half of all 25 post-encoding features:
#   `oldpeak_slope` 15/25, `heart_rate_reserve` 22/25, `rate_pressure_product`
#   24/25 (second-to-last). The top of the ranking is dominated by raw
#   features  -  `sex`, `ca`, `cp_4`, `thal_7`, `slope_2`  -  none of which are
#   engineered.
# - CV-vs-test consistency: cv_roc_auc_mean (0.9148-0.9313) sits 0.04-0.09
#   above test_roc_auc for every run. The gap is smallest and most stable for
#   logistic_regression (~0.04, both feature sets) and largest for xgboost
#   (~0.09-0.09), a mild overfitting signal for the boosted trees relative to
#   the linear model.
# - **Research question answered:** the derived clinical features (rate-
#   pressure product, heart-rate reserve, oldpeak×slope) do not clearly
#   improve prediction over the 13 raw features  -  the ROC-AUC effect is a
#   small, model-dependent wash (+0.006 to +0.010 for two models, -0.009 for
#   the third) that comes with a real recall cost for the tree-based models,
#   and the engineered features rank near the bottom of importance even in
#   the one model where clinical narrowly wins.

# %%
print("Comparison complete; figures in", FIG_DIR)
