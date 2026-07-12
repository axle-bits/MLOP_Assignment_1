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
# # Model Inference Pipeline Demo
#
# This notebook demonstrates how to load the exported model pipeline artifact 
# (`models/heart_disease_pipeline.joblib`) and perform inference on sample patient records.
# It validates that the pipeline correctly handles feature engineering, scaling, and 
# prediction without requiring a running API server or MLflow server.

# %%
import json
import sys
from pathlib import Path
import joblib
import pandas as pd

# Resolve repository root path
ROOT = Path.cwd()
if not (ROOT / "models").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

# Load the exported production pipeline
pipeline_path = ROOT / "models" / "heart_disease_pipeline.joblib"
metadata_path = ROOT / "models" / "model_metadata.json"
sample_request_path = ROOT / "api" / "sample_request.json"

print(f"Loading pipeline from: {pipeline_path.relative_to(ROOT)}")
pipeline = joblib.load(pipeline_path)

# Show exported model metadata details
with open(metadata_path, "r") as f:
    metadata = json.load(f)
print("\nExported Model Metadata:")
print(f"  Source Run ID: {metadata['run_id']}")
print(f"  Model Family:  {metadata['model_name']}")
print(f"  Test ROC-AUC:  {metadata['metrics']['test_roc_auc']:.4f}")

# %% [markdown]
# ## 1. Load Sample Patient Data
#
# Load the standard request payload defined in `api/sample_request.json` representing a patient record.

# %%
with open(sample_request_path, "r") as f:
    sample_data = json.load(f)

# Convert to DataFrame matching expected input columns
input_df = pd.DataFrame([sample_data])
print("Input Patient Features:")
print(input_df.to_string(index=False))

# %% [markdown]
# ## 2. Execute Inference Pipeline
#
# Pass the raw features through the loaded pipeline. The pipeline handles:
# - Derived feature calculation (Rate-Pressure Product, Heart Rate Reserve, etc.)
# - Standard scaling of continuous features
# - One-hot encoding of categorical variables
# - Model prediction

# %%
prediction = pipeline.predict(input_df)[0]
probability = pipeline.predict_proba(input_df)[0][1]

print("\nInference Results:")
print(f"  Risk Label Prediction: {prediction} ({'Heart Disease Risk Present' if prediction == 1 else 'No Risk Detected'})")
print(f"  Probability of Risk:   {probability:.4%}")

# %% [markdown]
# ## 3. Batch Inference on Clean Processed Dataset
#
# Load the processed dataset to showcase batch prediction and evaluate run-time inference performance.

# %%
processed_csv_path = ROOT / "data" / "processed" / "heart_cleveland_clean.csv"
if processed_csv_path.exists():
    df = pd.read_csv(processed_csv_path)
    X = df.drop(columns=["target"])
    y = df["target"]
    
    # Run batch prediction
    df["predicted_risk"] = pipeline.predict(X)
    df["predicted_probability"] = pipeline.predict_proba(X)[:, 1]
    
    print(f"\nBatch Inference run on {len(X)} records.")
    print("Sample of batch predictions (first 5 patients):")
    print(df[["predicted_risk", "predicted_probability", "target"]].head().to_string(index=False))
else:
    print(f"\nProcessed dataset not found at: {processed_csv_path}")
