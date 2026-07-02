"""Cleaning steps for the Cleveland heart-disease data.

Schema constants here are the single source of truth for later phases
(feature engineering, API input validation).
"""
from pathlib import Path

import pandas as pd

from ml.data.download import DEFAULT_RAW_PATH

# Integer-coded categorical clinical features vs true continuous measurements.
CATEGORICAL_COLS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
CONTINUOUS_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
TARGET_COL = "target"

DEFAULT_CLEAN_PATH = Path("data/processed/heart_cleveland_clean.csv")


def binarize_target(df: pd.DataFrame) -> pd.DataFrame:
    """Map raw `num` severity (0-4) to binary `target` (0 = no disease)."""
    out = df.copy()
    out[TARGET_COL] = (out["num"] > 0).astype("int64")
    return out.drop(columns=["num"])


def drop_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the rows with missing `ca`/`thal` (6 of 303; both categorical,
    imputing them would invent clinical facts)."""
    return df.dropna(subset=["ca", "thal"]).reset_index(drop=True)


def cast_types(df: pd.DataFrame) -> pd.DataFrame:
    """Restore integer dtype on `ca`/`thal` (float only because of the NaNs)."""
    out = df.copy()
    out["ca"] = out["ca"].astype("int64")
    out["thal"] = out["thal"].astype("int64")
    return out


def preprocess(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Full cleaning: binarize target, drop missing, fix dtypes."""
    return cast_types(drop_missing(binarize_target(raw_df)))


if __name__ == "__main__":
    clean = preprocess(pd.read_csv(DEFAULT_RAW_PATH))
    DEFAULT_CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(DEFAULT_CLEAN_PATH, index=False, lineterminator="\n")
    print(f"Wrote {DEFAULT_CLEAN_PATH} shape={clean.shape}")
    print(clean[TARGET_COL].value_counts().to_dict())
