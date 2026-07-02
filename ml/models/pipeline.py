"""Build the preprocessing + estimator sklearn Pipeline.

Encoding scheme (rationale in docs/DECISIONS.md):
- CONTINUOUS_COLS + ca (ordinal vessel count 0-3) -> StandardScaler
- cp, restecg, slope, thal (multi-class categoricals) -> OneHotEncoder
- sex, fbs, exang (already 0/1) -> passthrough
Derived clinical features are appended inside the pipeline so the persisted
model accepts the raw 13-column schema end-to-end.
"""
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from ml.data.preprocess import CONTINUOUS_COLS
from ml.features.clinical import CLINICAL_FEATURES, add_clinical_features

MULTICLASS_COLS = ["cp", "restecg", "slope", "thal"]
BINARY_COLS = ["sex", "fbs", "exang"]
ORDINAL_COLS = ["ca"]


def build_pipeline(model, include_clinical: bool) -> Pipeline:
    """Assemble (optional clinical features ->) preprocessing -> model."""
    scaled_cols = CONTINUOUS_COLS + ORDINAL_COLS + (
        CLINICAL_FEATURES if include_clinical else []
    )
    preprocess = ColumnTransformer(
        [
            ("scale", StandardScaler(), scaled_cols),
            ("onehot", OneHotEncoder(handle_unknown="ignore"), MULTICLASS_COLS),
            ("binary", "passthrough", BINARY_COLS),
        ]
    )
    steps = []
    if include_clinical:
        steps.append(("clinical", FunctionTransformer(add_clinical_features)))
    steps.append(("preprocess", preprocess))
    steps.append(("model", model))
    return Pipeline(steps)
