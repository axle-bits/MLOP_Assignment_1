"""Derived clinical features for the risk-stratification theme.

Embedded in the serving pipeline via FunctionTransformer, so
`add_clinical_features` must remain a module-level named function
(picklable). See docs/DECISIONS.md for the clinical rationale.
"""
import pandas as pd

CLINICAL_FEATURES = ["rate_pressure_product", "heart_rate_reserve", "oldpeak_slope"]


def add_clinical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append the three derived clinical features; input is not mutated.

    - rate_pressure_product: trestbps × thalach — index of myocardial
      oxygen demand used in stress testing.
    - heart_rate_reserve: 220 − age − thalach — unused chronotropic capacity.
    - oldpeak_slope: oldpeak × slope — ST-depression magnitude × slope
      pattern interaction.
    """
    out = df.copy()
    out["rate_pressure_product"] = out["trestbps"] * out["thalach"]
    out["heart_rate_reserve"] = 220 - out["age"] - out["thalach"]
    out["oldpeak_slope"] = out["oldpeak"] * out["slope"]
    return out
