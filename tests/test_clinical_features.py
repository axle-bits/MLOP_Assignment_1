"""Unit tests for the derived clinical features."""
import pandas as pd

from ml.features.clinical import CLINICAL_FEATURES, add_clinical_features


def sample_frame():
    return pd.DataFrame(
        {"age": [50], "trestbps": [120], "thalach": [150], "oldpeak": [2.0], "slope": [2]}
    )


def test_values_hand_computed():
    out = add_clinical_features(sample_frame())
    assert out.loc[0, "rate_pressure_product"] == 120 * 150  # 18000
    assert out.loc[0, "heart_rate_reserve"] == 220 - 50 - 150  # 20
    assert out.loc[0, "oldpeak_slope"] == 2.0 * 2  # 4.0


def test_adds_exactly_the_declared_columns_in_order():
    df = sample_frame()
    out = add_clinical_features(df)
    assert list(out.columns) == list(df.columns) + CLINICAL_FEATURES


def test_input_not_mutated():
    df = sample_frame()
    add_clinical_features(df)
    assert "rate_pressure_product" not in df.columns
