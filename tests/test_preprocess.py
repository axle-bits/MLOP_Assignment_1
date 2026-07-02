"""Unit tests for the data-cleaning functions (assignment Task 5 requires
unit tests for data processing; written test-first in Phase 1)."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml.data.preprocess import (
    CATEGORICAL_COLS,
    CONTINUOUS_COLS,
    TARGET_COL,
    binarize_target,
    cast_types,
    drop_missing,
    preprocess,
)

RAW_CSV = Path("data/raw/heart_cleveland_raw.csv")


def make_raw(num_values, ca=None, thal=None):
    """Minimal raw-schema frame with n rows controlled by num_values."""
    n = len(num_values)
    return pd.DataFrame(
        {
            "age": [50] * n,
            "sex": [1] * n,
            "cp": [3] * n,
            "trestbps": [120] * n,
            "chol": [200] * n,
            "fbs": [0] * n,
            "restecg": [0] * n,
            "thalach": [150] * n,
            "exang": [0] * n,
            "oldpeak": [1.0] * n,
            "slope": [2] * n,
            "ca": ca if ca is not None else [0.0] * n,
            "thal": thal if thal is not None else [3.0] * n,
            "num": num_values,
        }
    )


def test_schema_constants_cover_all_13_features_exactly_once():
    assert len(CATEGORICAL_COLS) + len(CONTINUOUS_COLS) == 13
    assert not set(CATEGORICAL_COLS) & set(CONTINUOUS_COLS)


def test_binarize_target_maps_zero_to_zero_and_one_to_four_to_one():
    out = binarize_target(make_raw([0, 1, 2, 3, 4]))
    assert list(out[TARGET_COL]) == [0, 1, 1, 1, 1]
    assert "num" not in out.columns


def test_binarize_target_does_not_mutate_input():
    df = make_raw([0, 2])
    binarize_target(df)
    assert "num" in df.columns


def test_drop_missing_removes_only_rows_with_nan_ca_or_thal():
    df = make_raw([0, 1, 2], ca=[0.0, np.nan, 1.0], thal=[3.0, 3.0, np.nan])
    out = drop_missing(df)
    assert len(out) == 1
    assert out.index.tolist() == [0]  # index reset after drop


def test_cast_types_yields_integer_ca_and_thal():
    out = cast_types(make_raw([0]))
    assert pd.api.types.is_integer_dtype(out["ca"])
    assert pd.api.types.is_integer_dtype(out["thal"])


@pytest.mark.skipif(not RAW_CSV.exists(), reason="raw CSV not downloaded")
def test_preprocess_end_to_end_on_real_data():
    clean = preprocess(pd.read_csv(RAW_CSV))
    assert clean.shape == (297, 14)
    assert not clean.isna().any().any()
    assert set(clean[TARGET_COL].unique()) == {0, 1}
    # column ORDER is not contractual; presence is
    assert set(clean.columns) == set(CONTINUOUS_COLS) | set(CATEGORICAL_COLS) | {TARGET_COL}
