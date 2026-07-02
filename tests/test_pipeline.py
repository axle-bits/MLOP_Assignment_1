"""Tests for the preprocessing + model pipeline builder."""
import pickle

import pandas as pd
from sklearn.linear_model import LogisticRegression

from ml.data.preprocess import DEFAULT_CLEAN_PATH, TARGET_COL
from ml.models.pipeline import build_pipeline


def load_xy():
    df = pd.read_csv(DEFAULT_CLEAN_PATH)
    return df.drop(columns=[TARGET_COL]), df[TARGET_COL]


def test_clinical_pipeline_fits_and_predicts_proba_on_real_data():
    X, y = load_xy()
    pipe = build_pipeline(LogisticRegression(max_iter=2000), include_clinical=True)
    pipe.fit(X, y)
    proba = pipe.predict_proba(X)
    assert proba.shape == (len(X), 2)


def test_raw_pipeline_has_no_clinical_step():
    pipe = build_pipeline(LogisticRegression(), include_clinical=False)
    names = dict(pipe.steps)
    assert "clinical" not in names
    assert "preprocess" in names and "model" in names


def test_clinical_pipeline_has_clinical_step_first():
    pipe = build_pipeline(LogisticRegression(), include_clinical=True)
    assert pipe.steps[0][0] == "clinical"


def test_pipeline_pickle_roundtrip_predicts():
    X, y = load_xy()
    pipe = build_pipeline(LogisticRegression(max_iter=2000), include_clinical=True)
    pipe.fit(X, y)
    restored = pickle.loads(pickle.dumps(pipe))
    assert restored.predict(X.head(5)).shape == (5,)
