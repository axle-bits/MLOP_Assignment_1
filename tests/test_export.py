"""Tests for best-run selection and export. Selection logic is tested on
hand-built frames (no store); store-level failure modes use an empty temp
sqlite store — no test here requires the developer's real MLflow store."""
from pathlib import Path

import pandas as pd
import pytest

from ml.models.export import export, pick_best


def runs_frame(rows):
    return pd.DataFrame(rows)


def full_run(run_id, auc):
    return {
        "run_id": run_id,
        "params.quick_mode": "False",
        "metrics.test_roc_auc": auc,
        "tags.model_name": "logistic_regression",
        "tags.feature_set": "clinical",
    }


def test_pick_best_takes_highest_test_roc_auc():
    runs = runs_frame([full_run("a", 0.85), full_run("b", 0.88), full_run("c", 0.83)])
    assert pick_best(runs)["run_id"] == "b"


def test_pick_best_excludes_quick_mode_runs():
    quick = dict(full_run("q", 0.99), **{"params.quick_mode": "True"})
    runs = runs_frame([quick, full_run("a", 0.85)])
    assert pick_best(runs)["run_id"] == "a"


def test_pick_best_raises_on_empty_frame():
    with pytest.raises(RuntimeError, match="No runs"):
        pick_best(pd.DataFrame())


def test_pick_best_raises_when_only_quick_runs():
    quick = dict(full_run("q", 0.99), **{"params.quick_mode": "True"})
    with pytest.raises(RuntimeError, match="full-grid"):
        pick_best(runs_frame([quick]))


def test_export_fails_clearly_when_experiment_missing(tmp_path):
    uri = f"sqlite:///{(tmp_path / 'empty.db').as_posix()}"
    with pytest.raises(RuntimeError, match="not found"):
        export(tracking_uri=uri, out_dir=tmp_path / "models")
