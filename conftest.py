"""Repo-root conftest: pytest inserts this directory onto sys.path so the
`ml` package imports everywhere, and the fixture below keeps process-global
MLflow state from leaking between tests (mlflow.set_tracking_uri is a
module-level global; several tests and export() set it)."""
import pytest


@pytest.fixture(autouse=True)
def restore_tracking_uri():
    import mlflow

    before = mlflow.get_tracking_uri()
    yield
    mlflow.set_tracking_uri(before)
