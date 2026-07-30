import sys
import types

import pytest

import ml_flow
from conftest import make_settings


class FakeRun:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeMlflow(types.ModuleType):
    def __init__(self, fail=False):
        super().__init__("mlflow")
        self.fail = fail
        self.params = {}
        self.metrics = {}
        self.texts = {}
        self.tracking_uri = None
        self.experiment = None

    def set_tracking_uri(self, uri):
        self.tracking_uri = uri

    def set_experiment(self, name):
        self.experiment = name

    def start_run(self, run_name=None):
        if self.fail:
            raise RuntimeError("tracking server unreachable")
        return FakeRun()

    def log_param(self, key, value):
        self.params[key] = value

    def log_metric(self, key, value):
        self.metrics[key] = value

    def log_text(self, text, artifact_file):
        self.texts[artifact_file] = text


@pytest.fixture
def fake_mlflow(monkeypatch):
    module = FakeMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", module)
    ml_flow.reset_tracing_state()
    yield module
    ml_flow.reset_tracing_state()


def test_logging_is_skipped_when_disabled(fake_mlflow):
    settings = make_settings(mlflow_enabled=False)
    assert ml_flow.log_query("q", "a", [], settings=settings) is False
    assert fake_mlflow.experiment is None


def test_logging_records_params_metrics_and_artifacts(fake_mlflow):
    settings = make_settings(
        mlflow_enabled=True,
        mlflow_tracking_uri="http://mlflow.internal:5000",
        mlflow_experiment="rag-tests",
    )
    sources = [{"source": "docs/a.md"}, {"source": "docs/b.md"}]

    assert ml_flow.log_query("q?", "an answer", sources, latency_ms=12.5, settings=settings)

    assert fake_mlflow.tracking_uri == "http://mlflow.internal:5000"
    assert fake_mlflow.experiment == "rag-tests"
    assert fake_mlflow.metrics["source_count"] == 2
    assert fake_mlflow.metrics["latency_ms"] == 12.5
    assert fake_mlflow.texts["answer.txt"] == "an answer"
    assert fake_mlflow.texts["sources.txt"] == "docs/a.md\ndocs/b.md"


def test_tracking_failures_do_not_propagate(monkeypatch):
    monkeypatch.setitem(sys.modules, "mlflow", FakeMlflow(fail=True))
    ml_flow.reset_tracing_state()
    settings = make_settings(mlflow_enabled=True)
    assert ml_flow.log_query("q", "a", [], settings=settings) is False


def test_missing_mlflow_package_is_tolerated(monkeypatch):
    monkeypatch.setitem(sys.modules, "mlflow", None)
    ml_flow.reset_tracing_state()
    settings = make_settings(mlflow_enabled=True)
    assert ml_flow.log_query("q", "a", [], settings=settings) is False
