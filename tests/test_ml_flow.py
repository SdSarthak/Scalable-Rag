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
    def __init__(self, fail=False, fail_configure=False):
        super().__init__("mlflow")
        self.fail = fail
        self.fail_configure = fail_configure
        self.params = {}
        self.metrics = {}
        self.texts = {}
        self.tracking_uri = None
        self.experiment = None
        self.run_attempts = 0
        self.configure_attempts = 0

    def set_tracking_uri(self, uri):
        self.tracking_uri = uri

    def set_experiment(self, name):
        self.configure_attempts += 1
        if self.fail_configure:
            raise RuntimeError("tracking server unreachable")
        self.experiment = name

    def start_run(self, run_name=None):
        self.run_attempts += 1
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


def test_configuration_failures_do_not_propagate(monkeypatch):
    """A tracking server that is down must not fail an otherwise good answer.

    ``set_experiment`` contacts the server, so it has to be inside the
    best-effort boundary just like ``start_run``.
    """
    module = FakeMlflow(fail_configure=True)
    monkeypatch.setitem(sys.modules, "mlflow", module)
    ml_flow.reset_tracing_state()
    settings = make_settings(mlflow_enabled=True, mlflow_tracking_uri="http://down:5000")

    assert ml_flow.log_query("q", "a", [], settings=settings) is False
    assert module.run_attempts == 0


def test_failures_suspend_tracing_instead_of_retrying_every_query(monkeypatch):
    module = FakeMlflow(fail=True)
    monkeypatch.setitem(sys.modules, "mlflow", module)
    ml_flow.reset_tracing_state()
    settings = make_settings(mlflow_enabled=True)

    for _ in range(5):
        assert ml_flow.log_query("q", "a", [], settings=settings) is False

    # Only the first query pays the cost of contacting the dead server.
    assert module.run_attempts == 1


def test_tracing_resumes_after_the_backoff_window(monkeypatch):
    module = FakeMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", module)
    ml_flow.reset_tracing_state()
    settings = make_settings(mlflow_enabled=True)

    module.fail = True
    assert ml_flow.log_query("q", "a", [], settings=settings) is False
    module.fail = False
    assert ml_flow.log_query("q", "a", [], settings=settings) is False  # still suspended

    monkeypatch.setattr(ml_flow, "_suspended_until", 0.0)
    assert ml_flow.log_query("q", "a", [], settings=settings) is True


def test_configuration_happens_once_across_queries(fake_mlflow):
    settings = make_settings(mlflow_enabled=True)
    for _ in range(3):
        assert ml_flow.log_query("q", "a", [], settings=settings) is True
    assert fake_mlflow.configure_attempts == 1
