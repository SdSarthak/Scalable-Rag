import pytest
from fastapi.testclient import TestClient

from config import get_settings
from conftest import make_settings
from main import app
from retrieval_chain import get_retrieval_service
from vector_store import VectorStoreNotFound


class StubService:
    def __init__(self, result=None, error=None, ready=True):
        self.result = result or {
            "answer": "42",
            "sources": [{"content": "ground truth", "source": "docs/a.md", "metadata": {}}],
            "latency_ms": 1.0,
            "model": "gpt-4o-mini",
        }
        self.error = error
        self.ready = ready
        self.calls = []

    def answer(self, question, top_k=None):
        self.calls.append((question, top_k))
        if self.error is not None:
            raise self.error
        return self.result

    def is_ready(self):
        return self.ready


def build_client(service=None, settings=None):
    app.dependency_overrides[get_settings] = lambda: settings or make_settings()
    app.dependency_overrides[get_retrieval_service] = lambda: service or StubService()
    return TestClient(app)


@pytest.fixture(autouse=True)
def clear_overrides():
    yield
    app.dependency_overrides.clear()


def test_health_is_always_available():
    response = build_client().get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_ready_reports_index_availability():
    assert build_client(StubService(ready=True)).get("/ready").status_code == 200

    response = build_client(StubService(ready=False)).get("/ready")
    assert response.status_code == 503
    assert response.json()["status"] == "index_unavailable"


def test_metrics_are_exposed_in_prometheus_format():
    response = build_client().get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    assert "query_requests_total" in response.text


def test_query_returns_answer_and_serialisable_sources():
    service = StubService()
    response = build_client(service).post("/query", json={"question": "what?", "top_k": 3})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "42"
    assert body["sources"][0]["source"] == "docs/a.md"
    assert body["model"] == "gpt-4o-mini"
    assert service.calls == [("what?", 3)]


def test_query_adds_a_process_time_header():
    response = build_client().post("/query", json={"question": "what?"})
    assert float(response.headers["X-Process-Time-Ms"]) >= 0


def test_query_rejects_an_empty_question():
    assert build_client().post("/query", json={"question": ""}).status_code == 422


def test_query_rejects_an_out_of_range_top_k():
    assert build_client().post("/query", json={"question": "q", "top_k": 0}).status_code == 422


def test_query_returns_503_when_the_index_is_missing():
    service = StubService(error=VectorStoreNotFound("no FAISS index"))
    response = build_client(service).post("/query", json={"question": "q"})
    assert response.status_code == 503
    assert "no FAISS index" in response.json()["detail"]


def test_query_returns_400_for_invalid_input():
    service = StubService(error=ValueError("question must not be empty"))
    response = build_client(service).post("/query", json={"question": "q"})
    assert response.status_code == 400


def test_query_returns_502_when_the_provider_fails():
    service = StubService(error=RuntimeError("openai timeout"))
    response = build_client(service).post("/query", json={"question": "q"})
    assert response.status_code == 502
    assert "openai timeout" not in response.text  # upstream detail is not leaked


def test_query_requires_a_token_when_auth_is_enabled():
    settings = make_settings(auth_enabled=True, jwt_secret="unit-test-secret")
    client = build_client(settings=settings)
    response = client.post("/query", json={"question": "q"})
    assert response.status_code == 401

    response = client.post(
        "/query", json={"question": "q"}, headers={"Authorization": "Bearer nonsense"}
    )
    assert response.status_code == 401
