import asyncio
import threading
import time

import httpx
import pytest
from fastapi.testclient import TestClient

import main
from config import get_settings
from conftest import make_settings
from main import MAX_QUESTION_CHARS, app
from retrieval_chain import get_retrieval_service
from vector_store import VectorStoreNotFound

#: How long the stubbed retrieval call blocks for in the concurrency test.
SLOW_QUERY_SECONDS = 3.0


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


def test_query_rejects_an_oversized_question_at_the_parser():
    response = build_client().post("/query", json={"question": "x" * (MAX_QUESTION_CHARS + 1)})
    assert response.status_code == 422


def test_query_traces_to_mlflow_after_the_response(monkeypatch):
    logged = []
    monkeypatch.setattr(main, "log_query", lambda *a, **kw: logged.append(a))

    response = build_client().post("/query", json={"question": "what?"})

    assert response.status_code == 200
    assert logged and logged[0][0] == "what?"


def test_a_slow_query_does_not_block_the_event_loop():
    """``/query`` performs blocking network I/O.

    If it is declared ``async def`` it runs on the event loop and every other
    request in the process — health probes included — waits behind it.
    """
    started = threading.Event()
    release = threading.Event()

    class SlowService(StubService):
        def answer(self, question, top_k=None):
            started.set()
            release.wait(timeout=SLOW_QUERY_SECONDS)
            return self.result

    service = SlowService()
    app.dependency_overrides[get_settings] = lambda: make_settings()
    app.dependency_overrides[get_retrieval_service] = lambda: service

    async def scenario():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            begun = time.monotonic()
            slow = asyncio.ensure_future(client.post("/query", json={"question": "q"}))
            for _ in range(300):
                if started.is_set():
                    break
                await asyncio.sleep(0.01)
            assert started.is_set(), "the request never reached the retrieval service"

            health = await client.get("/health")
            # The event loop stayed responsive while the query was in flight;
            # a blocking `async def` endpoint would only let us get here after
            # SLOW_QUERY_SECONDS.
            elapsed = time.monotonic() - begun
            release.set()
            return health, elapsed, await slow

    health, elapsed, slow = asyncio.run(scenario())
    assert health.status_code == 200
    assert slow.status_code == 200
    assert elapsed < SLOW_QUERY_SECONDS, (
        f"/health took {elapsed:.2f}s while /query was running; the event loop was blocked"
    )


def test_query_requires_a_token_when_auth_is_enabled():
    settings = make_settings(auth_enabled=True, jwt_secret="unit-test-secret")
    client = build_client(settings=settings)
    response = client.post("/query", json={"question": "q"})
    assert response.status_code == 401

    response = client.post(
        "/query", json={"question": "q"}, headers={"Authorization": "Bearer nonsense"}
    )
    assert response.status_code == 401
