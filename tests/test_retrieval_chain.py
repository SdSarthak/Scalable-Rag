import threading
import time

import pytest
from langchain_core.documents import Document

import retrieval_chain
from conftest import make_settings
from retrieval_chain import (
    MAX_CACHED_CHAINS,
    PROMPT_TEMPLATE,
    RetrievalService,
    serialize_document,
)


class FakeChain:
    """Stands in for a LangChain RetrievalQA chain."""

    def __init__(self, answer="42", documents=None):
        self.answer = answer
        self.documents = documents or []
        self.calls = []

    def invoke(self, payload):
        self.calls.append(payload)
        return {"result": self.answer, "source_documents": self.documents}


def build_service(**overrides):
    documents = [Document(page_content="ground truth", metadata={"source": "docs/a.md"})]
    chain = FakeChain(answer="  42  ", documents=documents)
    service = RetrievalService(make_settings(**overrides), chain=chain)
    return service, chain


def test_answer_returns_serialised_sources():
    service, chain = build_service()
    result = service.answer("what is the answer?")

    assert result["answer"] == "42"
    assert result["sources"] == [
        {
            "content": "ground truth",
            "source": "docs/a.md",
            "metadata": {"source": "docs/a.md"},
        }
    ]
    assert result["latency_ms"] >= 0
    assert chain.calls == [{"query": "what is the answer?"}]


def test_answer_rejects_empty_questions():
    service, _ = build_service()
    with pytest.raises(ValueError):
        service.answer("   ")


def test_answer_rejects_oversized_questions():
    service, _ = build_service(max_question_length=10)
    with pytest.raises(ValueError):
        service.answer("x" * 11)


def test_chain_is_reused_for_the_default_top_k():
    service, chain = build_service(retrieval_k=5)
    assert service.chain_for() is chain
    assert service.chain_for(5) is chain


def test_is_ready_reflects_a_built_chain():
    service, _ = build_service()
    assert service.is_ready() is True
    assert RetrievalService(make_settings(index_path="does-not-exist")).is_ready() is False


def test_not_ready_without_an_api_key(tmp_path):
    """A complete index is useless if the LLM credentials are missing."""
    index = tmp_path / "index"
    index.mkdir()
    (index / "index.faiss").write_bytes(b"stub")
    (index / "index.pkl").write_bytes(b"stub")

    ready = make_settings(index_path=str(index))
    assert RetrievalService(ready).is_ready() is True
    assert RetrievalService(make_settings(index_path=str(index), openai_api_key="")).is_ready() is False


def test_zero_top_k_is_rejected_rather_than_silently_defaulted():
    service, chain = build_service(retrieval_k=5)
    with pytest.raises(ValueError):
        service.chain_for(0)
    with pytest.raises(ValueError):
        service.chain_for(-3)


def test_chain_cache_is_bounded(monkeypatch):
    service, _ = build_service(retrieval_k=1)
    built = []

    def fake_build(top_k):
        built.append(top_k)
        return f"chain-{top_k}"

    monkeypatch.setattr(service, "_build_chain", fake_build)
    for top_k in range(2, MAX_CACHED_CHAINS + 6):
        service.chain_for(top_k)

    assert len(service._chains) <= MAX_CACHED_CHAINS
    assert built == list(range(2, MAX_CACHED_CHAINS + 6))


def test_concurrent_first_requests_build_the_chain_once(monkeypatch):
    """Requests run in a thread pool, so the lazy build has to be serialised."""
    service = RetrievalService(make_settings(retrieval_k=4))
    calls = []
    start = threading.Barrier(4)

    def slow_build(top_k):
        calls.append(top_k)
        time.sleep(0.05)
        return f"chain-{top_k}"

    monkeypatch.setattr(service, "_build_chain", slow_build)
    results = []

    def worker():
        start.wait(timeout=5)
        results.append(service.chain_for())

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    # All four callers share one chain; only the first paid to build it.
    assert calls == [4]
    assert results == ["chain-4"] * 4


def test_serialize_document_handles_missing_metadata():
    assert serialize_document(Document(page_content="x")) == {
        "content": "x",
        "source": "unknown",
        "metadata": {},
    }


def test_prompt_is_grounded():
    assert "{context}" in PROMPT_TEMPLATE and "{question}" in PROMPT_TEMPLATE
    assert "do not know" in PROMPT_TEMPLATE
