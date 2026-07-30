import pytest
from langchain_core.documents import Document

from conftest import make_settings
from retrieval_chain import PROMPT_TEMPLATE, RetrievalService, serialize_document


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


def test_serialize_document_handles_missing_metadata():
    assert serialize_document(Document(page_content="x")) == {
        "content": "x",
        "source": "unknown",
        "metadata": {},
    }


def test_prompt_is_grounded():
    assert "{context}" in PROMPT_TEMPLATE and "{question}" in PROMPT_TEMPLATE
    assert "do not know" in PROMPT_TEMPLATE
