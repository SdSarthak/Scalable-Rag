"""The retrieval-augmented generation pipeline.

``RetrievalService`` owns the FAISS-backed retriever and the grounded QA chain.
Everything is built lazily: importing this module never touches OpenAI or the
index, which keeps the API process able to start (and report ``/ready``) even
when the corpus has not been indexed yet.
"""

import logging
import time
from functools import lru_cache
from typing import Any, Dict, List, Optional

from config import Settings, get_settings
from vector_store import VectorStoreNotFound, index_exists, load_vector_store

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """You are a retrieval assistant. Answer the question using only the
context below. If the context does not contain the answer, say that you do not know
instead of guessing. Keep the answer concise and cite the source names you relied on.

Context:
{context}

Question: {question}

Answer:"""


def _build_prompt():
    try:
        from langchain_core.prompts import PromptTemplate
    except ImportError:  # pragma: no cover - older langchain layout
        from langchain.prompts import PromptTemplate
    return PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])


def build_llm(settings: Optional[Settings] = None):
    """Instantiate the chat model used to synthesise answers."""
    settings = settings or get_settings()
    settings.require_llm()
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:  # pragma: no cover - older langchain layout
        from langchain_community.chat_models import ChatOpenAI
    return ChatOpenAI(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        api_key=settings.openai_api_key,
    )


def serialize_document(document: Any) -> Dict[str, Any]:
    """Convert a LangChain ``Document`` into a JSON-serialisable dict."""
    metadata = dict(getattr(document, "metadata", {}) or {})
    return {
        "content": getattr(document, "page_content", ""),
        "source": metadata.get("source", "unknown"),
        "metadata": metadata,
    }


class RetrievalService:
    """Answers questions against the indexed corpus."""

    def __init__(self, settings: Optional[Settings] = None, chain: Any = None):
        self.settings = settings or get_settings()
        self._vector_store = None
        self._chains: Dict[int, Any] = {}
        if chain is not None:
            self._chains[self.settings.retrieval_k] = chain

    # --- wiring ----------------------------------------------------------
    @property
    def vector_store(self):
        if self._vector_store is None:
            self._vector_store = load_vector_store(self.settings)
        return self._vector_store

    def _build_chain(self, top_k: int):
        try:
            from langchain.chains import RetrievalQA
        except ImportError as exc:  # pragma: no cover - missing dependency
            raise RuntimeError("langchain is required to build the QA chain") from exc

        retriever = self.vector_store.as_retriever(search_kwargs={"k": top_k})
        return RetrievalQA.from_chain_type(
            llm=build_llm(self.settings),
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": _build_prompt()},
        )

    def chain_for(self, top_k: Optional[int] = None):
        top_k = top_k or self.settings.retrieval_k
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        if top_k not in self._chains:
            self._chains[top_k] = self._build_chain(top_k)
        return self._chains[top_k]

    def warmup(self) -> bool:
        """Pre-load the index and the default chain. Returns True on success."""
        try:
            self.chain_for()
            return True
        except VectorStoreNotFound as exc:
            logger.warning("no index available yet: %s", exc)
            return False
        except Exception as exc:  # the API still starts; /ready reports the failure
            logger.warning("retrieval service warmup failed: %s", exc)
            return False

    def is_ready(self) -> bool:
        return bool(self._chains) or index_exists(self.settings)

    # --- querying --------------------------------------------------------
    def answer(self, question: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        question = (question or "").strip()
        if not question:
            raise ValueError("question must not be empty")
        if len(question) > self.settings.max_question_length:
            raise ValueError(
                f"question exceeds {self.settings.max_question_length} characters"
            )

        chain = self.chain_for(top_k)
        started = time.perf_counter()
        payload = {"query": question}
        raw = chain.invoke(payload) if hasattr(chain, "invoke") else chain(payload)
        latency_ms = (time.perf_counter() - started) * 1000

        answer = (raw.get("result") or "").strip()
        documents: List[Any] = raw.get("source_documents") or []
        return {
            "answer": answer,
            "sources": [serialize_document(document) for document in documents],
            "latency_ms": round(latency_ms, 2),
            "model": self.settings.llm_model,
        }


@lru_cache(maxsize=1)
def get_retrieval_service() -> RetrievalService:
    """FastAPI dependency: the process-wide retrieval service."""
    return RetrievalService()


def reset_retrieval_service() -> None:
    get_retrieval_service.cache_clear()


def get_qa_chain(settings: Optional[Settings] = None):
    """Backwards-compatible helper returning the default RetrievalQA chain."""
    service = RetrievalService(settings) if settings else get_retrieval_service()
    return service.chain_for()
