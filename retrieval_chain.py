"""The retrieval-augmented generation pipeline.

``RetrievalService`` owns the FAISS-backed retriever and the grounded QA chain.
Everything is built lazily: importing this module never touches OpenAI or the
index, which keeps the API process able to start (and report ``/ready``) even
when the corpus has not been indexed yet.
"""

import logging
import threading
import time
from collections import OrderedDict
from functools import lru_cache
from typing import Any, Dict, List, Optional

from config import ConfigError, Settings, get_settings
from vector_store import VectorStoreNotFound, index_exists, load_vector_store

logger = logging.getLogger(__name__)

#: Each cached chain pins a retriever and an LLM client, and ``top_k`` comes
#: from the caller, so the cache is bounded and evicted least-recently-used.
MAX_CACHED_CHAINS = 8

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
        self._chains: "OrderedDict[int, Any]" = OrderedDict()
        # Requests are served from a thread pool, so building the index and the
        # chains has to be serialised: without this two concurrent first
        # requests each load the whole FAISS index into memory.
        self._lock = threading.RLock()
        if chain is not None:
            self._chains[self.settings.retrieval_k] = chain

    # --- wiring ----------------------------------------------------------
    @property
    def vector_store(self):
        if self._vector_store is None:
            with self._lock:
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
        if top_k is None:
            top_k = self.settings.retrieval_k
        if not isinstance(top_k, int) or isinstance(top_k, bool):
            raise ValueError("top_k must be an integer")
        if top_k <= 0:
            # ``top_k or default`` used to silently turn 0 into the default.
            raise ValueError("top_k must be greater than zero")

        with self._lock:
            chain = self._chains.get(top_k)
            if chain is None:
                chain = self._build_chain(top_k)
                self._chains[top_k] = chain
                while len(self._chains) > MAX_CACHED_CHAINS:
                    evicted, _ = self._chains.popitem(last=False)
                    logger.debug("evicted cached chain for top_k=%s", evicted)
            else:
                self._chains.move_to_end(top_k)
            return chain

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
        """Readiness means a query would actually succeed.

        A complete index on disk is not enough: without an API key every query
        fails with a 503, so a pod deployed with a missing secret must never
        pass its readiness probe and start taking traffic.
        """
        if self._chains:
            return True
        try:
            self.settings.require_llm()
        except ConfigError as exc:
            logger.warning("not ready: %s", exc)
            return False
        return index_exists(self.settings)

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
