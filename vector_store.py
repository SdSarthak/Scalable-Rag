"""FAISS index lifecycle: build, persist, load.

The heavy dependencies (faiss, the embedding client) are imported lazily so the
API process and the test suite can import this module without them.
"""

import logging
import os
from typing import List, Optional

from langchain_core.documents import Document

from config import Settings, get_settings
from document_loader import load_documents

logger = logging.getLogger(__name__)


class VectorStoreNotFound(RuntimeError):
    """Raised when the persisted index is missing or unreadable."""


def _faiss():
    try:
        from langchain_community.vectorstores import FAISS
    except ImportError:  # pragma: no cover - older langchain layout
        from langchain.vectorstores import FAISS
    return FAISS


def get_embeddings(settings: Optional[Settings] = None):
    """Build the embedding client used for both indexing and querying."""
    settings = settings or get_settings()
    settings.require_llm()
    try:
        from langchain_openai import OpenAIEmbeddings
    except ImportError:  # pragma: no cover - older langchain layout
        from langchain_community.embeddings import OpenAIEmbeddings
    return OpenAIEmbeddings(model=settings.embedding_model, api_key=settings.openai_api_key)


def create_vector_store(
    documents: Optional[List[Document]] = None, settings: Optional[Settings] = None
):
    """Embed the corpus and persist the FAISS index to ``INDEX_PATH``."""
    settings = settings or get_settings()
    documents = documents if documents is not None else load_documents(settings)
    if not documents:
        raise ValueError("cannot build a vector store from an empty corpus")

    embeddings = get_embeddings(settings)
    vector_store = _faiss().from_documents(documents, embeddings)
    os.makedirs(os.path.dirname(os.path.abspath(settings.index_path)), exist_ok=True)
    vector_store.save_local(settings.index_path)
    logger.info("indexed %d chunks into %s", len(documents), settings.index_path)
    return vector_store


def index_exists(settings: Optional[Settings] = None) -> bool:
    settings = settings or get_settings()
    return os.path.isfile(os.path.join(settings.index_path, "index.faiss"))


def load_vector_store(settings: Optional[Settings] = None):
    """Load the persisted FAISS index from disk."""
    settings = settings or get_settings()
    if not index_exists(settings):
        raise VectorStoreNotFound(
            f"no FAISS index at {os.path.abspath(settings.index_path)}; "
            "build one with `python ingest.py`"
        )

    embeddings = get_embeddings(settings)
    try:
        return _faiss().load_local(
            settings.index_path, embeddings, allow_dangerous_deserialization=True
        )
    except TypeError:  # pragma: no cover - langchain < 0.1.x has no such flag
        return _faiss().load_local(settings.index_path, embeddings)
