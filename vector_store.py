"""FAISS index lifecycle: build, persist, load.

The heavy dependencies (faiss, the embedding client) are imported lazily so the
API process and the test suite can import this module without them.
"""

import inspect
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


#: ``FAISS.save_local`` writes both of these. An index missing either one is
#: unusable, so both have to be present before we report the service ready.
INDEX_FILES = ("index.faiss", "index.pkl")


def missing_index_files(settings: Optional[Settings] = None) -> List[str]:
    """Return the index artefacts that are absent from ``INDEX_PATH``."""
    settings = settings or get_settings()
    return [
        name
        for name in INDEX_FILES
        if not os.path.isfile(os.path.join(settings.index_path, name))
    ]


def index_exists(settings: Optional[Settings] = None) -> bool:
    """True only when a *complete* index is on disk.

    Checking ``index.faiss`` alone made ``/ready`` claim the pod could serve
    traffic while a half-written index (no ``index.pkl``) blew up on the first
    query, so Kubernetes routed requests straight into 502s.
    """
    return not missing_index_files(settings)


def _load_local(settings: Settings, embeddings):
    faiss = _faiss()
    try:
        parameters = inspect.signature(faiss.load_local).parameters
    except (TypeError, ValueError):  # pragma: no cover - builtin//C callable
        parameters = {}
    if "allow_dangerous_deserialization" in parameters:
        return faiss.load_local(
            settings.index_path, embeddings, allow_dangerous_deserialization=True
        )
    return faiss.load_local(settings.index_path, embeddings)  # pragma: no cover


def load_vector_store(settings: Optional[Settings] = None):
    """Load the persisted FAISS index from disk."""
    settings = settings or get_settings()
    missing = missing_index_files(settings)
    if missing:
        raise VectorStoreNotFound(
            f"incomplete FAISS index at {os.path.abspath(settings.index_path)} "
            f"(missing {', '.join(missing)}); build one with `python ingest.py`"
        )

    embeddings = get_embeddings(settings)
    try:
        return _load_local(settings, embeddings)
    except Exception as exc:  # corrupt, truncated or embedding-mismatched index
        raise VectorStoreNotFound(
            f"the FAISS index at {os.path.abspath(settings.index_path)} could not be "
            f"loaded ({type(exc).__name__}: {exc}); rebuild it with `python ingest.py`"
        ) from exc
