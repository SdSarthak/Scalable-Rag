"""Corpus loading and chunking.

Documents can come from the local filesystem (handy for development and CI) or
from an S3 prefix (the production path). Both loaders return LangChain
``Document`` objects that are then split into overlapping chunks.
"""

import io
import logging
import os
from typing import Iterable, List, Optional

from langchain_core.documents import Document

from config import Settings, get_settings

try:  # langchain >= 0.2 moved the splitters into their own package
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:  # pragma: no cover - depends on the installed langchain
    from langchain.text_splitter import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


class DocumentLoadError(RuntimeError):
    """Raised when a corpus cannot be read."""


def _read_pdf(payload: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise DocumentLoadError(
            "pypdf is required to ingest PDF files; install it or drop '.pdf' "
            "from DOCUMENT_EXTENSIONS"
        ) from exc
    reader = PdfReader(io.BytesIO(payload))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def parse_document(name: str, payload: bytes, source: str) -> Optional[Document]:
    """Turn raw bytes into a ``Document``; returns ``None`` when it is empty."""
    extension = os.path.splitext(name)[1].lower()
    if extension == ".pdf":
        text = _read_pdf(payload)
    else:
        text = payload.decode("utf-8", errors="replace")
    text = text.strip()
    if not text:
        logger.warning("skipping empty document: %s", source)
        return None
    return Document(
        page_content=text,
        metadata={"source": source, "name": os.path.basename(name)},
    )


def load_local_documents(settings: Optional[Settings] = None) -> List[Document]:
    """Read every supported file below ``DOCS_DIR``."""
    settings = settings or get_settings()
    root = settings.docs_dir
    if not os.path.isdir(root):
        raise DocumentLoadError(f"document directory not found: {os.path.abspath(root)}")

    allowed = set(settings.allowed_extensions)
    documents: List[Document] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for filename in sorted(filenames):
            if os.path.splitext(filename)[1].lower() not in allowed:
                continue
            path = os.path.join(dirpath, filename)
            with open(path, "rb") as handle:
                document = parse_document(filename, handle.read(), source=path)
            if document is not None:
                documents.append(document)

    if not documents:
        raise DocumentLoadError(
            f"no documents with extensions {sorted(allowed)} found under {os.path.abspath(root)}"
        )
    logger.info("loaded %d local documents from %s", len(documents), root)
    return documents


def load_s3_documents(settings: Optional[Settings] = None, client=None) -> List[Document]:
    """Read every supported object under ``S3_PREFIX`` in ``S3_BUCKET_NAME``."""
    settings = settings or get_settings()
    settings.require_ingestion()

    if client is None:  # pragma: no cover - exercised against real AWS only
        import boto3

        client = boto3.client("s3", region_name=settings.aws_region)

    allowed = set(settings.allowed_extensions)
    documents: List[Document] = []
    for key in _iter_s3_keys(client, settings.s3_bucket_name, settings.s3_prefix):
        if os.path.splitext(key)[1].lower() not in allowed:
            continue
        body = client.get_object(Bucket=settings.s3_bucket_name, Key=key)["Body"].read()
        document = parse_document(key, body, source=f"s3://{settings.s3_bucket_name}/{key}")
        if document is not None:
            documents.append(document)

    if not documents:
        raise DocumentLoadError(
            f"no documents found in s3://{settings.s3_bucket_name}/{settings.s3_prefix}"
        )
    logger.info("loaded %d documents from s3://%s", len(documents), settings.s3_bucket_name)
    return documents


def _iter_s3_keys(client, bucket: str, prefix: str) -> Iterable[str]:
    """Yield object keys, transparently following pagination."""
    token = None
    while True:
        kwargs = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token
        response = client.list_objects_v2(**kwargs)
        for item in response.get("Contents", []):
            key = item["Key"]
            if not key.endswith("/"):
                yield key
        if not response.get("IsTruncated"):
            return
        token = response.get("NextContinuationToken")
        if not token:
            return


def split_documents(
    documents: List[Document], settings: Optional[Settings] = None
) -> List[Document]:
    """Split documents into overlapping chunks and tag each chunk with its index."""
    settings = settings or get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        add_start_index=True,
    )
    chunks = splitter.split_documents(documents)
    for position, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = position
    logger.info("split %d documents into %d chunks", len(documents), len(chunks))
    return chunks


def load_documents(settings: Optional[Settings] = None) -> List[Document]:
    """Load the configured corpus and return it as ready-to-embed chunks."""
    settings = settings or get_settings()
    if settings.document_source == "s3":
        documents = load_s3_documents(settings)
    else:
        documents = load_local_documents(settings)
    return split_documents(documents, settings)
