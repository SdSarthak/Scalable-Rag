"""Corpus loading and chunking.

Documents can come from the local filesystem (handy for development and CI) or
from an S3 prefix (the production path). Both loaders return LangChain
``Document`` objects that are then split into overlapping chunks.
"""

import codecs
import contextlib
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


def decode_text(payload: bytes) -> str:
    """Decode text bytes, honouring the byte-order marks Windows editors add.

    ``bytes.decode("utf-8")`` keeps a UTF-8 BOM as a literal ``\\ufeff`` at the
    start of the document and turns UTF-16 into interleaved NUL characters,
    both of which end up embedded in the index.
    """
    if payload.startswith(codecs.BOM_UTF8):
        return payload[len(codecs.BOM_UTF8) :].decode("utf-8", errors="replace")
    if payload.startswith(codecs.BOM_UTF16_LE) or payload.startswith(codecs.BOM_UTF16_BE):
        return payload.decode("utf-16", errors="replace")
    return payload.decode("utf-8", errors="replace")


def parse_document(name: str, payload: bytes, source: str) -> Optional[Document]:
    """Turn raw bytes into a ``Document``; returns ``None`` when it is empty."""
    extension = os.path.splitext(name)[1].lower()
    if extension == ".pdf":
        text = _read_pdf(payload)
    else:
        text = decode_text(payload)
    text = text.strip()
    if not text:
        logger.warning("skipping empty document: %s", source)
        return None
    return Document(
        page_content=text,
        metadata={"source": source, "name": os.path.basename(name)},
    )


def _too_large(size: int, settings: Settings, source: str) -> bool:
    limit = settings.max_document_bytes
    if limit and size > limit:
        logger.warning(
            "skipping %s: %d bytes exceeds MAX_DOCUMENT_MB (%d bytes)", source, size, limit
        )
        return True
    return False


def load_local_documents(settings: Optional[Settings] = None) -> List[Document]:
    """Read every supported file below ``DOCS_DIR``.

    A single unreadable or corrupt file is skipped with a warning rather than
    aborting the whole ingest — one bad PDF in a corpus of ten thousand should
    not cost you the other 9,999.
    """
    settings = settings or get_settings()
    root = settings.docs_dir
    if not os.path.isdir(root):
        raise DocumentLoadError(f"document directory not found: {os.path.abspath(root)}")

    allowed = set(settings.allowed_extensions)
    documents: List[Document] = []
    skipped = 0
    for dirpath, _dirnames, filenames in os.walk(root):
        for filename in sorted(filenames):
            if os.path.splitext(filename)[1].lower() not in allowed:
                continue
            path = os.path.join(dirpath, filename)
            try:
                if _too_large(os.path.getsize(path), settings, path):
                    skipped += 1
                    continue
                with open(path, "rb") as handle:
                    payload = handle.read()
                document = parse_document(filename, payload, source=path)
            except DocumentLoadError:
                raise  # a missing parser is a setup problem, not a bad file
            except Exception as exc:
                logger.warning("skipping unreadable document %s: %s", path, exc)
                skipped += 1
                continue
            if document is not None:
                documents.append(document)

    if not documents:
        raise DocumentLoadError(
            f"no readable documents with extensions {sorted(allowed)} found under "
            f"{os.path.abspath(root)} ({skipped} skipped)"
        )
    logger.info(
        "loaded %d local documents from %s (%d skipped)", len(documents), root, skipped
    )
    return documents


def load_s3_documents(settings: Optional[Settings] = None, client=None) -> List[Document]:
    """Read every supported object under ``S3_PREFIX`` in ``S3_BUCKET_NAME``."""
    settings = settings or get_settings()
    settings.require_ingestion()

    if client is None:  # pragma: no cover - exercised against real AWS only
        import boto3

        client = boto3.client("s3", region_name=settings.aws_region)

    allowed = set(settings.allowed_extensions)
    bucket = settings.s3_bucket_name
    documents: List[Document] = []
    skipped = 0
    try:
        for key, size in _iter_s3_keys(client, bucket, settings.s3_prefix):
            if os.path.splitext(key)[1].lower() not in allowed:
                continue
            source = f"s3://{bucket}/{key}"
            if _too_large(size, settings, source):
                skipped += 1
                continue
            try:
                document = parse_document(key, _read_object(client, bucket, key), source=source)
            except DocumentLoadError:
                raise
            except Exception as exc:
                logger.warning("skipping unreadable object %s: %s", source, exc)
                skipped += 1
                continue
            if document is not None:
                documents.append(document)
    except DocumentLoadError:
        raise
    except Exception as exc:  # listing failed: missing bucket, denied, no creds
        raise DocumentLoadError(
            f"could not list s3://{bucket}/{settings.s3_prefix} "
            f"({type(exc).__name__}: {exc})"
        ) from exc

    if not documents:
        raise DocumentLoadError(
            f"no readable documents found in s3://{bucket}/{settings.s3_prefix} "
            f"({skipped} skipped)"
        )
    logger.info(
        "loaded %d documents from s3://%s (%d skipped)", len(documents), bucket, skipped
    )
    return documents


def _read_object(client, bucket: str, key: str) -> bytes:
    """Read an object, always releasing the underlying connection."""
    body = client.get_object(Bucket=bucket, Key=key)["Body"]
    try:
        return body.read()
    finally:
        with contextlib.suppress(Exception):
            body.close()


def _iter_s3_keys(client, bucket: str, prefix: str) -> Iterable[tuple]:
    """Yield ``(key, size)`` pairs, transparently following pagination."""
    token = None
    seen = set()
    while True:
        kwargs = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token
        response = client.list_objects_v2(**kwargs)
        for item in response.get("Contents", []):
            key = item.get("Key")
            if not key or key.endswith("/"):
                continue
            if key in seen:  # a repeated continuation token would loop forever
                logger.warning("s3 listing returned %s twice; stopping", key)
                return
            seen.add(key)
            yield key, int(item.get("Size") or 0)
        if not response.get("IsTruncated"):
            return
        next_token = response.get("NextContinuationToken")
        if not next_token or next_token == token:
            return
        token = next_token


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
