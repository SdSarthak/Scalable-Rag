"""Build the FAISS index from the configured corpus.

    python ingest.py                      # use the settings from .env
    python ingest.py --source local --docs-dir ./docs
    python ingest.py --source s3 --bucket my-bucket --prefix docs/
    python ingest.py --dry-run            # count chunks without calling OpenAI
"""

import argparse
import logging
import sys

from config import ConfigError, Settings, get_settings
from document_loader import DocumentLoadError, load_documents
from vector_store import create_vector_store

logger = logging.getLogger("ingest")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Index a corpus for the RAG service")
    parser.add_argument("--source", choices=["local", "s3"], help="where documents live")
    parser.add_argument("--docs-dir", help="local directory to ingest")
    parser.add_argument("--bucket", help="S3 bucket name")
    parser.add_argument("--prefix", help="S3 key prefix")
    parser.add_argument("--index-path", help="where to write the FAISS index")
    parser.add_argument("--chunk-size", type=int, help="characters per chunk")
    parser.add_argument("--chunk-overlap", type=int, help="characters shared between chunks")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="load and split documents but skip embedding",
    )
    return parser


def settings_from_args(args: argparse.Namespace) -> Settings:
    """Apply CLI overrides on top of the environment-derived settings."""
    overrides = {
        "document_source": args.source,
        "docs_dir": args.docs_dir,
        "s3_bucket_name": args.bucket,
        "s3_prefix": args.prefix,
        "index_path": args.index_path,
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
    }
    overrides = {key: value for key, value in overrides.items() if value is not None}
    return get_settings().model_copy(update=overrides) if overrides else get_settings()


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    settings = settings_from_args(args)

    try:
        settings.require_ingestion()
        chunks = load_documents(settings)
        if args.dry_run:
            logger.info("dry run: %d chunks ready (nothing embedded)", len(chunks))
            return 0
        create_vector_store(chunks, settings)
        logger.info("index written to %s", settings.index_path)
        return 0
    except (ConfigError, DocumentLoadError, ValueError) as exc:
        logger.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
