import io

import pytest

from conftest import make_settings
from document_loader import (
    DocumentLoadError,
    load_documents,
    load_local_documents,
    load_s3_documents,
    parse_document,
    split_documents,
)


def test_local_loader_walks_subdirectories_and_filters_extensions(corpus_dir):
    settings = make_settings(docs_dir=str(corpus_dir))
    documents = load_local_documents(settings)

    names = sorted(document.metadata["name"] for document in documents)
    assert names == ["alpha.md", "beta.txt"]  # .bin ignored, empty file skipped
    assert all(document.metadata["source"] for document in documents)


def test_local_loader_reports_missing_directory(tmp_path):
    settings = make_settings(docs_dir=str(tmp_path / "nope"))
    with pytest.raises(DocumentLoadError):
        load_local_documents(settings)


def test_local_loader_reports_empty_directory(tmp_path):
    settings = make_settings(docs_dir=str(tmp_path))
    with pytest.raises(DocumentLoadError):
        load_local_documents(settings)


def test_parse_document_skips_blank_content():
    assert parse_document("a.txt", b"   \n", source="a.txt") is None


def test_parse_document_survives_invalid_utf8():
    document = parse_document("a.txt", b"caf\xe9", source="a.txt")
    assert document is not None and "caf" in document.page_content


def test_split_documents_chunks_and_labels(corpus_dir):
    settings = make_settings(docs_dir=str(corpus_dir), chunk_size=100, chunk_overlap=20)
    chunks = split_documents(load_local_documents(settings), settings)

    assert len(chunks) > 2
    assert all(len(chunk.page_content) <= 100 for chunk in chunks)
    assert [chunk.metadata["chunk_id"] for chunk in chunks] == list(range(len(chunks)))
    assert all("source" in chunk.metadata for chunk in chunks)


def test_load_documents_dispatches_to_local(corpus_dir):
    settings = make_settings(docs_dir=str(corpus_dir), chunk_size=200, chunk_overlap=0)
    assert load_documents(settings)


class FakeS3Client:
    """Minimal stand-in for boto3's S3 client, including pagination."""

    def __init__(self, pages, objects):
        self.pages = pages
        self.objects = objects
        self.requests = []

    def list_objects_v2(self, **kwargs):
        self.requests.append(kwargs)
        index = 0
        if "ContinuationToken" in kwargs:
            index = int(kwargs["ContinuationToken"])
        return self.pages[index]

    def get_object(self, Bucket, Key):  # noqa: N803 - boto3 signature
        return {"Body": io.BytesIO(self.objects[Key])}


def test_s3_loader_follows_pagination_and_filters_keys():
    pages = [
        {
            "Contents": [{"Key": "docs/one.md"}, {"Key": "docs/"}],
            "IsTruncated": True,
            "NextContinuationToken": "1",
        },
        {
            "Contents": [{"Key": "docs/two.txt"}, {"Key": "docs/image.png"}],
            "IsTruncated": False,
        },
    ]
    objects = {"docs/one.md": b"one", "docs/two.txt": b"two"}
    client = FakeS3Client(pages, objects)
    settings = make_settings(document_source="s3", s3_bucket_name="corpus", s3_prefix="docs/")

    documents = load_s3_documents(settings, client=client)

    assert [document.page_content for document in documents] == ["one", "two"]
    assert documents[0].metadata["source"] == "s3://corpus/docs/one.md"
    assert len(client.requests) == 2


def test_s3_loader_errors_when_prefix_is_empty():
    client = FakeS3Client([{"Contents": [], "IsTruncated": False}], {})
    settings = make_settings(document_source="s3", s3_bucket_name="corpus")
    with pytest.raises(DocumentLoadError):
        load_s3_documents(settings, client=client)
