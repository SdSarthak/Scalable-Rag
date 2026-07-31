import io

import pytest

import document_loader
from conftest import make_settings
from document_loader import (
    DocumentLoadError,
    decode_text,
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


def test_utf8_bom_is_stripped():
    assert decode_text(b"\xef\xbb\xbfhello") == "hello"


def test_utf16_content_is_decoded_not_riddled_with_nulls():
    assert decode_text("héllo".encode("utf-16")) == "héllo"


def test_unicode_survives_a_round_trip(tmp_path):
    (tmp_path / "u.md").write_text("naïve café — 日本語 🚀", encoding="utf-8")
    documents = load_local_documents(make_settings(docs_dir=str(tmp_path)))
    assert documents[0].page_content == "naïve café — 日本語 🚀"


def test_one_unreadable_file_does_not_abort_the_whole_corpus(tmp_path, monkeypatch):
    (tmp_path / "good.txt").write_text("usable content", encoding="utf-8")
    (tmp_path / "bad.txt").write_text("also fine", encoding="utf-8")

    real_open = open

    def flaky_open(path, *args, **kwargs):
        if str(path).endswith("bad.txt"):
            raise PermissionError("access denied")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", flaky_open)
    documents = load_local_documents(make_settings(docs_dir=str(tmp_path)))
    assert [document.metadata["name"] for document in documents] == ["good.txt"]


def test_a_corrupt_pdf_is_skipped_rather_than_fatal(tmp_path, monkeypatch):
    (tmp_path / "ok.txt").write_text("usable content", encoding="utf-8")
    (tmp_path / "broken.pdf").write_bytes(b"not really a pdf")
    monkeypatch.setattr(
        document_loader,
        "_read_pdf",
        lambda payload: (_ for _ in ()).throw(ValueError("EOF marker not found")),
    )

    documents = load_local_documents(make_settings(docs_dir=str(tmp_path)))
    assert [document.metadata["name"] for document in documents] == ["ok.txt"]


def test_a_corpus_of_only_broken_files_is_an_error(tmp_path, monkeypatch):
    (tmp_path / "broken.pdf").write_bytes(b"not really a pdf")
    monkeypatch.setattr(
        document_loader,
        "_read_pdf",
        lambda payload: (_ for _ in ()).throw(ValueError("EOF marker not found")),
    )
    with pytest.raises(DocumentLoadError) as excinfo:
        load_local_documents(make_settings(docs_dir=str(tmp_path)))
    assert "1 skipped" in str(excinfo.value)


def test_oversized_files_are_skipped(tmp_path):
    """Files are read whole, so one huge object must not OOM the ingest job."""
    (tmp_path / "small.txt").write_text("small", encoding="utf-8")
    (tmp_path / "huge.txt").write_text("x" * (2 * 1024 * 1024), encoding="utf-8")

    capped = load_local_documents(make_settings(docs_dir=str(tmp_path), max_document_mb=1))
    assert [document.metadata["name"] for document in capped] == ["small.txt"]

    uncapped = load_local_documents(make_settings(docs_dir=str(tmp_path), max_document_mb=0))
    assert len(uncapped) == 2  # 0 disables the cap


class FakeS3Client:
    """Minimal stand-in for boto3's S3 client, including pagination."""

    def __init__(self, pages, objects, fail_on=()):
        self.pages = pages
        self.objects = objects
        self.fail_on = set(fail_on)
        self.requests = []
        self.open_bodies = 0

    def list_objects_v2(self, **kwargs):
        self.requests.append(kwargs)
        index = 0
        if "ContinuationToken" in kwargs:
            index = int(kwargs["ContinuationToken"])
        page = self.pages[index]
        if isinstance(page, Exception):
            raise page
        return page

    def get_object(self, Bucket, Key):  # noqa: N803 - boto3 signature
        if Key in self.fail_on:
            raise RuntimeError("An error occurred (AccessDenied)")
        self.open_bodies += 1
        return {"Body": TrackingBody(self, self.objects[Key])}


class TrackingBody(io.BytesIO):
    def __init__(self, client, payload):
        super().__init__(payload)
        self._client = client

    def close(self):
        self._client.open_bodies -= 1
        super().close()


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


def test_s3_listing_failures_become_actionable_errors():
    """A denied or missing bucket must not surface as a raw botocore traceback."""
    client = FakeS3Client([RuntimeError("An error occurred (NoSuchBucket)")], {})
    settings = make_settings(document_source="s3", s3_bucket_name="gone")
    with pytest.raises(DocumentLoadError) as excinfo:
        load_s3_documents(settings, client=client)
    assert "s3://gone/" in str(excinfo.value)
    assert "NoSuchBucket" in str(excinfo.value)


def test_s3_loader_skips_objects_it_cannot_read():
    pages = [
        {
            "Contents": [{"Key": "docs/one.md"}, {"Key": "docs/two.txt"}],
            "IsTruncated": False,
        }
    ]
    client = FakeS3Client(pages, {"docs/one.md": b"one"}, fail_on={"docs/two.txt"})
    settings = make_settings(document_source="s3", s3_bucket_name="corpus")

    documents = load_s3_documents(settings, client=client)
    assert [document.page_content for document in documents] == ["one"]


def test_s3_loader_closes_every_object_body():
    pages = [{"Contents": [{"Key": "docs/one.md"}], "IsTruncated": False}]
    client = FakeS3Client(pages, {"docs/one.md": b"one"})

    load_s3_documents(make_settings(document_source="s3", s3_bucket_name="c"), client=client)
    assert client.open_bodies == 0


def test_s3_loader_skips_oversized_objects():
    pages = [
        {
            "Contents": [
                {"Key": "docs/small.md", "Size": 10},
                {"Key": "docs/huge.md", "Size": 50 * 1024 * 1024},
            ],
            "IsTruncated": False,
        }
    ]
    client = FakeS3Client(pages, {"docs/small.md": b"small"})
    settings = make_settings(
        document_source="s3", s3_bucket_name="corpus", max_document_mb=25
    )

    documents = load_s3_documents(settings, client=client)
    assert [document.page_content for document in documents] == ["small"]


def test_s3_loader_stops_on_a_repeating_continuation_token():
    """A provider that echoes the same token would otherwise loop forever."""
    pages = [
        {
            "Contents": [{"Key": "docs/one.md"}],
            "IsTruncated": True,
            "NextContinuationToken": "0",
        }
    ]
    client = FakeS3Client(pages, {"docs/one.md": b"one"})
    settings = make_settings(document_source="s3", s3_bucket_name="corpus")

    documents = load_s3_documents(settings, client=client)
    assert len(documents) == 1
    assert len(client.requests) == 2  # the repeat is detected, not followed forever
