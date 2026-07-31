import pytest

import ingest
from config import ConfigError
from conftest import make_settings


@pytest.fixture
def base_settings(monkeypatch, corpus_dir):
    """Pin the CLI to a throwaway corpus instead of the developer's .env."""
    settings = make_settings(docs_dir=str(corpus_dir), chunk_size=200, chunk_overlap=20)
    monkeypatch.setattr(ingest, "get_settings", lambda: settings)
    return settings


def parse(*argv):
    return ingest.build_parser().parse_args(list(argv))


def test_overrides_are_revalidated(base_settings):
    """model_copy(update=...) skips validators; the CLI must not."""
    with pytest.raises(ConfigError) as excinfo:
        ingest.settings_from_args(parse("--chunk-size", "100", "--chunk-overlap", "5000"))
    assert "chunk_overlap" in str(excinfo.value)


def test_non_positive_chunk_size_is_rejected(base_settings):
    with pytest.raises(ConfigError):
        ingest.settings_from_args(parse("--chunk-size", "0"))


def test_valid_overrides_are_applied(base_settings, tmp_path):
    settings = ingest.settings_from_args(
        parse("--index-path", str(tmp_path / "idx"), "--chunk-size", "500")
    )
    assert settings.index_path == str(tmp_path / "idx")
    assert settings.chunk_size == 500
    assert settings.docs_dir == base_settings.docs_dir  # untouched fields survive


def test_no_overrides_reuses_the_environment_settings(base_settings):
    assert ingest.settings_from_args(parse()) is base_settings


def test_invalid_options_exit_non_zero(base_settings):
    assert ingest.main(["--chunk-size", "10", "--chunk-overlap", "10"]) == 1


def test_dry_run_reports_chunks_without_embedding(base_settings, monkeypatch):
    def explode(*args, **kwargs):  # pragma: no cover - must not be reached
        raise AssertionError("dry run must not embed anything")

    monkeypatch.setattr(ingest, "create_vector_store", explode)
    assert ingest.main(["--dry-run"]) == 0


def test_missing_docs_directory_exits_non_zero(base_settings, tmp_path):
    assert ingest.main(["--dry-run", "--docs-dir", str(tmp_path / "nope")]) == 1


def test_s3_source_without_a_bucket_exits_non_zero(base_settings):
    assert ingest.main(["--dry-run", "--source", "s3"]) == 1


def test_a_missing_api_key_fails_before_reading_the_corpus(monkeypatch, corpus_dir):
    settings = make_settings(docs_dir=str(corpus_dir), openai_api_key="")
    monkeypatch.setattr(ingest, "get_settings", lambda: settings)

    def explode(*args, **kwargs):  # pragma: no cover - must not be reached
        raise AssertionError("the corpus must not be read without credentials")

    monkeypatch.setattr(ingest, "load_documents", explode)
    assert ingest.main([]) == 1


def test_a_successful_run_writes_the_index(base_settings, monkeypatch, tmp_path):
    written = {}

    def fake_create(chunks, settings):
        written["chunks"] = len(chunks)
        written["path"] = settings.index_path
        return object()

    monkeypatch.setattr(ingest, "create_vector_store", fake_create)
    assert ingest.main(["--index-path", str(tmp_path / "idx")]) == 0
    assert written["chunks"] > 0
    assert written["path"] == str(tmp_path / "idx")
