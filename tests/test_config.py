import pytest
from pydantic import ValidationError

from config import ConfigError
from conftest import make_settings


def test_defaults_are_usable():
    settings = make_settings()
    assert settings.document_source == "local"
    assert settings.retrieval_k > 0
    assert settings.chunk_overlap < settings.chunk_size


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValidationError):
        make_settings(chunk_size=100, chunk_overlap=100)


def test_invalid_document_source_rejected():
    with pytest.raises(ValidationError):
        make_settings(document_source="ftp")


def test_log_level_is_normalised():
    assert make_settings(log_level="debug").log_level == "DEBUG"


def test_list_properties_are_parsed():
    settings = make_settings(
        cors_allow_origins="https://a.example, https://b.example",
        document_extensions="txt, .md",
    )
    assert settings.allowed_origins == ["https://a.example", "https://b.example"]
    assert settings.allowed_extensions == [".txt", ".md"]


def test_require_llm_needs_a_key():
    with pytest.raises(ConfigError):
        make_settings(openai_api_key="").require_llm()
    make_settings(openai_api_key="sk-placeholder").require_llm()


def test_require_ingestion_needs_a_bucket_for_s3():
    with pytest.raises(ConfigError):
        make_settings(document_source="s3", s3_bucket_name="").require_ingestion()
    make_settings(document_source="s3", s3_bucket_name="corpus").require_ingestion()


def test_require_auth_checks_the_algorithm_specific_material():
    with pytest.raises(ConfigError):
        make_settings(auth_enabled=True, jwt_secret="").require_auth()
    with pytest.raises(ConfigError):
        make_settings(auth_enabled=True, jwt_algorithm="RS256", jwt_jwks_url="").require_auth()
    make_settings(auth_enabled=False, jwt_secret="").require_auth()
