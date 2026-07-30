import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Settings  # noqa: E402


def make_settings(**overrides) -> Settings:
    """Build settings that ignore the developer's local .env file."""
    defaults = {
        "openai_api_key": "test-key",
        "auth_enabled": False,
        "jwt_secret": "unit-test-secret",
        "mlflow_enabled": False,
        "docs_dir": "docs",
        "index_path": "faiss_index",
    }
    defaults.update(overrides)
    return Settings(_env_file=None, **defaults)


@pytest.fixture
def settings() -> Settings:
    return make_settings()


@pytest.fixture
def corpus_dir(tmp_path):
    """A small on-disk corpus used by the loader tests."""
    (tmp_path / "alpha.md").write_text("# Alpha\n" + "alpha content. " * 40, encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "beta.txt").write_text("beta content", encoding="utf-8")
    (nested / "ignored.bin").write_bytes(b"\x00\x01")
    (tmp_path / "empty.txt").write_text("   ", encoding="utf-8")
    return tmp_path
