import pytest

import vector_store
from conftest import make_settings
from vector_store import (
    VectorStoreNotFound,
    create_vector_store,
    index_exists,
    load_vector_store,
    missing_index_files,
)


def write_index(directory, *names):
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (directory / name).write_bytes(b"stub")
    return directory


def test_a_half_written_index_is_not_reported_as_present(tmp_path):
    """``FAISS.save_local`` writes index.faiss *and* index.pkl.

    Only checking index.faiss made /ready pass while every query crashed.
    """
    directory = write_index(tmp_path / "idx", "index.faiss")
    settings = make_settings(index_path=str(directory))

    assert missing_index_files(settings) == ["index.pkl"]
    assert index_exists(settings) is False


def test_a_complete_index_is_reported_as_present(tmp_path):
    directory = write_index(tmp_path / "idx", "index.faiss", "index.pkl")
    assert index_exists(make_settings(index_path=str(directory))) is True


def test_loading_a_missing_index_names_the_missing_files(tmp_path):
    directory = write_index(tmp_path / "idx", "index.faiss")
    with pytest.raises(VectorStoreNotFound) as excinfo:
        load_vector_store(make_settings(index_path=str(directory)))
    assert "index.pkl" in str(excinfo.value)
    assert "ingest.py" in str(excinfo.value)


def test_a_corrupt_index_raises_vector_store_not_found(tmp_path, monkeypatch):
    """A corrupt index must map to 503 (rebuild me), not an opaque 502."""
    directory = write_index(tmp_path / "idx", "index.faiss", "index.pkl")

    class BrokenFaiss:
        @staticmethod
        def load_local(path, embeddings, allow_dangerous_deserialization=False):
            raise EOFError("Ran out of input")

    monkeypatch.setattr(vector_store, "_faiss", lambda: BrokenFaiss)
    monkeypatch.setattr(vector_store, "get_embeddings", lambda settings: object())

    with pytest.raises(VectorStoreNotFound) as excinfo:
        load_vector_store(make_settings(index_path=str(directory)))
    assert "EOFError" in str(excinfo.value)


def test_load_omits_the_dangerous_flag_for_older_langchain(tmp_path, monkeypatch):
    directory = write_index(tmp_path / "idx", "index.faiss", "index.pkl")
    seen = {}

    class LegacyFaiss:
        @staticmethod
        def load_local(path, embeddings):
            seen["path"] = path
            return "store"

    monkeypatch.setattr(vector_store, "_faiss", lambda: LegacyFaiss)
    monkeypatch.setattr(vector_store, "get_embeddings", lambda settings: object())

    assert load_vector_store(make_settings(index_path=str(directory))) == "store"
    assert seen["path"] == str(directory)


def test_creating_a_store_from_an_empty_corpus_is_rejected():
    with pytest.raises(ValueError):
        create_vector_store([], make_settings())
