"""Shared fixtures.

Note: no test here calls the Anthropic API or the /ask endpoint — prompt
quality is verified manually (see README).
"""

import pytest

from app import config, vectorstore


@pytest.fixture
def fresh_store(tmp_path, monkeypatch):
    """Point Chroma at a temporary directory and reset cached singletons.

    The embedding model cache is intentionally kept — reloading it per test
    would be slow and the model itself is stateless.
    """
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    config.get_settings.cache_clear()
    vectorstore.get_vectorstore.cache_clear()
    yield
    config.get_settings.cache_clear()
    vectorstore.get_vectorstore.cache_clear()
