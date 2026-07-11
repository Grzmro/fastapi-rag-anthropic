from pathlib import Path

import pytest

from app.ingestion import chunk_documents, ingest_file, load_document

SAMPLE = Path(__file__).parent.parent / "data" / "samples" / "solar_system.md"


def test_load_document_sets_source_and_page():
    docs = load_document(SAMPLE)
    assert len(docs) == 1
    assert docs[0].metadata["source"] == "solar_system.md"
    assert docs[0].metadata["page"] == 1
    assert "Olympus Mons" in docs[0].page_content


def test_load_document_source_name_override(tmp_path):
    tmp_file = tmp_path / "tmp_upload_123.txt"
    tmp_file.write_text("some content", encoding="utf-8")
    docs = load_document(tmp_file, source_name="original.txt")
    assert docs[0].metadata["source"] == "original.txt"


def test_load_document_rejects_unsupported_extension(tmp_path):
    bad = tmp_path / "file.docx"
    bad.write_bytes(b"not really a docx")
    with pytest.raises(ValueError, match="Unsupported file type"):
        load_document(bad)


def test_chunk_documents_preserves_metadata_and_assigns_ids():
    docs = load_document(SAMPLE)
    chunks = chunk_documents(docs)

    assert len(chunks) > 1  # sample is longer than one chunk_size
    for i, chunk in enumerate(chunks):
        assert chunk.metadata["source"] == "solar_system.md"
        assert chunk.metadata["page"] == 1
        assert chunk.metadata["chunk_id"] == f"solar_system.md:{i}"


def test_chunks_respect_size_limit():
    chunks = ingest_file(SAMPLE)
    from app.config import get_settings

    max_size = get_settings().chunk_size
    assert all(len(c.page_content) <= max_size for c in chunks)
