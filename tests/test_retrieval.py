from pathlib import Path

from app import vectorstore
from app.ingestion import ingest_file

SAMPLE = Path(__file__).parent.parent / "data" / "samples" / "solar_system.md"


def test_index_and_search_returns_relevant_chunk(fresh_store):
    chunks = ingest_file(SAMPLE)
    added = vectorstore.add_documents(chunks)
    assert added == len(chunks)

    results = vectorstore.search("Which planet has the largest volcano?", k=2)
    assert len(results) == 2
    assert any("Olympus Mons" in doc.page_content for doc in results)


def test_search_results_carry_citation_metadata(fresh_store):
    vectorstore.add_documents(ingest_file(SAMPLE))
    results = vectorstore.search("rings of Saturn", k=1)
    meta = results[0].metadata
    assert meta["source"] == "solar_system.md"
    assert isinstance(meta["page"], int)
    assert "chunk_id" in meta


def test_reingest_is_idempotent(fresh_store):
    chunks = ingest_file(SAMPLE)
    vectorstore.add_documents(chunks)
    count_first = vectorstore.count_chunks()

    # Same chunk_ids -> upsert, not duplication.
    vectorstore.add_documents(chunks)
    assert vectorstore.count_chunks() == count_first


def test_list_sources(fresh_store):
    assert vectorstore.list_sources() == []
    vectorstore.add_documents(ingest_file(SAMPLE))
    assert vectorstore.list_sources() == ["solar_system.md"]
