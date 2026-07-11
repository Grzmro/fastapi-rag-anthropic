"""Chroma vector store with local sentence-transformers embeddings."""

from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import get_settings


@lru_cache
def get_embeddings() -> HuggingFaceEmbeddings:
    settings = get_settings()
    return HuggingFaceEmbeddings(model_name=settings.embedding_model)


@lru_cache
def get_vectorstore() -> Chroma:
    settings = get_settings()
    return Chroma(
        collection_name=settings.chroma_collection,
        embedding_function=get_embeddings(),
        persist_directory=settings.chroma_persist_dir,
    )


def add_documents(docs: list[Document]) -> int:
    """Upsert chunks into Chroma. Returns the number of chunks added."""
    if not docs:
        return 0
    store = get_vectorstore()
    store.add_documents(docs, ids=[doc.metadata["chunk_id"] for doc in docs])
    return len(docs)


def search(query: str, k: int | None = None) -> list[Document]:
    """Semantic similarity search over the indexed chunks."""
    settings = get_settings()
    store = get_vectorstore()
    return store.similarity_search(query, k=k or settings.top_k)


def list_sources() -> list[str]:
    """Distinct source filenames currently in the index."""
    store = get_vectorstore()
    result = store.get(include=["metadatas"])
    sources = {meta["source"] for meta in result["metadatas"] if meta and "source" in meta}
    return sorted(sources)


def count_chunks() -> int:
    return len(get_vectorstore().get(include=[])["ids"])
