from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Anthropic API
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-haiku-4-5"
    max_tokens: int = 1024

    # Embeddings (local sentence-transformers model)
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Chunking & retrieval
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 4

    # Chroma persistence
    chroma_persist_dir: str = "./chroma_data"
    chroma_collection: str = "documents"


@lru_cache
def get_settings() -> Settings:
    return Settings()
