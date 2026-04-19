from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    openai_api_key: str
    openai_embedding_model: str = "text-embedding-3-small"
    openai_embedding_dimensions: Optional[int] = None

    opensearch_host: str = "localhost"
    opensearch_port: int = 9200
    opensearch_username: str = "admin"
    opensearch_password: str = "admin"
    opensearch_use_ssl: bool = False
    opensearch_verify_certs: bool = False
    opensearch_index: str = "rag_chunks"

    default_top_k: int = 10
    hybrid_keyword_weight: float = 0.4
    hybrid_semantic_weight: float = 0.6

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    """Return a cached settings object for reuse across requests."""
    return Settings()
