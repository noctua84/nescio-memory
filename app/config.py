from typing import Literal
from urllib.parse import urlparse

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Database
    database_url: str

    # Embeddings backend
    # "ollama" posts to an Ollama server over HTTP; "local" runs
    # sentence-transformers in-process and needs the local-embeddings extra.
    embedding_backend: Literal["ollama", "local"] = "ollama"
    # 384 dimensions, matching the default embedding_dimension below.
    local_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Ollama
    ollama_url: str = "http://localhost:11434/api/embeddings"
    ollama_model: str = "qwen3-embedding:0.6b"
    embedding_dimension: int = 384

    # langfuse:
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    # app
    app_name: str = "Nescio Semantic Memory API"
    log_level: str = "INFO"

    # chunks
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # server
    host: str = "localhost"
    port: int = 8080

    # env_file is resolved against the working directory, not this file's
    # location, so it stays ".env" even though this module lives in app/.
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # SQLAlchemy 2.1 changed the default PostgreSQL driver for a bare
    # postgresql:// URL to psycopg 3, but this project ships psycopg2; an
    # explicit driver in the URL is respected, so rewrite the bare scheme once.
    @field_validator("database_url", mode="before")
    @classmethod
    def _normalize_database_url_scheme(cls, v: str) -> str:
        if isinstance(v, str) and v.startswith("postgresql://"):
            return v.replace("postgresql://", "postgresql+psycopg2://", 1)
        return v

    # Failing here rather than only in chunk_text() means a bad .env crash-loops
    # the pod at startup instead of returning a 500 on every ingest request.
    @model_validator(mode="after")
    def _chunk_window_must_advance(self) -> "Settings":
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError(
                "CHUNK_OVERLAP must satisfy 0 <= CHUNK_OVERLAP < CHUNK_SIZE, got "
                f"CHUNK_SIZE={self.chunk_size}, CHUNK_OVERLAP={self.chunk_overlap}; "
                "otherwise chunk_text's sliding window never advances"
            )
        return self

    # Same reasoning as above: a malformed OLLAMA_URL otherwise surfaces per
    # request as a 503 advertising a retry that can never succeed, or as a bare
    # 500 when httpx raises something outside its own exception tree.
    @model_validator(mode="after")
    def _ollama_url_must_be_http(self) -> "Settings":
        if self.embedding_backend != "ollama":
            return self
        parsed = urlparse(self.ollama_url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise ValueError(
                "OLLAMA_URL must be an absolute http:// or https:// URL with a "
                f"host, got {self.ollama_url!r}"
            )
        return self


settings = Settings()