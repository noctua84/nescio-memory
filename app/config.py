from typing import Literal
from urllib.parse import urlparse

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_VALID_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")

class Settings(BaseSettings):
    # Database
    database_url: str
    # Caps only the vector-search statement (set via SET LOCAL in
    # LearningRepository.search), not the whole request or connection. 0 would
    # disable the timeout in Postgres, which is never what we want here.
    statement_timeout_ms: int = 5000

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
    # A typo'd level (e.g. "WARN", "verbose") should crash at startup rather
    # than silently log at a surprising level. "WARN" is rejected on purpose:
    # it's a stdlib alias for WARNING, but we keep exactly one accepted
    # spelling to avoid ambiguity in config and docs.
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

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

    # Normalize case before pydantic's Literal check runs, so "info"/"Info"
    # work from the environment (env vars arrive as plain strings, case and
    # all). Pydantic's own Literal error names the field "log_level" in
    # lower case, which wouldn't satisfy tooling/tests that grep for the
    # ENV VAR name, so an invalid value is rejected here instead with a
    # message naming LOG_LEVEL explicitly.
    @field_validator("log_level", mode="before")
    @classmethod
    def _log_level_case_insensitive(cls, value: object) -> object:
        if isinstance(value, str):
            value = value.strip().upper()
        if value not in _VALID_LOG_LEVELS:
            raise ValueError(
                f"LOG_LEVEL must be one of {', '.join(_VALID_LOG_LEVELS)}, "
                f"got {value!r}"
            )
        return value

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

    # A non-positive value would disable the statement timeout in Postgres
    # (0) or be rejected by set_config (negative), silently removing the cap
    # that issue #17 added rather than failing loudly at startup.
    @model_validator(mode="after")
    def _statement_timeout_must_be_positive(self) -> "Settings":
        if self.statement_timeout_ms <= 0:
            raise ValueError(
                "STATEMENT_TIMEOUT_MS must be > 0, got "
                f"{self.statement_timeout_ms}; 0 disables the timeout in Postgres"
            )
        return self


settings = Settings()