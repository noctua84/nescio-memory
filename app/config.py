from typing import Literal
from urllib.parse import urlparse

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Database
    database_url: str
    # Caps only the vector-search statement (set via SET LOCAL in
    # LearningRepository.search), not the whole request or connection. 0 would
    # disable the timeout in Postgres, which is never what we want here.
    statement_timeout_ms: int = 5000
    # Connect timeout per resolved address in seconds. libpq silently raises any
    # connect_timeout below 2 to 2 seconds, so a smaller value would claim a
    # bound that is not applied; 0 would disable the timeout entirely.
    db_connect_timeout_s: int = 10

    # Embeddings backend
    # "ollama" posts to an Ollama server over HTTP; "local" runs
    # sentence-transformers in-process and needs the local-embeddings extra.
    embedding_backend: Literal["ollama", "local"] = "ollama"
    # 384 dimensions, which does NOT match the embedding_dimension default
    # below. all-MiniLM-L6-v2 is genuinely 384-wide and the Ollama default is
    # genuinely 1024-wide, so the two backends are not interchangeable against
    # one database: selecting "local" also means EMBEDDING_DIMENSION=384 and a
    # vector(384) column. See the Embedding backends section of the README.
    local_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Ollama
    ollama_url: str = "http://localhost:11434/api/embeddings"
    ollama_model: str = "qwen3-embedding:0.6b"
    # qwen3-embedding:0.6b emits 1024 floats -- measured, not inferred:
    # /api/show reports qwen3.embedding_length 1024 and /api/embeddings returns
    # a 1024-element vector. Nothing in the suite compares this number against
    # a real model, so `uv run python -m scripts.check_embedding_dimension`
    # exists to do exactly that. (Was 384 until #36, which made the default
    # configuration unable to embed anything at all.)
    embedding_dimension: int = 1024

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

    # Total characters of expanded context one /api/v1/search response may
    # carry, across all regions. A cap, not a target: regions are built in
    # descending order of the best similarity among the results referencing
    # them and stop once this would be exceeded. 100,000 is roughly 25 notes
    # at this corpus's average size.
    max_context_chars: int = 100_000

    # How many distinct notes one /api/v1/search response may expand. The
    # character budget above is spent after the rows are already in memory, so
    # it alone does not bound the FETCH: `context: "document"` with top_k=50
    # hits in 50 different notes asks the database for every chunk of all 50,
    # however large they are, only to discard most of it (#48). This caps the
    # documents, in rank order, before anything is fetched. 25 pairs with the
    # 100,000-character budget above, which is roughly 25 notes' worth.
    max_context_documents: int = 25

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

    # 0 is not merely a smaller budget. The budget is spent by whole regions:
    # a region is built only if it fits, so a budget of 0 admits no region at
    # all and `context: "neighbors"|"document"` silently returns an empty
    # `contexts` list with every `context_ref` null -- indistinguishable, to a
    # caller, from `context: "none"` or from a service where the feature is
    # broken. Someone who wants the feature off asks for "none" per request;
    # someone who sets this to 0 has misconfigured it, and finding that out at
    # startup beats finding it out from silently context-free search results.
    # Negative is the same mistake with a sign on it.
    @model_validator(mode="after")
    def _max_context_chars_must_be_positive(self) -> "Settings":
        if self.max_context_chars <= 0:
            raise ValueError(
                "MAX_CONTEXT_CHARS must be > 0, got "
                f"{self.max_context_chars}; 0 does not shrink the context "
                "budget, it disables context expansion while still accepting "
                "requests that ask for it"
            )
        return self

    # 0 fails for the same reason MAX_CONTEXT_CHARS=0 does, one step earlier in
    # the pipeline: no document is planned, so `context: "neighbors"|"document"`
    # returns an empty `contexts` list with every `context_ref` null and looks,
    # to a caller, exactly like `context: "none"` or a broken service. Turning
    # the feature off is a per-request decision ("none"), not a deployment-wide
    # one that still accepts requests asking for it. Negative is the same
    # mistake with a sign on it.
    @model_validator(mode="after")
    def _max_context_documents_must_be_positive(self) -> "Settings":
        if self.max_context_documents <= 0:
            raise ValueError(
                "MAX_CONTEXT_DOCUMENTS must be > 0, got "
                f"{self.max_context_documents}; 0 does not narrow context "
                "expansion, it disables it while still accepting requests "
                "that ask for it"
            )
        return self

    # libpq silently raises any connect_timeout below 2 to 2 seconds, so a
    # smaller value would claim a bound that is not actually applied; 0 would
    # disable the timeout entirely. Reject smaller values to fail loudly at
    # startup rather than silently applying a different timeout than configured.
    @model_validator(mode="after")
    def _db_connect_timeout_must_be_valid(self) -> "Settings":
        if self.db_connect_timeout_s < 2:
            raise ValueError(
                "DB_CONNECT_TIMEOUT_S must be >= 2, got "
                f"{self.db_connect_timeout_s}; libpq silently raises any "
                "connect_timeout below 2 to 2 seconds"
            )
        return self

    # A non-positive value would disable the statement timeout in Postgres
    # (0) or be rejected by set_config (negative), silently removing the cap
    # that issue #17 added rather than failing loudly at startup. A value above
    # 60000 is a misconfiguration guard: the ceiling prevents a typo from
    # reinstating the unbounded search mentioned in #17 (closed); search and
    # context expansion each get one statement, so 60 seconds per statement is
    # the ceiling.
    @model_validator(mode="after")
    def _statement_timeout_must_be_positive(self) -> "Settings":
        if not 1 <= self.statement_timeout_ms <= 60000:
            raise ValueError(
                "STATEMENT_TIMEOUT_MS must be in range 1..60000, got "
                f"{self.statement_timeout_ms}; per statement, not per request — "
                "search and context expansion each get one"
            )
        return self


settings = Settings()