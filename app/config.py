import math
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
    # The batch endpoint: it takes `input` (a list) and returns `embeddings`.
    # The legacy /api/embeddings (single `prompt`) is rejected at startup by
    # _ollama_url_must_be_http below.
    ollama_url: str = "http://localhost:11434/api/embed"
    ollama_model: str = "qwen3-embedding:0.6b"
    # qwen3-embedding:0.6b emits 1024 floats -- measured, not inferred:
    # /api/show reports qwen3.embedding_length 1024 and the embedding endpoint
    # returns 1024-element vectors. Nothing in the suite compares this number against
    # a real model, so `uv run python -m scripts.check_embedding_dimension`
    # exists to do exactly that. (Was 384 until #36, which made the default
    # configuration unable to embed anything at all.)
    embedding_dimension: int = 1024

    # Search and query embedding
    # How many candidates each arm (vector, full-text) contributes to rank
    # fusion. Larger pools give fusion more to reorder at the cost of a wider
    # index scan; 50 comfortably covers the page sizes callers ask for.
    search_candidates: int = 50
    # Reciprocal Rank Fusion damping constant. 60 is the value from the
    # original RRF paper; it keeps a single arm's top rank from dominating.
    rrf_k: int = 60
    # Per-arm multipliers on the fused score. 0 switches an arm off, which is
    # how a single-arm baseline is measured.
    fts_weight: float = 1.0
    vector_weight: float = 1.0
    # Qwen3-Embedding is instruction-tuned on the query side only, so queries
    # get this prefix and passages never do. Empty string disables the prefix,
    # which is what a model without an instruction format needs.
    query_instruction: str = (
        "Given a web search query, retrieve relevant passages that answer the query"
    )
    # A search is interactive, so waiting the 30 s ingest timeout on a dead
    # Ollama would be worse than degrading; this bounds the query embedding.
    query_embed_timeout_s: float = 5.0
    # Texts per /api/embed request. Bounds each request under the 30 s ingest
    # timeout on a CPU Ollama and bounds how long ingest holds its row lock.
    embed_batch_size: int = 16

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
        # Fail fast: Ollama still answers the legacy endpoint, but with a
        # different request/response shape, so a stale .env would otherwise
        # surface per request as a bad-response 503 that retrying never fixes.
        if parsed.path.rstrip("/").endswith("/api/embeddings"):
            raise ValueError(
                "OLLAMA_URL points at the legacy /api/embeddings endpoint, which "
                "is no longer supported: the service now uses the batch endpoint "
                "/api/embed, which takes `input` and returns `embeddings`; "
                f"set OLLAMA_URL=<same host>/api/embed, got {self.ollama_url!r}"
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

    # 0 disables the timeout in Postgres entirely, and a negative value is
    # rejected by set_config -- either way the cap that issue #17 added would
    # be silently removed rather than failing loudly at startup. A value above
    # 60000 is a misconfiguration guard: this is a PER-STATEMENT timeout, not
    # per request -- a search with context runs up to three such statements
    # (the vector search, the context measurement, the context fetch) -- so
    # the ceiling prevents a typo from reinstating the unbounded search
    # mentioned in #17 (closed) under a wall-clock worse than it looks.
    @model_validator(mode="after")
    def _statement_timeout_must_be_in_range(self) -> "Settings":
        if not 1 <= self.statement_timeout_ms <= 60000:
            raise ValueError(
                "STATEMENT_TIMEOUT_MS must be in range 1..60000, got "
                f"{self.statement_timeout_ms}; 0 disables the timeout in "
                "Postgres, and above 60000 risks reinstating the unbounded "
                "search #17 closed -- a search with context runs up to three "
                "statements under this ceiling, not one"
            )
        return self

    # Each knob below fails at startup naming its env var, like the validators
    # above, rather than surfacing as a malformed query or request later.
    @model_validator(mode="after")
    def _search_knobs_must_be_in_range(self) -> "Settings":
        if not 1 <= self.search_candidates <= 1000:
            raise ValueError(
                "SEARCH_CANDIDATES must be in range 1..1000, got "
                f"{self.search_candidates}; 0 leaves fusion nothing to rank and "
                "a very large pool makes every search scan that many rows"
            )
        if self.rrf_k < 1:
            raise ValueError(
                f"RRF_K must be >= 1, got {self.rrf_k}; the fused score is "
                "1 / (RRF_K + rank), which is undefined at rank 0 for RRF_K=0"
            )
        for name, value in (
            ("FTS_WEIGHT", self.fts_weight),
            ("VECTOR_WEIGHT", self.vector_weight),
        ):
            # NaN compares false against everything, so `< 0` alone would let it through.
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and >= 0, got {value}")
        if self.fts_weight == 0 and self.vector_weight == 0:
            raise ValueError(
                "FTS_WEIGHT and VECTOR_WEIGHT must not both be 0, got "
                "FTS_WEIGHT=0 and VECTOR_WEIGHT=0; with both arms off every "
                "search would return nothing"
            )
        return self

    @model_validator(mode="after")
    def _embedding_knobs_must_be_in_range(self) -> "Settings":
        # The upper bound mirrors OLLAMA_TIMEOUT_SECONDS (30) in
        # app.core.embeddings, inlined because config must not import embeddings:
        # a query may never be allowed to wait longer than an ingest call.
        if not (
            math.isfinite(self.query_embed_timeout_s)
            and 0 < self.query_embed_timeout_s <= 30
        ):
            raise ValueError(
                "QUERY_EMBED_TIMEOUT_S must satisfy 0 < x <= 30 "
                f"(OLLAMA_TIMEOUT_SECONDS), got {self.query_embed_timeout_s}"
            )
        if not 1 <= self.embed_batch_size <= 256:
            raise ValueError(
                "EMBED_BATCH_SIZE must be in range 1..256, got "
                f"{self.embed_batch_size}; 0 would never advance through the "
                "texts and a huge batch defeats the per-request timeout bound"
            )
        return self


settings = Settings()