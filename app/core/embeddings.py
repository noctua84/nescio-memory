from typing import NamedTuple

import httpx

from app.config import settings
from app.core.embedding_validation import (  # noqa: F401  re-exported
    FLOAT32_MAX,
    FLOAT32_MIN_NORMAL,
    UnusableComponent,
    _as_float32,
    _component_fault,
    _float32_sum_of_squares,
    _validated,
)
from app.core.errors import (
    EmbeddingBackendBadResponse,
    EmbeddingBackendError,
    EmbeddingBackendMisconfigured,
)

_local_model = None

# Kept hardcoded deliberately. Because the first failure aborts the request, this
# bounds a failed ingest at roughly one timeout rather than one per chunk.
OLLAMA_TIMEOUT_SECONDS = 30.0


def _get_local_model():
    global _local_model
    if _local_model is None:
        # Imported lazily so the local-embeddings extra, and the torch it
        # pulls in, are only needed when that backend is selected.
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            # The install hint is for the operator reading logs; the client sees
            # only the class's fixed detail.
            raise EmbeddingBackendMisconfigured(
                "EMBEDDING_BACKEND=local needs the local-embeddings extra; "
                "install it with `uv sync --extra local-embeddings`"
            ) from exc
        try:
            _local_model = SentenceTransformer(settings.local_embedding_model)
        except Exception as exc:
            # Accepted trade-off: a one-off download hiccup on first load is
            # also reported as a 500, and a retry or restart fixes it. The
            # alternative, the transient base class, would let search degrade
            # to FTS-only for ever on a typo'd LOCAL_EMBEDDING_MODEL.
            raise EmbeddingBackendMisconfigured(
                "Could not load LOCAL_EMBEDDING_MODEL "
                f"{settings.local_embedding_model!r}: {exc}"
            ) from exc
    return _local_model


def _fetch_batch(texts: list[str], timeout: float) -> list[object]:
    """Ask the configured backend for one embedding per text, unvalidated.

    The single raw transport for both backends. Nothing here checks the count
    or the contents of what comes back; _embed does, so the probe can still
    measure a backend whose output _validated would reject.
    """
    if settings.embedding_backend == "local":
        try:
            return [row.tolist() for row in _get_local_model().encode(texts)]
        except EmbeddingBackendError:
            # _get_local_model raises EmbeddingBackendMisconfigured for a missing
            # extra or an unloadable model. That is a subclass, so re-raise it
            # unchanged rather than reclassifying a permanent fault as a
            # transient one.
            raise
        except Exception as exc:
            # Everything else here is a runtime fault in the in-process model --
            # memory pressure, for one. Treated as transient: unlike a missing
            # extra or a bad model name, a later attempt may succeed.
            raise EmbeddingBackendError(
                f"Local embedding backend failed: {exc}"
            ) from exc

    # truncate=True matches the silent truncation of the legacy endpoint. With
    # false, an over-long passage fails as an HTTP 400, which the status
    # classification below would report as a misconfiguration.
    payload = {"model": settings.ollama_model, "input": texts, "truncate": True}
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(settings.ollama_url, json=payload)
            response.raise_for_status()
    except httpx.InvalidURL as exc:
        # InvalidURL inherits from Exception rather than httpx.HTTPError, so the
        # clauses below do not catch it. A malformed URL is a deployment
        # mistake, not a transient condition, so it must not promise a retry.
        raise EmbeddingBackendMisconfigured(
            f"OLLAMA_URL is not a valid URL: {settings.ollama_url!r}"
        ) from exc
    except httpx.HTTPStatusError as exc:
        # Must precede the generic clause. Ollama answers 404 for a model that
        # was never pulled and for a build without /api/embed, and 400 for a
        # non-embedding model; retrying never fixes either. Left on the base
        # class they would make search degrade to FTS-only for good while
        # claiming to be transient. 408 and 429 are the 4xx codes that do mean
        # "try again later". The message carries the status and URL, never the
        # input texts.
        code = exc.response.status_code
        if 400 <= code < 500 and code not in (408, 429):
            raise EmbeddingBackendMisconfigured(
                f"Ollama rejected the request with HTTP {code} at "
                f"{settings.ollama_url!r}; check that OLLAMA_MODEL is pulled "
                "and is an embedding model, and that Ollama supports /api/embed"
            ) from exc
        raise EmbeddingBackendError(f"Ollama request failed: HTTP {code}") from exc
    except httpx.HTTPError as exc:
        # httpx.HTTPError is the base of RequestError (connection, timeout), so
        # every remaining transport failure is caught here and the httpx
        # dependency stops at this module's edge.
        raise EmbeddingBackendError(f"Ollama request failed: {exc}") from exc

    try:
        embeddings = response.json()["embeddings"]
    except (ValueError, KeyError, TypeError) as exc:
        # ValueError covers a non-JSON body (JSONDecodeError subclasses it);
        # KeyError and TypeError cover JSON that is not the shape we expect.
        raise EmbeddingBackendBadResponse(
            f"Ollama response was not a usable embedding batch: {exc}"
        ) from exc
    if not isinstance(embeddings, list):
        raise EmbeddingBackendBadResponse(
            f"Ollama response 'embeddings' was {type(embeddings).__name__}, "
            "expected a list"
        )
    return embeddings


def _fetch(text: str) -> object:
    """Ask the configured backend for one embedding, without validating it.

    Split out of get_embedding so probe_embedding_dimension() can measure what
    the backend actually emits. Routing it through get_embedding instead would
    mean the dimension check rejects the very vector being measured, which is
    the one thing the probe must not do. The probe and
    scripts/check_embedding_dimension.py still call this; tests that want the
    validation chain patch _fetch_batch instead.
    """
    vectors = _fetch_batch([text], OLLAMA_TIMEOUT_SECONDS)
    if not vectors:
        # An empty batch for one input would otherwise surface as IndexError.
        raise EmbeddingBackendBadResponse("backend returned no embedding for 1 input")
    return vectors[0]


def _embed(texts: list[str], timeout: float) -> list[list[float]]:
    """Embed texts in order, in sub-batches, validating every vector."""
    if not texts:
        return []
    # Sub-batches bound each request under the 30 s timeout on a CPU Ollama,
    # and bound how long ingest holds its FOR UPDATE lock across these calls
    # (see app/api/v2/ingest.py).
    size = settings.embed_batch_size
    vectors: list[list[float]] = []
    for start in range(0, len(texts), size):
        batch = texts[start : start + size]
        result = _fetch_batch(batch, timeout)
        if len(result) != len(batch):
            # Counts only: the texts are note content and must not be logged.
            raise EmbeddingBackendBadResponse(
                f"backend returned {len(result)} embeddings for {len(batch)} inputs"
            )
        vectors.extend(_validated(vector) for vector in result)
    return vectors


def get_embeddings(texts: list[str]) -> list[list[float]]:
    return _embed(texts, OLLAMA_TIMEOUT_SECONDS)


def get_embedding(text: str) -> list[float]:
    return get_embeddings([text])[0]


def get_query_embedding(query: str) -> list[float]:
    # Qwen3-Embedding is instruction-tuned on the query side only. Passages stay
    # unprefixed, so the prefix does not enter pipeline_version() and changing
    # QUERY_INSTRUCTION never invalidates stored embeddings.
    if settings.query_instruction:
        query = f"Instruct: {settings.query_instruction}\nQuery:{query}"
    return _embed([query], settings.query_embed_timeout_s)[0]


# Short and fixed. The probe's only job is to make the backend emit one vector,
# and a constant keeps the measurement comparable between runs.
PROBE_TEXT = "nescio embedding dimension probe"


class EmbeddingProbe(NamedTuple):
    """One measurement of the backend: how wide it is, and what it cannot store.

    Both halves come from a single call, because two calls could disagree -- and
    an operator told the width by one request and the usability by another has
    no grounds to believe the two describe the same backend state.
    """

    dimensions: int
    unusable: tuple[UnusableComponent, ...]


def probe_embedding() -> EmbeddingProbe:
    """Measure what the backend emits, reporting faults instead of raising.

    Reporting rather than rejecting is the whole point. _validated() stops at
    the first bad component because the request is already lost; a probe is the
    opposite -- a backend emitting NaNs may *also* be the wrong width, and
    raising here would withhold the one measurement this exists to produce. So
    every fault comes back as data and the caller decides what it means.

    See probe_embedding_dimension() for why this is a script and a contract
    test rather than a startup check.
    """
    vector = _fetch(PROBE_TEXT)
    if not isinstance(vector, list):
        # The one fault a probe still cannot report around: with no list there
        # is no width to measure and no components to inspect.
        raise EmbeddingBackendBadResponse(
            f"embedding was {type(vector).__name__}, expected a list"
        )
    faults = (_component_fault(i, component) for i, component in enumerate(vector))
    return EmbeddingProbe(len(vector), tuple(f for f in faults if f is not None))


def probe_embedding_dimension() -> int:
    """How many floats the configured backend actually returns.

    This is the number nothing else in this project measures. The startup check
    in app.core.schema_checks compares EMBEDDING_DIMENSION against the database
    column, and the test suite's fake embedder returns EMBEDDING_DIMENSION
    floats by construction, so a configured width that disagrees with the real
    model stays invisible until the first ingest fails -- which is exactly how
    issue #36 survived a green suite.

    Deliberately not called at startup: that would make boot depend on the
    embedding backend being reachable, and a liveness probe that fails because
    Ollama is restarting is worse than the mismatch it guards against. Called
    instead by the live-model contract test, and -- through probe_embedding(),
    which also reports whether the values are storable -- by
    scripts/check_embedding_dimension.py.
    """
    return probe_embedding().dimensions
