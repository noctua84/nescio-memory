import math
from typing import NamedTuple

import httpx

from app.config import settings
from app.core.errors import (
    EmbeddingBackendBadResponse,
    EmbeddingBackendError,
    EmbeddingBackendMisconfigured,
)

_local_model = None

# Kept hardcoded deliberately. Because the first failure aborts the request, this
# bounds a failed ingest at roughly one timeout rather than one per chunk.
OLLAMA_TIMEOUT_SECONDS = 30.0

# pgvector's `vector` column type is float4 (float32), not float8. A component
# that is finite in Python's float64 but beyond this magnitude is just as
# unusable as a NaN -- pgvector rejects it with SQLSTATE 22003 ("out of range
# for type vector"), which the error layer maps to a client error even though
# the backend produced the bad value. This is float32's largest finite value;
# it round-trips through float32 exactly, so it is the correct inclusive bound.
FLOAT32_MAX = 3.4028234663852886e38


class UnusableComponent(NamedTuple):
    """One component of a vector that pgvector's float4 column cannot hold.

    `message` is the sentence the API surfaces, so the two callers of the rule
    below cannot end up describing the same fault differently. `index` is what
    a diagnostic needs and a request does not: _validated() stops at the first
    fault, while a probe reports every one of them and has to say where.
    """

    index: int
    message: str
    # Set only for a component that could not be evaluated at all, so
    # _validated() can still chain the OverflowError the verdict came from.
    # Without it the traceback would stop at this module and lose the reason.
    cause: BaseException | None = None


def _component_fault(index: int, component: object) -> UnusableComponent | None:
    """Judge one component against float4, without raising.

    The single home of the rule. _validated() turns the first fault into the
    error the caller sees; probe_embedding() collects them all and still reports
    the width. A second copy of this predicate would drift from the first, and
    the probe needs the one thing _validated() must not do -- to measure a
    vector it would reject.
    """
    # bool is a subclass of int, so check it out explicitly rather than
    # letting True sail through as 1.0.
    if isinstance(component, bool) or not isinstance(component, (int, float)):
        return UnusableComponent(
            index,
            f"embedding contained a {type(component).__name__}, expected numbers",
        )
    # NaN passes isinstance(..., float), so the type check above is
    # insufficient to guarantee a usable value, and math.isfinite is still
    # not sufficient on its own: the destination column is float32, narrower
    # than a Python float, so a finite float64 can still be unrepresentable
    # there, and an unbounded int (valid JSON, decoded by stdlib json into a
    # Python int of arbitrary size) makes math.isfinite itself raise rather
    # than return False.
    try:
        finite = math.isfinite(component)
    except OverflowError as exc:
        return UnusableComponent(
            index,
            "embedding component was too large to evaluate as a float",
            exc,
        )
    if not finite or abs(component) > FLOAT32_MAX:
        return UnusableComponent(
            index,
            f"embedding component was {component}, expected finite float32 values",
        )
    return None


def _validated(vector: object) -> list[float]:
    """Reject anything that is not a usable embedding.

    The `embedding` key being present says nothing about its value. Without this,
    a wrongly sized or wrongly typed vector reaches pgvector and fails as an
    opaque 500 -- and a null one produced runnable SQL that returned a row, which
    is worse than an error because it is silently wrong.
    """
    expected = settings.embedding_dimension
    if not isinstance(vector, list):
        raise EmbeddingBackendBadResponse(
            f"embedding was {type(vector).__name__}, expected a list"
        )
    if len(vector) != expected:
        raise EmbeddingBackendBadResponse(
            f"embedding had {len(vector)} dimensions, expected {expected}"
        )
    for index, component in enumerate(vector):
        fault = _component_fault(index, component)
        if fault is None:
            continue
        # First fault only: a request has nothing to gain from the rest of the
        # vector once one component makes it unstorable.
        if fault.cause is not None:
            raise EmbeddingBackendBadResponse(fault.message) from fault.cause
        raise EmbeddingBackendBadResponse(fault.message)
    return vector


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
        _local_model = SentenceTransformer(settings.local_embedding_model)
    return _local_model


def _fetch(text: str) -> object:
    """Ask the configured backend for an embedding, without validating it.

    Split out of get_embedding so probe_embedding_dimension() can measure what
    the backend actually emits. Routing it through get_embedding instead would
    mean the dimension check rejects the very vector being measured, which is
    the one thing the probe must not do.
    """
    if settings.embedding_backend == "local":
        try:
            vector = _get_local_model().encode(text).tolist()
        except EmbeddingBackendError:
            # _get_local_model raises EmbeddingBackendMisconfigured for a missing
            # extra. That is a subclass, so re-raise it unchanged rather than
            # reclassifying a permanent fault as a transient one.
            raise
        except Exception as exc:
            # Everything else here is a runtime fault in the in-process model --
            # memory pressure, a bad model name, a failed download. Treated as
            # transient: unlike a missing extra, a later attempt may succeed.
            raise EmbeddingBackendError(
                f"Local embedding backend failed: {exc}"
            ) from exc
        return vector

    payload = {"model": settings.ollama_model, "prompt": text}
    try:
        with httpx.Client(timeout=OLLAMA_TIMEOUT_SECONDS) as client:
            response = client.post(settings.ollama_url, json=payload)
            response.raise_for_status()
    except httpx.InvalidURL as exc:
        # InvalidURL inherits from Exception rather than httpx.HTTPError, so the
        # clause below does not catch it. A malformed URL is a deployment
        # mistake, not a transient condition, so it must not promise a retry.
        raise EmbeddingBackendMisconfigured(
            f"OLLAMA_URL is not a valid URL: {settings.ollama_url!r}"
        ) from exc
    except httpx.HTTPError as exc:
        # httpx.HTTPError is the base of RequestError (connection, timeout) and
        # HTTPStatusError, so every transport failure is caught here and the
        # httpx dependency stops at this module's edge.
        raise EmbeddingBackendError(f"Ollama request failed: {exc}") from exc

    try:
        vector = response.json()["embedding"]
    except (ValueError, KeyError, TypeError) as exc:
        # ValueError covers a non-JSON body (JSONDecodeError subclasses it);
        # KeyError and TypeError cover JSON that is not the shape we expect.
        raise EmbeddingBackendBadResponse(
            f"Ollama response was not a usable embedding: {exc}"
        ) from exc
    return vector


def get_embedding(text: str) -> list[float]:
    return _validated(_fetch(text))


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
