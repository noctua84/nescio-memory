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


def get_embedding(text: str) -> list[float]:
    if settings.embedding_backend == "local":
        try:
            return _get_local_model().encode(text).tolist()
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
        return response.json()["embedding"]
    except (ValueError, KeyError, TypeError) as exc:
        # ValueError covers a non-JSON body (JSONDecodeError subclasses it);
        # KeyError and TypeError cover JSON that is not the shape we expect.
        raise EmbeddingBackendBadResponse(
            f"Ollama response was not a usable embedding: {exc}"
        ) from exc
