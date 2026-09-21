import httpx

from app.config import settings

_local_model = None


def _get_local_model():
    global _local_model
    if _local_model is None:
        # Imported lazily so the local-embeddings extra, and the torch it
        # pulls in, are only needed when that backend is selected.
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "EMBEDDING_BACKEND=local needs the local-embeddings extra; "
                "install it with `uv sync --extra local-embeddings`"
            ) from exc
        _local_model = SentenceTransformer(settings.local_embedding_model)
    return _local_model


def get_embedding(text: str) -> list[float]:
    if settings.embedding_backend == "local":
        return _get_local_model().encode(text).tolist()

    payload = {"model": settings.ollama_model, "prompt": text}
    with httpx.Client(timeout=30.0) as client:
        response = client.post(settings.ollama_url, json=payload)
        response.raise_for_status()
        return response.json()["embedding"]