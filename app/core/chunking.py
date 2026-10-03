from app.config import settings


def chunk_text(
    text: str,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[str]:
    # `or` would turn an explicit overlap=0 (or chunk_size=0, though that is
    # rejected below) into the configured default, silently discarding the
    # caller's choice. Now that ingest passes the window explicitly this is
    # load-bearing: an ingest run configured at CHUNK_OVERLAP=0 must actually
    # chunk at overlap 0, not silently fall back to settings.chunk_overlap.
    size = chunk_size if chunk_size is not None else settings.chunk_size
    overlap = overlap if overlap is not None else settings.chunk_overlap
    # Without this guard the loop below never terminates: a step of zero or
    # less leaves start standing still, or moving backwards.
    if not 0 <= overlap < size:
        raise ValueError(
            "chunk_text requires 0 <= overlap < chunk_size so the window "
            f"advances, got chunk_size={size}, overlap={overlap}"
        )
    step = size - overlap

    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start : start + size])
        start += step
    return chunks