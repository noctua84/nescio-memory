from app.config import settings


def chunk_text(
    text: str,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[str]:
    size = chunk_size or settings.chunk_size
    overlap = overlap or settings.chunk_overlap
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