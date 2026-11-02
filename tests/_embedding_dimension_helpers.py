"""Shared staging for the embedding-dimension script's tests.

Used by `test_embedding_dimension_script_verdicts.py` and
`test_embedding_dimension_script_degenerate.py`: the width that is
unmistakably not the configured one, a vector builder, and the seam at
`app.core.embeddings._fetch` that both files stage a backend response through.
Kept in one place so the two files cannot drift on what "a clean vector" or
"the wrong width" mean.
"""

from app.config import settings
from app.core import embeddings as embeddings_module

# A width that is unmistakably not the configured one. Derived from the setting
# rather than written as a literal so it stays wrong for any configured width.
OFF_WIDTH = settings.embedding_dimension + 7


def _clean_vector(width: int | None = None) -> list[float]:
    """A vector of components pgvector's float4 column holds without complaint."""
    return [0.01] * (settings.embedding_dimension if width is None else width)


def _stage(monkeypatch, vector):
    monkeypatch.setattr(embeddings_module, "_fetch", lambda text: vector)
