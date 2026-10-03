"""Deterministic stand-ins for the one thing these tests do not exercise.

Only the Ollama embedding call is faked. The database, the pgvector extension,
the vector(N) column, the <=> cosine operator and the HNSW index are all
real. Width comes from settings.embedding_dimension, so this file states no
opinion about any model's output size -- which is precisely why it could not
have caught issue #36.
"""
import hashlib
import math
import random

from app.config import settings


def fake_embedding(text: str) -> list[float]:
    """A deterministic unit vector derived from `text`.

    The same text always yields the same vector and different text yields a
    different one, which is all the ingest tests need. Seeded from a SHA-256
    digest rather than Python's hash(), because the latter is salted per
    process and would make results differ between runs.
    """
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    vector = [rng.uniform(-1.0, 1.0) for _ in range(settings.embedding_dimension)]
    magnitude = math.sqrt(sum(component * component for component in vector))
    return [component / magnitude for component in vector]


def unit_vector(axis: int) -> list[float]:
    """A basis vector: 1.0 on `axis`, 0.0 elsewhere.

    Two basis vectors on different axes are orthogonal, so their cosine
    distance is exactly 1.0, and any vector's distance to itself is exactly
    0.0. That makes search ranking assertable against arithmetic we control
    rather than a model's opaque output.
    """
    vector = [0.0] * settings.embedding_dimension
    vector[axis] = 1.0
    return vector
