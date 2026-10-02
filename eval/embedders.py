"""The two ways this harness can turn text into a vector, and only one is real.

`real` calls the service's own app.core.embeddings.get_embedding, so the
measurement is of the model production actually uses. That is the whole point:
see this package's __init__ docstring.

`stub` exists ONLY to prove the plumbing -- corpus loading, both strategies,
the similarity scan, the metrics, the report writers -- in an environment with
no Ollama server. It is not an embedding model and its numbers are not a
finding. Every output path labels which one produced it.
"""
import hashlib
import re
import sys
from typing import Protocol

import eval.appenv  # noqa: F401  -- must precede any app. import; see appenv

from app.config import settings
from app.core.embeddings import get_embedding

# A word is the longest run of letters/digits/underscore. Good enough for a
# deliberately crude lexical stub; nothing real depends on it.
_WORD = re.compile(r"\w+", re.UNICODE)


class Embedder(Protocol):
    label: str
    is_real: bool

    def __call__(self, text: str) -> list[float]: ...


class RealEmbedder:
    """app.core.embeddings.get_embedding, with a same-text cache.

    The cache is a correctness-neutral speed-up: get_embedding is a pure
    function of its input for a fixed model, and a full run issues one call per
    chunk plus one per note plus one per query -- a few thousand round trips to
    Ollama. Identical text does occur across strategies and across repeated
    queries, and paying for it twice measures nothing.

    Errors are deliberately NOT caught. app.core.embeddings raises
    EmbeddingBackendError / ...Misconfigured for a backend that is down or
    wrongly deployed, and a measurement run that silently substituted a zero
    vector for a failed call would produce a plausible-looking report built on
    missing data. A hand-run tool should stop and say so.
    """

    label = "real"
    is_real = True

    def __init__(self) -> None:
        self._cache: dict[str, list[float]] = {}
        self.calls = 0
        self.cache_hits = 0

    def __call__(self, text: str) -> list[float]:
        cached = self._cache.get(text)
        if cached is not None:
            self.cache_hits += 1
            return cached
        vector = get_embedding(text)
        self._cache[text] = vector
        self.calls += 1
        if self.calls % 100 == 0:
            # stderr, so it never lands in a redirected report.
            print(f"  ... {self.calls} embeddings", file=sys.stderr, flush=True)
        return vector

    def describe(self) -> str:
        backend = settings.embedding_backend
        model = (
            settings.local_embedding_model
            if backend == "local"
            else settings.ollama_model
        )
        return f"{backend}:{model} ({settings.embedding_dimension}d)"


class LexicalHashingStub:
    """A hashing bag-of-words vectorizer. NOT a model. NOT semantic.

    Each word is hashed into one of `dimension` buckets and counted. Two texts
    are close only to the extent that they repeat the same literal words, so
    this measures lexical overlap and nothing else: it cannot match a
    paraphrase, a synonym, or a question against the answer's prose, which is
    the entire capability under evaluation.

    Why this and not a hash-seeded random vector (the shape tests/fakes.py
    uses)? A random vector per text makes every ranking pure noise, so a stub
    run cannot distinguish "the metrics are wired up correctly" from "the
    metrics always return zero". Lexical overlap produces a non-degenerate,
    reproducible ranking, which exercises the similarity scan, the rank
    bookkeeping and the metric arithmetic for real. It still tells you nothing
    about which indexing strategy is better.
    """

    label = "stub"
    is_real = False

    def __init__(self, dimension: int | None = None) -> None:
        self.dimension = dimension or settings.embedding_dimension
        self.calls = 0

    def __call__(self, text: str) -> list[float]:
        self.calls += 1
        vector = [0.0] * self.dimension
        for word in _WORD.findall(text.lower()):
            digest = hashlib.sha256(word.encode("utf-8")).digest()
            vector[int.from_bytes(digest[:8], "big") % self.dimension] += 1.0
        if not any(vector):
            # A text with no word characters at all. Park it on one fixed axis
            # rather than returning a zero vector, which vectors.l2_normalize
            # rejects outright.
            vector[0] = 1.0
        return vector

    def describe(self) -> str:
        return f"lexical-hashing-stub ({self.dimension}d, NOT a model)"


def build_embedder(kind: str) -> Embedder:
    if kind == "real":
        return RealEmbedder()
    if kind == "stub":
        return LexicalHashingStub()
    raise ValueError(f"unknown embedder {kind!r}")
