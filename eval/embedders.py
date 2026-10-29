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
import time
from typing import Protocol

import eval.appenv  # noqa: F401  -- must precede any app. import; see appenv

from app.config import settings
from app.core.embeddings import get_embedding
from app.core.errors import EmbeddingBackendError, EmbeddingBackendMisconfigured

# A full real-corpus run is ~3,000 serial HTTP calls over roughly an hour and a
# half, and a local Ollama will occasionally answer one of them with a 500 for
# its own reasons. Without a retry, one such blip discards the entire run --
# which is exactly what happened on the first attempt at the 440-note brain,
# after forty minutes of embedding.
#
# This is a RETRY, not a fallback: it asks the same question again and
# substitutes nothing. The rule this package is built on -- never write a
# report over missing data -- is untouched, because exhausting the attempts
# still raises and still aborts the run. The one thing added is that a flaky
# backend costs seconds instead of an hour, and the retry count is reported so
# "the backend wobbled" never passes silently.
EMBED_ATTEMPTS = 3
EMBED_RETRY_SLEEP_SECONDS = 2.0

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

    A transient backend error is retried EMBED_ATTEMPTS times and nothing is
    ever substituted for a failed call. Once the attempts are exhausted the
    exception propagates and the run dies without writing a report, because a
    measurement built on missing data is the worst outcome available to a
    measurement tool. EmbeddingBackendMisconfigured is not retried at all: a
    bad URL or a missing extra is permanent, and sleeping on it only delays
    the same failure.
    """

    label = "real"
    is_real = True

    def __init__(self) -> None:
        self._cache: dict[str, list[float]] = {}
        self.calls = 0
        self.cache_hits = 0
        self.retries = 0

    def __call__(self, text: str) -> list[float]:
        cached = self._cache.get(text)
        if cached is not None:
            self.cache_hits += 1
            return cached
        vector = self._embed_with_retry(text)
        self._cache[text] = vector
        self.calls += 1
        if self.calls % 100 == 0:
            # stderr, so it never lands in a redirected report.
            print(f"  ... {self.calls} embeddings", file=sys.stderr, flush=True)
        return vector

    def _embed_with_retry(self, text: str) -> list[float]:
        for attempt in range(1, EMBED_ATTEMPTS + 1):
            try:
                return get_embedding(text)
            except EmbeddingBackendMisconfigured:
                # Permanent by construction. Fail now rather than three times.
                raise
            except EmbeddingBackendError as exc:
                if attempt == EMBED_ATTEMPTS:
                    raise
                self.retries += 1
                print(
                    f"  ... embedding attempt {attempt}/{EMBED_ATTEMPTS} failed "
                    f"({exc}); retrying in {EMBED_RETRY_SLEEP_SECONDS:g}s",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(EMBED_RETRY_SLEEP_SECONDS)
        # Unreachable: the final attempt either returns or re-raises above.
        raise AssertionError("retry loop fell through")

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
