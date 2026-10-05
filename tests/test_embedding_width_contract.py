"""The one contract the rest of the suite cannot check: the real model's width.

Everything else here embeds through `tests.fakes.fake_embedding`, which returns
`settings.embedding_dimension` floats by construction. That makes the suite's
agreement with the configured dimension a tautology, and it is why issue #36 --
EMBEDDING_DIMENSION=384 against a model that emits 1024, so the default
configuration could not ingest or search at all -- survived 93 passing tests.

So this module talks to the configured backend for real. It needs no database
and no container; it needs a reachable embedding backend, and it SKIPS rather
than fails when there is none, because CI has no Ollama. `addopts = "-ra"` in
pyproject.toml means that skip is printed in the short summary of every run:
a silent skip would recreate the hole this test exists to close.

Width is necessary and not sufficient. A model emitting EMBEDDING_DIMENSION
NaNs matches on width and cannot embed anything, and the fake embedder cannot
expose that either -- it returns genuine finite floats by construction. So the
second test below asks the same backend whether the values it emits are ones
pgvector's float4 column can actually hold.
"""
import pytest

from app.config import settings
from app.core.embeddings import (
    PROBE_TEXT,
    probe_embedding,
    probe_embedding_dimension,
)
from app.core.errors import (
    EmbeddingBackendBadResponse,
    EmbeddingBackendError,
    EmbeddingBackendMisconfigured,
)


def _backend_description() -> str:
    if settings.embedding_backend == "local":
        return f"local/{settings.local_embedding_model}"
    return f"ollama {settings.ollama_url} model {settings.ollama_model}"


def test_the_configured_dimension_matches_what_the_backend_emits():
    try:
        actual = probe_embedding_dimension()
    except EmbeddingBackendBadResponse:
        # The backend answered with something that is not a vector at all.
        # That is a genuine failure, not an absent backend, so it must not be
        # laundered into a skip by the broader clause below.
        raise
    except EmbeddingBackendMisconfigured as exc:
        pytest.skip(
            "LIVE WIDTH CHECK SKIPPED -- backend misconfigured for this "
            f"environment ({_backend_description()}): {exc}"
        )
    except EmbeddingBackendError as exc:
        pytest.skip(
            "LIVE WIDTH CHECK SKIPPED -- no reachable embedding backend "
            f"({_backend_description()}): {exc}. EMBEDDING_DIMENSION="
            f"{settings.embedding_dimension} is therefore UNVERIFIED against a "
            "real model in this run. Start the backend, or run "
            "`uv run python -m scripts.check_embedding_dimension` where it is "
            "reachable."
        )

    assert actual == settings.embedding_dimension, (
        f"{_backend_description()} returned {actual} floats for "
        f"{PROBE_TEXT!r} but EMBEDDING_DIMENSION is "
        f"{settings.embedding_dimension}. Every ingest and every search fails "
        "in this configuration: app.core.embeddings rejects the vector before "
        "it reaches pgvector. Correct EMBEDDING_DIMENSION and migrate the "
        "learnings.embedding column to match, or configure a model of the "
        "configured width."
    )


def test_the_backend_emits_values_pgvector_can_actually_store():
    """The other half of the live contract: usable values, not merely enough of them.

    The width test above passes against a backend returning
    EMBEDDING_DIMENSION NaNs -- right count, nothing storable -- and that is
    not a hypothetical shape: it is what a wrong quantisation, a truncated
    model download or an error payload shaped like a vector produces. pgvector's
    `vector` column is float4, so app.core.embeddings rejects such a vector
    before the database sees it and every ingest and every search fails with a
    503 while the width check reports agreement.

    A separate test rather than an extra assertion on the one above, for two
    reasons. The width assertion and its failure message are the regression
    guard for issue #36 and are left exactly as they were. And the two failures
    want different reports: "the configured width is wrong, change it and
    migrate" and "the model is broken, do not change the width" are opposite
    remedies, and a single test could only name one of them at a time.

    The skip structure is repeated verbatim rather than factored out because it
    is the load-bearing part of this module: EmbeddingBackendBadResponse is
    re-raised rather than skipped (a backend answering with something that is
    not a vector is a genuine failure, not an absent backend), and the two
    skips are worded for the situation an operator is actually in. A shared
    helper would make all of that one edit away from being quietly loosened for
    both tests at once.
    """
    try:
        probe = probe_embedding()
    except EmbeddingBackendBadResponse:
        # Same as above: not an absent backend, so it must not be laundered
        # into a skip by the broader clause below.
        raise
    except EmbeddingBackendMisconfigured as exc:
        pytest.skip(
            "LIVE USABILITY CHECK SKIPPED -- backend misconfigured for this "
            f"environment ({_backend_description()}): {exc}"
        )
    except EmbeddingBackendError as exc:
        pytest.skip(
            "LIVE USABILITY CHECK SKIPPED -- no reachable embedding backend "
            f"({_backend_description()}): {exc}. Whether this model emits "
            "values pgvector can store is therefore UNVERIFIED in this run. "
            "Start the backend, or run "
            "`uv run python -m scripts.check_embedding_dimension` where it is "
            "reachable."
        )

    assert probe.unusable == (), (
        f"{_backend_description()} returned {len(probe.unusable)} of "
        f"{probe.dimensions} components that pgvector cannot store for "
        f"{PROBE_TEXT!r}; the first few are "
        f"{[(fault.index, fault.message) for fault in probe.unusable[:3]]}. "
        "Every ingest and every search fails with a 503 in this "
        "configuration: the learnings.embedding column is float4 and "
        "app.core.embeddings rejects the vector before it reaches pgvector. "
        "EMBEDDING_DIMENSION is not the fault -- do not change it to match a "
        "backend in this state; fix or replace the model."
    )
