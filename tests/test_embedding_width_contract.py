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
"""
import pytest

from app.config import settings
from app.core.embeddings import PROBE_TEXT, probe_embedding_dimension
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
