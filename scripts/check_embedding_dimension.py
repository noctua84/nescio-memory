# scripts/check_embedding_dimension.py
"""Compare EMBEDDING_DIMENSION against what the configured backend emits.

This is the one check nothing else in the project performs. `verify_embedding_
dimension` at startup compares the setting with the database column, and the
test suite's fake embedder produces EMBEDDING_DIMENSION floats by construction,
so a setting that disagrees with the real model is invisible to both -- which
is how issue #36 shipped a default configuration that could not embed anything
while 93 tests stayed green.

Deliberately a script rather than a startup check: a boot-time probe would make
the API's liveness depend on the embedding backend being reachable, so a
restarting Ollama would take the API down with it. Run this after changing
OLLAMA_MODEL, EMBEDDING_DIMENSION, EMBEDDING_BACKEND or LOCAL_EMBEDDING_MODEL,
and in any deployment pipeline that can reach the backend.

    uv run python -m scripts.check_embedding_dimension

Invoked with -m, not as a path: `python scripts/check_embedding_dimension.py`
puts scripts/ on sys.path instead of the repository root, so `import app`
fails.

Exit codes: 0 match, 1 mismatch, 2 backend unreachable or misconfigured.
"""
import sys

from app.config import settings
from app.core.embeddings import probe_embedding_dimension
from app.core.errors import EmbeddingBackendError

EXIT_OK = 0
EXIT_MISMATCH = 1
EXIT_UNREACHABLE = 2


def _describe_backend() -> str:
    if settings.embedding_backend == "local":
        return f"local backend, model {settings.local_embedding_model}"
    return f"ollama backend at {settings.ollama_url}, model {settings.ollama_model}"


def main() -> int:
    # The console is cp1252 on Windows and the backend name may not be ASCII.
    sys.stdout.reconfigure(encoding="utf-8")

    backend = _describe_backend()
    expected = settings.embedding_dimension

    try:
        actual = probe_embedding_dimension()
    except EmbeddingBackendError as exc:
        # Includes EmbeddingBackendMisconfigured. Distinguished from a mismatch
        # by its exit code: a pipeline may reasonably tolerate "could not
        # check" while never tolerating "checked, and it is wrong".
        print(f"COULD NOT CHECK  {backend}")
        print(f"  {type(exc).__name__}: {exc}")
        return EXIT_UNREACHABLE

    if actual != expected:
        print(f"MISMATCH  {backend}")
        print(f"  emits                {actual} dimensions")
        print(f"  EMBEDDING_DIMENSION  {expected}")
        print(
            "  Every ingest and every search will fail. Set EMBEDDING_DIMENSION "
            f"to {actual} and migrate the learnings.embedding column to "
            f"vector({actual}), or configure a model of width {expected}."
        )
        return EXIT_MISMATCH

    print(f"OK  {backend}")
    print(f"  emits {actual} dimensions, matching EMBEDDING_DIMENSION")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
