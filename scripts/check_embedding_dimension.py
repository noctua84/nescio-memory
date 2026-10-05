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

Exit codes: 0 match, 1 mismatch, 2 backend unreachable or misconfigured,
3 values pgvector cannot store (reported ahead of a mismatch).
"""
import sys

from app.config import settings
from app.core.embeddings import probe_embedding
from app.core.errors import EmbeddingBackendError

EXIT_OK = 0
EXIT_MISMATCH = 1
EXIT_UNREACHABLE = 2
EXIT_UNUSABLE = 3

# A backend that emits one unusable component is usually emitting nothing else,
# so listing every fault would be a thousand near-identical lines with the
# remediation scrolled off the top. Three is enough to show the pattern and to
# find the values again in the backend's own output.
MAX_REPORTED_COMPONENTS = 3


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
        probe = probe_embedding()
    except EmbeddingBackendError as exc:
        # Includes EmbeddingBackendMisconfigured. Distinguished from a mismatch
        # by its exit code: a pipeline may reasonably tolerate "could not
        # check" while never tolerating "checked, and it is wrong".
        print(f"COULD NOT CHECK  {backend}")
        print(f"  {type(exc).__name__}: {exc}")
        return EXIT_UNREACHABLE

    actual = probe.dimensions

    if probe.unusable:
        # Unusable values are reported ahead of a width mismatch, and a run
        # with both faults says so. Both have the same cause -- the wrong model
        # -- but opposite remedies, and MISMATCH's is the harmful one to act on
        # first: setting EMBEDDING_DIMENSION to this width and migrating the
        # column adopts the shape of a backend whose output cannot be stored at
        # any width, so it fixes nothing and migrates the column for nothing.
        print(f"UNUSABLE VALUES  {backend}")
        print(f"  emits                {actual} dimensions")
        print(f"  EMBEDDING_DIMENSION  {expected}")
        print(f"  unusable components  {len(probe.unusable)} of {actual}")
        for fault in probe.unusable[:MAX_REPORTED_COMPONENTS]:
            print(f"    index {fault.index}: {fault.message}")
        remaining = len(probe.unusable) - MAX_REPORTED_COMPONENTS
        if remaining > 0:
            print(f"    ... and {remaining} more")
        print(
            "  Every ingest and every search will fail with a 503 while this "
            "holds: the learnings.embedding column is float4 and pgvector "
            "cannot store these values, so app.core.embeddings rejects the "
            "vector before the database sees it. EMBEDDING_DIMENSION is not "
            "the fault here -- fix or replace the model, then re-run."
        )
        if actual != expected:
            print(
                f"  The width is wrong too ({actual}, expected {expected}), but "
                "do not change EMBEDDING_DIMENSION to match a backend in this "
                "state; re-run once it emits usable values and treat whatever "
                "width it reports then as the real one."
            )
        return EXIT_UNUSABLE

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
