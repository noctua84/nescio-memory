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
3 values pgvector cannot store (reported ahead of a mismatch), 64 bad
command-line usage (argparse's own EX_USAGE, deliberately not 2 -- see
EXIT_USAGE below).

Additional flags:
  --degenerate  Report the norm and _validated() verdict for degenerate inputs
                (blank, newlines, punctuation, emoji, single char). Invoked
                with this flag, the exit code is still 0 unless the base
                dimension check fails (the degenerate report is informational).
                #56: if a backend returns zero-norm for non-blank text, the
                zero-norm query case should become a 400 instead of 503.
"""
import argparse
import math
import sys

from app.config import settings
from app.core.embeddings import (
    _component_fault,
    _fetch,
    _float32_sum_of_squares,
    _validated,
    probe_embedding,
)
from app.core.errors import EmbeddingBackendBadResponse, EmbeddingBackendError

EXIT_OK = 0
EXIT_MISMATCH = 1
EXIT_UNREACHABLE = 2
EXIT_UNUSABLE = 3
# argparse's own usage-error exit code, which would otherwise collide with
# EXIT_UNREACHABLE above: argparse.ArgumentParser.error() calls
# sys.exit(2) by default, so "unknown flag" and "backend unreachable" would
# be indistinguishable to a pipeline reading the exit code alone. 64 is
# EX_USAGE from BSD sysexits.h, the conventional code for "the command was
# invoked wrong".
EXIT_USAGE = 64


class _UsageErrorExitCode(argparse.ArgumentParser):
    """ArgumentParser that exits EXIT_USAGE instead of argparse's default 2.

    Overriding error() rather than catching SystemExit around parse_args():
    the default implementation already prints the usage message and the
    error to stderr before exiting, and this keeps that behaviour -- only
    the exit code changes.
    """

    def error(self, message: str) -> None:
        self.print_usage(sys.stderr)
        self.exit(EXIT_USAGE, f"{self.prog}: error: {message}\n")


# A backend that emits one unusable component is usually emitting nothing else,
# so listing every fault would be a thousand near-identical lines with the
# remediation scrolled off the top. Three is enough to show the pattern and to
# find the values again in the backend's own output.
MAX_REPORTED_COMPONENTS = 3


def _describe_backend() -> str:
    if settings.embedding_backend == "local":
        return f"local backend, model {settings.local_embedding_model}"
    return f"ollama backend at {settings.ollama_url}, model {settings.ollama_model}"


def _test_degenerate_inputs() -> None:
    """Test and report embedding norms for degenerate and edge-case inputs.

    #56: synthetic inputs that a real backend might return with zero norm
    or very small norm. Reports L2 norm in float64 (Python's native), the
    norm after float32 rounding with subnormals flushed, and whether
    _validated() would accept the vector. Never prints the vector itself.
    The decision whether to map zero-norm queries to 400 (client error) or
    leave them to become 503 (service error) is for the maintainer.
    """
    degenerate_inputs = [
        " ",
        "\n\n",
        "...",
        "🙂",
        "a",
    ]

    print(f"DEGENERATE INPUTS  {_describe_backend()}")
    for text in degenerate_inputs:
        # _fetch, not get_embedding: get_embedding validates first and would
        # raise on exactly the zero-norm vectors this report exists to measure.
        try:
            vector = _fetch(text)
        except EmbeddingBackendError as exc:
            print(f"  {text!r:20} -> backend error: {exc}")
            continue

        if not isinstance(vector, list):
            print(f"  {text!r:20} -> not a list of numbers")
            continue

        # Judge every component with the same rule _validated() uses, and
        # before computing any norm: an unbounded int (valid JSON; the stdlib
        # json module decodes a huge integer literal into a Python int of
        # arbitrary size) or a non-finite float makes math.isfinite raise
        # OverflowError rather than return a verdict (app/core/embeddings.py
        # :76-78), and the same value would make math.sqrt below and
        # struct.pack inside _float32_sum_of_squares raise too. Checking the
        # shape first turns that crash into a reported verdict line instead.
        faults = [
            fault
            for index, component in enumerate(vector)
            if (fault := _component_fault(index, component)) is not None
        ]
        if faults:
            print(f"  {text!r:20} -> {faults[0].message}")
            continue

        # L2 norm in float64
        norm_f64 = math.sqrt(sum(x * x for x in vector))

        # L2 norm after float32 rounding with subnormals flushed
        norm_f32_sq = _float32_sum_of_squares(vector)
        norm_f32 = math.sqrt(norm_f32_sq) if norm_f32_sq > 0 else 0.0

        # Whether _validated() would accept it
        try:
            _validated(vector)
            accepted = True
        except EmbeddingBackendBadResponse:
            accepted = False

        verdict = "accepted" if accepted else "rejected by _validated()"
        print(
            f"  {text!r:20} -> norm_f64={norm_f64:.6e}, "
            f"norm_f32={norm_f32:.6e}, {verdict}"
        )


def main(argv: list[str] | None = None) -> int:
    # The console is cp1252 on Windows and the backend name may not be ASCII.
    sys.stdout.reconfigure(encoding="utf-8")

    parser = _UsageErrorExitCode(
        description="Check embedding dimension and backend configuration"
    )
    parser.add_argument(
        "--degenerate",
        action="store_true",
        help="Additionally test degenerate inputs (blank, newlines, etc)",
    )
    args = parser.parse_args(argv or [])

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

    if args.degenerate:
        print()
        _test_degenerate_inputs()

    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main(argv=sys.argv[1:]))
