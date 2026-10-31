"""Storability rules for embedding vectors, shared by the backend and the probe.

Split out of app.core.embeddings so the rule can be read and tested without the
HTTP and model-loading code around it. app.core.embeddings re-exports every name
here, so existing imports keep working.
"""
import math
import struct
from typing import NamedTuple

from app.config import settings
from app.core.errors import EmbeddingBackendBadResponse

# pgvector's `vector` column type is float4 (float32), not float8. A component
# that is finite in Python's float64 but beyond this magnitude is just as
# unusable as a NaN -- pgvector rejects it with SQLSTATE 22003 ("out of range
# for type vector"), which the error layer maps to a client error even though
# the backend produced the bad value. This is float32's largest finite value;
# it round-trips through float32 exactly, so it is the correct inclusive bound.
FLOAT32_MAX = 3.4028234663852886e38

# float4's smallest normal value. Anything with a smaller magnitude is either
# a subnormal or exactly zero; both are candidates for the zero-norm check
# below, because pgvector's internal float4 arithmetic treats them as
# contributing nothing to a vector's norm. Verified against the real
# container (pgvector/pgvector:pg17): a vector of all-subnormal components
# (every |x| < this bound) stores unchanged but its self cosine-distance
# comes back NaN -- confirming the norm pgvector computes for it is zero even
# though Python's float64 arithmetic on the same stored values is not.
FLOAT32_MIN_NORMAL = 1.17549435e-38


class UnusableComponent(NamedTuple):
    """One component of a vector that pgvector's float4 column cannot hold.

    `message` is the sentence the API surfaces, so the two callers of the rule
    below cannot end up describing the same fault differently. `index` is what
    a diagnostic needs and a request does not: _validated() stops at the first
    fault, while a probe reports every one of them and has to say where.
    """

    index: int
    message: str
    # Set only for a component that could not be evaluated at all, so
    # _validated() can still chain the OverflowError the verdict came from.
    # Without it the traceback would stop at this module and lose the reason.
    cause: BaseException | None = None


def _component_fault(index: int, component: object) -> UnusableComponent | None:
    """Judge one component against float4, without raising.

    The single home of the rule. _validated() turns the first fault into the
    error the caller sees; probe_embedding() collects them all and still reports
    the width. A second copy of this predicate would drift from the first, and
    the probe needs the one thing _validated() must not do -- to measure a
    vector it would reject.
    """
    # bool is a subclass of int, so check it out explicitly rather than
    # letting True sail through as 1.0.
    if isinstance(component, bool) or not isinstance(component, (int, float)):
        return UnusableComponent(
            index,
            f"embedding contained a {type(component).__name__}, expected numbers",
        )
    # NaN passes isinstance(..., float), so the type check above is
    # insufficient to guarantee a usable value, and math.isfinite is still
    # not sufficient on its own: the destination column is float32, narrower
    # than a Python float, so a finite float64 can still be unrepresentable
    # there, and an unbounded int (valid JSON, decoded by stdlib json into a
    # Python int of arbitrary size) makes math.isfinite itself raise rather
    # than return False.
    try:
        finite = math.isfinite(component)
    except OverflowError as exc:
        return UnusableComponent(
            index,
            "embedding component was too large to evaluate as a float",
            exc,
        )
    if not finite or abs(component) > FLOAT32_MAX:
        return UnusableComponent(
            index,
            f"embedding component was {component}, expected finite float32 values",
        )
    return None


def _as_float32(value: float) -> float:
    """Round `value` through float32, matching pgvector's float4 column.

    struct.pack/unpack rather than numpy: numpy is not otherwise a dependency
    of this module, and the standard library already round-trips a float
    through IEEE 754 single precision exactly.
    """
    return struct.unpack("f", struct.pack("f", value))[0]


def _float32_sum_of_squares(vector: list[float]) -> float:
    """The vector's squared norm, accumulated the way float4 arithmetic sees it.

    Doing this in Python's native float64 would miss exactly what #42 is
    about: a component whose float64 square is a tiny but nonzero number can
    still underflow to exactly 0 once it is squared in float32 (confirmed
    empirically -- see FLOAT32_MIN_NORMAL above and the #42 self-distance
    tests), and accumulating in float64 would report a nonzero total where
    pgvector computes an exact zero. Each step below is rounded through
    float32 so the running total cannot "rescue" an underflowed square with
    float64 precision it would not actually have in storage.
    """
    total = 0.0
    for component in vector:
        value = _as_float32(component)
        # Subnormals round-trip through struct.pack unchanged (Python/IEEE 754
        # do not flush them to zero), but pgvector's own arithmetic does
        # not preserve their contribution to a norm -- confirmed empirically:
        # an all-subnormal vector's self cosine-distance comes back NaN. So a
        # subnormal component is treated as zero here too, ahead of squaring.
        if abs(value) < FLOAT32_MIN_NORMAL:
            value = 0.0
        total = _as_float32(total + _as_float32(value * value))
    return total


def _validated(vector: object) -> list[float]:
    """Reject anything that is not a usable embedding.

    The `embedding` key being present says nothing about its value. Without this,
    a wrongly sized or wrongly typed vector reaches pgvector and fails as an
    opaque 500 -- and a null one produced runnable SQL that returned a row, which
    is worse than an error because it is silently wrong.
    """
    expected = settings.embedding_dimension
    if not isinstance(vector, list):
        raise EmbeddingBackendBadResponse(
            f"embedding was {type(vector).__name__}, expected a list"
        )
    if len(vector) != expected:
        raise EmbeddingBackendBadResponse(
            f"embedding had {len(vector)} dimensions, expected {expected}"
        )
    for index, component in enumerate(vector):
        fault = _component_fault(index, component)
        if fault is None:
            continue
        # First fault only: a request has nothing to gain from the rest of the
        # vector once one component makes it unstorable.
        if fault.cause is not None:
            raise EmbeddingBackendBadResponse(fault.message) from fault.cause
        raise EmbeddingBackendBadResponse(fault.message)
    # #42: every component above passed as a usable float32, but the VECTOR as
    # a whole can still be unusable -- a zero vector, an all-subnormal one, or
    # one whose components are individually normal yet each square underflows
    # float32 (so the accumulated norm is zero even though no single component
    # was rejected above). cosine_distance divides by the norm of both sides,
    # so a zero-norm row produces NaN, which search() cannot turn back into a
    # `similarity: float`. Caught here rather than left to the repository
    # because the backend already returned something architecturally
    # unusable; retrying the same input will not help, so this is a 503
    # fault like the component checks above, not a request-shape error.
    # Use <= not == because _float32_sum_of_squares is always >= 0 (sum of
    # squares) and cannot be NaN here (non-finite components are rejected
    # above), so this catches the zero case without float equality comparison.
    if _float32_sum_of_squares(vector) <= 0.0:
        raise EmbeddingBackendBadResponse(
            "embedding had zero norm once rounded to float32, so cosine "
            "distance against it is undefined; retrying with the same "
            "input will not help"
        )
    return vector
