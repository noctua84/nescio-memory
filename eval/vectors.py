"""Cosine similarity over plain Python lists, using only the stdlib.

No numpy. The corpus this harness targets is ~424 notes yielding at most a few
thousand vectors of 384 floats, so a brute-force scan in pure Python is well
under a second -- not worth adding a dependency to do arithmetic. numpy is not
currently installed in this project's environment at all.

Vectors are L2-normalized once when the index is built, after which cosine
similarity is just a dot product. That keeps the hot loop a single `sum(...)`
and removes any per-query magnitude work.
"""
import math


class ZeroVectorError(ValueError):
    """A vector of magnitude zero has no direction, so no cosine to anything.

    Raised rather than silently returning zeros: a zero vector reaching the
    index would rank identically against every query and quietly depress the
    scores of whichever strategy produced it, which is precisely the kind of
    silent wrongness this harness exists to avoid.
    """


def l2_normalize(vector: list[float]) -> list[float]:
    """Scale `vector` to unit length."""
    magnitude = math.sqrt(sum(component * component for component in vector))
    if magnitude == 0.0:
        raise ZeroVectorError("cannot normalize a zero-magnitude vector")
    return [component / magnitude for component in vector]


def dot(left: list[float], right: list[float]) -> float:
    """Dot product. For unit vectors this is the cosine similarity."""
    if len(left) != len(right):
        raise ValueError(
            f"dimension mismatch: {len(left)} against {len(right)}"
        )
    return sum(a * b for a, b in zip(left, right))
