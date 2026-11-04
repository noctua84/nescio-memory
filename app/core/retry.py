"""Bounded retry with exponential backoff and full jitter.

Knows nothing about embeddings: the caller says which failures are transient and
how long the server asked to be left alone. What this module owns is the bound.
Three independent limits stack, and the retry stops at whichever bites first:

- ``attempts``: tries per call, the first one included.
- ``budget``: retries shared by every call that holds the same RetryBudget, so a
  request made of many calls cannot retry once per call without limit.
- ``deadline``: a monotonic timestamp; a retry that could not finish sleeping
  before it is not started.
"""
import random
import time
from collections.abc import Callable


class RetryBudget:
    """A pool of retries shared across the calls of one request."""

    def __init__(self, retries: int) -> None:
        self.remaining = max(0, retries)

    def take(self) -> bool:
        """Spend one retry; False, spending nothing, once the pool is empty."""
        if self.remaining <= 0:
            return False
        self.remaining -= 1
        return True


def backoff_delay(
    retry_index: int,
    base_delay_s: float,
    max_delay_s: float,
    rand: Callable[[], float] = random.random,
) -> float:
    """Full jitter: uniform in [0, min(max, base * 2**retry_index)).

    retry_index is 0 for the first retry. The exponent is capped so a large
    index cannot overflow the float before min() clamps it.
    """
    ceiling = min(max_delay_s, base_delay_s * 2 ** min(retry_index, 30))
    return rand() * ceiling


def retry_transient[T](
    fn: Callable[[], T],
    *,
    retryable: Callable[[Exception], bool],
    attempts: int,
    base_delay_s: float,
    max_delay_s: float,
    budget: RetryBudget | None = None,
    deadline: float | None = None,
    retry_after: Callable[[Exception], float | None] | None = None,
    on_retry: Callable[[int, float, Exception], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    rand: Callable[[], float] = random.random,
    clock: Callable[[], float] = time.monotonic,
) -> T:
    """Call fn, retrying failures that ``retryable`` accepts.

    The last exception is re-raised unchanged when a limit is reached, so the
    caller's own error handling sees the real failure, not a wrapper.

    ``retry_after`` may return the number of seconds the server asked us to wait.
    It is honoured but capped at ``max_delay_s``: a server cannot stretch the
    wait past what the operator configured.
    """
    attempt = 1
    while True:
        try:
            return fn()
        except Exception as exc:
            if attempt >= attempts or not retryable(exc):
                raise
            hint = retry_after(exc) if retry_after is not None else None
            if hint is not None:
                delay = min(max_delay_s, max(0.0, hint))
            else:
                delay = backoff_delay(attempt - 1, base_delay_s, max_delay_s, rand)
            # Deadline before budget, so a retry that is refused for time does
            # not spend a retry another call could have used.
            if deadline is not None and clock() + delay >= deadline:
                raise
            if budget is not None and not budget.take():
                raise
            if on_retry is not None:
                on_retry(attempt, delay, exc)
            sleep(delay)
            attempt += 1
