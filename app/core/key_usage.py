"""api_keys.last_used_at, written on its own short transaction.

Never on the request's session. Ingest holds one transaction from the auth SELECT
through embedding to its commit, so a touch inside it would keep the key's row
locked for as long as that takes, and every other request with that key would
queue behind it. On a separate connection the row lock lasts for one statement.

An ingest that later rolls back still records the use, which is correct: the key
*was* used.

Injected through get_key_usage_recorder because the suite's sessions live inside
a transaction a separate connection cannot see; tests/conftest.py overrides it.
"""
import logging
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

from sqlalchemy import text

from app.config import settings
from app.core.db import get_engine

logger = logging.getLogger(__name__)

KeyUsageRecorder = Callable[[int], None]

# The throttle lives in the statement so that concurrent requests agree on one
# clock and at most one of them rewrites the row per window.
_TOUCH_KEY = text(
    "UPDATE api_keys SET last_used_at = now() "
    "WHERE id = :id "
    "AND (last_used_at IS NULL "
    "OR last_used_at < now() - make_interval(secs => :throttle))"
)


def key_use_is_due(last_used_at: datetime | None) -> bool:
    """Whether a touch could change anything, judged from the row auth just read.

    This is what makes the extra connection rare. The statement above is throttled
    but still needs a connection to find that out, and at that moment the request
    is already holding one for its auth SELECT: taking a second on every request
    would let a burst the size of the pool hold them all while each waits for
    another. The app's clock only decides whether to ask; the statement stays the
    authority.
    """
    if last_used_at is None:
        return True
    elapsed = datetime.now(UTC) - last_used_at
    return elapsed >= timedelta(seconds=settings.key_last_used_throttle_s)


def record_key_use(key_id: int) -> None:
    """Touch the key's last_used_at; a failure logs a WARNING and nothing else.

    Bookkeeping must not fail the request it describes, so this catches everything
    (a refused connection, an exhausted pool, a canceled statement). The exception
    class is logged rather than its message, which carries connection details.
    """
    try:
        with get_engine().begin() as connection:
            connection.execute(
                _TOUCH_KEY,
                {"id": key_id, "throttle": settings.key_last_used_throttle_s},
            )
    except Exception as exc:
        logger.warning(
            "could not record the use of API key id %s (%s); last_used_at is stale",
            key_id,
            type(exc).__name__,
        )


def get_key_usage_recorder() -> KeyUsageRecorder:
    return record_key_use
