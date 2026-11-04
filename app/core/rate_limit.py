"""Per-tenant limits: a concurrency cap on ingest, a request rate on search.

Both live in Postgres, because that is what every replica already shares, and both
are keyed by client_name, so the overlapping keys of a rotation cannot multiply a
budget. A database failure in either surfaces like any other (503): a limit that
cannot be checked does not silently stop applying.

Ingest is capped by concurrency alone, since what it can exhaust is the embedding
backend for as long as it holds a slot. It has no per-request counter on purpose:
ingest keeps one transaction open from the auth SELECT through embedding to its
commit, so a counter row updated inside it would stay locked for all of that, and
every other request of that client would queue behind it.
"""
import hashlib
import logging

from fastapi import Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.config import settings
from app.core.db import get_db
from app.core.errors import RateLimited
from app.core.security import get_current_client
from app.models.api_key import ApiKey
from app.schemas.errors import ErrorDetail

logger = logging.getLogger(__name__)


def rate_limited_response(description: str) -> dict:
    """A route's OpenAPI entry for 429, in the shape of the router's 503."""
    return {
        "model": ErrorDetail,
        "description": description,
        "headers": {
            "Retry-After": {
                "description": "Seconds to wait before retrying.",
                "schema": {"type": "integer"},
            }
        },
    }


# --- Ingest -------------------------------------------------------------------

# The "try" variant never waits, so it cannot deadlock against the note's FOR
# UPDATE however the two are ordered. The xact variant is released by the commit
# or rollback that ends the ingest, including a dropped connection, so a slot
# cannot leak. The casts are needed because psycopg binds an int as the smallest
# type that fits, and this signature takes exactly two int4.
_TRY_SLOT = text(
    "SELECT pg_try_advisory_xact_lock(CAST(:tenant AS integer), CAST(:slot AS integer))"
)


def _tenant_lock_key(client_name: str) -> int:
    """client_name as the signed int4 Postgres wants for an advisory lock key.

    SHA-256, not hash(): Python randomises str hashes per process, and every
    replica has to arrive at the same key or each would get its own slots. Two
    clients colliding in 32 bits would share their slots, which fails safe and is
    not worth a lookup table at any plausible number of clients.
    """
    digest = hashlib.sha256(client_name.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big", signed=True)


def acquire_ingest_slot(db: Session, client_name: str) -> None:
    """Take a free ingest slot until this transaction ends, or raise RateLimited.

    One statement per slot rather than one over generate_series: a set-returning
    query gives no guarantee of stopping at the first lock it wins (the Postgres
    manual warns about exactly this with LIMIT), and any extra lock would be a slot
    silently taken from the client's own budget.

    Advisory locks are reentrant within a session, which is harmless here: an
    ingest asks once, and each request has its own connection.
    """
    tenant = _tenant_lock_key(client_name)
    for slot in range(settings.rate_limit_ingest_concurrency):
        if db.execute(_TRY_SLOT, {"tenant": tenant, "slot": slot}).scalar_one():
            return
    logger.warning(
        "ingest concurrency limit reached for client %r (%d slots in use)",
        client_name,
        settings.rate_limit_ingest_concurrency,
    )
    raise RateLimited(settings.rate_limit_ingest_retry_after_s)


# --- Search -------------------------------------------------------------------

# clock_timestamp(), not now(): the database's clock is the one every replica
# shares, and now() would be frozen at the start of a transaction that began at
# the auth SELECT. A boundary crossed while waiting for this row's lock can leave
# seconds_left at zero or below, which limit_search raises to 1.
_COUNT_REQUEST = text(
    """
    WITH current_window AS (
        SELECT date_bin(
            make_interval(secs => :window_s), clock_timestamp(), TIMESTAMPTZ 'epoch'
        ) AS window_start
    )
    INSERT INTO rate_limit_windows (client_name, window_start, count)
    SELECT CAST(:client_name AS text), window_start, 1 FROM current_window
    ON CONFLICT (client_name, window_start)
    DO UPDATE SET count = rate_limit_windows.count + 1
    RETURNING
        count,
        window_start,
        CAST(
            ceil(extract(epoch FROM
                window_start + make_interval(secs => :window_s) - clock_timestamp()
            )) AS integer
        ) AS seconds_left
    """
)

# Across all clients on purpose, unlike every other statement in this service: an
# expired window belongs to nobody, this reveals nothing to the caller, and a
# client that stopped searching would otherwise never have its last row removed.
# SKIP LOCKED because this transaction already holds its counter row: two sweeps
# waiting on each other's rows could deadlock, and a skipped row is simply left for
# the next sweep.
_DELETE_EXPIRED = text(
    """
    DELETE FROM rate_limit_windows
    WHERE (client_name, window_start) IN (
        SELECT client_name, window_start
        FROM rate_limit_windows
        WHERE window_start < :current_window
        FOR UPDATE SKIP LOCKED
    )
    """
)


def limit_search(
    db: Session = Depends(get_db),
    client: ApiKey = Depends(get_current_client),
) -> None:
    """Count this request against the client's window; over the limit, raise.

    Runs inside the auth transaction and ENDS it, a little before the commit in
    app/api/v1/search.py would have. That is what keeps the counter row's lock to
    milliseconds. FastAPI runs this dependency and the route in separate threadpool
    tasks, so if the route's commit were the one releasing it, a burst could fill
    the threadpool with same-client requests waiting on the row while the holder
    queues for a thread to commit from. Those waiters have no lock timeout, so only
    the connection pool's own timeout would break the stall.

    Two consequences. The commit expires `client`, so the route's next read of it
    re-selects the row, and search.py's early commit still ends that transaction
    before embedding. And it runs ahead of body validation, so a request answered
    with 422 has been counted, which is right for a limit on request rate.

    A refused request commits as well, so that the sweep sticks and it holds no lock
    while the response is written. Its count keeps climbing past the limit, which
    changes nothing: only the comparison below reads it.
    """
    client_name = client.client_name
    row = db.execute(
        _COUNT_REQUEST,
        {"client_name": client_name, "window_s": settings.rate_limit_window_s},
    ).one()
    # Only the first request of a client's window sweeps, so it costs one scan per
    # client per window rather than one per search.
    if row.count == 1:
        db.execute(_DELETE_EXPIRED, {"current_window": row.window_start})
    db.commit()

    if row.count > settings.rate_limit_search_requests:
        logger.warning(
            "search rate limit reached for client %r (%d requests per %d s)",
            client_name,
            settings.rate_limit_search_requests,
            settings.rate_limit_window_s,
        )
        raise RateLimited(max(1, row.seconds_left))
