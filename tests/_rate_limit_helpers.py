"""Shared by the tests/test_rate_limit*.py modules."""
import time
from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import text
from sqlalchemy.engine import Engine

from app.core.rate_limit import _tenant_lock_key

SEARCH_PAYLOAD = {"q": "anything", "top_k": 5}


def auth(key: str) -> dict[str, str]:
    return {"X-API-Key": key}


def search(client, key: str, payload: dict = SEARCH_PAYLOAD):
    return client.post("/api/v1/search", json=payload, headers=auth(key))


def ingest(client, key: str, content: str, path: str = "docs/note.md", **overrides):
    payload = {"scope": "repo_a", "path_or_uri": path, "content": content, **overrides}
    return client.post("/api/v1/ingest", json=payload, headers=auth(key))


@contextmanager
def held_slots(engine: Engine, client_name: str, *slots: int) -> Iterator[None]:
    """Hold these ingest slots from a second connection, like another request.

    It has to be a second connection: advisory locks are reentrant within a
    session, so the test's own connection would simply be granted them again.
    Rolling back at the end is what releases them.
    """
    with engine.connect() as connection:
        for slot in slots:
            connection.execute(
                text(
                    "SELECT pg_advisory_xact_lock("
                    "CAST(:tenant AS integer), CAST(:slot AS integer))"
                ),
                {"tenant": _tenant_lock_key(client_name), "slot": slot},
            )
        yield
        connection.rollback()


def clear_of_window_boundary(engine: Engine, window_s: int, margin_s: float = 5.0) -> None:
    """Sleep past the next window boundary if it is less than margin_s away.

    A test that counts requests in one window would otherwise fail, once in a long
    while, by crossing into the next. Only ever sleeps that rarely.
    """
    with engine.connect() as connection:
        left = connection.execute(
            text(
                "SELECT extract(epoch FROM "
                "date_bin(make_interval(secs => :w), clock_timestamp(), TIMESTAMPTZ 'epoch')"
                " + make_interval(secs => :w) - clock_timestamp())"
            ),
            {"w": window_s},
        ).scalar_one()
    if float(left) < margin_s:
        time.sleep(float(left) + 0.05)
