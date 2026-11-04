"""The limits with requests that really overlap, on real separate connections.

Unlike the rest of the suite, nothing here runs in the rolled-back fixture
transaction: each request gets its own session, as in production, so that commits,
rollbacks and the locks they release are the real ones. Everything committed belongs
to one uniquely named client and is deleted in the fixture's `finally`.
"""
import threading
import time
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete, select, text
from sqlalchemy.orm import sessionmaker

from app.api.v1 import ingest as ingest_module
from app.api.v1 import search as search_module
from app.config import settings
from app.core.db import get_db
from app.core.key_usage import get_key_usage_recorder
from app.core.security import _hash_key
from app.main import app
from app.models.api_key import ApiKey
from app.models.ingest_event import IngestEvent
from app.models.note import Note
from app.models.rate_limit import RateLimitWindow
from tests._rate_limit_helpers import clear_of_window_boundary, ingest, search
from tests.fakes import fake_embedding, fake_embeddings

# Generous, and only ever waited out when the test is about to fail.
WAIT_S = 15
BODY = "The marmoset throughput ledger for the quarter, with no headings at all."


class Tenant:
    def __init__(self) -> None:
        self.name = f"rate-limit-{uuid.uuid4().hex}"
        self.key = f"nm_{self.name}"


@pytest.fixture
def tenant(engine, monkeypatch):
    """A committed client, and real per-request sessions on the test database."""
    factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    def real_get_db():
        db = factory()
        try:
            yield db
        finally:
            db.close()

    created = Tenant()
    with engine.begin() as connection:
        connection.execute(
            ApiKey.__table__.insert().values(
                key_prefix=created.key[:15],
                key_hash=_hash_key(created.key),
                client_name=created.name,
            )
        )
    monkeypatch.setattr(ingest_module, "get_embeddings", fake_embeddings)
    monkeypatch.setattr(search_module, "get_query_embedding", fake_embedding)
    app.dependency_overrides[get_db] = real_get_db
    app.dependency_overrides[get_key_usage_recorder] = lambda: lambda key_id: None
    try:
        yield created
    finally:
        app.dependency_overrides.clear()
        with engine.begin() as connection:
            for model in (IngestEvent, Note, RateLimitWindow, ApiKey):
                connection.execute(delete(model).where(model.client_name == created.name))


class Gate:
    """A stand-in embedder that parks every caller until released."""

    def __init__(self, result) -> None:
        self.result = result
        self.release = threading.Event()
        self._entered = threading.Semaphore(0)
        self.calls = 0
        self._lock = threading.Lock()

    def __call__(self, arg):
        with self._lock:
            self.calls += 1
        self._entered.release()
        assert self.release.wait(WAIT_S), "the test never released the embedder"
        return self.result(arg)

    def wait_for_callers(self, count: int) -> None:
        for _ in range(count):
            assert self._entered.acquire(timeout=WAIT_S), "a caller never reached the embedder"


class Call(threading.Thread):
    """Run one request on its own thread and TestClient; `.response` once joined.

    A thread even when the test only waits for it, so that a regression which blocks
    fails the test instead of hanging the suite.
    """

    def __init__(self, request, *args, **kwargs) -> None:
        super().__init__(daemon=True)
        self._call = lambda: request(TestClient(app), *args, **kwargs)
        self.response = None

    def run(self) -> None:
        self.response = self._call()

    def result(self):
        self.join(WAIT_S)
        assert not self.is_alive(), "the request is still blocked"
        return self.response


def _started(request, *args, **kwargs) -> Call:
    call = Call(request, *args, **kwargs)
    call.start()
    return call


def _window_count(engine, client_name):
    with engine.connect() as connection:
        return connection.execute(
            select(RateLimitWindow.count).where(RateLimitWindow.client_name == client_name)
        ).scalar_one_or_none()


def test_only_n_ingests_embed_at_once_and_a_finished_one_frees_its_slot(
    engine, tenant, monkeypatch
):
    gate = Gate(fake_embeddings)
    monkeypatch.setattr(ingest_module, "get_embeddings", gate)
    embedding = [
        _started(ingest, tenant.key, f"{BODY} {n}", path=f"docs/{n}.md") for n in range(2)
    ]
    gate.wait_for_callers(2)

    refused = _started(ingest, tenant.key, f"{BODY} third", path="docs/third.md").result()

    assert refused.status_code == 429
    assert refused.headers["Retry-After"] == str(settings.rate_limit_ingest_retry_after_s)
    assert gate.calls == 2
    with engine.connect() as connection:
        assert (
            connection.execute(
                select(IngestEvent).where(IngestEvent.client_name == tenant.name)
            ).first()
            is None
        )

    gate.release.set()
    assert [call.result().status_code for call in embedding] == [200, 200]

    # Released by the commits above, on connections now back in the pool.
    after = _started(ingest, tenant.key, f"{BODY} third", path="docs/third.md").result()
    assert after.status_code == 200
    with engine.connect() as connection:
        actions = connection.execute(
            select(IngestEvent.action).where(IngestEvent.client_name == tenant.name)
        ).scalars().all()
    assert actions == ["created", "created", "created"]


def test_a_refused_ingest_releases_the_note_lock_and_changes_nothing(
    engine, tenant, monkeypatch
):
    assert _started(ingest, tenant.key, BODY).result().status_code == 200
    gate = Gate(fake_embeddings)
    monkeypatch.setattr(ingest_module, "get_embeddings", gate)
    embedding = [
        _started(ingest, tenant.key, f"{BODY} {n}", path=f"docs/{n}.md") for n in range(2)
    ]
    gate.wait_for_callers(2)

    # A changed body for the existing note: it locks the note, then finds no slot.
    refused = _started(ingest, tenant.key, f"{BODY} changed").result()
    assert refused.status_code == 429

    with engine.connect() as connection:
        connection.execute(text("SET LOCAL lock_timeout = '5s'"))
        stored = connection.execute(
            select(Note.body)
            .where(Note.client_name == tenant.name, Note.path_or_uri == "docs/note.md")
            .with_for_update()
        ).scalar_one()
        connection.rollback()
    assert "changed" not in stored

    gate.release.set()
    assert [call.result().status_code for call in embedding] == [200, 200]


def test_a_skip_succeeds_while_every_slot_is_embedding(tenant, monkeypatch):
    assert _started(ingest, tenant.key, BODY).result().status_code == 200
    gate = Gate(fake_embeddings)
    monkeypatch.setattr(ingest_module, "get_embeddings", gate)
    embedding = [
        _started(ingest, tenant.key, f"{BODY} {n}", path=f"docs/{n}.md") for n in range(2)
    ]
    gate.wait_for_callers(2)

    skipped = _started(ingest, tenant.key, BODY).result()

    assert skipped.status_code == 200
    assert skipped.json()["skipped"] is True
    gate.release.set()
    for call in embedding:
        call.result()


def test_a_long_ingest_does_not_block_same_client_searches(tenant, monkeypatch):
    gate = Gate(fake_embeddings)
    monkeypatch.setattr(ingest_module, "get_embeddings", gate)
    long_ingest = _started(ingest, tenant.key, BODY)
    gate.wait_for_callers(1)

    found = _started(search, tenant.key).result()

    assert found.status_code == 200
    assert long_ingest.is_alive()
    gate.release.set()
    assert long_ingest.result().status_code == 200


def test_a_search_waiting_on_the_embedder_holds_no_counter_lock(engine, tenant, monkeypatch):
    """The slow part of a search must not sit on the row every other search needs."""
    gate = Gate(fake_embedding)
    monkeypatch.setattr(search_module, "get_query_embedding", gate)
    first = _started(search, tenant.key)
    gate.wait_for_callers(1)

    # A second search would get parked by the gate after the counter, so the row
    # is probed directly: it has to be free right now.
    with engine.connect() as connection:
        connection.execute(text("SET LOCAL lock_timeout = '5s'"))
        counted = connection.execute(
            select(RateLimitWindow.count)
            .where(RateLimitWindow.client_name == tenant.name)
            .with_for_update()
        ).scalar_one()
        connection.rollback()

    assert counted == 1
    gate.release.set()
    assert first.result().status_code == 200


def test_the_counter_is_committed_before_the_route_runs(engine, tenant):
    """A request answered with 422 never reaches the route's commit, and still counts.

    This is what proves the limiter ends the transaction itself, rather than leaving
    the row locked until the route's thread comes round.
    """
    invalid = _started(search, tenant.key, {"q": "anything", "top_k": 0}).result()

    assert invalid.status_code == 422
    assert _window_count(engine, tenant.name) == 1


def test_concurrent_searches_never_lose_a_count_and_admit_exactly_the_limit(
    engine, tenant, monkeypatch
):
    monkeypatch.setattr(settings, "rate_limit_search_requests", 5)
    monkeypatch.setattr(settings, "rate_limit_window_s", 3600)
    clear_of_window_boundary(engine, 3600)

    calls = [_started(search, tenant.key) for _ in range(12)]
    statuses = sorted(call.result().status_code for call in calls)

    assert statuses == [200] * 5 + [429] * 7
    assert _window_count(engine, tenant.name) == 12


def test_a_sweep_skips_a_locked_row_rather_than_waiting_for_it(engine, tenant):
    """Two sweeps each holding a counter row must not be able to deadlock."""
    stale_owner = f"{tenant.name}-stale"
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO rate_limit_windows (client_name, window_start, count) "
                "VALUES (:name, TIMESTAMPTZ '2001-01-01 00:00:00+00', 1)"
            ),
            {"name": stale_owner},
        )
    locker = engine.connect()
    try:
        locker.execute(
            select(RateLimitWindow)
            .where(RateLimitWindow.client_name == stale_owner)
            .with_for_update()
        )

        # The first request of the client's window sweeps, and the row is locked.
        assert _started(search, tenant.key).result().status_code == 200

        assert _window_count(engine, stale_owner) == 1
        locker.rollback()

        # Locked no longer, the next sweep takes it. Any new window would do; this
        # forces one by removing the client's current row.
        with engine.begin() as connection:
            connection.execute(
                delete(RateLimitWindow).where(RateLimitWindow.client_name == tenant.name)
            )
        assert _started(search, tenant.key).result().status_code == 200
        assert _window_count(engine, stale_owner) is None
    finally:
        locker.close()
        with engine.begin() as connection:
            connection.execute(
                delete(RateLimitWindow).where(RateLimitWindow.client_name == stale_owner)
            )


def test_a_boundary_crossed_while_waiting_for_the_row_still_sends_a_positive_retry_after(
    engine, tenant, monkeypatch
):
    """Waiting on the row can outlast the window, leaving no seconds to report."""
    window_s = 2
    monkeypatch.setattr(settings, "rate_limit_search_requests", 1)
    monkeypatch.setattr(settings, "rate_limit_window_s", window_s)
    # Starting early in a window gives the request time to reach the row in it. If
    # it missed, it would start a new window and get a 200, failing this loudly.
    clear_of_window_boundary(engine, window_s, margin_s=1.5)
    holder = engine.connect()
    try:
        # An uncommitted row for the current window makes the request's upsert wait.
        holder.execute(
            text(
                "INSERT INTO rate_limit_windows (client_name, window_start, count) "
                "VALUES (:name, date_bin(make_interval(secs => :w), clock_timestamp(), "
                "TIMESTAMPTZ 'epoch'), 5)"
            ),
            {"name": tenant.name, "w": window_s},
        )
        waiting = _started(search, tenant.key)
        # Long enough for the window to be over, whatever was left of it.
        time.sleep(window_s + 0.2)
        holder.commit()
    finally:
        holder.close()

    refused = waiting.result()

    assert refused.status_code == 429
    assert int(refused.headers["Retry-After"]) == 1
