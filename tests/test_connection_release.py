"""The invariant in app/core/db.py's get_db, checked deterministically, with no load:
a request's session holds a connection only while a thread is running on it.

A Session can hold a connection only while session.in_transaction() is True, so False
at the moment of a hop means none is held. tests/test_burst_stall.py is the same
invariant as the outage it prevents; this module is what names the hop that broke it.

Like tests/test_rate_limit_concurrency.py, nothing here runs in the rolled-back
fixture transaction, whose session would not release anything: each request gets its
own session, built as the application builds them. Committed rows belong to one
uniquely named client and are deleted in the fixture's `finally`.

States are recorded at the hop and asserted afterwards, not asserted at the hop: an
AssertionError raised inside a request would surface as an unrelated 500.
"""
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field

import anyio.to_thread
import fastapi.routing
import pytest
from fastapi.routing import iter_route_contexts
from fastapi.testclient import TestClient
from sqlalchemy import delete, select
from sqlalchemy.orm import Session, sessionmaker

from app.api.v1 import ingest as ingest_module
from app.api.v1 import search as search_module
from app.config import settings
from app.core.db import get_db
from app.core.errors import RateLimited
from app.core.key_usage import get_key_usage_recorder
from app.core.rate_limit import limit_search
from app.core.security import _hash_key, get_current_client
from app.main import app, create_app
from app.models.api_key import ApiKey
from app.models.ingest_event import IngestEvent
from app.models.note import Note
from app.models.rate_limit import RateLimitWindow
from tests.fakes import fake_embedding, fake_embeddings

API = "/api/v1"
SCOPE = "repo_a"
NOTE_PATH = "docs/note.md"
BODY = "The pelicanium isotope decays slowly in cool storage."


@dataclass
class Harness:
    """One committed client, and what each request did with its session."""

    http: TestClient
    key: str
    client_name: str
    key_id: int
    factory: sessionmaker[Session]
    sessions: list[Session] = field(default_factory=list)
    # Session state on entering fastapi.routing.serialize_response.
    at_validation: list[bool] = field(default_factory=list)
    # Session state on taking a token from anyio's limiter, for any reason.
    at_thread_hop: list[bool] = field(default_factory=list)
    # Session state as get_db's cleanup starts, before it closes the session.
    at_close: list[bool] = field(default_factory=list)

    def forget_requests(self) -> None:
        for recorded in (self.sessions, self.at_validation, self.at_thread_hop, self.at_close):
            recorded.clear()

    def request(self, method: str, url: str, **kwargs):
        return self.http.request(method, url, headers={"X-API-Key": self.key}, **kwargs)

    def ingest(self, content: str = BODY):
        payload = {"scope": SCOPE, "path_or_uri": NOTE_PATH, "content": content}
        return self.request("POST", f"{API}/ingest", json=payload)


@pytest.fixture
def harness(engine, monkeypatch):
    factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    name = f"connection-release-{uuid.uuid4().hex}"
    key = f"nm_{name}"
    with engine.begin() as connection:
        key_id = connection.execute(
            ApiKey.__table__.insert()
            .values(key_prefix=key[:15], key_hash=_hash_key(key), client_name=name)
            .returning(ApiKey.id)
        ).scalar_one()

    built = Harness(TestClient(app), key, name, key_id, factory)

    def real_get_db():
        db = factory()
        built.sessions.append(db)
        try:
            yield db
        finally:
            built.at_close.append(db.in_transaction())
            db.close()

    real_serialize = fastapi.routing.serialize_response

    async def recording_serialize(**kwargs):
        built.at_validation.append(built.sessions[-1].in_transaction())
        return await real_serialize(**kwargs)

    real_run_sync = anyio.to_thread.run_sync

    async def recording_run_sync(func, *args, limiter=None, **kwargs):
        # No limiter means the shared one, the 40 tokens that deadlock. Yield
        # dependencies exit on a limiter of their own, so that is not this hop.
        # Before the first session exists there is nothing to hold: that is get_db
        # itself being entered.
        if limiter is None and built.sessions:
            built.at_thread_hop.append(built.sessions[-1].in_transaction())
        return await real_run_sync(func, *args, limiter=limiter, **kwargs)

    monkeypatch.setattr(ingest_module, "get_embeddings", fake_embeddings)
    monkeypatch.setattr(search_module, "get_query_embedding", fake_embedding)
    monkeypatch.setattr(fastapi.routing, "serialize_response", recording_serialize)
    monkeypatch.setattr(anyio.to_thread, "run_sync", recording_run_sync)
    app.dependency_overrides[get_db] = real_get_db
    app.dependency_overrides[get_key_usage_recorder] = lambda: lambda key_id: None
    try:
        yield built
    finally:
        app.dependency_overrides.clear()
        with engine.begin() as connection:
            for model in (IngestEvent, Note, RateLimitWindow, ApiKey):
                connection.execute(delete(model).where(model.client_name == name))


# --- Every /api/v1 route ------------------------------------------------------


@dataclass(frozen=True)
class Scenario:
    """A successful call of one route. `prepare` returns the URL's path parameters."""

    name: str
    method: str
    path: str
    prepare: Callable[[Harness], dict]
    kwargs: dict
    status: int = 200
    check: Callable[[dict], bool] = lambda body: True
    # False for a route that returns a Response itself: FastAPI then never validates,
    # so the route's return is no hop, and test_nothing_is_left_for_the_session_close_to_end
    # is what reads the state it leaves. Giving such a route a response model fails
    # the validation test until this is corrected.
    validated: bool = True


def _nothing(harness: Harness) -> dict:
    return {}


def _stored_note(harness: Harness) -> dict:
    stored = harness.ingest()
    assert stored.status_code == 200, stored.text
    return {"note_id": stored.json()["note_id"]}


_INGEST = {"json": {"scope": SCOPE, "path_or_uri": NOTE_PATH, "content": BODY}}
_SEARCH = {"json": {"q": "pelicanium", "top_k": 5}}
_BY_PATH = {"params": {"scope": SCOPE, "path_or_uri": NOTE_PATH}}
_BY_SCOPE = {"params": {"scope": SCOPE}}
_NOTE = f"{API}/notes/{{note_id}}"


def _one_deleted(body: dict) -> bool:
    return body["notes_deleted"] == 1


SCENARIOS = [
    Scenario("search", "POST", f"{API}/search", _stored_note, _SEARCH,
             check=lambda body: len(body["results"]) >= 1),
    Scenario("ingest_fresh", "POST", f"{API}/ingest", _nothing, _INGEST,
             check=lambda body: body["skipped"] is False),
    Scenario("ingest_skipped", "POST", f"{API}/ingest", _stored_note, _INGEST,
             check=lambda body: body["skipped"] is True),
    Scenario("get_note", "GET", _NOTE, _stored_note, {},
             check=lambda body: body["passage_count"] >= 1),
    Scenario("delete_note", "DELETE", _NOTE, _stored_note, {}, status=204, validated=False),
    Scenario("delete_by_path", "DELETE", f"{API}/notes", _stored_note, _BY_PATH, check=_one_deleted),
    Scenario("delete_scope", "DELETE", f"{API}/scopes", _stored_note, _BY_SCOPE, check=_one_deleted),
]
ALL_SCENARIOS = pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.name)


def _run(harness: Harness, scenario: Scenario) -> None:
    path_parameters = scenario.prepare(harness)
    harness.forget_requests()

    response = harness.request(
        scenario.method, scenario.path.format(**path_parameters), **scenario.kwargs
    )

    assert response.status_code == scenario.status, response.text
    if response.content:
        assert scenario.check(response.json()), response.text
    assert len(harness.sessions) == 1


def _api_routes(application) -> set[tuple[str, str]]:
    # iter_route_contexts, not app.routes alone: FastAPI keeps an included router as
    # one opaque entry there, so app.routes by itself finds no /api/v1 route at all,
    # and a check built on it would pass whatever was added.
    return {
        (method, context.path_format)
        for context in iter_route_contexts(application.routes)
        if context.path_format.startswith(API)
        for method in context.methods
    }


def test_every_api_route_has_a_scenario():
    """A new route is a new place to return with a transaction open, so it has to say
    how it behaves here before it can ship: add it to SCENARIOS. Equality, so that the
    check also fails if it stops seeing the real routes."""
    assert _api_routes(app) == {(s.method, s.path) for s in SCENARIOS}


def test_a_new_route_would_be_noticed():
    """A check that cannot fail protects nothing, and with app.routes alone this one
    could not (see _api_routes)."""

    def brand_new() -> None: ...

    application = create_app()
    application.add_api_route(f"{API}/brand-new", brand_new, methods=["PUT"])

    assert ("PUT", f"{API}/brand-new") in _api_routes(application)


@ALL_SCENARIOS
def test_no_transaction_is_open_when_response_validation_starts(harness, scenario):
    """The route's return is a hop. A route that has only read and does not commit
    would queue for a thread to validate its response, connection in hand."""
    _run(harness, scenario)

    assert harness.at_validation == ([False] if scenario.validated else [])


@ALL_SCENARIOS
def test_no_transaction_is_open_at_any_thread_hop_of_the_request(harness, scenario):
    """Every token taken from the shared limiter, so that a dependency added later is
    covered without being named. This is also the hop from auth to the next dependency,
    seen from inside a real request."""
    _run(harness, scenario)

    # Auth and the route at the very least, or this is not watching what it claims to.
    assert len(harness.at_thread_hop) >= 2
    assert not any(harness.at_thread_hop), harness.at_thread_hop


@ALL_SCENARIOS
def test_nothing_is_left_for_the_session_close_to_end(harness, scenario):
    """The state a route leaves behind. For delete_note, whose 204 skips validation,
    this is the check that the route itself ended its transaction."""
    _run(harness, scenario)

    assert harness.at_close == [False]


# --- Auth ---------------------------------------------------------------------


def test_auth_returns_with_no_transaction_and_a_record_readable_without_one(harness):
    """The next dependency is a separate threadpool task. Reading client_name or id must
    not re-SELECT, or the transaction just released reopens on the very next line."""
    with harness.factory() as db:
        record = get_current_client(api_key=harness.key, db=db, record_use=lambda key_id: None)
        assert db.in_transaction() is False

        assert record.client_name == harness.client_name
        assert record.id == harness.key_id
        assert db.in_transaction() is False


def test_the_key_use_is_recorded_only_after_the_auth_transaction_ended(harness):
    """Inside the transaction the touch would nest a second checkout under the first,
    and a burst the size of the pool could hold every connection waiting for another."""
    with harness.factory() as db:
        assert (
            db.execute(select(ApiKey.last_used_at).where(ApiKey.id == harness.key_id)).scalar_one()
            is None
        )
        db.rollback()
        seen = []

        get_current_client(
            api_key=harness.key,
            db=db,
            record_use=lambda key_id: seen.append((key_id, db.in_transaction())),
        )

    assert seen == [(harness.key_id, False)]


def test_a_real_request_records_the_key_use_with_its_transaction_ended(harness):
    """The same through the real dependency wiring, where the session is the request's."""
    seen = []
    app.dependency_overrides[get_key_usage_recorder] = lambda: lambda key_id: seen.append(
        (key_id, harness.sessions[-1].in_transaction())
    )

    response = harness.request("POST", f"{API}/search", json={"q": "anything"})

    assert response.status_code == 200, response.text
    assert seen == [(harness.key_id, False)]


# --- limit_search -------------------------------------------------------------


def _authenticated(harness: Harness, db: Session) -> ApiKey:
    return get_current_client(api_key=harness.key, db=db, record_use=lambda key_id: None)


def test_limit_search_leaves_no_transaction_open(harness):
    """The transaction also holds the counter row's lock, which every other search of
    this client needs, so the hop to the route must not happen with it open."""
    with harness.factory() as db:
        limit_search(db=db, client=_authenticated(harness, db))

        assert db.in_transaction() is False


def test_a_refused_search_leaves_no_transaction_open_either(harness, monkeypatch):
    """A refused request has counted first, so it holds the same lock until it commits."""
    monkeypatch.setattr(settings, "rate_limit_search_requests", 0)
    with harness.factory() as db:
        client = _authenticated(harness, db)

        with pytest.raises(RateLimited):
            limit_search(db=db, client=client)

        assert db.in_transaction() is False
