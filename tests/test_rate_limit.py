"""The limits themselves, through the real routes against the real database.

These use the conftest `client`, whose session lives in one rolled-back transaction,
so counters and advisory locks vanish with the test. Other connections cannot see a
key or a counter made in it, which is why anything about separate requests really
overlapping is in tests/test_rate_limit_concurrency.py.
"""
import logging
import math
from datetime import timedelta

import pytest
from sqlalchemy import select, text, update

from app.api.v1 import ingest as ingest_module
from app.api.v1 import search as search_module
from app.config import Settings, settings
from app.core import rate_limit as rate_limit_module
from app.core.errors import RATE_LIMITED_DETAIL, RateLimited
from app.core.pipeline import content_hash
from app.core.rate_limit import _tenant_lock_key, limit_search
from app.models.api_key import ApiKey
from app.models.ingest_event import IngestEvent
from app.models.note import Note
from app.models.rate_limit import RateLimitWindow
from tests._rate_limit_helpers import (
    clear_of_window_boundary,
    held_slots,
    ingest,
    search,
)
from tests.factories import make_api_key, make_note, make_passage
from tests.fakes import fake_embedding, fake_embeddings

BODY = "The marmoset throughput ledger for the quarter, with no headings at all."
HOUR = 3600


def _events(db_session, client_name):
    db_session.expire_all()
    return (
        db_session.execute(
            select(IngestEvent).where(IngestEvent.client_name == client_name)
        )
        .scalars()
        .all()
    )


def _windows(db_session, client_name):
    db_session.expire_all()
    return (
        db_session.execute(
            select(RateLimitWindow)
            .where(RateLimitWindow.client_name == client_name)
            .order_by(RateLimitWindow.window_start)
        )
        .scalars()
        .all()
    )


# --- Ingest: the concurrency cap ----------------------------------------------


def test_the_next_ingest_is_refused_while_every_slot_is_held(client, db_session, engine):
    key = make_api_key(db_session, "acme")

    with held_slots(engine, "acme", 0, 1):
        response = ingest(client, key, BODY)

    assert response.status_code == 429
    assert response.json() == {"detail": RATE_LIMITED_DETAIL}
    assert response.headers["Retry-After"] == "5"
    assert settings.rate_limit_ingest_concurrency == 2


def test_the_retry_after_is_the_setting(client, db_session, engine, monkeypatch):
    monkeypatch.setattr(settings, "rate_limit_ingest_retry_after_s", 37)
    key = make_api_key(db_session, "acme")

    with held_slots(engine, "acme", 0, 1):
        response = ingest(client, key, BODY)

    assert response.headers["Retry-After"] == "37"


def test_the_last_free_slot_is_found(client, db_session, engine, monkeypatch):
    monkeypatch.setattr(settings, "rate_limit_ingest_concurrency", 3)
    key = make_api_key(db_session, "acme")

    with held_slots(engine, "acme", 0, 1):
        assert ingest(client, key, BODY).status_code == 200


def test_the_cap_is_the_setting(client, db_session, engine, monkeypatch):
    monkeypatch.setattr(settings, "rate_limit_ingest_concurrency", 1)
    key = make_api_key(db_session, "acme")

    with held_slots(engine, "acme", 0):
        assert ingest(client, key, BODY).status_code == 429


def test_a_refused_new_note_leaves_no_note_no_event_and_spends_no_embedding(
    client, db_session, engine, monkeypatch
):
    key = make_api_key(db_session, "acme")
    embedded = []
    monkeypatch.setattr(
        ingest_module,
        "get_embeddings",
        lambda texts: embedded.append(texts) or fake_embeddings(texts),
    )

    with held_slots(engine, "acme", 0, 1):
        assert ingest(client, key, BODY).status_code == 429

    assert embedded == []
    assert _events(db_session, "acme") == []
    assert (
        db_session.execute(select(Note).where(Note.client_name == "acme")).first() is None
    )


@pytest.mark.parametrize(
    "stored_hash, overrides",
    [
        pytest.param(None, {"force": True}, id="forced"),
        pytest.param("0" * 64, {}, id="changed"),
    ],
)
def test_a_refused_rewrite_leaves_the_note_as_it_was(
    client, db_session, engine, stored_hash, overrides
):
    key = make_api_key(db_session, "acme")
    note = make_note(
        db_session,
        "acme",
        body=BODY,
        content_hash=content_hash(BODY) if stored_hash is None else stored_hash,
    )
    make_passage(db_session, note)
    before = note.content_hash

    with held_slots(engine, "acme", 0, 1):
        response = ingest(client, key, BODY, path=note.path_or_uri, **overrides)

    assert response.status_code == 429
    db_session.expire_all()
    assert note.content_hash == before
    assert _events(db_session, "acme") == []


def test_a_skipped_reingest_succeeds_with_every_slot_held(client, db_session, engine):
    key = make_api_key(db_session, "acme")
    note = make_note(db_session, "acme", body=BODY)
    make_passage(db_session, note)

    with held_slots(engine, "acme", 0, 1):
        response = ingest(client, key, BODY, path=note.path_or_uri)

    assert response.status_code == 200, response.text
    assert response.json()["skipped"] is True
    [event] = _events(db_session, "acme")
    assert event.action == "skipped"


def test_a_skipped_reingest_takes_no_slot(client, db_session, engine):
    """Not merely "succeeds": the limiter is not reached at all."""
    key = make_api_key(db_session, "acme")
    note = make_note(db_session, "acme", body=BODY)
    make_passage(db_session, note)

    assert ingest(client, key, BODY, path=note.path_or_uri).json()["skipped"] is True

    locks = db_session.execute(
        text(
            "SELECT count(*) FROM pg_locks WHERE locktype = 'advisory' "
            "AND pid = pg_backend_pid()"
        )
    ).scalar_one()
    assert locks == 0


def test_a_successful_ingest_takes_exactly_one_slot(client, db_session):
    key = make_api_key(db_session, "acme")

    assert ingest(client, key, BODY).status_code == 200

    # The fixture's commit releases only a SAVEPOINT, so the slot is still held
    # here, where a real commit would have released it.
    slots = db_session.execute(
        text(
            "SELECT classid::bigint AS tenant, objid::bigint AS slot FROM pg_locks "
            "WHERE locktype = 'advisory' AND pid = pg_backend_pid()"
        )
    ).all()
    # pg_locks shows the int4 key as an unsigned oid.
    assert [(row.tenant, row.slot) for row in slots] == [
        (_tenant_lock_key("acme") & 0xFFFFFFFF, 0)
    ]


def test_a_refusal_logs_a_warning_naming_the_client(client, db_session, engine, caplog):
    key = make_api_key(db_session, "acme")

    with (
        caplog.at_level(logging.WARNING, logger=rate_limit_module.logger.name),
        held_slots(engine, "acme", 0, 1),
    ):
        ingest(client, key, BODY)

    [record] = [r for r in caplog.records if r.name == rate_limit_module.logger.name]
    assert record.levelno == logging.WARNING
    assert "acme" in record.getMessage()


def test_the_lock_key_is_stable_and_fits_postgres_int4():
    # A literal, because the property is "the same in every process": a key built on
    # Python's hash() would pass every other test here and give each replica its
    # own slots.
    assert _tenant_lock_key("acme") == -2111097939
    assert _tenant_lock_key("globex") == 1539416205
    assert -(2**31) <= _tenant_lock_key("a" * 500) < 2**31


# --- Search: the fixed window -------------------------------------------------


@pytest.fixture
def three_per_hour(engine, monkeypatch):
    monkeypatch.setattr(settings, "rate_limit_search_requests", 3)
    monkeypatch.setattr(settings, "rate_limit_window_s", HOUR)
    clear_of_window_boundary(engine, HOUR)


def test_searches_up_to_the_limit_pass_and_the_next_is_refused(
    client, db_session, three_per_hour
):
    key = make_api_key(db_session, "acme")

    assert [search(client, key).status_code for _ in range(3)] == [200, 200, 200]
    refused = search(client, key)

    assert refused.status_code == 429
    assert refused.json() == {"detail": RATE_LIMITED_DETAIL}


def test_the_retry_after_is_the_time_left_in_the_window(client, db_session, three_per_hour):
    key = make_api_key(db_session, "acme")
    for _ in range(3):
        search(client, key)

    retry_after = int(search(client, key).headers["Retry-After"])

    [window] = _windows(db_session, "acme")
    left_now = db_session.execute(
        text(
            "SELECT extract(epoch FROM "
            "CAST(:start AS timestamptz) + make_interval(secs => :w) - clock_timestamp())"
        ),
        {"start": window.window_start, "w": HOUR},
    ).scalar_one()
    # Measured a moment after the response, so at most a little smaller than it.
    assert math.ceil(left_now) <= retry_after <= math.ceil(left_now) + 2
    # Epoch aligned, so it is a whole number of windows from the epoch.
    assert window.window_start.timestamp() % HOUR == 0


def test_the_next_window_starts_a_fresh_count_and_sweeps_the_old_one(
    client, db_session, three_per_hour
):
    key = make_api_key(db_session, "acme")
    for _ in range(3):
        search(client, key)
    assert search(client, key).status_code == 429
    # Moving the row back one window is the same as the clock moving forward one.
    db_session.execute(
        update(RateLimitWindow)
        .where(RateLimitWindow.client_name == "acme")
        .values(window_start=RateLimitWindow.window_start - timedelta(seconds=HOUR))
    )

    assert search(client, key).status_code == 200

    [window] = _windows(db_session, "acme")
    assert window.count == 1


def test_only_the_expired_windows_are_swept(client, db_session, three_per_hour):
    key = make_api_key(db_session, "acme")
    search(client, key)
    [current] = _windows(db_session, "acme")
    start = current.window_start
    db_session.add_all(
        [
            RateLimitWindow(
                client_name="dormant", window_start=start - timedelta(hours=9), count=7
            ),
            RateLimitWindow(client_name="globex", window_start=start, count=2),
        ]
    )
    db_session.flush()
    db_session.delete(current)
    db_session.flush()

    # The first request of acme's window is the one that sweeps.
    assert search(client, key).status_code == 200

    assert _windows(db_session, "dormant") == []
    assert [w.count for w in _windows(db_session, "globex")] == [2]


def test_a_refused_search_does_no_search_work(
    client, db_session, three_per_hour, monkeypatch
):
    key = make_api_key(db_session, "acme")
    for _ in range(3):
        search(client, key)
    embedded = []
    monkeypatch.setattr(
        search_module,
        "get_query_embedding",
        lambda q: embedded.append(q) or fake_embedding(q),
    )

    assert search(client, key).status_code == 429

    assert embedded == []


def test_a_failed_authentication_is_not_counted(client, db_session, three_per_hour):
    key = make_api_key(db_session, "acme")
    revoked = make_api_key(db_session, "acme", plaintext="nm_test_acme_old", revoked=True)

    assert search(client, revoked).status_code == 401
    assert search(client, "nm_not_a_real_key").status_code == 401
    assert _windows(db_session, "acme") == []
    assert search(client, key).status_code == 200


def test_no_transaction_is_open_when_the_limiter_returns(client, db_session, three_per_hour):
    """What keeps the counter row's lock to milliseconds, whatever comes next.

    The dependency and the route run in separate threadpool tasks, so the row must
    not stay locked until the route gets a thread to commit from.
    """
    make_api_key(db_session, "acme")
    row = db_session.execute(select(ApiKey).where(ApiKey.client_name == "acme")).scalar_one()
    assert db_session.in_transaction()

    limit_search(db=db_session, client=row)

    assert not db_session.in_transaction()


def test_a_refused_search_leaves_no_transaction_open_either(
    client, db_session, three_per_hour
):
    make_api_key(db_session, "acme")
    row = db_session.execute(select(ApiKey).where(ApiKey.client_name == "acme")).scalar_one()
    for _ in range(3):
        limit_search(db=db_session, client=row)
    db_session.execute(select(ApiKey.id)).first()
    assert db_session.in_transaction()

    with pytest.raises(RateLimited):
        limit_search(db=db_session, client=row)

    assert not db_session.in_transaction()


# --- Settings -----------------------------------------------------------------


@pytest.mark.parametrize(
    "field, env_name, low, high",
    [
        ("rate_limit_ingest_concurrency", "RATE_LIMIT_INGEST_CONCURRENCY", 1, 32),
        ("rate_limit_ingest_retry_after_s", "RATE_LIMIT_INGEST_RETRY_AFTER_S", 1, 60),
        ("rate_limit_search_requests", "RATE_LIMIT_SEARCH_REQUESTS", 1, 10000),
        ("rate_limit_window_s", "RATE_LIMIT_WINDOW_S", 1, 3600),
    ],
)
def test_each_limit_is_bounded_and_the_error_names_its_variable(field, env_name, low, high):
    base = {"database_url": "postgresql+psycopg://u:p@localhost/db", "_env_file": None}

    assert getattr(Settings(**base, **{field: low}), field) == low
    assert getattr(Settings(**base, **{field: high}), field) == high
    for bad in (low - 1, high + 1):
        with pytest.raises(ValueError, match=env_name):
            Settings(**base, **{field: bad})


def test_the_defaults_are_the_documented_ones():
    fields = Settings.model_fields
    assert fields["rate_limit_ingest_concurrency"].default == 2
    assert fields["rate_limit_ingest_retry_after_s"].default == 5
    assert fields["rate_limit_search_requests"].default == 120
    assert fields["rate_limit_window_s"].default == 60
