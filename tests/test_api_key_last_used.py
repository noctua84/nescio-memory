"""api_keys.last_used_at: the real recorder, on real separate connections.

Unlike the rest of the suite, the keys here are committed. The recorder writes on
its own connection, which cannot see a row inside a test's SAVEPOINT transaction.
Every committed key is deleted in the fixture's `finally`.

`get_key_usage_recorder` is overridden by the conftest `client` fixture. The tests
that go through HTTP assert on what that override captured (`key_uses`), or take
the override out to run the real recorder end to end.
"""
import logging
import threading
from contextlib import nullcontext
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import create_engine, delete, select, text, update

from app.api.v1 import ingest as ingest_module
from app.config import settings
from app.core import key_usage as key_usage_module
from app.core.errors import EmbeddingBackendError
from app.core.key_usage import get_key_usage_recorder, record_key_use
from app.core.security import _hash_key
from app.main import app
from app.models.api_key import ApiKey
from scripts import manage_api_keys
from tests.factories import make_api_key

# Generous, and only ever waited out when the test is about to fail.
JOIN_TIMEOUT_S = 15
SEARCH_PAYLOAD = {"q": "anything", "top_k": 5}


@pytest.fixture
def committed_key(engine, monkeypatch):
    """(id, plaintext) of a key committed on its own connection, with the real
    recorder pointed at the test database."""
    monkeypatch.setattr(key_usage_module, "get_engine", lambda: engine)
    plaintext = "nm_last_used_probe"
    with engine.begin() as connection:
        key_id = connection.execute(
            ApiKey.__table__.insert()
            .values(
                key_prefix=plaintext[:15],
                key_hash=_hash_key(plaintext),
                client_name="last-used-probe",
            )
            .returning(ApiKey.id)
        ).scalar_one()
    try:
        yield key_id, plaintext
    finally:
        with engine.begin() as connection:
            connection.execute(delete(ApiKey).where(ApiKey.id == key_id))


def _last_used(engine, key_id):
    with engine.connect() as connection:
        return connection.execute(
            select(ApiKey.last_used_at).where(ApiKey.id == key_id)
        ).scalar_one()


def _set_last_used(engine, key_id, value):
    with engine.begin() as connection:
        connection.execute(
            update(ApiKey).where(ApiKey.id == key_id).values(last_used_at=value)
        )


def test_the_first_use_sets_last_used_at(engine, committed_key):
    key_id, _ = committed_key
    assert _last_used(engine, key_id) is None

    record_key_use(key_id)

    assert _last_used(engine, key_id) is not None


def test_a_use_inside_the_throttle_window_does_not_rewrite(engine, committed_key):
    key_id, _ = committed_key
    record_key_use(key_id)
    first = _last_used(engine, key_id)

    record_key_use(key_id)

    assert _last_used(engine, key_id) == first


def test_a_use_after_the_throttle_window_rewrites(engine, committed_key):
    key_id, _ = committed_key
    stale = datetime.now(UTC) - timedelta(seconds=settings.key_last_used_throttle_s + 60)
    _set_last_used(engine, key_id, stale)

    record_key_use(key_id)

    assert _last_used(engine, key_id) > stale


def test_a_zero_throttle_touches_on_every_use(engine, committed_key, monkeypatch):
    monkeypatch.setattr(settings, "key_last_used_throttle_s", 0)
    key_id, _ = committed_key
    earlier = datetime.now(UTC) - timedelta(milliseconds=50)
    _set_last_used(engine, key_id, earlier)

    record_key_use(key_id)

    assert _last_used(engine, key_id) > earlier


def test_a_long_open_request_transaction_does_not_block_the_touch(engine, committed_key):
    """The reason the touch has its own connection.

    The first connection plays ingest's session: it has read the key for auth and
    stays in that transaction, as ingest does across embedding. The touch runs in
    a thread so that a regression fails this test rather than hanging the suite.
    """
    key_id, _ = committed_key
    with engine.connect() as request_session:
        request_session.execute(select(ApiKey).where(ApiKey.id == key_id))
        assert request_session.in_transaction()

        touch = threading.Thread(target=record_key_use, args=(key_id,), daemon=True)
        touch.start()
        touch.join(JOIN_TIMEOUT_S)
        assert not touch.is_alive(), "the touch waited for the request's transaction"

        # Committed and visible from the request's own transaction as well, which
        # is READ COMMITTED: the touch did not wait for it to end.
        visible = request_session.execute(
            select(ApiKey.last_used_at).where(ApiKey.id == key_id)
        ).scalar_one()
        assert visible is not None
        request_session.rollback()


def test_a_failed_touch_logs_a_warning_and_raises_nothing(monkeypatch, caplog):
    dead = create_engine(
        "postgresql+psycopg://x:x@localhost:1/x", connect_args={"connect_timeout": 2}
    )
    monkeypatch.setattr(key_usage_module, "get_engine", lambda: dead)

    with caplog.at_level(logging.WARNING, logger=key_usage_module.logger.name):
        record_key_use(4242)

    records = [r for r in caplog.records if r.name == key_usage_module.logger.name]
    assert len(records) == 1
    assert records[0].levelno == logging.WARNING
    message = records[0].getMessage()
    assert "4242" in message
    assert "OperationalError" in message
    # The driver's message names the host and user; only the class is logged.
    assert "localhost" not in message


def test_the_default_recorder_is_the_real_one():
    assert get_key_usage_recorder() is record_key_use


# --- Through the auth dependency ----------------------------------------------


def _search(client, key):
    return client.post("/api/v1/search", json=SEARCH_PAYLOAD, headers={"X-API-Key": key})


def _id_of(db_session, plaintext):
    return db_session.execute(
        select(ApiKey.id).where(ApiKey.key_hash == _hash_key(plaintext))
    ).scalar_one()


def test_an_authenticated_request_records_the_callers_key(client, db_session, key_uses):
    key = make_api_key(db_session, "acme")
    make_api_key(db_session, "globex")

    assert _search(client, key).status_code == 200

    assert key_uses == [_id_of(db_session, key)]


def test_a_rejected_request_records_nothing(client, db_session, key_uses):
    revoked = make_api_key(db_session, "retired", revoked=True)

    assert _search(client, revoked).status_code == 401
    assert _search(client, "nm_not_a_real_key").status_code == 401

    assert key_uses == []


def test_a_recent_use_takes_no_extra_connection(client, db_session, key_uses):
    """Inside the window the recorder, and so its connection, is never reached."""
    key = make_api_key(db_session, "acme")
    db_session.execute(
        update(ApiKey)
        .where(ApiKey.id == _id_of(db_session, key))
        .values(last_used_at=datetime.now(UTC))
    )
    db_session.expire_all()

    assert _search(client, key).status_code == 200

    assert key_uses == []


def test_a_stale_use_is_recorded(client, db_session, key_uses):
    key = make_api_key(db_session, "acme")
    key_id = _id_of(db_session, key)
    stale = datetime.now(UTC) - timedelta(seconds=settings.key_last_used_throttle_s + 60)
    db_session.execute(update(ApiKey).where(ApiKey.id == key_id).values(last_used_at=stale))
    db_session.expire_all()

    assert _search(client, key).status_code == 200

    assert key_uses == [key_id]


def test_a_zero_throttle_records_every_request(client, db_session, key_uses, monkeypatch):
    monkeypatch.setattr(settings, "key_last_used_throttle_s", 0)
    key = make_api_key(db_session, "acme")
    key_id = _id_of(db_session, key)
    db_session.execute(
        update(ApiKey).where(ApiKey.id == key_id).values(last_used_at=datetime.now(UTC))
    )
    db_session.expire_all()

    assert _search(client, key).status_code == 200

    assert key_uses == [key_id]


def test_a_rolled_back_ingest_still_records_the_use(
    client, engine, committed_key, monkeypatch
):
    """End to end with the real recorder: the embedding 503 ends the request, and
    the touch has already committed on its own connection."""
    key_id, plaintext = committed_key
    del app.dependency_overrides[get_key_usage_recorder]

    def backend_down(texts):
        raise EmbeddingBackendError("backend down")

    monkeypatch.setattr(ingest_module, "get_embeddings", backend_down)

    response = client.post(
        "/api/v1/ingest",
        json={"scope": "repo_a", "path_or_uri": "docs/x.md", "content": "Some body."},
        headers={"X-API-Key": plaintext},
    )

    assert response.status_code == 503
    assert _last_used(engine, key_id) is not None


# --- scripts/manage_api_keys.py list ------------------------------------------


def _list(db_session, capsys):
    argv = ["list", "--client", "acme"]
    assert manage_api_keys.main(argv, session_factory=lambda: nullcontext(db_session)) == 0
    return capsys.readouterr().out


def test_list_shows_last_used(db_session, capsys):
    key = make_api_key(db_session, "acme")
    used_at = datetime(2026, 11, 4, 12, 30, tzinfo=UTC)
    db_session.execute(
        update(ApiKey).where(ApiKey.id == _id_of(db_session, key)).values(last_used_at=used_at)
    )
    db_session.execute(text("SET LOCAL TIME ZONE 'UTC'"))
    db_session.expire_all()

    out = _list(db_session, capsys)

    assert "LAST USED" in out
    assert "2026-11-04T12:30:00" in out


def test_list_shows_a_dash_for_a_key_never_used(db_session, capsys):
    make_api_key(db_session, "acme")

    row = _list(db_session, capsys).splitlines()[1]

    # CREATED, then REVOKED and LAST USED both "-".
    assert row.split()[-2:] == ["-", "-"]
