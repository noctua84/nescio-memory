"""The ingest_events table: one event per change, in the change's transaction.

Everything goes through the real routes against the real database. Each assertion
on an event reads every column it cares about, because a wrong before/after value
is exactly the failure this table exists to prevent: it would answer an operator's
question confidently and wrongly.

Recipes, pruning, Langfuse and settings are in tests/test_audit_trail_ops.py.
"""
import hashlib

import pytest
from sqlalchemy import delete, insert, inspect, select
from sqlalchemy.exc import IntegrityError

from app.api.v1 import ingest as ingest_module
from app.core.errors import EmbeddingBackendError
from app.core.pipeline import content_hash
from app.core.security import _hash_key
from app.models.api_key import ApiKey
from app.models.ingest_event import IngestEvent
from app.models.note import Note
from app.repositories.audit import path_sha256
from tests.factories import make_api_key, make_note, make_passage

PATH = "people/ada-lovelace.md"
PATH_HASH = hashlib.sha256(PATH.encode("utf-8")).hexdigest()
SCOPE = "repo_a"
BODY = "The marmoset throughput ledger for the quarter, with no headings at all."
# Long sections: the splitter merges short ones, and a replace has to change the
# passage count for its before and after to be told apart.
NEW_BODY = (
    "# One\n\n" + "First rewritten section. " * 80
    + "\n\n# Two\n\n" + "Second rewritten section. " * 80 + "\n"
)


def _auth(key):
    return {"X-API-Key": key}


def _ingest(client, key, content=BODY, **overrides):
    payload = {"scope": SCOPE, "path_or_uri": PATH, "content": content, **overrides}
    return client.post("/api/v1/ingest", json=payload, headers=_auth(key))


def _events(db_session, **filters):
    db_session.expire_all()
    stmt = select(IngestEvent).order_by(IngestEvent.id).filter_by(**filters)
    return db_session.execute(stmt).scalars().all()


def _key_id(db_session, plaintext):
    return db_session.execute(
        select(ApiKey.id).where(ApiKey.key_hash == _hash_key(plaintext))
    ).scalar_one()


def _seed(db_session, client_name="acme", path_or_uri=PATH, scope=SCOPE, passages=2):
    note = make_note(db_session, client_name, scope=scope, path_or_uri=path_or_uri)
    for ordinal in range(passages):
        make_passage(db_session, note, ordinal=ordinal, content=f"{path_or_uri} {ordinal}")
    return note


# --- Ingest -------------------------------------------------------------------


def test_a_new_note_writes_one_created_event(client, db_session):
    key = make_api_key(db_session, "acme")

    response = _ingest(client, key)

    assert response.status_code == 200, response.text
    body = response.json()
    [event] = _events(db_session)
    assert event.action == "created"
    assert event.client_name == "acme"
    assert event.api_key_id == _key_id(db_session, key)
    assert event.scope == SCOPE
    assert event.note_id == body["note_id"]
    assert event.path_or_uri == PATH
    assert event.path_sha256 == PATH_HASH
    assert event.passages_before is None
    assert event.passages_after == body["passages"]
    assert event.content_hash_before is None
    assert event.content_hash_after == content_hash(BODY)
    assert event.at is not None


def test_changed_content_writes_one_replaced_event_with_what_it_held(client, db_session):
    key = make_api_key(db_session, "acme")
    first = _ingest(client, key).json()

    second = _ingest(client, key, content=NEW_BODY).json()

    assert second["passages"] != first["passages"]
    created, replaced = _events(db_session)
    assert created.action == "created"
    assert replaced.action == "replaced"
    assert replaced.note_id == first["note_id"]
    assert replaced.path_or_uri == PATH
    assert replaced.path_sha256 == PATH_HASH
    assert replaced.passages_before == first["passages"]
    assert replaced.passages_after == second["passages"]
    assert replaced.content_hash_before == content_hash(BODY)
    assert replaced.content_hash_after == content_hash(NEW_BODY)


def test_unchanged_content_writes_one_skipped_event(client, db_session):
    key = make_api_key(db_session, "acme")
    first = _ingest(client, key).json()

    second = _ingest(client, key).json()

    assert second["skipped"] is True
    assert second["passages"] == first["passages"]
    _, skipped = _events(db_session)
    assert skipped.action == "skipped"
    assert skipped.note_id == first["note_id"]
    assert skipped.path_or_uri == PATH
    assert skipped.passages_before == first["passages"]
    assert skipped.passages_after == first["passages"]
    assert skipped.content_hash_before == content_hash(BODY)
    assert skipped.content_hash_after == content_hash(BODY)


def test_a_forced_reingest_of_the_same_content_is_a_replace(client, db_session):
    key = make_api_key(db_session, "acme")
    _ingest(client, key)

    _ingest(client, key, force=True)

    _, forced = _events(db_session)
    assert forced.action == "replaced"
    assert forced.content_hash_before == forced.content_hash_after == content_hash(BODY)


def test_losing_the_first_ingest_race_is_recorded_as_a_replace(
    client, db_session, monkeypatch
):
    """lock_note saw no row, then upsert_note overwrote one a concurrent first
    ingest had just committed. `created` would be a false statement."""
    key = make_api_key(db_session, "acme")
    winner = _seed(db_session, passages=3)
    monkeypatch.setattr(
        ingest_module.NoteRepository, "lock_note", lambda self, scope, path: None
    )

    response = _ingest(client, key, content=NEW_BODY)

    assert response.json()["note_id"] == winner.id
    [event] = _events(db_session)
    assert event.action == "replaced"
    assert event.passages_before == 3
    # Overwritten by the upsert before anything read it. NULL, not a guess.
    assert event.content_hash_before is None


def test_a_rejected_ingest_writes_no_event(client, db_session):
    key = make_api_key(db_session, "acme")

    assert _ingest(client, key, path_or_uri="/etc/passwd").status_code == 400

    assert _events(db_session) == []


def test_a_rolled_back_ingest_writes_no_event(client, db_session, monkeypatch):
    """An embedding 503 is the real way an ingest fails after it started writing."""
    key = make_api_key(db_session, "acme")

    def backend_down(texts):
        raise EmbeddingBackendError("backend down")

    monkeypatch.setattr(ingest_module, "get_embeddings", backend_down)

    assert _ingest(client, key).status_code == 503

    # What closing the request's session does in production.
    db_session.rollback()
    assert _events(db_session) == []


def test_an_event_does_not_survive_the_rollback_of_its_change(client, db_session, monkeypatch):
    """The event really is in the change's transaction: a failure after it was
    written takes the note and the event back together."""
    key = make_api_key(db_session, "acme")
    db_session.commit()

    def commit_fails():
        assert len(_events(db_session)) == 1, "the event should be written by now"
        raise RuntimeError("commit failed")

    with monkeypatch.context() as patch:
        patch.setattr(db_session, "commit", commit_fails)
        with pytest.raises(RuntimeError, match="commit failed"):
            _ingest(client, key)

    db_session.rollback()
    assert _events(db_session) == []
    assert db_session.scalars(select(Note)).all() == []


def test_events_hold_no_content(client, db_session):
    key = make_api_key(db_session, "acme")

    _ingest(client, key, content="Marmosetsecretword appears in the body only.")

    [event] = _events(db_session)
    values = {c.name: getattr(event, c.name) for c in IngestEvent.__table__.columns}
    assert "Marmosetsecretword" not in repr(values)
    assert not {"content", "body", "name", "description", "meta"} & set(values)


# --- Delete -------------------------------------------------------------------


def _assert_deleted_event(event, note_id, key_id, passages, stored_hash, path=PATH):
    assert event.action == "deleted"
    assert event.client_name == "acme"
    assert event.api_key_id == key_id
    assert event.scope == SCOPE
    assert event.note_id == note_id
    assert event.path_or_uri is None
    assert event.path_sha256 == path_sha256(path)
    assert event.passages_before == passages
    assert event.passages_after is None
    assert event.content_hash_before == stored_hash
    assert event.content_hash_after is None


def test_delete_by_id_writes_one_hashed_event(client, db_session):
    key = make_api_key(db_session, "acme")
    note = _seed(db_session, passages=3)
    note_id, stored_hash = note.id, note.content_hash

    response = client.delete(f"/api/v1/notes/{note_id}", headers=_auth(key))

    assert response.status_code == 204
    [event] = _events(db_session)
    _assert_deleted_event(event, note_id, _key_id(db_session, key), 3, stored_hash)


def test_delete_by_path_writes_one_hashed_event(client, db_session):
    key = make_api_key(db_session, "acme")
    note = _seed(db_session, passages=2)
    _seed(db_session, path_or_uri="docs/other.md")
    note_id, stored_hash = note.id, note.content_hash

    response = client.delete(
        "/api/v1/notes", params={"scope": SCOPE, "path_or_uri": PATH}, headers=_auth(key)
    )

    assert response.status_code == 200, response.text
    [event] = _events(db_session)
    _assert_deleted_event(event, note_id, _key_id(db_session, key), 2, stored_hash)


def test_delete_scope_writes_one_event_per_note_with_its_own_counts(client, db_session):
    key = make_api_key(db_session, "acme")
    three = _seed(db_session, path_or_uri="docs/three.md", passages=3)
    one = _seed(db_session, path_or_uri="docs/one.md", passages=1)
    none = _seed(db_session, path_or_uri="docs/none.md", passages=0)
    _seed(db_session, path_or_uri="docs/elsewhere.md", scope="repo_b")
    expected = {
        note.id: (note.path_or_uri, note.content_hash, passages)
        for note, passages in ((three, 3), (one, 1), (none, 0))
    }

    response = client.delete("/api/v1/scopes", params={"scope": SCOPE}, headers=_auth(key))

    assert response.status_code == 200, response.text
    events = _events(db_session)
    assert len(events) == 3
    key_id = _key_id(db_session, key)
    for event in events:
        path, stored_hash, passages = expected[event.note_id]
        _assert_deleted_event(event, event.note_id, key_id, passages, stored_hash, path=path)


def test_delete_scope_of_an_empty_scope_writes_no_event(client, db_session):
    key = make_api_key(db_session, "acme")

    response = client.delete("/api/v1/scopes", params={"scope": "nothing"}, headers=_auth(key))

    assert response.status_code == 200
    assert _events(db_session) == []


def test_a_delete_that_matched_nothing_writes_no_event(client, db_session):
    key = make_api_key(db_session, "acme")

    assert client.delete("/api/v1/notes/999999", headers=_auth(key)).status_code == 404
    assert (
        client.delete(
            "/api/v1/notes", params={"scope": SCOPE, "path_or_uri": PATH}, headers=_auth(key)
        ).status_code
        == 404
    )

    assert _events(db_session) == []


def test_the_database_refuses_a_plain_path_on_a_deleted_event(db_session):
    row = {
        "client_name": "acme",
        "api_key_id": 1,
        "scope": SCOPE,
        "note_id": 1,
        "action": "deleted",
        "path_or_uri": PATH,
        "path_sha256": PATH_HASH,
    }
    with (
        pytest.raises(IntegrityError, match="ck_ingest_events_deleted_path_hashed"),
        db_session.begin_nested(),
    ):
        db_session.execute(insert(IngestEvent).values(**row))


def test_the_earlier_events_of_a_deleted_note_are_found_by_hash(client, db_session):
    """The documented residual, and the reason every event carries the hash."""
    key = make_api_key(db_session, "acme")
    note_id = _ingest(client, key).json()["note_id"]
    client.delete(f"/api/v1/notes/{note_id}", headers=_auth(key))

    created, deleted = _events(db_session, path_sha256=PATH_HASH)

    assert (created.action, created.path_or_uri) == ("created", PATH)
    assert (deleted.action, deleted.path_or_uri) == ("deleted", None)


# --- Isolation ----------------------------------------------------------------


def test_an_event_carries_the_callers_client_name_and_key(client, db_session):
    acme = make_api_key(db_session, "acme")
    globex = make_api_key(db_session, "globex")

    _ingest(client, acme)
    _ingest(client, globex)

    [acme_event] = _events(db_session, client_name="acme")
    [globex_event] = _events(db_session, client_name="globex")
    assert acme_event.api_key_id == _key_id(db_session, acme)
    assert globex_event.api_key_id == _key_id(db_session, globex)
    assert acme_event.note_id != globex_event.note_id


def test_another_tenants_delete_writes_no_event_and_touches_nothing(client, db_session):
    make_api_key(db_session, "acme")
    globex = make_api_key(db_session, "globex")
    note = _seed(db_session, "acme")
    note_id = note.id

    assert client.delete(f"/api/v1/notes/{note_id}", headers=_auth(globex)).status_code == 404
    assert (
        client.delete(
            "/api/v1/notes", params={"scope": SCOPE, "path_or_uri": PATH}, headers=_auth(globex)
        ).status_code
        == 404
    )
    response = client.delete("/api/v1/scopes", params={"scope": SCOPE}, headers=_auth(globex))

    assert response.json()["notes_deleted"] == 0
    assert _events(db_session) == []
    assert db_session.get(Note, note_id) is not None


def test_a_scope_delete_counts_only_the_callers_passages(client, db_session):
    # Same scope name, other tenant: its passages must not reach this event.
    key = make_api_key(db_session, "acme")
    _seed(db_session, "globex", passages=5)
    _seed(db_session, "acme", passages=1)

    client.delete("/api/v1/scopes", params={"scope": SCOPE}, headers=_auth(key))

    [event] = _events(db_session)
    assert event.client_name == "acme"
    assert event.passages_before == 1


# --- The schema's decisions ---------------------------------------------------


def test_ingest_events_has_no_foreign_keys(engine):
    # An FK to api_keys would take a key-share lock inside ingest's transaction,
    # and an FK to notes could not survive the delete it records.
    assert inspect(engine).get_foreign_keys("ingest_events") == []


def test_the_event_outlives_its_key(client, db_session):
    key = make_api_key(db_session, "acme")
    _ingest(client, key)
    key_id = _key_id(db_session, key)

    db_session.execute(delete(ApiKey).where(ApiKey.id == key_id))

    [event] = _events(db_session)
    assert event.api_key_id == key_id
