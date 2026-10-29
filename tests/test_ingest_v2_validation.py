"""POST /api/v2/ingest: input validation, the content cap, and large bodies.

The other half of tests/test_ingest_v2.py, split out to keep each module under
the length the plan asks for; the `_ingest` helper and the small readers are
repeated here rather than shared, so neither module depends on the other.

Both validations run before upsert_note and before replace_passages' DELETE,
so every rejection here must also leave the database untouched.
"""
from sqlalchemy import func, select

from app.api.v2.ingest import MAX_CONTENT_CHARS
from app.models.note import Note
from app.models.passage import Passage
from tests.factories import make_api_key

BODY = "The marmoset throughput ledger for the quarter, with no headings at all."


def _ingest(client, key, **overrides):
    payload = {
        "scope": "repo_a",
        "path_or_uri": "docs/note.md",
        "content": BODY,
    }
    payload.update(overrides)
    return client.post("/api/v2/ingest", json=payload, headers={"X-API-Key": key})


def _note(db, note_id) -> Note:
    db.expire_all()
    return db.execute(select(Note).where(Note.id == note_id)).scalar_one()


def _passage_ids(db, note_id) -> list[int]:
    db.expire_all()
    return list(
        db.scalars(
            select(Passage.id)
            .where(Passage.note_id == note_id)
            .order_by(Passage.ordinal)
        )
    )


def test_an_absolute_path_or_uri_is_rejected(client, db_session):
    key = make_api_key(db_session, "acme")

    response = _ingest(client, key, path_or_uri="/etc/passwd")

    assert response.status_code == 400
    assert "relative path" in response.json()["detail"]
    db_session.expire_all()
    assert db_session.scalar(select(func.count()).select_from(Note)) == 0


def test_a_traversing_path_or_uri_is_rejected(client, db_session):
    key = make_api_key(db_session, "acme")

    response = _ingest(client, key, path_or_uri="docs/../../secrets.md")

    assert response.status_code == 400
    assert "relative path" in response.json()["detail"]
    db_session.expire_all()
    assert db_session.scalar(select(func.count()).select_from(Note)) == 0


def test_an_http_uri_is_accepted(client, db_session):
    key = make_api_key(db_session, "acme")
    uri = "https://example.com/docs/adr-001"

    response = _ingest(client, key, path_or_uri=uri)

    assert response.status_code == 200
    assert _note(db_session, response.json()["note_id"]).path_or_uri == uri


def test_validation_runs_before_any_write(client, db_session):
    # Both validations run before upsert_note and before replace_passages'
    # DELETE, so a rejected request leaves what is already stored alone.
    key = make_api_key(db_session, "acme")
    first = _ingest(client, key).json()
    before = _passage_ids(db_session, first["note_id"])

    assert _ingest(client, key, path_or_uri="/etc/passwd").status_code == 400

    db_session.expire_all()
    assert db_session.scalar(select(func.count()).select_from(Note)) == 1
    assert _passage_ids(db_session, first["note_id"]) == before


def test_content_over_the_cap_is_rejected(client, db_session):
    key = make_api_key(db_session, "acme")

    response = _ingest(client, key, content="x" * (MAX_CONTENT_CHARS + 1))

    assert response.status_code == 400
    assert "too large" in response.json()["detail"]
    db_session.expire_all()
    assert db_session.scalar(select(func.count()).select_from(Note)) == 0


def test_content_exactly_at_the_cap_is_accepted(client, db_session):
    # The cap is inclusive.
    key = make_api_key(db_session, "acme")

    response = _ingest(client, key, content="x" * MAX_CONTENT_CHARS)

    assert response.status_code == 200
    assert response.json()["passages"] > 1


def test_a_large_multibyte_body_is_accepted(client, db_session):
    # What justifies the JSON body: v1's form route could not take this at all
    # (Starlette's 1024KB per-field limit fires first, and urlencoding inflates
    # multi-byte characters further). Here MAX_CONTENT_CHARS counts characters
    # and is the only limit that applies.
    key = make_api_key(db_session, "acme")

    response = _ingest(client, key, content="中" * 200_000)

    assert response.status_code == 200
    assert response.json()["passages"] > 1
    assert len(_note(db_session, response.json()["note_id"]).body) == 200_000
