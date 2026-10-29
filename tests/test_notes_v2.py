"""GET and DELETE /api/v2/notes/{note_id}: the read and erase paths.

Tenant isolation for these routes lives in tests/test_isolation_v2.py and is
deliberately not duplicated here; everything below runs as a single client and
asserts the non-isolation contract -- which fields come back, what the 404
covers, and that a delete really erases the passages rather than orphaning
them.
"""
from datetime import date

from sqlalchemy import select

from app.core.splitter import split_note
from app.models.passage import Passage
from tests.factories import make_api_key, make_note, make_passage

NOTE_WITH_FRONTMATTER = """---
name: ADR-0001 Pelican routing
description: Why the pelican router replaced the albatross one.
type: adr
---

# ADR-0001

[Source: user override — 2026-02-03]

The pelican router is chosen because it keeps the hop count bounded.
"""


def _ingest(client, key, **overrides):
    payload = {
        "scope": "repo_a",
        "path_or_uri": "docs/note.md",
        "content": NOTE_WITH_FRONTMATTER,
    }
    payload.update(overrides)
    return client.post("/api/v2/ingest", json=payload, headers={"X-API-Key": key})


def _get(client, key, note_id):
    return client.get(f"/api/v2/notes/{note_id}", headers={"X-API-Key": key})


def _delete(client, key, note_id):
    return client.delete(f"/api/v2/notes/{note_id}", headers={"X-API-Key": key})


def _search(client, key, q, **overrides):
    payload = {"q": q, "top_k": 10}
    payload.update(overrides)
    return client.post("/api/v2/search", json=payload, headers={"X-API-Key": key})


def test_get_note_returns_every_stored_field(client, db_session):
    key = make_api_key(db_session, "acme")
    note = make_note(
        db_session,
        "acme",
        scope="repo_a",
        path_or_uri="docs/architecture/adr-001.md",
        name="ADR-001: Use PostgreSQL",
        description="Why Postgres and not a document store.",
        type_="adr",
        provenance_kind="user override",
        provenance_date=date(2026, 2, 3),
        status="approved",
        body="the stored body of the note",
        meta={"source": "promotion"},
    )
    make_passage(db_session, note, ordinal=0, content="first passage")

    response = _get(client, key, note.id)

    assert response.status_code == 200
    body = response.json()
    assert body == {
        "id": note.id,
        "scope": "repo_a",
        "path_or_uri": "docs/architecture/adr-001.md",
        "name": "ADR-001: Use PostgreSQL",
        "description": "Why Postgres and not a document store.",
        "type": "adr",
        "provenance_kind": "user override",
        "provenance_date": "2026-02-03",
        "status": "approved",
        "content_hash": note.content_hash,
        "pipeline_version": note.pipeline_version,
        "body": "the stored body of the note",
        "meta": {"source": "promotion"},
        "passage_count": 1,
        "created_at": body["created_at"],
        "updated_at": body["updated_at"],
    }
    # Asserted separately: the timestamps are server-generated, so the dict
    # comparison above cannot pin their values -- only that they are present
    # and populated.
    assert body["created_at"] is not None
    assert body["updated_at"] is not None


def test_get_note_returns_the_passage_count(client, db_session):
    key = make_api_key(db_session, "acme")
    note = make_note(db_session, "acme")
    for ordinal in range(3):
        make_passage(db_session, note, ordinal=ordinal, content=f"passage {ordinal}")
    # A second note with its own passage, so a count that ignored note_id
    # would come back as 4.
    other = make_note(db_session, "acme", path_or_uri="docs/other.md")
    make_passage(db_session, other, ordinal=0, content="unrelated passage")

    response = _get(client, key, note.id)

    assert response.status_code == 200
    assert response.json()["passage_count"] == 3


def test_get_note_returns_404_for_an_unknown_id(client, db_session):
    key = make_api_key(db_session, "acme")

    response = _get(client, key, 999_999)

    assert response.status_code == 404
    assert response.json() == {"detail": "Note not found"}


def test_get_note_rejects_a_non_integer_id(client, db_session):
    key = make_api_key(db_session, "acme")

    response = client.get("/api/v2/notes/not-an-int", headers={"X-API-Key": key})

    assert response.status_code == 422


def test_a_note_ingested_through_v2_is_retrievable_by_the_returned_id(
    client, db_session
):
    key = make_api_key(db_session, "acme")

    ingested = _ingest(client, key)
    assert ingested.status_code == 200
    note_id = ingested.json()["note_id"]

    response = _get(client, key, note_id)

    assert response.status_code == 200
    body = response.json()
    # The round trip: what GET returns is what the splitter produced, not the
    # raw submitted content (the frontmatter block is not part of the body).
    expected = split_note(NOTE_WITH_FRONTMATTER)
    assert body["body"] == expected.body
    assert body["name"] == expected.name
    assert body["description"] == expected.description
    assert body["type"] == expected.type
    assert body["path_or_uri"] == "docs/note.md"
    assert body["scope"] == "repo_a"
    assert body["passage_count"] == ingested.json()["passages"]


def test_delete_note_returns_204_and_an_empty_body(client, db_session):
    key = make_api_key(db_session, "acme")
    note = make_note(db_session, "acme")

    response = _delete(client, key, note.id)

    assert response.status_code == 204
    assert response.content == b""


def test_delete_note_cascades_to_its_passages(client, db_session):
    key = make_api_key(db_session, "acme")
    note = make_note(db_session, "acme")
    note_id = note.id
    for ordinal in range(3):
        make_passage(db_session, note, ordinal=ordinal, content=f"passage {ordinal}")

    response = _delete(client, key, note_id)

    assert response.status_code == 204
    db_session.expire_all()
    remaining = db_session.scalars(
        select(Passage).where(Passage.note_id == note_id)
    ).all()
    assert remaining == []


def test_delete_note_returns_404_for_an_unknown_id(client, db_session):
    key = make_api_key(db_session, "acme")

    response = _delete(client, key, 999_999)

    assert response.status_code == 404
    assert response.json() == {"detail": "Note not found"}


def test_deleting_twice_returns_404_the_second_time(client, db_session):
    key = make_api_key(db_session, "acme")
    note = make_note(db_session, "acme")
    note_id = note.id

    first = _delete(client, key, note_id)
    second = _delete(client, key, note_id)

    assert first.status_code == 204
    assert second.status_code == 404
    assert second.json() == {"detail": "Note not found"}


def test_delete_note_rejects_a_non_integer_id(client, db_session):
    key = make_api_key(db_session, "acme")

    response = client.delete("/api/v2/notes/not-an-int", headers={"X-API-Key": key})

    assert response.status_code == 422


def test_a_deleted_notes_passages_no_longer_appear_in_search(client, db_session):
    # The erasure that matters. A cascade that left the passage rows behind
    # would still be a leak through search, and a row count taken on `notes`
    # alone would not see it.
    key = make_api_key(db_session, "acme")
    ingested = _ingest(
        client,
        key,
        content="Body text about the pelicanium isotope and nothing else.\n",
    )
    note_id = ingested.json()["note_id"]

    before = _search(client, key, "pelicanium")
    assert before.status_code == 200
    assert [r["note_id"] for r in before.json()["results"]] == [note_id]

    assert _delete(client, key, note_id).status_code == 204

    after = _search(client, key, "pelicanium")
    assert after.status_code == 200
    assert after.json()["results"] == []


def test_the_identity_is_reusable_after_a_delete(client, db_session):
    key = make_api_key(db_session, "acme")

    first = _ingest(client, key)
    first_id = first.json()["note_id"]
    assert _delete(client, key, first_id).status_code == 204

    # Same (scope, path_or_uri): the unique constraint must not remember the
    # deleted row.
    second = _ingest(client, key)

    assert second.status_code == 200
    second_id = second.json()["note_id"]
    assert second_id != first_id
    assert _get(client, key, second_id).status_code == 200
