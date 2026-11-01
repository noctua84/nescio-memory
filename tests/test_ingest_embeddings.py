"""POST /api/v1/ingest: what it hands the embedder, and what an embedding
failure leaves behind.

Split out of tests/test_ingest.py only to keep each module under the length
the plan asks for; the request helpers are repeated here rather than shared,
so neither module depends on the other.

The route embeds through get_embeddings, ONE batched call per ingest carrying
every passage's embedding input in order. These tests patch that name in the
route's own namespace (app.api.v1.ingest), the binding the route actually
calls -- see the `client` fixture in tests/conftest.py for why patching
app.core.embeddings would have no effect.
"""
from sqlalchemy import select

from app.api.v1 import ingest as ingest_module
from app.config import settings
from app.core.errors import EmbeddingBackendError
from app.models.note import Note
from app.models.passage import Passage
from tests.factories import make_api_key

# Short enough to stay a single passage (under the splitter's TARGET_CHARS).
BODY = "The marmoset throughput ledger for the quarter, with no headings at all."

FRONTMATTER_NOTE = (
    "---\nname: Quokka Ledger\n"
    "description: Pangolin accounting commentary\ntype: adr\n---\n\n"
    "The marmoset throughput ledger. [Source: empirical — 2026-03-04]\n"
)


def _ingest(client, key, **overrides):
    payload = {"scope": "repo_a", "path_or_uri": "docs/note.md", "content": BODY}
    payload.update(overrides)
    return client.post("/api/v1/ingest", json=payload, headers={"X-API-Key": key})


def _structured_body(marker: str) -> str:
    """A body over TARGET_CHARS whose four `##` sections each clear
    MIN_SECTION_CHARS, so the splitter yields four passages rather than one."""
    return "\n\n".join(
        f"## Section {i}\n\n{marker} section {i}. " + "filler text here. " * 30
        for i in range(4)
    )


def _note(db, note_id) -> Note:
    db.expire_all()
    return db.execute(select(Note).where(Note.id == note_id)).scalar_one()


def _passages(db, note_id):
    db.expire_all()
    return db.execute(
        select(Passage).where(Passage.note_id == note_id).order_by(Passage.ordinal)
    ).scalars().all()


def _recorder(monkeypatch):
    """Replace the route's get_embeddings with one that records each batch:
    one entry per call, holding the list of texts that call was given."""
    calls: list[list[str]] = []

    def record(texts: list[str]) -> list[list[float]]:
        calls.append(list(texts))
        return [[0.0] * (settings.embedding_dimension - 1) + [1.0] for _ in texts]

    monkeypatch.setattr(ingest_module, "get_embeddings", record)
    return calls


def test_the_description_is_not_part_of_the_embedding_input(
    client, db_session, monkeypatch
):
    # Decision 6, asserted rather than assumed.
    key = make_api_key(db_session, "acme")
    calls = _recorder(monkeypatch)

    _ingest(client, key, content=FRONTMATTER_NOTE)

    # One call for the whole note, carrying the single passage's input.
    assert len(calls) == 1
    seen = calls[0]
    assert len(seen) == 1
    assert all("Pangolin" not in recorded for recorded in seen)
    # The name IS part of it, so this is not passing merely because nothing
    # was recorded in a usable form.
    assert any("Quokka Ledger" in recorded for recorded in seen)


def test_an_embedding_failure_leaves_the_previous_passages_intact(
    client, db_session, monkeypatch
):
    key = make_api_key(db_session, "acme")
    first = _ingest(client, key).json()
    before = [
        (passage.id, passage.ordinal, passage.content)
        for passage in _passages(db_session, first["note_id"])
    ]

    def boom(texts: list[str]) -> list[list[float]]:
        raise EmbeddingBackendError("backend down")

    monkeypatch.setattr(ingest_module, "get_embeddings", boom)
    response = _ingest(client, key, content="a different body entirely")

    assert response.status_code == 503
    assert response.headers["Retry-After"]
    # No commit happened on the failing request, so the route's own writes are
    # discarded exactly as a real request teardown would discard them.
    db_session.rollback()
    assert [
        (passage.id, passage.ordinal, passage.content)
        for passage in _passages(db_session, first["note_id"])
    ] == before
    assert _note(db_session, first["note_id"]).body == BODY.rstrip()


def test_a_skipped_ingest_spends_no_embeddings(client, db_session, monkeypatch):
    # The economy the lock ordering exists for: the skip check runs before any
    # embedding is requested. A multi-passage note, so the first ingest also
    # shows the batching: ONE call carrying every passage, not one per passage.
    key = make_api_key(db_session, "acme")
    calls = _recorder(monkeypatch)
    body = _structured_body("alpha")
    first = _ingest(client, key, content=body).json()
    assert first["passages"] > 1
    assert [len(batch) for batch in calls] == [first["passages"]]

    assert _ingest(client, key, content=body).json()["skipped"] is True
    assert len(calls) == 1


def test_an_embedding_failure_on_a_first_ingest_persists_no_note(
    client, db_session, monkeypatch
):
    # The route upserts the note row BEFORE it embeds, so this is the case
    # where a failure has something of its own to leave behind.
    key = make_api_key(db_session, "acme")

    def boom(texts: list[str]) -> list[list[float]]:
        raise EmbeddingBackendError("backend down")

    monkeypatch.setattr(ingest_module, "get_embeddings", boom)
    response = _ingest(client, key, path_or_uri="docs/never-stored.md")

    assert response.status_code == 503
    db_session.rollback()
    db_session.expire_all()
    assert db_session.execute(
        select(Note.id).where(Note.path_or_uri == "docs/never-stored.md")
    ).all() == []
    assert db_session.execute(
        select(Passage.id).where(Passage.client_name == "acme")
    ).all() == []
