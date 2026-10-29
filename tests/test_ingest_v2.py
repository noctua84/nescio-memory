"""POST /api/v2/ingest: persistence, the versioned skip, force, and embedding.

Shaped like tests/test_ingest.py -- one local `_ingest` helper, then one
behaviour per test named after the behaviour.

Input validation (path_or_uri, the content cap) lives in
tests/test_ingest_v2_validation.py, split out only to keep each module under
the length the plan asks for.
"""
from types import SimpleNamespace

from sqlalchemy import Text, cast, select

from app.api.v2 import ingest as ingest_v2
from app.config import settings
from app.core import pipeline
from app.core.errors import EmbeddingBackendError
from app.models.note import Note
from app.models.passage import Passage
from tests.factories import make_api_key

# Short enough to stay a single passage (under the splitter's TARGET_CHARS) and
# deliberately free of the sentinel words the frontmatter tests use.
BODY = "The marmoset throughput ledger for the quarter, with no headings at all."

FRONTMATTER_NOTE = (
    "---\nname: Quokka Ledger\n"
    "description: Pangolin accounting commentary\ntype: adr\n---\n\n"
    "The marmoset throughput ledger. [Source: empirical — 2026-03-04]\n"
)


def _ingest(client, key, **overrides):
    payload = {"scope": "repo_a", "path_or_uri": "docs/note.md", "content": BODY}
    payload.update(overrides)
    return client.post("/api/v2/ingest", json=payload, headers={"X-API-Key": key})


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


def _passage_ids(db, note_id) -> list[int]:
    return [passage.id for passage in _passages(db, note_id)]


def _tsv_weights(db, passage_id, lexeme) -> set[str]:
    """The weight labels Postgres stored for `lexeme` in that passage's tsv.

    Reads the column back as text (`'quokka':1A,7B`) and returns {'A', 'B'}.
    An unweighted (default D) entry carries no label and yields an empty set,
    so this fails on a tsv built without setweight at all.
    """
    db.expire_all()
    tsv_text = db.execute(
        select(cast(Passage.tsv, Text)).where(Passage.id == passage_id)
    ).scalar_one()
    for entry in tsv_text.split(" "):
        lex, _, positions = entry.partition(":")
        if lex == f"'{lexeme}'":
            return {character for character in positions if character.isalpha()}
    return set()


def _stale_pipeline_settings(monkeypatch, **changes) -> None:
    """Rebind app.core.pipeline's own `settings` to a stand-in.

    pipeline_version() dereferences that module global at call time, so this is
    the binding the route actually reads -- patching a re-exported copy
    elsewhere would pass against a broken implementation. A stand-in rather
    than app.config.settings in place because embedding_dimension is ALSO what
    types the `embedding` bindparam in replace_passages and what the vector(N)
    column was migrated to: changing it globally would fail the INSERT on a
    width mismatch instead of exercising the skip key. In production a
    dimension change comes with a column migration; here only the version
    string moves.
    """
    stand_in = SimpleNamespace(
        embedding_backend=settings.embedding_backend,
        embedding_dimension=settings.embedding_dimension,
    )
    for attribute, value in changes.items():
        setattr(stand_in, attribute, value)
    monkeypatch.setattr(pipeline, "settings", stand_in)


def _recorder(monkeypatch):
    """Replace the route's get_embedding with one that records its input."""
    seen: list[str] = []

    def record(text: str) -> list[float]:
        seen.append(text)
        return [0.0] * (settings.embedding_dimension - 1) + [1.0]

    monkeypatch.setattr(ingest_v2, "get_embedding", record)
    return seen


def test_ingest_v2_persists_a_note_and_its_passages(client, db_session):
    key = make_api_key(db_session, "acme")

    response = _ingest(client, key)

    assert response.status_code == 200
    body = response.json()
    note = _note(db_session, body["note_id"])
    assert (note.client_name, note.scope, note.path_or_uri) == (
        "acme",
        "repo_a",
        "docs/note.md",
    )
    assert note.body == BODY.rstrip()
    assert [passage.content for passage in _passages(db_session, note.id)] == [
        BODY.rstrip()
    ]
    assert body["passages"] == 1


def test_a_first_ingest_writes_passages(client, db_session):
    # The fresh-insert path specifically: a naive skip check would see the
    # matching hash on the row it just inserted and write no passages at all.
    key = make_api_key(db_session, "acme")

    body = _ingest(client, key).json()

    assert body["skipped"] is False
    assert body["passages"] >= 1
    assert len(_passages(db_session, body["note_id"])) == body["passages"]


def test_reingesting_identical_content_is_skipped(client, db_session):
    key = make_api_key(db_session, "acme")
    first = _ingest(client, key).json()
    before = _passage_ids(db_session, first["note_id"])

    second = _ingest(client, key).json()

    assert second["note_id"] == first["note_id"]
    assert second["skipped"] is True
    # The ids, not merely the count: a rewrite that happened to produce the
    # same number of passages would pass a count assertion.
    assert _passage_ids(db_session, first["note_id"]) == before
    assert second["passages"] == len(before)


def test_reingesting_changed_content_replaces_the_passages(client, db_session):
    key = make_api_key(db_session, "acme")
    first = _ingest(client, key, content=_structured_body("alpha")).json()
    before = set(_passage_ids(db_session, first["note_id"]))

    second = _ingest(client, key, content=_structured_body("beta")).json()

    assert second["note_id"] == first["note_id"]
    assert second["skipped"] is False
    rows = _passages(db_session, first["note_id"])
    assert set(passage.id for passage in rows).isdisjoint(before)
    assert [passage.ordinal for passage in rows] == list(range(len(rows)))


def test_a_splitter_version_bump_forces_a_reingest(client, db_session, monkeypatch):
    key = make_api_key(db_session, "acme")
    first = _ingest(client, key).json()
    stale = _note(db_session, first["note_id"]).pipeline_version

    monkeypatch.setattr(pipeline, "SPLITTER_VERSION", "99")
    second = _ingest(client, key).json()

    assert second["skipped"] is False
    assert _note(db_session, first["note_id"]).pipeline_version != stale


def test_an_fts_version_bump_forces_a_reingest(client, db_session, monkeypatch):
    key = make_api_key(db_session, "acme")
    first = _ingest(client, key).json()
    stale = _note(db_session, first["note_id"]).pipeline_version

    monkeypatch.setattr(pipeline, "FTS_VERSION", "99")
    second = _ingest(client, key).json()

    assert second["skipped"] is False
    assert _note(db_session, first["note_id"]).pipeline_version != stale


def test_an_embed_input_version_bump_forces_a_reingest(client, db_session, monkeypatch):
    key = make_api_key(db_session, "acme")
    first = _ingest(client, key).json()
    stale = _note(db_session, first["note_id"]).pipeline_version

    monkeypatch.setattr(pipeline, "EMBED_INPUT_VERSION", "99")
    second = _ingest(client, key).json()

    assert second["skipped"] is False
    assert _note(db_session, first["note_id"]).pipeline_version != stale


def test_an_embedding_dimension_change_forces_a_reingest(
    client, db_session, monkeypatch
):
    key = make_api_key(db_session, "acme")
    first = _ingest(client, key).json()
    stale = _note(db_session, first["note_id"]).pipeline_version

    _stale_pipeline_settings(
        monkeypatch, embedding_dimension=settings.embedding_dimension + 1
    )
    second = _ingest(client, key).json()

    assert second["skipped"] is False
    assert _note(db_session, first["note_id"]).pipeline_version != stale


def test_force_reingests_unchanged_content(client, db_session):
    key = make_api_key(db_session, "acme")
    first = _ingest(client, key).json()
    before = set(_passage_ids(db_session, first["note_id"]))

    second = _ingest(client, key, force=True).json()

    assert second["note_id"] == first["note_id"]
    assert second["skipped"] is False
    assert set(_passage_ids(db_session, first["note_id"])).isdisjoint(before)


def test_force_does_not_change_the_stored_pipeline_version_or_hash(client, db_session):
    # force is extra work, not a different outcome.
    key = make_api_key(db_session, "acme")
    first = _ingest(client, key).json()
    note = _note(db_session, first["note_id"])
    before = (note.content_hash, note.pipeline_version)

    _ingest(client, key, force=True)

    after = _note(db_session, first["note_id"])
    assert (after.content_hash, after.pipeline_version) == before


def test_force_defaults_to_false(client, db_session):
    key = make_api_key(db_session, "acme")
    _ingest(client, key)

    assert _ingest(client, key).json()["skipped"] is True


def test_the_note_row_carries_the_frontmatter_and_provenance(client, db_session):
    key = make_api_key(db_session, "acme")

    body = _ingest(
        client, key, content=FRONTMATTER_NOTE, status="approved",
        meta={"origin": "test"},
    ).json()

    note = _note(db_session, body["note_id"])
    assert note.name == "Quokka Ledger"
    assert note.description == "Pangolin accounting commentary"
    assert note.type == "adr"
    assert note.provenance_kind == "empirical"
    assert note.provenance_date.isoformat() == "2026-03-04"
    assert note.status == "approved"
    assert note.meta == {"origin": "test"}


def test_scope_defaults_to_the_empty_string(client, db_session):
    # '' rather than NULL, because a NULL would defeat the uniqueness of
    # uq_notes_client_scope_path.
    key = make_api_key(db_session, "acme")

    response = client.post(
        "/api/v2/ingest",
        json={"path_or_uri": "docs/note.md", "content": BODY},
        headers={"X-API-Key": key},
    )

    assert response.status_code == 200
    assert _note(db_session, response.json()["note_id"]).scope == ""


def test_repo_name_is_accepted_as_an_alias_for_scope(client, db_session):
    key = make_api_key(db_session, "acme")

    response = client.post(
        "/api/v2/ingest",
        json={"repo_name": "repo_b", "path_or_uri": "docs/note.md", "content": BODY},
        headers={"X-API-Key": key},
    )

    assert response.status_code == 200
    assert _note(db_session, response.json()["note_id"]).scope == "repo_b"


def test_passages_carry_an_embedding_of_the_configured_width(client, db_session):
    key = make_api_key(db_session, "acme")

    body = _ingest(client, key).json()

    for passage in _passages(db_session, body["note_id"]):
        assert len(passage.embedding) == settings.embedding_dimension


def test_passages_carry_a_three_weight_tsv(client, db_session):
    # The weighting itself, not merely a populated column: a length(tsv) > 0
    # assertion would pass on an unweighted one.
    key = make_api_key(db_session, "acme")

    body = _ingest(client, key, content=FRONTMATTER_NOTE).json()

    passage_id = _passage_ids(db_session, body["note_id"])[0]
    # 'quokka' comes only from the note's name (A), 'marmoset' only from the
    # passage content (B), 'pangolin' only from the description (C).
    assert "A" in _tsv_weights(db_session, passage_id, "quokka")
    assert "B" in _tsv_weights(db_session, passage_id, "marmoset")
    assert "C" in _tsv_weights(db_session, passage_id, "pangolin")


def test_a_note_without_a_name_or_description_still_gets_a_usable_tsv(client, db_session):
    # The NULL-concatenation hazard on both the A and C sections: in Postgres
    # 'x' || NULL is NULL, so composing the leads in SQL would blank the whole
    # tsv for a note with no frontmatter.
    key = make_api_key(db_session, "acme")

    body = _ingest(client, key).json()

    note = _note(db_session, body["note_id"])
    assert (note.name, note.description) == (None, None)
    passage_id = _passage_ids(db_session, note.id)[0]
    assert "B" in _tsv_weights(db_session, passage_id, "marmoset")
    assert "B" in _tsv_weights(db_session, passage_id, "ledger")


def test_the_description_is_not_part_of_the_embedding_input(
    client, db_session, monkeypatch
):
    # Decision 6, asserted rather than assumed.
    key = make_api_key(db_session, "acme")
    seen = _recorder(monkeypatch)

    _ingest(client, key, content=FRONTMATTER_NOTE)

    assert seen
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

    def boom(text: str) -> list[float]:
        raise EmbeddingBackendError("backend down")

    monkeypatch.setattr(ingest_v2, "get_embedding", boom)
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


def test_a_skipped_ingest_spends_no_embeddings(client, db_session, monkeypatch):
    # The economy the lock ordering exists for: the skip check runs before any
    # embedding is requested.
    key = make_api_key(db_session, "acme")
    seen = _recorder(monkeypatch)
    _ingest(client, key)
    after_first = len(seen)

    assert _ingest(client, key).json()["skipped"] is True
    assert after_first > 0
    assert len(seen) == after_first
