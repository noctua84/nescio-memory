"""Row builders for the integration suite.

These write through the test's own session, so the rows live inside the
transaction that the `connection` fixture rolls back afterwards. They flush
rather than commit: flushing makes the rows visible to subsequent queries on
the same session, including the auth dependency, without ending anything.
"""
from datetime import date, datetime, timezone
from typing import Any

from sqlalchemy import func, literal_column
from sqlalchemy.orm import Session

from app.config import settings
from app.core.pipeline import (
    content_hash as compute_content_hash,
    fts_lead_a,
    fts_lead_c,
    pipeline_version as compute_pipeline_version,
)
from app.core.security import _hash_key
from app.models.api_key import ApiKey
from app.models.learning import Learning
from app.models.note import Note
from app.models.passage import Passage
from tests.fakes import fake_embedding


def make_api_key(
    db: Session,
    client_name: str,
    plaintext: str | None = None,
    revoked: bool = False,
) -> str:
    """Insert an ApiKey row; return the plaintext key to send as X-API-Key.

    Hashing goes through the application's own _hash_key rather than a
    reimplementation, so that changing the hash in production breaks these
    tests instead of silently leaving them passing against a stale scheme.
    """
    plaintext = plaintext or f"nm_test_{client_name}"
    db.add(
        ApiKey(
            key_prefix=plaintext[:15],
            key_hash=_hash_key(plaintext),
            client_name=client_name,
            revoked_at=datetime.now(timezone.utc) if revoked else None,
        )
    )
    db.flush()
    return plaintext


def make_learning(
    db: Session,
    client_name: str,
    repo_name: str = "repo_a",
    file_path: str = "docs/note.md",
    content: str = "some remembered content",
    embedding: list[float] | None = None,
    chunk_index: int = 0,
    record_window: bool = True,
) -> Learning:
    """Insert a Learning row directly, bypassing the ingest endpoint.

    client_name is set explicitly here precisely because this bypasses
    LearningRepository.add, which is what stamps it in production. Seeding
    another client's data is the whole point of the isolation tests.

    `record_window` controls whether meta carries the chunk_size/chunk_overlap
    the chunk was cut with, the way app/api/v1/ingest.py now stamps them:

    - True (the default) mirrors what ingest writes today, so a factory-seeded
      note resolves to a RECORDED window in app/core/context.py's
      _resolve_window (window_recorded=True) exactly as an API-ingested one
      does. This is the default precisely so that a test which does not care
      about window provenance gets production's shape rather than an obsolete
      one.
    - False writes the pre-provenance meta -- no window keys at all -- which
      is the ONLY way to reach the legacy fallback path in
      app/core/context.py's _resolve_window (window_recorded=False, meaning
      the window is the service's current configuration rather than a recorded
      fact). That path is where invariant 0 and overlap agreement still do
      their work, since a note whose rows record their own window is immune to
      a later reconfiguration. Rows seeded this way stand in for rows written
      before the field existed.

    The values stamped come from `settings` at call time, matching ingest.
    Tests that monkeypatch the window must therefore seed BEFORE patching if
    they want rows recording the original window.
    """
    meta = {
        "file_name": file_path.rsplit("/", 1)[-1],
        "relative_path": file_path,
        "chunk_index": chunk_index,
    }
    if record_window:
        meta["chunk_size"] = settings.chunk_size
        meta["chunk_overlap"] = settings.chunk_overlap

    row = Learning(
        client_name=client_name,
        repo_name=repo_name,
        file_path=file_path,
        content=content,
        meta=meta,
        embedding=fake_embedding(content) if embedding is None else embedding,
    )
    db.add(row)
    db.flush()
    return row


def make_note(
    db: Session,
    client_name: str,
    *,
    scope: str = "repo_a",
    path_or_uri: str = "docs/note.md",
    name: str | None = None,
    description: str | None = None,
    type_: str | None = None,
    provenance_kind: str | None = None,
    provenance_date: date | None = None,
    status: str | None = None,
    body: str = "some remembered content",
    content_hash: str | None = None,
    pipeline_version: str | None = None,
    meta: dict[str, Any] | None = None,
) -> Note:
    """Insert a Note row directly, bypassing POST /api/v2/ingest.

    client_name is set explicitly here precisely because this bypasses
    NoteRepository.upsert_note, which is what stamps it from the authenticated
    client in production. Seeding another client's data is the whole point of
    the v2 isolation tests, and no route will ever write a tenant it was not
    authenticated as.

    `content_hash` and `pipeline_version` default to exactly what the ingest
    route would have written -- content_hash(body) and pipeline_version() --
    rather than to placeholder strings. That makes a factory-seeded note
    skip-equivalent to an API-ingested one: re-ingesting the same body without
    `force` takes the skip path, as it does in production. This is the default
    so that a test which does not care about the skip path still gets
    production's shape; a test that wants a STALE note passes an explicit
    mismatching value, which is the only way to reach the re-ingest path from
    the factory.

    pipeline_version() reads settings at call time, matching the route, so a
    test that monkeypatches the embedding backend or dimension must seed BEFORE
    patching if it wants a note recording the original version.

    Flushes rather than commits, like the builders above: the rows become
    visible to subsequent queries on the same session without ending the
    transaction the `connection` fixture rolls back.
    """
    row = Note(
        client_name=client_name,
        scope=scope,
        path_or_uri=path_or_uri,
        name=name,
        description=description,
        type=type_,
        provenance_kind=provenance_kind,
        provenance_date=provenance_date,
        status=status,
        body=body,
        content_hash=compute_content_hash(body) if content_hash is None else content_hash,
        pipeline_version=(
            compute_pipeline_version() if pipeline_version is None else pipeline_version
        ),
        meta={} if meta is None else meta,
    )
    db.add(row)
    db.flush()
    return row


def make_passage(
    db: Session,
    note: Note,
    *,
    ordinal: int = 0,
    heading_path: str | None = None,
    content: str = "a remembered passage",
    embedding: list[float] | None = None,
    client_name: str | None = None,
) -> Passage:
    """Insert a Passage row directly, bypassing NoteRepository.replace_passages.

    `tsv` is computed here with the SAME three weighted sections the repository
    writes -- A = fts_lead_a(note.name, heading_path), B = content,
    C = fts_lead_c(note.description), all under the 'english' configuration --
    and the lead strings come from app.core.pipeline rather than being
    reimplemented, so a change to that contract breaks these tests instead of
    leaving them passing against a stale shape.

    Computing it matters twice over. tsv is NOT NULL, so leaving it out fails
    the insert; but worse, a wrong-but-valid tsv would make a factory-seeded
    passage invisible to (or mis-ranked by) FTS and quietly weaken every search
    test that relies on this builder. Omitting the C section specifically would
    make the description-ranking tests untestable from the factory, since
    description reaches the index only through that section.

    The weight labels go through literal_column, not bound parameters, for the
    reason app/repositories/note.py:_weighted documents: setweight's second
    argument is Postgres's `"char"` and a bound string renders as VARCHAR, for
    which no overload exists.

    `client_name` defaults to the note's -- which is what replace_passages
    stamps, copying it from the note it writes under -- and is overridable
    specifically so a test can attempt a cross-tenant passage and assert the
    composite FK (note_id, client_name) -> notes(id, client_name) rejects it
    with an IntegrityError on flush.

    Flushes rather than commits, like the builders above.
    """
    row = Passage(
        note_id=note.id,
        client_name=note.client_name if client_name is None else client_name,
        ordinal=ordinal,
        heading_path=heading_path,
        content=content,
        embedding=fake_embedding(content) if embedding is None else embedding,
        tsv=(
            func.setweight(
                func.to_tsvector("english", fts_lead_a(note.name, heading_path)),
                literal_column("'A'"),
            )
            .op("||")(
                func.setweight(
                    func.to_tsvector("english", content), literal_column("'B'")
                )
            )
            .op("||")(
                func.setweight(
                    func.to_tsvector("english", fts_lead_c(note.description)),
                    literal_column("'C'"),
                )
            )
        ),
    )
    db.add(row)
    db.flush()
    return row
