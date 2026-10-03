"""Row builders for the integration suite.

These write through the test's own session, so the rows live inside the
transaction that the `connection` fixture rolls back afterwards. They flush
rather than commit: flushing makes the rows visible to subsequent queries on
the same session, including the auth dependency, without ending anything.
"""
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.config import settings
from app.core.security import _hash_key
from app.models.api_key import ApiKey
from app.models.learning import Learning
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
