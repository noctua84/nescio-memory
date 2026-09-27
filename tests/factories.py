"""Row builders for the integration suite.

These write through the test's own session, so the rows live inside the
transaction that the `connection` fixture rolls back afterwards. They flush
rather than commit: flushing makes the rows visible to subsequent queries on
the same session, including the auth dependency, without ending anything.
"""
from datetime import datetime, timezone

from sqlalchemy.orm import Session

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
) -> Learning:
    """Insert a Learning row directly, bypassing the ingest endpoint.

    client_name is set explicitly here precisely because this bypasses
    LearningRepository.add, which is what stamps it in production. Seeding
    another client's data is the whole point of the isolation tests.
    """
    row = Learning(
        client_name=client_name,
        repo_name=repo_name,
        file_path=file_path,
        content=content,
        meta={
            "file_name": file_path.rsplit("/", 1)[-1],
            "relative_path": file_path,
            "chunk_index": chunk_index,
        },
        embedding=fake_embedding(content) if embedding is None else embedding,
    )
    db.add(row)
    db.flush()
    return row
